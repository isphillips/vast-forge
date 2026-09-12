"""
Hunyuan3D 2.1 engine for vast-forge (FORGE_ENGINE=hunyuan; built by Dockerfile.hunyuan).

Same contract as server.run_trellis: PIL.Image -> mobile-ready GLB bytes. Stages:
    rembg -> shape (DiT flow matching, 50 steps, octree 384) -> floater/degenerate cleanup -> FaceReducer to
    TARGET_FACES -> depth sanity re-roll -> paint (6 views @ 512, Real-ESRGAN upscale, PBR bake) -> GLB.

Why Hunyuan (bake-off 2026-09-12, RTX 3090, same three images as TRELLIS.2): every export was a single closed
surface (0 open edges, 1 piece) against 345-4,312 pieces from TRELLIS; 151-169 s per facet against 245-410 s;
peak 7.6 GB in the shape stage and 13.8 GB in paint, so the 24 GB card never needs a retry ladder.

Its one observed failure mode is a wrong reading rather than a broken surface: on a frontal image with ambiguous
depth the shape model can extrude a body behind the subject (the mask came out 3.9x deeper than wide once, against
1.4-1.7x on four fixed-seed re-rolls; a real car is 2.4x). Because the mesh is closed that shows up in one number,
so `_shape` re-rolls with a new seed while depth/width exceeds HY_MAX_DEPTH_RATIO and keeps the least deep result.

Install notes that this module relies on (all applied in Dockerfile.hunyuan): the repo's torchvision_fix must run
before the paint imports (basicsr wants a module torchvision 0.20 removed); setuptools < 81 (the paint stage
imports pkg_resources); bpy has no py3.10 wheel, so the painted OBJ is converted to GLB here with trimesh instead
of the repo's Blender step; the paint pipeline's own remesh is a decimation to a hard-coded 40k faces, so it is
called with use_remesh=False and the shape mesh is reduced to TARGET_FACES beforehand.
"""
import gc
import os
import random
import shutil
import sys
import tempfile
import time

HY_ROOT = os.environ.get("HY_ROOT", "/app/hy")
HY_MODEL = os.environ.get("HY_MODEL_ID", "tencent/Hunyuan3D-2.1")
TARGET_FACES = int(os.environ.get("TARGET_FACES", "120000"))
TEXTURE_SIZE = int(os.environ.get("TEXTURE_SIZE", "1024"))
STEPS = int(os.environ.get("HY_STEPS", "50"))
GUIDANCE = float(os.environ.get("HY_GUIDANCE", "5.0"))
OCTREE = int(os.environ.get("HY_OCTREE", "384"))
VIEWS = int(os.environ.get("HY_VIEWS", "6"))            # paint views (6-9); 6 @ 512 measured at 13.8 GB peak
RES = int(os.environ.get("HY_RES", "512"))              # paint view resolution (512 or 768)
MAX_DEPTH_RATIO = float(os.environ.get("HY_MAX_DEPTH_RATIO", "3.0"))
SHAPE_RETRIES = int(os.environ.get("HY_SHAPE_RETRIES", "2"))
# Park the shape model on the CPU while painting (and vice versa) so the two stages' peaks don't add up on a
# 24 GB card: ~1-2 s of PCIe traffic per stage against a resident sum near the card's limit.
SWAP_STAGES = os.environ.get("HY_SWAP_STAGES", "1") == "1"

_REMBG = None
_SHAPE = None
_PAINT = None


def _log(msg):
    print(f"[hunyuan] {msg}", flush=True)


def loaded() -> bool:
    return _SHAPE is not None and _PAINT is not None


def load():
    """Load both stages once (called under the server's GPU lock). First load on a cold node downloads the
    weights (~10 GB Hunyuan3D-2.1 + ~4.5 GB facebook/dinov2-giant) into HF_HOME=/models."""
    global _REMBG, _SHAPE, _PAINT
    if loaded():
        return
    # The paint pipeline reads its config, custom pipeline and Real-ESRGAN checkpoint relative to the repo root,
    # and the repo's own demo puts both sub-packages on sys.path rather than installing them.
    os.chdir(HY_ROOT)
    for p in (HY_ROOT, f"{HY_ROOT}/hy3dshape", f"{HY_ROOT}/hy3dpaint"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from torchvision_fix import apply_fix
    apply_fix()
    from hy3dshape.rembg import BackgroundRemover
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    t0 = time.time()
    _REMBG = BackgroundRemover()
    _SHAPE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(HY_MODEL)
    conf = Hunyuan3DPaintConfig(max_num_view=VIEWS, resolution=RES)
    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    _PAINT = Hunyuan3DPaintPipeline(conf)
    if SWAP_STAGES:
        _move(_SHAPE, "cpu")
    _log(f"loaded shape + paint in {time.time() - t0:.0f}s")


def _move(pipeline, device):
    """Best effort: the shape pipeline exposes .to(); if a future version doesn't, stay resident."""
    try:
        pipeline.to(device)
    except Exception as e:  # noqa: BLE001
        _log(f"could not move shape pipeline to {device} ({type(e).__name__}); leaving it resident")


def _shape_once(img_rgba, seed):
    import torch
    from hy3dshape import FaceReducer, FloaterRemover, DegenerateFaceRemover

    g = torch.Generator().manual_seed(seed)
    mesh = _SHAPE(image=img_rgba, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                  octree_resolution=OCTREE, generator=g)[0]
    raw = len(mesh.faces)
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, max_facenum=TARGET_FACES)
    ext = mesh.bounds[1] - mesh.bounds[0]
    ratio = float(ext[2] / max(float(ext[0]), 1e-6))    # depth (Z) over width (X); the front faces +Z
    return mesh, raw, ratio


def _shape(img_rgba):
    """Shape stage with the depth sanity re-roll (see module docstring). Returns the reduced trimesh."""
    best = None
    for attempt in range(SHAPE_RETRIES + 1):
        seed = random.randrange(2 ** 31)
        t0 = time.time()
        mesh, raw, ratio = _shape_once(img_rgba, seed)
        _log(f"shape seed {seed}: {time.time() - t0:.0f}s, raw {raw} -> {len(mesh.faces)} faces, depth/width {ratio:.2f}")
        if best is None or ratio < best[1]:
            best = (mesh, ratio)
        if ratio <= MAX_DEPTH_RATIO:
            return mesh
        if attempt < SHAPE_RETRIES:
            _log(f"depth/width {ratio:.2f} > {MAX_DEPTH_RATIO}: re-rolling the shape stage with a new seed")
    _log(f"keeping the least deep of {SHAPE_RETRIES + 1} attempts (depth/width {best[1]:.2f})")
    return best[0]


def _to_glb(obj_path: str) -> bytes:
    """Painted OBJ (+MTL, albedo/metallic/roughness JPEGs) -> GLB with the albedo only, like the app shades it
    (metalness 0, roughness 0.9; the extra maps are dropped), texture capped at TEXTURE_SIZE, WebP-compressed,
    and the model scaled into the unit cube: Hunyuan normalises its longest axis to [-1, 1], TRELLIS exported into
    [-0.5, 0.5], and the app's facets were sized for the latter."""
    import trimesh
    from PIL import Image

    tm = trimesh.load(obj_path, force="mesh", process=False)
    tex = getattr(tm.visual.material, "image", None)
    if tex is None:
        raise RuntimeError("painted mesh has no albedo texture")
    tex = tex.convert("RGB")
    if max(tex.size) > TEXTURE_SIZE:
        tex = tex.resize((TEXTURE_SIZE, TEXTURE_SIZE), Image.LANCZOS)
    tm.visual.material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=tex, metallicFactor=0.0, roughnessFactor=0.9, name="facet")
    lo, hi = tm.bounds
    tm.apply_translation(-(lo + hi) / 2.0)
    tm.apply_scale(1.0 / max(float((hi - lo).max()), 1e-6))
    try:
        return tm.export(file_type="glb", extension_webp=True)   # WebP textures — the app's loaders read EXT_texture_webp
    except TypeError:
        return tm.export(file_type="glb")


def run(image) -> bytes:
    """PIL.Image (RGB) -> GLB bytes. The caller (server.py) holds the GPU lock."""
    import torch

    load()
    torch.cuda.reset_peak_memory_stats()
    work = tempfile.mkdtemp(prefix="hy_")
    try:
        # The paint stage wants the ORIGINAL image (it cuts the background itself); the shape stage wants RGBA.
        img_png = os.path.join(work, "input.png")
        image.save(img_png)
        img_rgba = _REMBG(image.convert("RGB"))

        t0 = time.time()
        if SWAP_STAGES:
            _move(_SHAPE, "cuda")
        mesh = _shape(img_rgba)
        shape_glb = os.path.join(work, "shape.glb")
        mesh.export(shape_glb)
        del mesh
        if SWAP_STAGES:
            _move(_SHAPE, "cpu")
        gc.collect()
        torch.cuda.empty_cache()
        t1 = time.time()

        obj = os.path.join(work, "textured.obj")
        _PAINT(shape_glb, image_path=img_png, output_mesh_path=obj, save_glb=False, use_remesh=False)
        t2 = time.time()
        glb = _to_glb(obj)
        _log(f"shape {t1 - t0:.0f}s, paint {t2 - t1:.0f}s, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB, "
             f"{len(glb) / 1e6:.1f} MB")
        return glb
    finally:
        shutil.rmtree(work, ignore_errors=True)
        gc.collect()
        torch.cuda.empty_cache()
