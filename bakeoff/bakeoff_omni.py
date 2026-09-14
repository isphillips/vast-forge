"""
Bake-off 2: Hunyuan3D-Omni bounding-box control vs the shipped Hunyuan3D 2.1 engine, on a forge node built from
Dockerfile.hunyuan (system python has torch 2.5.1 + the 2.1 paint stage at /app/hy; Omni cloned to /opt/omni).

Omni replaces only the SHAPE stage: shape (Omni, bbox-conditioned) -> cleanup -> FaceReducer 120k -> the SAME paint
stage as engine_hunyuan (6 views @ 512) -> GLB. So texture quality is held constant and the comparison is geometry.

Runs per image (tags match the first bake-off; the handoff's car/turtle URLs were swapped, so tag "car" IS the
Leonardo mask and tag "turtle" IS the car):
    <tag>_omni_free   Omni, no control (does Omni match 2.1 unconstrained?)
    <tag>_omni_bbox   Omni, bbox = [w, h, d] with w:h from the cut-out's alpha bounds and d from a per-category prior
    car_omni_point    the mask only: point-cloud control with a canonical head-proxy ellipsoid (the "skeleton for
                      the face" idea in the only form Omni supports)
Two processes: Omni's `hy3dshape` package shadows 2.1's, so shapes are made by a child process with /opt/omni on
sys.path, then painted here with the image's own 2.1 paint pipeline.

    python3 /app/bakeoff/bakeoff_omni.py [--skip-paint] [--images '{"tag": ["url", "prompt", depth_ratio]}']
Outputs in /tmp/bake2/: <tag>_<variant>.glb, s_<tag>_<variant>.glb.gz, bake2_stats.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

OUT = "/tmp/bake2"
os.makedirs(OUT, exist_ok=True)
OMNI = "/opt/omni/Hunyuan3D-Omni"
sys.path.insert(0, "/app/bakeoff")
import bakeoff  # noqa: E402 — fetch_image / mesh_stats / slim / obj_to_glb from the first bake-off
bakeoff.OUT = OUT

# tag: (image url, label, depth prior as a fraction of width — the bake-off-1 2.1 outputs and real objects:
#       glasses+mustache ~0.4, a mask ~0.6 (a full head would be ~1.15), a car front view = length/width 2.4)
IMAGES = {
    "mustache": ("https://v3b.fal.media/files/b/0aa99f56/YMWJSfckiHqpe9SspVjYb_bFA6r8CP.png", "mustache and glasses", 0.4),
    "car":      ("https://v3b.fal.media/files/b/0aaa25cd/KDMi0X20iv_kcHt1RhtbZ_aJZ1lRL3.png", "Leonardo mask", 0.6),
    "turtle":   ("https://v3b.fal.media/files/b/0aaa23fa/hTUPe579zxjlmu8kNDyOQ_SMfjbmD3.png", "Autumn Jones car", 2.4),
}


def log(msg):
    print(f"[bake2] {msg}", flush=True)


def alpha_wh(tag):
    """Cut the background (same rembg as the engine) and return the subject's width:height from the alpha bounds."""
    import numpy as np
    from PIL import Image
    from rembg import remove
    p = f"{OUT}/{tag}.png"
    rgba = remove(Image.open(p).convert("RGB"))
    rgba.save(f"{OUT}/{tag}_rgba.png")
    a = np.asarray(rgba)[:, :, 3] > 64
    ys, xs = np.nonzero(a)
    w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    return float(w), float(h)


def bbox_for(tag, depth_ratio):
    """Omni's bbox is [x, y, z] extents with the longest axis = 1 (demos: a flat drawing is [0.96, 1.0, 0.356]);
    our front faces +Z, so z is depth. w:h from the image, depth = depth_ratio * width."""
    w, h = alpha_wh(tag)
    b = [w, h, depth_ratio * w]
    m = max(b)
    b = [round(v / m, 4) for v in b]
    log(f"{tag}: alpha {w:.0f}x{h:.0f} px, depth prior {depth_ratio} -> bbox {b}")
    return b


# ── child process: Omni shapes ─────────────────────────────────────────────────────────────────────────────
OMNI_CHILD = r'''
import json, os, sys, time, math
sys.path.insert(0, "%(omni)s"); os.chdir("%(omni)s")
import numpy as np, torch, trimesh
from PIL import Image
from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
from hy3dshape.postprocessors import FloaterRemover, DegenerateFaceRemover, FaceReducer
jobs = json.load(open(sys.argv[1]))
pipe = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-Omni")
stats = {}
for j in jobs:
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    kw = {}   # the image goes in as an RGBA path (the preprocessor reads the alpha as the mask), like inference.py
    if j.get("bbox"):
        kw["bbox"] = torch.FloatTensor(j["bbox"]).unsqueeze(0).unsqueeze(0).to(pipe.device).to(pipe.dtype)
    if j.get("point"):
        pts = np.load(j["point"]); kw["point"] = torch.from_numpy(pts.astype(np.float32)).unsqueeze(0).to(pipe.device).to(pipe.dtype)
    g = torch.Generator("cuda").manual_seed(j.get("seed", 1234))
    res = pipe(image=j["image"], num_inference_steps=50, octree_resolution=%(octree)d, mc_level=0, guidance_scale=4.5,
               generator=g, fast_decode=True, **kw)
    mesh = res["shapes"][0][0]
    raw = len(mesh.faces)
    mesh = FloaterRemover()(mesh); mesh = DegenerateFaceRemover()(mesh); mesh = FaceReducer()(mesh, max_facenum=120000)
    ext = (mesh.bounds[1] - mesh.bounds[0]).tolist()
    mesh.export(j["out"])
    stats[j["name"]] = {"omni_shape_s": round(time.time() - t0), "omni_raw_faces": raw,
                        "omni_shape_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                        "extents": [round(v, 3) for v in ext], "depth_over_width": round(ext[2] / max(ext[0], 1e-6), 2),
                        "control": "point" if j.get("point") else ("bbox" if j.get("bbox") else "none"), "bbox": j.get("bbox")}
    print("[omni] %%s: %%ds raw %%d -> %%d faces, extents %%s, d/w %%.2f, peak %%.1f GB" %% (
        j["name"], stats[j["name"]]["omni_shape_s"], raw, len(mesh.faces), stats[j["name"]]["extents"],
        stats[j["name"]]["depth_over_width"], stats[j["name"]]["omni_shape_peak_gb"]), flush=True)
json.dump(stats, open(sys.argv[2], "w"), indent=2)
print("[omni] DONE", flush=True)
'''


def head_proxy_points(n=81920, path=f"{OUT}/head_proxy.npy"):
    """A canonical head-shaped ellipsoid (w:h:d = 1 : 1.25 : 1.1), front at +Z, fitted to Omni's 0.98 canonical
    scale like inference.py's normalize_mesh. Surface points only; the model conforms the mask to it."""
    import numpy as np
    import trimesh
    e = trimesh.creation.icosphere(subdivisions=6)          # 40,962 vertices, used as inference.py uses mesh.vertices
    e.apply_scale([1.0, 1.25, 1.1])
    lo, hi = e.bounds
    e.apply_translation(-(lo + hi) / 2)
    e.apply_scale(1.0 / max(hi - lo) * 2 * 0.98)            # normalize_mesh(scale=0.98): longest axis in [-0.98, 0.98]
    np.save(path, np.asarray(e.vertices, dtype=np.float32))
    return path


def run_omni_shapes(jobs, octree):
    spec = f"{OUT}/omni_jobs.json"; st = f"{OUT}/omni_stats.json"
    json.dump(jobs, open(spec, "w"))
    script = f"{OUT}/omni_child.py"
    open(script, "w").write(OMNI_CHILD % {"omni": OMNI, "octree": octree})
    r = subprocess.run([sys.executable, script, spec, st], env={**os.environ, "HF_HOME": os.environ.get("HF_HOME", "/models"),
                       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    if r.returncode != 0:
        raise SystemExit(f"omni child failed ({r.returncode})")
    return json.load(open(st))


# ── paint with the image's own 2.1 stage (same settings as engine_hunyuan) ─────────────────────────────────
def paint_all(names, images):
    os.chdir("/app/hy"); sys.path[:0] = ["/app/hy", "/app/hy/hy3dshape", "/app/hy/hy3dpaint"]
    from torchvision_fix import apply_fix; apply_fix()
    import torch
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    paint = Hunyuan3DPaintPipeline(conf)
    stats = {}
    for name in names:
        tag = name.split("_")[0]
        out = f"{OUT}/{name}.glb"
        if os.path.exists(out):
            continue
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        obj = f"{OUT}/{name}.obj"
        paint(f"{OUT}/{name}_shape.glb", image_path=f"{OUT}/{tag}.png", output_mesh_path=obj, save_glb=False, use_remesh=False)
        stats[name] = {"paint_s": round(time.time() - t0), "paint_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        bakeoff.obj_to_glb(obj, out)
        log(f"{name}: paint {stats[name]['paint_s']}s, peak {stats[name]['paint_peak_gb']} GB")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help='JSON {"tag": ["url", "label", depth_ratio]} to add/override')
    ap.add_argument("--octree", type=int, default=384)      # 2.1 engine uses 384; Omni's demo uses 512
    ap.add_argument("--skip-paint", action="store_true")
    ap.add_argument("--no-free", action="store_true", help="skip the unconstrained Omni runs")
    a = ap.parse_args()
    images = dict(IMAGES)
    if a.images:
        images.update({k: tuple(v) for k, v in json.loads(a.images).items()})
    for tag, (url, _, _) in images.items():
        bakeoff.fetch_image(tag, url)

    jobs = []
    boxes = {tag: bbox_for(tag, depth) for tag, (_, _, depth) in images.items()}   # also writes <tag>_rgba.png
    for tag, (_, label, depth) in images.items():
        img = f"{OUT}/{tag}_rgba.png"
        if not a.no_free:
            jobs.append({"name": f"{tag}_omni_free", "image": img, "out": f"{OUT}/{tag}_omni_free_shape.glb"})
        jobs.append({"name": f"{tag}_omni_bbox", "image": img, "out": f"{OUT}/{tag}_omni_bbox_shape.glb", "bbox": boxes[tag]})
    if "car" in images:   # the mask: point-cloud head proxy
        jobs.append({"name": "car_omni_point", "image": f"{OUT}/car_rgba.png", "out": f"{OUT}/car_omni_point_shape.glb",
                     "point": head_proxy_points()})
    jobs = [j for j in jobs if not os.path.exists(j["out"])]
    stats = {}
    if jobs:
        stats.update(run_omni_shapes(jobs, a.octree))
    names = [f"{t}_omni_free" for t in images if not a.no_free] + [f"{t}_omni_bbox" for t in images] + (["car_omni_point"] if "car" in images else [])
    if not a.skip_paint:
        for n, d in paint_all(names, images).items():
            stats.setdefault(n, {}).update(d)
    for n in names:
        p = f"{OUT}/{n}.glb"
        if os.path.exists(p):
            stats.setdefault(n, {}).update(bakeoff.mesh_stats(p))
            bakeoff.slim(p, f"{OUT}/s_{n}.glb.gz")
    prev = f"{OUT}/bake2_stats.json"
    if os.path.exists(prev):
        old = json.load(open(prev))
        for k, d in stats.items():
            old.setdefault(k, {}).update(d)
        stats = old
    json.dump(stats, open(prev, "w"), indent=2)
    log("DONE\n" + json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
