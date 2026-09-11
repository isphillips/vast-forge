"""
vast-forge — TRELLIS 2 image→3D service for Vidrip Facet Forge.

Contract (called by the facet-generate edge function):
    POST /generate  { "image_url": "https://.../qwen-image.png" }
      -> { "glbUrl": "https://<r2-public>/forge/<id>.glb" }

The image is made upstream (Qwen-Image-2512 on fal); this service only turns it into a mobile-ready GLB:
TRELLIS 2 -> decimate + flatten to baseColor -> upload to R2. Everything here is complete EXCEPT `run_trellis`,
which holds the TRELLIS 2 inference call to fill from the repo's example (marked clearly below).
"""
import io
import os
import threading
import uuid

import boto3
import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# Serialize all GPU work. /generate is a SYNC FastAPI endpoint, so FastAPI runs it in a threadpool — without this,
# two requests that arrive close together run PIPE.run concurrently on the ONE GPU, doubling peak VRAM → CUDA OOM
# (a single generation fits a 24 GB 3090; two don't). This lock makes generations run one-at-a-time; it also guards
# the one-time model load against two cold requests both trying to load it. Requests queue instead of failing.

# ── config (all from env; see .env.example) ─────────────────────────────────────────────────────────────────
R2_ENDPOINT = os.environ["R2_ENDPOINT"]          # https://<accountid>.r2.cloudflarestorage.com
R2_KEY = os.environ["R2_KEY"]
R2_SECRET = os.environ["R2_SECRET"]
R2_BUCKET = os.environ["R2_BUCKET"]              # e.g. channel-clips
R2_PUBLIC_BASE = os.environ["R2_PUBLIC_BASE"].rstrip("/")  # public https base that serves the bucket
FORGE_TOKEN = os.environ["FORGE_TOKEN"]          # shared secret; must match the edge function
TARGET_FACES = int(os.environ.get("TARGET_FACES", "120000"))  # triangle budget (to_glb decimation_target); 40k lost hair-thin
                                                              # geometry (84% of the surface kept → eyebrow/mustache holes), 120k keeps 93%
TEXTURE_SIZE = int(os.environ.get("TEXTURE_SIZE", "1024"))    # mobile-friendly texture size (repo default is 4096)
# Remesh resolution ladder (2026-09-08). The fork's exporter remeshes with a dual-contour pass whose resolution it
# reads from OVOXEL_DC_MAX_RES at call time (our Dockerfile patch; the fork hard-coded 512 while the model generates
# at 1024). Measured on the 3090 (same generation, 40k faces): 512/no fill = 89 open edges, 96.9% of the raw surface
# kept, 0.8 GB; 768 + fill = 8 open edges, 98.4%, 1.5 GB; 1024 + fill = 59 open edges, 98.8%, 2.5 GB. We run the
# model's full 1024 and step down on a CUDA OOM (1024 → 768 → 512) — a huge object degrades instead of failing the
# job. Generation is already done by then (the fork offloads it), so the exporter never takes the box down.
REMESH_RES = int(os.environ.get("FORGE_REMESH_RES", "1024"))
REMESH_FALLBACKS = [int(x) for x in os.environ.get("FORGE_REMESH_FALLBACKS", "768,512").split(",") if x.strip()]
# Solidify (2026-09-09): close the model's own voxel grid before export so hair-like sheets become solid masses
# (see solidify.py). Radius in voxels; 0 turns it off and exports through the remesh ladder alone. Any failure in
# the solidify path also falls back to the remesh ladder, so a job never dies because of it.
SOLIDIFY_RADIUS = int(os.environ.get("FORGE_SOLIDIFY_RADIUS", "2"))

s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_KEY,
    aws_secret_access_key=R2_SECRET,
    region_name="auto",
)

app = FastAPI(title="vast-forge")

_GPU_LOCK = threading.Lock()


# ── model: TRELLIS.2 (microsoft/TRELLIS.2-4B), loaded ONCE at import so a warm worker keeps it resident ───────
# https://github.com/microsoft/TRELLIS.2 . On Turing (RTX 8000) force fp16 + SDPA/xformers attention (flash-attn
# needs Ampere+); on AMD use a ROCm port. See README.
PIPE = None


def _load_pipeline():
    global PIPE
    if PIPE is not None:
        return
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    # from_pretrained ALSO pulls the GATED encoder facebook/dinov3-vitl16-pretrain-lvd1689m. That download 401s
    # unless HF_TOKEN (a classic read token whose account has accepted the DINOv3 gate) is in the env — huggingface_hub
    # reads it automatically. First load on a cold node downloads ~8-16 GB to HF_HOME=/models; mount a persistent
    # volume there so it survives node restarts. See .env.example.
    PIPE = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    PIPE.cuda()


def _is_cuda_oom(e: BaseException) -> bool:
    # torch raises torch.OutOfMemoryError; cumesh's C++ side surfaces the same condition as a RuntimeError.
    return type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()


def _recover_gpu(torch):
    """Put the GPU back the way a finished job leaves it. The fork's low-VRAM pipeline moves each model to the GPU
    for its stage and back to the CPU afterwards — but only on the happy path. A CUDA OOM mid-stage (seen 2026-09-08
    on a dense object: 20 GB in the shape decoder) left that model stranded on the GPU, so every later request
    failed until a manual restart. Move every model home explicitly (NOT PIPE.cpu(): the fork overrides `to()` to
    only relabel the pipeline's device, which would send the next job to the CPU), then drop the cache."""
    import gc
    try:
        if PIPE is not None:
            for m in PIPE.models.values():
                try:
                    m.cpu()
                except Exception:  # noqa: BLE001
                    pass
    finally:
        gc.collect()
        torch.cuda.empty_cache()


# Generation ladder. A large, dense object can need more decode memory at 1024 than the card has; rather than fail
# the job, retry with a smaller token budget (the cascade then lowers ITS resolution only as far as needed), and
# as a last resort at 512. Each step follows a full GPU recovery.
GEN_FALLBACKS = [
    {},                                   # as configured
    {"max_num_tokens": 24576},            # half the token cap → coarser only where the object is too dense
    {"pipeline_type": "512"},             # last resort: the 512 pipeline (chunkier, always fits)
]


def _generate(image, torch):
    last = None
    for i, kw in enumerate(GEN_FALLBACKS):
        try:
            return PIPE.run(image, **kw)[0]
        except Exception as e:  # noqa: BLE001 — only OOM is retried; everything else propagates
            if not _is_cuda_oom(e):
                _recover_gpu(torch)
                raise
            last = e
            _recover_gpu(torch)
            print(f"[forge] generation hit CUDA OOM (attempt {i + 1}); retrying with {kw or 'lower settings'}", flush=True)
    raise last


def _export_solid(mesh, o_voxel, torch):
    """Solidify path: closing on the voxel grid → marching cubes → to_glb WITHOUT the remesh (the surface is already
    clean), baked with the original colours extended onto the added voxels."""
    import solidify

    sv, sf, closed_coords = solidify.solidify(mesh, radius=SOLIDIFY_RADIUS, log=lambda s: print("[forge] " + s, flush=True))
    attrs, coords = solidify.extend_attrs(mesh, closed_coords, log=lambda s: print("[forge] " + s, flush=True))
    del closed_coords
    return o_voxel.postprocess.to_glb(
        vertices=sv,
        faces=sf,
        attr_volume=attrs,
        coords=coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=TARGET_FACES,
        texture_size=TEXTURE_SIZE,
        remesh=False,
    )


def _export_glb(mesh, o_voxel, torch):
    """Solidify first (when enabled), else/then to_glb at REMESH_RES, stepping down through REMESH_FALLBACKS on CUDA
    OOM. Tensors are cloned per attempt so a failed attempt can't have moved or freed what the retry needs."""
    if SOLIDIFY_RADIUS > 0:
        try:
            return _export_solid(mesh, o_voxel, torch)
        except Exception as e:  # noqa: BLE001 — the remesh ladder below is the safety net
            import traceback
            traceback.print_exc()
            print(f"[forge] solidify export failed ({type(e).__name__}: {str(e)[:120]}); falling back to remesh", flush=True)
            torch.cuda.empty_cache()
    last = None
    for res in [REMESH_RES] + [r for r in REMESH_FALLBACKS if r < REMESH_RES]:
        os.environ["OVOXEL_DC_MAX_RES"] = str(res)
        try:
            return o_voxel.postprocess.to_glb(
                vertices=mesh.vertices.clone(),
                faces=mesh.faces.clone(),
                attr_volume=mesh.attrs.clone(),
                coords=mesh.coords.clone(),
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=TARGET_FACES,   # mobile budget — repo default 1,000,000 is desktop-scale
                texture_size=TEXTURE_SIZE,        # repo default 4096 → drop to 1024 for phones
                remesh=True,                      # clean topology (hole fill restored inside the exporter)
            )
        except Exception as e:  # noqa: BLE001 — anything that isn't an OOM propagates unchanged
            if not _is_cuda_oom(e):
                raise
            last = e
            print(f"[forge] export hit CUDA OOM at remesh {res}; retrying one step down", flush=True)
            torch.cuda.empty_cache()
    raise last


def run_trellis(image) -> bytes:
    """PIL.Image -> mobile-ready GLB bytes via TRELLIS.2. `to_glb` does the decimation + remesh + texture bake
    itself, so there's no separate trimesh pass (re-exporting would strip the WebP textures)."""
    import gc
    import tempfile

    import torch
    import o_voxel

    # One generation on the GPU at a time (see _GPU_LOCK). Concurrent requests queue here instead of racing for VRAM.
    with _GPU_LOCK:
        _load_pipeline()
        # A long-lived warm worker fragments GPU memory across requests, so a later generation can OOM in the mesh
        # decode (CuMesh.get_edges) on a 24 GB card where earlier ones fit. Free the previous run's cache first, and
        # again in `finally` so this run's tensors are released promptly for the next request.
        torch.cuda.empty_cache()
        try:
            mesh = _generate(image, torch)
            glb = _export_glb(mesh, o_voxel, torch)
            path = tempfile.mktemp(suffix=".glb")
            glb.export(path, extension_webp=True)  # WebP textures — the on-device loaders read EXT_texture_webp
            try:
                with open(path, "rb") as fh:
                    return fh.read()
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
        except BaseException:
            # Whatever failed, leave the GPU as a finished job would (no stranded models, no cached blocks).
            _recover_gpu(torch)
            raise
        finally:
            # Drop this run's Python refs (mesh/glb hold GPU tensors) BEFORE emptying the cache, so the VRAM is
            # actually reclaimable for the next queued request.
            mesh = glb = None
            gc.collect()
            torch.cuda.empty_cache()


def r2_put(key: str, data: bytes, content_type: str) -> str:
    s3.put_object(Bucket=R2_BUCKET, Key=key, Body=data, ContentType=content_type)
    return f"{R2_PUBLIC_BASE}/{key}"


def forge_from_url(image_url: str) -> str:
    """image URL → public GLB URL on R2. Shared by the push endpoint (/generate) and the pull worker."""
    from PIL import Image
    r = requests.get(image_url, timeout=30)
    r.raise_for_status()
    image = Image.open(io.BytesIO(r.content)).convert("RGB")
    glb = run_trellis(image)
    return r2_put(f"forge/{uuid.uuid4().hex}.glb", glb, "model/gltf-binary")


# ── pull worker (the dispatcher's node side; see worker.py) ─────────────────────────────────────────────────
# With FORGE_CONTROL_URL set, this process ALSO pulls jobs from the shared queue: heartbeat + claim through the
# forge-worker edge function, forge, hand back the GLB. The /generate push endpoint stays for manual tests and
# for the edge function's inline fallback. Both share PIPE and _GPU_LOCK, so a pulled job and a pushed one never
# run concurrently on the card.
WORKER = None


@app.on_event("startup")
def _start_worker():
    global WORKER
    import worker
    WORKER = worker.start(run_job=forge_from_url, model_loaded=lambda: PIPE is not None)


class GenReq(BaseModel):
    image_url: str


@app.get("/health")
def health():
    import torch
    return {
        "ok": True, "cuda": torch.cuda.is_available(), "model_loaded": PIPE is not None,
        "node": WORKER.id if WORKER else None,
        "busy": bool(WORKER and WORKER.current),
        "jobs_done": WORKER.jobs_done if WORKER else 0,
    }


@app.post("/generate")
def generate(req: GenReq, authorization: str = Header(default="")):
    if authorization != f"Bearer {FORGE_TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")
    try:
        return {"glbUrl": forge_from_url(req.image_url)}
    except HTTPException:
        raise
    except Exception as e:
        # Full traceback to stdout (→ container log) so a 500 is diagnosable without re-running; the HTTP body
        # still carries only a short reason, which the edge function maps to a friendly user message.
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"generate failed: {e}")
