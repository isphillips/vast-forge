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
import uuid

import boto3
import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ── config (all from env; see .env.example) ─────────────────────────────────────────────────────────────────
R2_ENDPOINT = os.environ["R2_ENDPOINT"]          # https://<accountid>.r2.cloudflarestorage.com
R2_KEY = os.environ["R2_KEY"]
R2_SECRET = os.environ["R2_SECRET"]
R2_BUCKET = os.environ["R2_BUCKET"]              # e.g. channel-clips
R2_PUBLIC_BASE = os.environ["R2_PUBLIC_BASE"].rstrip("/")  # public https base that serves the bucket
FORGE_TOKEN = os.environ["FORGE_TOKEN"]          # shared secret; must match the edge function
TARGET_FACES = int(os.environ.get("TARGET_FACES", "40000"))   # mobile triangle budget (to_glb decimation_target)
TEXTURE_SIZE = int(os.environ.get("TEXTURE_SIZE", "1024"))    # mobile-friendly texture size (repo default is 4096)

s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_KEY,
    aws_secret_access_key=R2_SECRET,
    region_name="auto",
)

app = FastAPI(title="vast-forge")


# ── model: TRELLIS.2 (microsoft/TRELLIS.2-4B), loaded ONCE at import so a warm worker keeps it resident ───────
# https://github.com/microsoft/TRELLIS.2 . On Turing (RTX 8000) force fp16 + SDPA/xformers attention (flash-attn
# needs Ampere+); on AMD use a ROCm port. See README.
PIPE = None


def _load_pipeline():
    global PIPE
    if PIPE is not None:
        return
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    PIPE = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    PIPE.cuda()


def run_trellis(image) -> bytes:
    """PIL.Image -> mobile-ready GLB bytes via TRELLIS.2. `to_glb` does the decimation + remesh + texture bake
    itself, so there's no separate trimesh pass (re-exporting would strip the WebP textures)."""
    import tempfile

    import o_voxel

    _load_pipeline()
    mesh = PIPE.run(image)[0]
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=TARGET_FACES,   # mobile budget — repo default 1,000,000 is desktop-scale
        texture_size=TEXTURE_SIZE,        # repo default 4096 → drop to 1024 for phones
        remesh=True,                      # clean topology
    )
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


def r2_put(key: str, data: bytes, content_type: str) -> str:
    s3.put_object(Bucket=R2_BUCKET, Key=key, Body=data, ContentType=content_type)
    return f"{R2_PUBLIC_BASE}/{key}"


class GenReq(BaseModel):
    image_url: str


@app.get("/health")
def health():
    import torch
    return {"ok": True, "cuda": torch.cuda.is_available(), "model_loaded": PIPE is not None}


@app.post("/generate")
def generate(req: GenReq, authorization: str = Header(default="")):
    if authorization != f"Bearer {FORGE_TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")
    try:
        from PIL import Image
        r = requests.get(req.image_url, timeout=30)
        r.raise_for_status()
        image = Image.open(io.BytesIO(r.content)).convert("RGB")

        glb = run_trellis(image)
        gid = uuid.uuid4().hex
        glb_url = r2_put(f"forge/{gid}.glb", glb, "model/gltf-binary")
        return {"glbUrl": glb_url}
    except HTTPException:
        raise
    except Exception as e:
        # Surface a short reason; the edge function maps any failure to a friendly user message.
        raise HTTPException(status_code=500, detail=f"generate failed: {e}")
