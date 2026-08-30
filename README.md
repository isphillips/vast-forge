# vast-forge

TRELLIS 2 **image→3D** service for Vidrip Facet Forge, self-hosted on Vast. The image is generated upstream
(Qwen-Image-2512 on fal); this service turns that image into a mobile-ready GLB.

```
POST /generate  { "image_url": "https://.../qwen.png" }   (Authorization: Bearer $FORGE_TOKEN)
  -> { "glbUrl": "https://<r2-public>/forge/<id>.glb" }
GET  /health -> { ok, cuda, model_loaded }
```

Pipeline inside: download image → **TRELLIS 2** → decimate to ~40k tris + baseColor GLB → upload to R2.

## Status: wired to the real TRELLIS.2 API

`server.py` calls the actual [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) API — no code blanks:

```python
from trellis2.pipelines import Trellis2ImageTo3DPipeline
PIPE = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B"); PIPE.cuda()
mesh = PIPE.run(image)[0]
glb = o_voxel.postprocess.to_glb(vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]], decimation_target=40000, texture_size=1024, remesh=True)
glb.export(path, extension_webp=True)   # WebP textures — the app's loaders read EXT_texture_webp
```

We pass a **mobile** `decimation_target` (40k, vs the repo's 1,000,000) and `texture_size` (1024, vs 4096), so
`to_glb` produces a phone-ready GLB directly — no trimesh post-pass.

**The thing to verify is the BUILD, not the code.** TRELLIS.2's `setup.sh` is written around conda + `--new-env`;
the Dockerfile installs into the system python (no `--new-env`). If the installer hard-requires conda, switch to
a miniconda base and activate the env in `CMD`. Build on the target GPU arch, then confirm both
`from trellis2.pipelines import Trellis2ImageTo3DPipeline` and `import o_voxel` import before you push.

## Build & test locally (need an NVIDIA GPU)

```bash
cp .env.example .env      # fill R2 creds + FORGE_TOKEN
docker build -t vast-forge:1 .
docker run --gpus all -p 8000:8000 --env-file .env vast-forge:1

curl -s localhost:8000/health
curl -s -X POST localhost:8000/generate \
  -H "authorization: Bearer $FORGE_TOKEN" -H "content-type: application/json" \
  -d '{"image_url":"https://picsum.photos/1024"}'
```

## Push to a registry

```bash
echo $GHCR_PAT | docker login ghcr.io -u <you> --password-stdin
docker tag vast-forge:1 ghcr.io/<you>/vast-forge:1
docker push ghcr.io/<you>/vast-forge:1
```

## Deploy on Vast Serverless (scale-to-zero)

Create a Serverless endpoint pointing at `ghcr.io/<you>/vast-forge:1`:
- **GPU:** RTX 3090 (24 GB) — TRELLIS 2 alone fits comfortably.
- **Autoscaler:** min workers **0** (scale to zero), max 2–3.
- **Env:** the same `R2_*` + `FORGE_TOKEN` as `.env`.
- Copy the endpoint URL → that's `VAST_URL`.

To **never cold-pull**, set min workers **1** and make the reserve worker **on-demand** (not interruptible) — ~$110/mo,
the price of always-warm. See the plan artifacts for the full cost/cold-vs-warm breakdown.

## Wire the edge function

In `supabase/functions/facet-generate/index.ts`, put the 3D call behind a flag so cutting over is env-only:

```ts
const mesh3d = Deno.env.get("FORGE_3D") === "vast"
  ? await fetch(Deno.env.get("VAST_URL")! + "/generate", {
      method: "POST",
      headers: { authorization: `Bearer ${Deno.env.get("FORGE_TOKEN")}`, "content-type": "application/json" },
      body: JSON.stringify({ image_url: imageUrl }),
    }).then(r => r.json())
  : await fal("fal-ai/trellis-2", { image_url: imageUrl });   // fal fallback
const glbUrl = mesh3d.glbUrl ?? (mesh3d.model_glb ?? mesh3d.model_mesh)?.url;
```

```bash
supabase secrets set FORGE_3D=vast VAST_URL=https://<endpoint> FORGE_TOKEN=<secret>
supabase functions deploy facet-generate
```

Flip `FORGE_3D=fal` any time to fall back to fal's TRELLIS 2.

## Running on a friend's 48 GB card instead of Vast

Same container; three GPU-specific notes:
- **NVIDIA L20 (Ada):** runs as-is — CUDA + bf16 + flash-attn 2 all native.
- **Quadro RTX 8000 (Turing):** force **fp16** (no bf16) and swap TRELLIS's attention to **SDPA/xformers**
  (flash-attn 2 needs Ampere+). Slower.
- **AMD W7900 (RDNA3):** use a **ROCm port** of TRELLIS 2 (e.g. `toastmanAu/trellis-2-rocm-comfyui`, tested on
  gfx1100) and a torch-ROCm base instead of the CUDA one; nvdiffrast runs via its OpenGL backend.

Expose it to `facet-generate` with a tunnel (Cloudflare Tunnel / Tailscale) and point `VAST_URL` at that.

## Notes
- **Sync now, async later:** `/generate` blocks ~30–60 s (warm). If cold starts outlast the edge-function
  timeout, add a job queue (`/generate` → id, `/result/<id>` poll) and push "facet ready" — see the plan.
- **Weights baked in** → the image is large (10–20 GB) but nodes boot warm. First registry push is slow.
