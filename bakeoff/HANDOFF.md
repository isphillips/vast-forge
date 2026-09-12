# Bake-off handoff: TRELLIS.2 vs Hunyuan3D 2.1 on a forge box

Written 2026-09-12 for a Claude Code session on a machine WITHOUT the work VPN (the VPN drops every
non-22/443 port, so the box can't be reached from the primary Mac). Everything below is self-contained.

## Goal

Decide whether Hunyuan3D 2.1 (watertight SDF meshes + PBR paint) should replace TRELLIS.2 on the Vidrip facet
forge boxes. Forge the same three images through BOTH engines on one box, measure, and put the six models side by
side in a viewer page (an Artifact) with the numbers, so the founder can judge visually. Recommend.

Context: TRELLIS generates sparse voxel *surfaces* — thin things become sheets, hair becomes torn sheets, and a
dense object (a car) overflows 24 GB in the shape decoder, which triggers a retry ladder that degrades to the
512 pipeline (blobby geometry, smeared texture — see the "Autumn jones car" job). Hunyuan3D 2.1 makes closed
surfaces by construction. Face accessories (masks, hats, glasses) want smooth and closed.

## The box

Any idle forge node works. Last known reachable line (ports change after a restart; get a fresh one from the
Vast console or the /ops/forge node card, which shows `sshN.vast.ai:PORT`):

    ssh -p 35369 root@142.172.103.36

The box runs the TRELLIS forge server (`/app/server.py`, uvicorn on :8000) plus the pull worker. While all
forging is routed to fal (the "Route everything to fal" switch in /ops/forge is ON), the box is idle — keep it
that way during the run (the paint stage needs ~21 GB; the idle TRELLIS server holds ~0 GB on the GPU). To be
safe, flip the node's AUTOSCALE toggle off in /ops/forge so the pool can't stop it mid-run.

RTX 3090 24 GB, 64 GB disk (~45 GB free), CUDA 12.4 devel toolchain (nvcc present), Python 3.10, system torch
2.6 (Hunyuan wants 2.5.1 → its own venv at /opt/hy/venv). HF_HOME=/models.

## Steps

1. Copy the scripts (from the vast-forge repo checkout):

       scp -P <PORT> bakeoff/setup_hy.sh bakeoff/bakeoff.py root@<HOST>:/app/bakeoff/

2. Install Hunyuan3D 2.1 (~15 min; torch wheel 2.5 GB, two CUDA extensions, Real-ESRGAN weights):

       ssh -p <PORT> root@<HOST> 'bash /app/bakeoff/setup_hy.sh 2>&1 | tee /tmp/setup_hy.log | tail -30'

   Ends with `[setup_hy] OK — torch 2.5.1 ...`. If `custom_rasterizer` or `DifferentiableRenderer` fail to
   compile: `export TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=8` and re-run those two steps by hand inside the venv.

3. Run the bake-off (model weights ~10 GB download on first use; then ~2–4 min per image per engine). Run it
   detached and poll — the SSH session may drop:

       ssh -p <PORT> root@<HOST> 'cd /app/bakeoff && setsid nohup /opt/hy/venv/bin/python bakeoff.py --images '"'"'{
         "car":    ["https://v3b.fal.media/files/b/0aaa25cd/KDMi0X20iv_kcHt1RhtbZ_aJZ1lRL3.png", "Autumn jones car"],
         "turtle": ["https://v3b.fal.media/files/b/0aaa23fa/hTUPe579zxjlmu8kNDyOQ_SMfjbmD3.png", "Ninja turtles Leonardo mask"]
       }'"'"' > /tmp/bake/run.log 2>&1 < /dev/null &'

   (the mustache image is built into the script.) Poll with `tail -5 /tmp/bake/run.log`; done when it prints
   `[bake] DONE` followed by the stats JSON. Outputs in `/tmp/bake/`:
   `<tag>_trellis.glb`, `<tag>_hy.glb`, `s_<tag>_<engine>.glb.gz` (viewer-slim), `bake_stats.json`.

4. Pull the slim models + stats back to the laptop:

       scp -P <PORT> 'root@<HOST>:/tmp/bake/s_*.glb.gz' root@<HOST>:/tmp/bake/bake_stats.json ./bake/

5. Build the viewer page. There is a ready template in the primary machine's session scratchpad
   (`forge-fidelity-lab.template.html`) — if it isn't available, write a fresh one: three.js r128 UMD from
   cdn.jsdelivr.net (three.min.js, GLTFLoader.js, OrbitControls.js) + pako from cdnjs; models embedded as
   base64 gzipped GLBs in `window.FORGE_MODELS = { id: { label, desc, b64 } }`; a tab per model
   (mustache/car/turtle × trellis/hy); matte shading (metalness 0, roughness 0.9, DoubleSide); an "open edges"
   overlay (merge vertices by position, edges used by one face → red LineSegments); a stats table from
   bake_stats.json (faces, open edges, parts, watertight, MB, low-alpha %, gen/paint seconds, peak VRAM).
   IMPORTANT for Artifacts: hide `window.createImageBitmap` before loading (the sandbox blocks fetch() of blob:
   URLs, so three must fall back to <img> for textures) and keep textures as data: JPEG URIs (the slim step
   already does this). Page must stay under 16 MB.

6. Publish the page as an Artifact ("Forge Bake-off"), and write the recommendation.

## What to look at

- Hair/fur and thin rims: mustache + eyebrows. TRELLIS tears; Hunyuan should be closed but softer.
- The car: TRELLIS likely blobby (512 fallback); Hunyuan should hold the silhouette. Watch texture sharpness.
- The turtle mask: cracks/seams in TRELLIS (the fal one had many); Hunyuan watertight?
- Numbers that matter: open_edges (0 = watertight), parts (fragments), low_alpha_pct (grey-haze texels;
  0% is clean), faces vs the 120k budget, seconds, peak GB (must fit 24 GB with margin).

## Known pitfalls

- Hunyuan's paint stage returns a GLB via `create_glb_with_pbr_materials` (albedo + metallic + roughness JPEGs);
  the slim step drops the extra maps and forces metalness 0 so it renders like the app.
- If the paint stage OOMs at 21 GB: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--views 6 --res 512`
  (defaults), and make sure nothing else is on the GPU (`nvidia-smi`).
- `hy3dpaint` imports need cwd = `/opt/hy/Hunyuan3D-2.1` (the script chdirs); FaceReducer needs pymeshlab
  (in requirements).
- The TRELLIS side calls the live server's `/generate` with the box's own FORGE_TOKEN (read from the uvicorn
  process env by the script) and uploads to R2 like a real job; those GLB URLs are fine to keep.
- Never print or copy the box's env secrets (R2_*, FORGE_TOKEN, HF_TOKEN) — names only.

## If Hunyuan wins

Next steps (not part of the bake-off): a Dockerfile variant with Hunyuan3D 2.1, a `server.py` engine switch
(`FORGE_ENGINE=hunyuan|trellis`) keeping the same `/generate` + pull-worker contract, the same 120k-face and
1024-texture budgets, then the bake-off images through the new image end-to-end, then the Vast template.
