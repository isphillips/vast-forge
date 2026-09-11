# vast-forge — TRELLIS 2 image→3D service for Vidrip Facet Forge (self-hosted on Vast).
# Only the 3D step lives here; the image is generated upstream (Qwen-Image-2512 on fal) and passed in as a URL.
#
# 3D model code = the StableProjectorz fork (IgorAherne/TRELLIS.2-stableprojectorz). Same `trellis2` package,
# same Trellis2ImageTo3DPipeline, same microsoft/TRELLIS.2-4B weights — but its sparse-conv path computes the
# neighbour map, parks it on CPU and streams feats back, which is a large VRAM win. MEASURED on the 3090
# (same 774px image, same to_glb params, 1024_cascade):
#     microsoft : full-path peak 7.14 GiB allocated / 11773 MiB total GPU / run 67.5s
#     fork      : full-path peak 3.13 GiB allocated /  4071 MiB total GPU / run 82.6s
# → ~2.9x less GPU for ~22% more time. That headroom is why we switched (the 3090 was OOMing on detailed inputs).
#
# TWO source patches are required and applied below — do not drop either:
#   1. DINOv3 layer path (same fix microsoft needs on transformers 5.16.x).
#   2. `needs_grad`: the fork calls FlexGEMM's SubMConv3dFunction._compute_neighbor_cache with 4 args (it was
#      written against an older FlexGEMM). Current FlexGEMM requires a 5th `needs_grad`. Rather than pin an
#      8-month-old FlexGEMM (its pins ship only as Windows wheels), we pass torch.is_grad_enabled(). Validated
#      end-to-end on the 3090 incl. to_glb. Everything else stays on CURRENT upstream — no extension pinning.
#
# Build:  docker build -t vast-forge:2 .
# Run:    docker run --gpus all -p 8000:8000 --env-file .env vast-forge:2

# DEVEL base (not runtime): we COMPILE TRELLIS.2's CUDA extensions (flash-attn, nvdiffrast, cumesh, o-voxel, …),
# which need nvcc + CUDA headers — only the devel image has them. (Bigger final image; a multi-stage devel→runtime
# build would slim it later.)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    HUGGINGFACE_HUB_CACHE=/models \
    # Reduce CUDA memory fragmentation on a long-lived warm worker (24 GB 3090 is tight for the 4B model + mesh
    # decode; without this a later request can OOM in CuMesh.get_edges where earlier ones fit).
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    # Only compile CUDA kernels for the GPUs we deploy on (Ampere 8.0/8.6 incl. 3090, Ada 8.9 incl. L20/4090).
    # Add 7.5 for Turing (RTX 8000). Trims a very long compile and avoids building for arches you don't use.
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" \
    # Cap parallel compile jobs — flash-attn's build is a RAM hog and OOMs cloud runners at full parallelism.
    MAX_JOBS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3-pip python3-dev build-essential ninja-build git ca-certificates \
      libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*
# python3-dev → Python.h (required to compile o_voxel/nvdiffrast/cumesh/flexgemm C++/CUDA extensions);
# build-essential → g++/make; ninja-build → fast parallel compiles (setup.sh falls back to slow distutils without it).

WORKDIR /app

# Torch pinned to EXACTLY what TRELLIS.2's setup.sh installs (torch 2.6.0 → triton 3.2). This is not cosmetic:
# flex_gemm subclasses triton's Autotuner and passes 14 positional args, which only triton 3.2's signature accepts
# — torch 2.4 (triton 3.0) makes `import o_voxel` die with "Autotuner.__init__() takes 7 to 13 args but 14 given".
# Compiled CUDA extensions (o_voxel, flexgemm, cumesh…) are also ABI-bound to this torch, so it must match.
RUN pip3 install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# TRELLIS.2 base runtime deps — normally pulled by `setup.sh --basic`, but that step doesn't reliably land under
# Docker's system-python (no conda env), and o_voxel is pip-installed with --no-build-isolation so it won't drag
# its own deps in either. Installing them explicitly makes `import trellis2, o_voxel` deterministic instead of a
# runtime game of whack-a-mole (trimesh→plyfile→zstandard→cv2→… were each a separate ModuleNotFoundError).
# plyfile isn't in --basic's list but o_voxel imports it; gradio/pillow-simd are omitted (demo-only / build-flaky).
RUN pip3 install --no-cache-dir \
      imageio imageio-ffmpeg tqdm easydict opencv-python-headless trimesh transformers tensorboard \
      pandas lpips zstandard kornia timm plyfile moderngl scipy scikit-image
# gradio/pillow-simd from --basic are omitted (demo-only / build-flaky); plyfile added (o_voxel imports it).
# scikit-image: marching cubes for solidify.py (the pre-export closing of hair-like geometry).
# moderngl + scipy are utils3d's runtime deps. utils3d itself is NOT pip-installed: its flat-layout pyproject has
# no explicit package list, so modern setuptools auto-discovery installs NOTHING from `pip install git+…` yet still
# exits 0 (→ `import utils3d` fails at runtime). We vendor the source onto PYTHONPATH below instead — same as trellis2.

# Service deps for our FastAPI wrapper (everything EXCEPT the model itself).
RUN pip3 install --no-cache-dir \
      fastapi "uvicorn[standard]" pydantic boto3 pillow requests numpy huggingface_hub

# ── TRELLIS.2 + its CUDA extensions ───────────────────────────────────────────────────────────────────────
# We deliberately DO NOT run TRELLIS.2's setup.sh. It aborts on its very first step with "No supported GPU found"
# (a bare `command -v nvidia-smi` check, no bypass flag) — and CI build runners have no GPU. That abort, swallowed
# by the old `|| echo` mask, is why NOTHING (neither --basic nor any extension) ever installed. nvcc cross-compiles
# for the arches in TORCH_CUDA_ARCH_LIST with no GPU present, so we run setup.sh's per-flag steps by hand instead.
#
# --recursive: o-voxel is a submodule of the repo (setup.sh's --o-voxel does `cp -r o-voxel`); it also pulls eigen.
# StableProjectorz fork (see header) — the fork's own o-voxel carries C++ changes, and --recursive pulls the
# fork's pinned submodule, so `pip install ./trellis2/o-voxel` below builds the fork's version. Dir stays
# `trellis2` so every downstream path (PYTHONPATH, ./trellis2/o-voxel, both source patches) is unchanged.
RUN git clone --recursive https://github.com/IgorAherne/TRELLIS.2-stableprojectorz.git trellis2

# utils3d — vendored on PYTHONPATH (pip discovery no-ops it, see base-deps note) at the exact commit TRELLIS.2 pins.
RUN git clone https://github.com/EasternJournalist/utils3d.git utils3d_src \
    && git -C utils3d_src checkout 9a4eb15e4021b67b12c460c7057d642626897ec8

# flash-attn (setup.sh --flash-attn): pip selects a prebuilt wheel for torch2.6/cu124/cp310 when one exists;
# otherwise it compiles (MAX_JOBS caps the RAM so the runner doesn't OOM).
RUN pip3 install --no-cache-dir flash-attn==2.7.3

# The from-source CUDA extensions, each exactly as setup.sh installs it. `--no-build-isolation` builds them against
# the torch already in this image; TORCH_CUDA_ARCH_LIST drives arch selection so no GPU is needed to COMPILE.
# KEEP the cloned sources under /tmp/ext (no `rm`) — nvdiffrast in particular is imported from its source tree.
# Order matters for the import check below: flexgemm before o-voxel, since `import o_voxel` imports flex_gemm.
#
# nvdiffrast is a three-part mess at this ref: (1) its packaging metadata is broken, so `pip install` builds a wheel
# named UNKNOWN that installs ONLY the compiled `_nvdiffrast_c.so` (which nvdiffrast.torch imports) and NOT the
# `nvdiffrast/` Python package; (2) so we get the package from the source tree via PYTHONPATH instead; (3) but that
# package's __init__ reads its version from dist metadata (`version('nvdiffrast')`) which doesn't exist (dist is
# named UNKNOWN) → crash, so we sed the version to a literal. All three are needed together. Validated live on GPU.
RUN git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/ext/nvdiffrast \
    && pip3 install --no-cache-dir /tmp/ext/nvdiffrast --no-build-isolation
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/ext/nvdiffrec \
    && pip3 install --no-cache-dir /tmp/ext/nvdiffrec --no-build-isolation
RUN git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/ext/CuMesh \
    && pip3 install --no-cache-dir /tmp/ext/CuMesh --no-build-isolation
RUN git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/ext/FlexGEMM \
    && pip3 install --no-cache-dir /tmp/ext/FlexGEMM --no-build-isolation
# PATCH 3 — exporter quality (2026-09-08). The fork's o-voxel (a) hard-codes its dual-contour remesh at 512 while
# the model generates at 1024, and (b) removed Microsoft's three mesh.fill_holes() cleanup calls (fork commit
# 8b1476f, chasing a "vertical lines" artifact we never saw). Measured on the 3090: 512/no-fill = 89 open edges and
# blobby thin parts; 768 + fill = 8 open edges with 98.4% of the raw surface kept, ~1.5 GB during export. Make the
# cap env-tunable (OVOXEL_DC_MAX_RES, set in ENV below; server.py steps it down on OOM) and put fill_holes back
# after every small-component prune. Applied to the SOURCE before the pip build below, and asserted so a fork
# restructure fails the build instead of silently shipping the old behaviour.
RUN sed -i "s/dc_resolution = min(int(resolution), 512)/dc_resolution = min(int(resolution), int(os.environ.get('OVOXEL_DC_MAX_RES', '512')))/" /app/trellis2/o-voxel/o_voxel/postprocess.py \
    && sed -i 's/^\(\s*\)mesh\.remove_small_connected_components(1e-5)$/&\n\1mesh.fill_holes(max_hole_perimeter=3e-2)/' /app/trellis2/o-voxel/o_voxel/postprocess.py \
    && grep -q "OVOXEL_DC_MAX_RES" /app/trellis2/o-voxel/o_voxel/postprocess.py \
    && [ "$(grep -c 'fill_holes(max_hole_perimeter=3e-2)' /app/trellis2/o-voxel/o_voxel/postprocess.py)" = "3" ]
RUN pip3 install --no-cache-dir ./trellis2/o-voxel --no-build-isolation

# Patch nvdiffrast's version line in a SEPARATE RUN so the expensive extension-compile layers above stay
# cache-valid. Its __init__ reads `version('nvdiffrast')` from dist metadata that doesn't exist (the wheel
# installed as UNKNOWN), which crashes at import — hardcode the version instead. See the block comment above.
RUN sed -i 's/^__version__ = version(.*/__version__ = "0.3.3"/' /tmp/ext/nvdiffrast/nvdiffrast/__init__.py

# TRELLIS.2-4B (~a year old) predates transformers' DINOv3 restructuring: its image feature extractor iterates
# `self.model.layer`, but in transformers 5.16.1 `DINOv3ViTModel` puts its encoder under `self.model` (a
# DINOv3ViTEncoder whose blocks are `.layer`) — so the blocks are at `self.model.model.layer`. Without this,
# inference dies with "'DINOv3ViTModel' object has no attribute 'layer'". (Upstream PR #148 used the intermediate
# `.encoder.layer`; 5.16.x moved it again.) Separate RUN so the extension-compile layers above stay cache-valid.
RUN sed -i 's/self\.model\.layer)/self.model.model.layer)/' /app/trellis2/trellis2/modules/image_feature_extractor.py

# PATCH 2 — `needs_grad` (see header). The fork's memory-saving sparse-conv path calls FlexGEMM's
# SubMConv3dFunction._compute_neighbor_cache(coords, shape, kernel, dilation) — 4 args, matching a FlexGEMM
# from before commit 122980b (2026-01-15) added a 5th `needs_grad`. Current FlexGEMM requires it, so without
# this the very first sparse conv dies with "missing 1 required positional argument: 'needs_grad'". We pass
# torch.is_grad_enabled() (False under the pipeline's @torch.no_grad, so no backward buffers are allocated).
# The assert makes the build fail loudly if the fork restructures these call sites, rather than silently
# shipping a broken image. Separate RUN so the extension-compile layers above stay cache-valid.
RUN python3 -c "p='/app/trellis2/trellis2/modules/sparse/conv/conv_flex_gemm.py'; s=open(p).read(); old='                (Kw, Kh, Kd),\n                self.dilation\n            )'; new='                (Kw, Kh, Kd),\n                self.dilation,\n                torch.is_grad_enabled()\n            )'; n=s.count(old); assert n==2, 'expected 2 _compute_neighbor_cache call sites, found %d' % n; open(p,'w').write(s.replace(old,new)); print('patched %d needs_grad call sites' % n)"

# trellis2, utils3d, and the nvdiffrast Python package are all imported from their source trees via PYTHONPATH.
# (nvdiffrast's compiled _nvdiffrast_c.so lives in site-packages from the UNKNOWN wheel; only the .py package needs
# this path. PYTHONPATH is searched before site-packages, so the patched source __init__ wins.)
ENV PYTHONPATH="/app/trellis2:/app/utils3d_src:/tmp/ext/nvdiffrast"
# Export quality dials (see PATCH 3 + server.py's remesh ladder): remesh at full 1024 (the model's own resolution),
# falling back to 768 and then 512 on a CUDA OOM. Measured 2026-09-08 on the 3090: 1024 = 2.5 GB during export,
# 98.8% of the raw surface kept; 768 = 1.5 GB / 98.4%; 512 = 0.8 GB / 96.9%.
ENV OVOXEL_DC_MAX_RES=1024 FORGE_REMESH_RES=1024 FORGE_REMESH_FALLBACKS=768,512
# Triangle budget (2026-09-08, "B"): 120k. On hair-heavy facets 40k kept only 84% of the generated surface (eyebrow
# and mustache holes); 120k keeps 93% for ~30 s more export and a ~6.6 MB file (downloaded once, cached on device).
ENV TARGET_FACES=120000

# Build-time sanity check WITHOUT a GPU. Note we can't fully `import o_voxel` here: it pulls flex_gemm's triton
# autotuner, which initializes a GPU driver at import ("RuntimeError: 0 active drivers" on a GPU-less builder).
# So we (a) assert the GPU-bound packages are INSTALLED via find_spec — which locates them without executing their
# __init__, catching a silent pip failure — and (b) fully import the pure-Python deps to catch a half-broken one.
# nvdiffrast needs BOTH checked: the .py package (from PYTHONPATH) and its compiled _nvdiffrast_c.so (site-packages).
# If _nvdiffrast_c is missing, the GPU-less builder failed to compile it — better to fail here than at runtime.
RUN python3 -c "import importlib.util as u; missing=[m for m in ['trellis2','o_voxel','flex_gemm','flash_attn','nvdiffrast','_nvdiffrast_c'] if u.find_spec(m) is None]; assert not missing, 'NOT INSTALLED: '+str(missing); import trimesh, plyfile, cv2, zstandard, kornia, timm, transformers, utils3d; print('build check OK: extensions installed + base deps import')"

# Weights are NOT baked in — the 4B model is ~8-16 GB and buildkit needs ~2× that transiently, which overruns
# the builder's disk. Instead they download on first use (server.py's from_pretrained → HF_HOME=/models on the
# Vast node's larger disk). Trade: a cold node's FIRST request pays the weight download. To make cold-starts warm
# again later, mount a persistent Vast volume at /models so the download survives worker restarts, or pre-download
# with download_models.py on a builder that has the disk headroom.
COPY server.py solidify.py worker.py ./
# Pre-export solidify radius in voxels (see solidify.py); 0 = off (remesh ladder only).
ENV FORGE_SOLIDIFY_RADIUS=2
# Pull worker: every box built from this image joins the shared job queue on boot (worker.py → the forge-worker
# edge function, authenticated with FORGE_TOKEN). Not a secret — the project's public functions base URL.
ENV FORGE_CONTROL_URL=https://ltpscwticavqutbzrrjb.supabase.co/functions/v1/forge-worker

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120"]
