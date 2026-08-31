# vast-forge — TRELLIS 2 image→3D service for Vidrip Facet Forge (self-hosted on Vast).
# Only the 3D step lives here; the image is generated upstream (Qwen-Image-2512 on fal) and passed in as a URL.
#
# Build:  docker build -t vast-forge:1 .
# Run:    docker run --gpus all -p 8000:8000 --env-file .env vast-forge:1

# DEVEL base (not runtime): we COMPILE TRELLIS.2's CUDA extensions (flash-attn, nvdiffrast, cumesh, o-voxel, …),
# which need nvcc + CUDA headers — only the devel image has them. (Bigger final image; a multi-stage devel→runtime
# build would slim it later.)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    HUGGINGFACE_HUB_CACHE=/models \
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
      pandas lpips zstandard kornia timm plyfile moderngl scipy
# gradio/pillow-simd from --basic are omitted (demo-only / build-flaky); plyfile added (o_voxel imports it).
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
RUN git clone --recursive https://github.com/microsoft/TRELLIS.2.git trellis2

# utils3d — vendored on PYTHONPATH (pip discovery no-ops it, see base-deps note) at the exact commit TRELLIS.2 pins.
RUN git clone https://github.com/EasternJournalist/utils3d.git utils3d_src \
    && git -C utils3d_src checkout 9a4eb15e4021b67b12c460c7057d642626897ec8

# flash-attn (setup.sh --flash-attn): pip selects a prebuilt wheel for torch2.6/cu124/cp310 when one exists;
# otherwise it compiles (MAX_JOBS caps the RAM so the runner doesn't OOM).
RUN pip3 install --no-cache-dir flash-attn==2.7.3

# The from-source CUDA extensions, each exactly as setup.sh installs it. `--no-build-isolation` builds them against
# the torch already in this image; TORCH_CUDA_ARCH_LIST drives arch selection so no GPU is needed to COMPILE.
# KEEP the cloned sources under /tmp/ext (no `rm`): nvdiffrast installs in-place and JIT-compiles its CUDA plugin
# from the source tree at runtime, so deleting it makes `import nvdiffrast` fail (find_spec → None). The extra
# image size is cheap next to a silently-broken extension; the others are copied into site-packages but kept too.
# Order matters for the import check below: flexgemm before o-voxel, since `import o_voxel` imports flex_gemm.
RUN git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/ext/nvdiffrast \
    && pip3 install --no-cache-dir /tmp/ext/nvdiffrast --no-build-isolation
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/ext/nvdiffrec \
    && pip3 install --no-cache-dir /tmp/ext/nvdiffrec --no-build-isolation
RUN git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/ext/CuMesh \
    && pip3 install --no-cache-dir /tmp/ext/CuMesh --no-build-isolation
RUN git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/ext/FlexGEMM \
    && pip3 install --no-cache-dir /tmp/ext/FlexGEMM --no-build-isolation
RUN pip3 install --no-cache-dir ./trellis2/o-voxel --no-build-isolation

# The server imports trellis2 in-place from the cloned repo; utils3d is vendored the same way.
ENV PYTHONPATH="/app/trellis2:/app/utils3d_src"

# Build-time sanity check WITHOUT a GPU. Note we can't fully `import o_voxel` here: it pulls flex_gemm's triton
# autotuner, which initializes a GPU driver at import ("RuntimeError: 0 active drivers" on a GPU-less builder).
# So we (a) assert the GPU-bound packages are INSTALLED via find_spec — which locates them without executing their
# __init__, catching a silent pip failure — and (b) fully import the pure-Python deps to catch a half-broken one.
# nvdiffrast is intentionally NOT asserted: its setup produces no find_spec-visible top-level package, AND it's off
# our critical path — on the live box PIPE.run() succeeded and to_glb never touched it (o_voxel's chain is
# flex_gemm/cv2/trimesh/plyfile/zstandard). If a runtime path ever needs it, the first /generate says so on the GPU.
RUN python3 -c "import importlib.util as u; missing=[m for m in ['trellis2','o_voxel','flex_gemm','flash_attn'] if u.find_spec(m) is None]; assert not missing, 'NOT INSTALLED: '+str(missing); import trimesh, plyfile, cv2, zstandard, kornia, timm, transformers, utils3d; print('build check OK: extensions installed + base deps import')"

# Weights are NOT baked in — the 4B model is ~8-16 GB and buildkit needs ~2× that transiently, which overruns
# the builder's disk. Instead they download on first use (server.py's from_pretrained → HF_HOME=/models on the
# Vast node's larger disk). Trade: a cold node's FIRST request pays the weight download. To make cold-starts warm
# again later, mount a persistent Vast volume at /models so the download survives worker restarts, or pre-download
# with download_models.py on a builder that has the disk headroom.
COPY server.py ./

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120"]
