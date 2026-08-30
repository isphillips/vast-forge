# vast-forge — TRELLIS 2 image→3D service for Vidrip Facet Forge (self-hosted on Vast).
# Only the 3D step lives here; the image is generated upstream (Qwen-Image-2512 on fal) and passed in as a URL.
#
# Build:  docker build -t vast-forge:1 .
# Run:    docker run --gpus all -p 8000:8000 --env-file .env vast-forge:1

# DEVEL base (not runtime): TRELLIS.2's setup.sh COMPILES CUDA extensions (flash-attn, nvdiffrast, cumesh, …),
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
      python3.10 python3-pip git ca-certificates \
      libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch matched to the CUDA 12.4 base. (Ada/Ampere/Turing all supported by cu124 wheels.)
RUN pip3 install --no-cache-dir torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu124

# Service deps (everything EXCEPT the model itself).
RUN pip3 install --no-cache-dir \
      fastapi "uvicorn[standard]" pydantic boto3 pillow requests numpy huggingface_hub

# ── TRELLIS.2 ─────────────────────────────────────────────────────────────────────────────────────────────
# github.com/microsoft/TRELLIS.2 — its setup.sh pulls the CUDA extensions (flash-attn, nvdiffrast/nvdiffrec,
# cumesh, o_voxel, flexgemm) and pip-installs the `trellis2` + `o_voxel` packages the server imports.
# NOTE: the repo's setup.sh is written around conda + `--new-env`; in Docker we install into the system python
# (no --new-env). If setup.sh hard-requires conda, switch this base to a miniconda image and activate the env in
# CMD. Pin a specific commit for reproducible rebuilds.
RUN git clone https://github.com/microsoft/TRELLIS.2.git trellis2 \
    && cd trellis2 \
    && bash setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm \
    || echo "!! TRELLIS.2 setup returned nonzero — run its setup.sh manually + re-pin; see README (conda caveat)."

# Pre-download the TRELLIS 2 weights INTO the image so a cold Vast node boots warm instead of pulling ~GBs on
# first request. (Fill the exact model id in download_models.py.)
COPY download_models.py server.py ./
RUN python3 download_models.py || echo "!! weight pre-download skipped — set the model id in download_models.py"

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120"]
