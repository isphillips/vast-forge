#!/bin/bash
# Bake-off: install Hunyuan3D-2.1 NEXT TO the TRELLIS server on a forge box, in its own venv (its torch is 2.5.1;
# the box's is 2.6). Nothing here touches /app or the running server. Run once as root on the box:
#     bash /app/bakeoff/setup_hy.sh 2>&1 | tee /tmp/setup_hy.log
# ~15 min: torch wheel (2.5 GB), two CUDA extensions (nvcc is in the devel base image), Real-ESRGAN weights.
# Model weights (~10 GB, tencent/Hunyuan3D-2.1) download on first use into HF_HOME.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export HF_HOME="${HF_HOME:-/models}"
export TORCH_CUDA_ARCH_LIST="8.6"          # RTX 3090 (Ampere)
export MAX_JOBS="${MAX_JOBS:-8}"

mkdir -p /opt/hy && cd /opt/hy
if [ ! -d Hunyuan3D-2.1 ]; then
  git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
fi
if [ ! -x /opt/hy/venv/bin/python ]; then
  python3 -m venv /opt/hy/venv
fi
# shellcheck disable=SC1091
source /opt/hy/venv/bin/activate
pip install -q --upgrade pip wheel setuptools ninja
pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

cd /opt/hy/Hunyuan3D-2.1
pip install -q -r requirements.txt
( cd hy3dpaint/custom_rasterizer && pip install -q -e . )
( cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh )
mkdir -p hy3dpaint/ckpt
[ -f hy3dpaint/ckpt/RealESRGAN_x4plus.pth ] || \
  wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt

python - <<'PY'
import torch, sys
sys.path.insert(0, '/opt/hy/Hunyuan3D-2.1')
import hy3dshape  # noqa: F401
from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline  # noqa: F401
print(f"[setup_hy] OK — torch {torch.__version__}, cuda {torch.cuda.is_available()}, {torch.cuda.get_device_name(0)}")
PY
