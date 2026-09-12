#!/bin/bash
# Remainder of setup_hy.sh, run after torch 2.5.1 is in /opt/hy/venv. Packages whose setup.py imports torch
# (basicsr, deepspeed, custom_rasterizer) are installed with --no-build-isolation so pip can't pull a second torch.
set -euo pipefail
export HF_HOME=/models TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=8 DEBIAN_FRONTEND=noninteractive
source /opt/hy/venv/bin/activate
cd /opt/hy/Hunyuan3D-2.1
echo "[rest] 1/6 build tools + numpy"
pip install -q numpy==1.24.4 pybind11==2.13.4 ninja==1.11.1.1 cython
echo "[rest] 2/6 basicsr (no build isolation)"
pip install -q --no-build-isolation basicsr==1.4.2
echo "[rest] 3/6 requirements.txt (minus deepspeed)"
grep -vE "^[[:space:]]*(deepspeed|bpy)" requirements.txt > /tmp/req_nods.txt
pip install -q -r /tmp/req_nods.txt
pip install -q --no-build-isolation deepspeed || echo "[rest] deepspeed skipped (not needed for inference)"
echo "[rest] 4/6 custom_rasterizer"
( cd hy3dpaint/custom_rasterizer && pip install -q --no-build-isolation -e . )
echo "[rest] 5/6 DifferentiableRenderer"
( cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh )
mkdir -p hy3dpaint/ckpt
[ -f hy3dpaint/ckpt/RealESRGAN_x4plus.pth ] || \
  wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt
echo "[rest] 6/6 verify"
python - <<'PY'
import torch, sys
sys.path.insert(0, '/opt/hy/Hunyuan3D-2.1')
import hy3dshape  # noqa: F401
from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline  # noqa: F401
print(f"[setup_hy] OK — torch {torch.__version__}, cuda {torch.cuda.is_available()}, {torch.cuda.get_device_name(0)}")
PY
