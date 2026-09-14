#!/bin/bash
# Bake-off 2: Hunyuan3D-Omni (controllable shape stage) NEXT TO the Hunyuan 2.1 engine on a forge node built from
# Dockerfile.hunyuan. Omni is the 2.1 shape DiT plus control encoders, same torch 2.5.1 / cu124 stack, so it goes
# into the image's system python; only the repo is cloned. Its `hy3dshape` package shadows 2.1's, so Omni is only
# ever imported in its own process with /opt/omni on sys.path (see bakeoff_omni.py).
# Weights: tencent/Hunyuan3D-Omni, ~25.7 GB, ungated, downloaded on first use into HF_HOME=/models.
#     bash /app/bakeoff/setup_omni.sh 2>&1 | tee /tmp/setup_omni.log
set -euo pipefail
export HF_HOME="${HF_HOME:-/models}"
mkdir -p /opt/omni && cd /opt/omni
[ -d Hunyuan3D-Omni ] || git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni.git
cd Hunyuan3D-Omni
echo "[setup_omni] commit $(git rev-parse --short HEAD)"
# Anything Omni needs that the 2.1 image lacks. Skip the training/demo weight (deepspeed, gradio, open3d, cupy, bpy).
grep -vE '^\s*(#|$|--extra-index|deepspeed|gradio|open3d|cupy|bpy|dash|plotly|pythreejs|torch|torchvision|torchaudio)' requirements.txt > /tmp/req_omni.txt || true
pip3 install --no-cache-dir --no-deps -r /tmp/req_omni.txt 2>&1 | tail -3 || echo "[setup_omni] some pins skipped (already satisfied by the image)"
df -h /models / | tail -2
python3 - <<'PY'
import sys, os
sys.path.insert(0, "/opt/omni/Hunyuan3D-Omni")
import inspect, hy3dshape
from hy3dshape import pipelines
names = [n for n in dir(pipelines) if "Pipeline" in n]
print("[setup_omni] pipelines:", names)
cls = getattr(pipelines, "Hunyuan3DOmniSiTFlowMatchingPipeline")
print("[setup_omni] __call__ signature:", inspect.signature(cls.__call__))
print("[setup_omni] from_pretrained signature:", inspect.signature(cls.from_pretrained))
print("[setup_omni] OK — hy3dshape from", os.path.dirname(hy3dshape.__file__))
PY
