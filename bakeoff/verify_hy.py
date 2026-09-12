import os, sys, torch
os.chdir("/opt/hy/Hunyuan3D-2.1")
sys.path.insert(0, "/opt/hy/Hunyuan3D-2.1"); from torchvision_fix import apply_fix; apply_fix()
sys.path.insert(0, "/opt/hy/Hunyuan3D-2.1/hy3dshape"); sys.path.insert(0, "/opt/hy/Hunyuan3D-2.1/hy3dpaint")
from hy3dshape.rembg import BackgroundRemover  # noqa
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa
from hy3dshape import FaceReducer, FloaterRemover, DegenerateFaceRemover  # noqa
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig  # noqa
import custom_rasterizer  # noqa
import pymeshlab, xatlas  # noqa
print(f"[verify_hy] OK — torch {torch.__version__}, cuda {torch.cuda.is_available()}, {torch.cuda.get_device_name(0)}")
