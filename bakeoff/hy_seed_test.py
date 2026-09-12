"""Seed sensitivity of the Hunyuan shape stage on the Leonardo mask image (/tmp/bake/car.png; the handoff's URLs were
swapped). The first run extruded a whole turtle body behind the face (depth/width 3.9). Try fixed seeds, log extents."""
import os, sys, time
os.chdir("/opt/hy/Hunyuan3D-2.1"); sys.path.insert(0, "/opt/hy/Hunyuan3D-2.1")
from torchvision_fix import apply_fix; apply_fix()
sys.path.insert(0, "/opt/hy/Hunyuan3D-2.1/hy3dshape"); sys.path.insert(0, "/opt/hy/Hunyuan3D-2.1/hy3dpaint")
import torch
from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape import FaceReducer, FloaterRemover, DegenerateFaceRemover

rembg = BackgroundRemover()
shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
img = rembg(Image.open("/tmp/bake/car.png").convert("RGB"))
for seed in [1, 2, 3, 4]:
    t0 = time.time()
    g = torch.Generator().manual_seed(seed)
    mesh = shape(image=img, num_inference_steps=50, guidance_scale=5.0, octree_resolution=384, generator=g)[0]
    mesh = FloaterRemover()(mesh); mesh = DegenerateFaceRemover()(mesh); mesh = FaceReducer()(mesh, max_facenum=120000)
    ext = mesh.bounds[1] - mesh.bounds[0]
    mesh.export(f"/tmp/bake/mask_seed{seed}_hy_shape.glb")
    print(f"[seed] {seed}: {time.time()-t0:.0f}s faces {len(mesh.faces)} extents x {ext[0]:.2f} y {ext[1]:.2f} z {ext[2]:.2f} depth/width {ext[2]/ext[0]:.2f}", flush=True)
print("[seed] DONE", flush=True)
