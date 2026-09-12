"""Paint one Hunyuan shape mesh with the same settings as bakeoff.py, then measure + slim it with bakeoff's own code.
Usage: hy_paint_one.py <shape.glb> <image.png> <out_tag>   -> /tmp/bake/<out_tag>_hy.glb, s_<out_tag>_hy.glb.gz, stats JSON line"""
import json, os, sys, time
shape_glb, image_png, out_tag = sys.argv[1:4]
sys.path.insert(0, "/app/bakeoff")
import bakeoff  # noqa: E402  (defines OUT=/tmp/bake, obj_to_glb, mesh_stats, slim)
os.chdir("/opt/hy/Hunyuan3D-2.1"); sys.path.insert(0, "/opt/hy/Hunyuan3D-2.1")
from torchvision_fix import apply_fix; apply_fix()
sys.path.insert(0, "/opt/hy/Hunyuan3D-2.1/hy3dshape"); sys.path.insert(0, "/opt/hy/Hunyuan3D-2.1/hy3dpaint")
import torch
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
paint = Hunyuan3DPaintPipeline(conf)
torch.cuda.reset_peak_memory_stats(); t0 = time.time()
obj = f"/tmp/bake/{out_tag}_hy.obj"; out = f"/tmp/bake/{out_tag}_hy.glb"
paint(shape_glb, image_path=image_png, output_mesh_path=obj, save_glb=False, use_remesh=False)
secs = time.time() - t0
bakeoff.obj_to_glb(obj, out)
st = bakeoff.mesh_stats(out); st["hy_paint_s"] = round(secs); st["hy_paint_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
bakeoff.slim(out, f"/tmp/bake/s_{out_tag}_hy.glb.gz")
json.dump(st, open(f"/tmp/bake/{out_tag}_stats.json", "w"), indent=2)
print("[paint1] DONE " + json.dumps(st), flush=True)
