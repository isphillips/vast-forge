"""
Bake-off: TRELLIS.2 (the running forge server, push mode) vs Hunyuan3D-2.1 (the /opt/hy venv) on the same images.

Run on a forge box, INSIDE the hy venv (torch 2.5.1; the TRELLIS side is reached over HTTP so its torch never
loads here):
    /opt/hy/venv/bin/python /app/bakeoff/bakeoff.py [--skip-trellis] [--skip-hy] [--views 6] [--res 512]

Outputs in /tmp/bake/:  <tag>_trellis.glb, <tag>_hy.glb (full), s_<tag>_<engine>.glb.gz (viewer-slim: normals
stripped, textures as JPEG data URIs, gzipped) and bake_stats.json. Images are listed in IMAGES below.
"""
import argparse
import base64
import gzip
import io
import json
import os
import struct
import sys
import time

import numpy as np
import requests
import trimesh
from PIL import Image, ImageDraw

OUT = "/tmp/bake"
os.makedirs(OUT, exist_ok=True)

IMAGES = {
    # tag: (image url, prompt shown in the viewer)
    "mustache": ("https://v3b.fal.media/files/b/0aa99f56/YMWJSfckiHqpe9SspVjYb_bFA6r8CP.png", "mustache and glasses"),
    # filled in from the jobs the user pointed at (car, turtle mask) — set via --images JSON or edit here
}


def log(msg):
    print(f"[bake] {msg}", flush=True)


def fetch_image(tag, url):
    p = f"{OUT}/{tag}.png"
    if not os.path.exists(p):
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        open(p, "wb").write(r.content)
    return p


# ── TRELLIS side: the live server on this box (push endpoint), token from its own environment ───────────────
def forge_token():
    tok = os.environ.get("FORGE_TOKEN", "")
    if tok:
        return tok
    # not in our venv env → read it from the running server process
    import subprocess
    pid = subprocess.run(["pgrep", "-f", "uvicorn server:app"], capture_output=True, text=True).stdout.split()
    for p in pid:
        try:
            env = open(f"/proc/{p}/environ", "rb").read().split(b"\0")
            for kv in env:
                if kv.startswith(b"FORGE_TOKEN="):
                    return kv.split(b"=", 1)[1].decode()
        except OSError:
            pass
    raise SystemExit("FORGE_TOKEN not found (is the forge server running?)")


def run_trellis(tag, url):
    out = f"{OUT}/{tag}_trellis.glb"
    if os.path.exists(out):
        return out, None
    t0 = time.time()
    r = requests.post("http://127.0.0.1:8000/generate", json={"image_url": url},
                      headers={"Authorization": f"Bearer {forge_token()}"}, timeout=1200)
    r.raise_for_status()
    glb_url = r.json()["glbUrl"]
    secs = time.time() - t0
    open(out, "wb").write(requests.get(glb_url, timeout=120).content)
    log(f"{tag}: TRELLIS {secs:.0f}s → {glb_url}")
    return out, secs


# ── Hunyuan3D side ──────────────────────────────────────────────────────────────────────────────────────────
def run_hy_all(tags, views, res, target_faces):
    import torch
    sys.path.insert(0, "/opt/hy/Hunyuan3D-2.1")
    os.chdir("/opt/hy/Hunyuan3D-2.1")
    from hy3dshape.rembg import BackgroundRemover
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape import FaceReducer, FloaterRemover, DegenerateFaceRemover
    from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    stats = {}
    # 1) shape for every image, then free the shape model before the (bigger) paint model loads
    rembg = BackgroundRemover()
    shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
    for tag in tags:
        sp = f"{OUT}/{tag}_hy_shape.glb"
        if os.path.exists(sp):
            continue
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        img = Image.open(f"{OUT}/{tag}.png").convert("RGB")
        img = rembg(img)                                    # RGBA, background cut — the pipeline wants this
        mesh = shape(image=img, num_inference_steps=50, guidance_scale=5.0, octree_resolution=384)[0]
        raw_faces = len(mesh.faces)
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=target_faces)
        mesh.export(sp)
        gen_s = time.time() - t0
        stats[tag] = {"hy_shape_s": round(gen_s), "hy_raw_faces": raw_faces, "hy_shape_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        log(f"{tag}: HY shape {gen_s:.0f}s raw {raw_faces} → {len(mesh.faces)} faces, peak {stats[tag]['hy_shape_peak_gb']} GB")
    del shape, rembg
    torch.cuda.empty_cache()

    # 2) paint
    conf = Hunyuan3DPaintConfig(max_num_view=views, resolution=res)
    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    paint = Hunyuan3DPaintPipeline(conf)
    for tag in tags:
        out = f"{OUT}/{tag}_hy.glb"
        if os.path.exists(out):
            continue
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        # the paint stage wants the ORIGINAL image (it cuts the background itself) and the shape mesh path
        paint(f"{OUT}/{tag}_hy_shape.glb", image_path=f"{OUT}/{tag}.png", output_mesh_path=out)
        secs = time.time() - t0
        stats.setdefault(tag, {})["hy_paint_s"] = round(secs)
        stats[tag]["hy_paint_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
        log(f"{tag}: HY paint {secs:.0f}s ({views} views @ {res}), peak {stats[tag]['hy_paint_peak_gb']} GB")
    del paint
    torch.cuda.empty_cache()
    return stats


# ── measurement (same numbers as the fidelity lab) ──────────────────────────────────────────────────────────
def glb_chunks(b):
    total = struct.unpack("<I", b[8:12])[0]
    off, chunks = 12, []
    while off < total:
        ln, typ = struct.unpack("<II", b[off:off + 8])
        chunks.append((typ, b[off + 8:off + 8 + ln]))
        off += 8 + ln
    return chunks


def low_alpha_inside_uv(path):
    """% of texels inside UV islands with alpha < 50% — the grey-haze signal. None when the texture has no alpha."""
    try:
        b = open(path, "rb").read()
        chunks = glb_chunks(b)
        js = json.loads(chunks[0][1]); bin_ = chunks[1][1]
        mat = js["materials"][0]
        ti = mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]
        tex = js["textures"][ti]
        src_i = tex.get("source", tex.get("extensions", {}).get("EXT_texture_webp", {}).get("source"))
        img = js["images"][src_i]
        if "bufferView" not in img:
            return None
        bv = js["bufferViews"][img["bufferView"]]
        im = Image.open(io.BytesIO(bin_[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]))
        if im.mode not in ("RGBA", "LA"):
            return None
        a = np.asarray(im.convert("RGBA"))
        tm = trimesh.load(path, force="mesh", process=False)
        uv = np.asarray(tm.visual.uv); f = np.asarray(tm.faces)
        m = Image.new("L", im.size, 0); d = ImageDraw.Draw(m); W, H = im.size
        for tri in f:
            d.polygon([(float(uv[i, 0] * W), float((1 - uv[i, 1]) * H)) for i in tri], fill=255)
        cov = np.asarray(m) > 0
        return round(float((a[cov][:, 3] < 128).mean() * 100), 1)
    except Exception as e:  # noqa: BLE001
        log(f"low-alpha check failed for {path}: {e}")
        return None


def mesh_stats(path):
    tm = trimesh.load(path, force="mesh")
    faces = len(tm.faces)
    t2 = tm.copy(); t2.merge_vertices(merge_tex=True, merge_norm=True)
    _, c = np.unique(t2.edges_sorted, axis=0, return_counts=True)
    open_edges = int((c == 1).sum())
    parts = len(t2.split(only_watertight=False))
    watertight = bool(t2.is_watertight)
    return {"faces": faces, "open_edges": open_edges, "parts": parts, "watertight": watertight,
            "mb": round(os.path.getsize(path) / 1e6, 1), "low_alpha_pct": low_alpha_inside_uv(path)}


# ── viewer-slim: strip normals + extra PBR maps, textures → JPEG data URIs, gzip ────────────────────────────
def slim(path, out_gz, jpeg_q=82):
    b = open(path, "rb").read()
    chunks = glb_chunks(b)
    js = json.loads(chunks[0][1]); bin_ = chunks[1][1] if len(chunks) > 1 else b""
    # textures → data URIs (JPEG), keep only baseColor
    for mat in js.get("materials", []):
        pbr = mat.get("pbrMetallicRoughness", {})
        pbr.pop("metallicRoughnessTexture", None)
        mat.pop("normalTexture", None); mat.pop("occlusionTexture", None); mat.pop("emissiveTexture", None)
        pbr["metallicFactor"] = 0.0; pbr["roughnessFactor"] = 0.9
    used_tex = {m["pbrMetallicRoughness"]["baseColorTexture"]["index"] for m in js.get("materials", [])
                if "baseColorTexture" in m.get("pbrMetallicRoughness", {})}
    for ti, tex in enumerate(js.get("textures", [])):
        if ti not in used_tex:
            continue
        src_i = tex.get("source", tex.get("extensions", {}).get("EXT_texture_webp", {}).get("source"))
        tex["source"] = src_i; tex.pop("extensions", None)
        img = js["images"][src_i]
        if "bufferView" in img:
            bv = js["bufferViews"][img["bufferView"]]
            raw = bin_[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            if max(im.size) > 1024:
                im = im.resize((1024, 1024))
            buf = io.BytesIO(); im.save(buf, "JPEG", quality=jpeg_q)
            js["images"][src_i] = {"uri": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), "mimeType": "image/jpeg"}
    js["extensionsUsed"] = [e for e in js.get("extensionsUsed", []) if e != "EXT_texture_webp"]
    js["extensionsRequired"] = [e for e in js.get("extensionsRequired", []) if e != "EXT_texture_webp"]
    if not js["extensionsRequired"]:
        js.pop("extensionsRequired", None)
    # drop normals; rebuild the buffer with only the bufferViews still referenced
    for mesh in js.get("meshes", []):
        for prim in mesh["primitives"]:
            prim["attributes"].pop("NORMAL", None); prim["attributes"].pop("TANGENT", None)
    used_acc = set()
    for mesh in js.get("meshes", []):
        for prim in mesh["primitives"]:
            used_acc.update(prim["attributes"].values())
            if "indices" in prim:
                used_acc.add(prim["indices"])
    used_bv = {js["accessors"][a]["bufferView"] for a in used_acc if "bufferView" in js["accessors"][a]}
    new_bin = bytearray(); remap = {}
    for i, bv in enumerate(js["bufferViews"]):
        if i not in used_bv:
            continue
        data = bin_[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]
        while len(new_bin) % 4:
            new_bin.append(0)
        remap[i] = len(remap)
        nbv = {k: v for k, v in bv.items() if k not in ("byteOffset", "buffer")}
        nbv["buffer"] = 0; nbv["byteOffset"] = len(new_bin); nbv["byteLength"] = len(data)
        js.setdefault("_nbv", []).append(nbv); new_bin += data
    js["bufferViews"] = js.pop("_nbv", [])
    js["accessors"] = [a for i, a in enumerate(js["accessors"]) if i in used_acc]
    # accessor indices changed → remap primitives
    acc_remap = {old: new for new, old in enumerate(sorted(used_acc))}
    for a in js["accessors"]:
        if "bufferView" in a:
            a["bufferView"] = remap[a["bufferView"]]
    for mesh in js.get("meshes", []):
        for prim in mesh["primitives"]:
            prim["attributes"] = {k: acc_remap[v] for k, v in prim["attributes"].items()}
            if "indices" in prim:
                prim["indices"] = acc_remap[prim["indices"]]
    js["buffers"] = [{"byteLength": len(new_bin)}]
    while len(new_bin) % 4:
        new_bin.append(0)
    jb = json.dumps(js, separators=(",", ":")).encode()
    while len(jb) % 4:
        jb += b" "
    out = b"glTF" + struct.pack("<II", 2, 12 + 8 + len(jb) + 8 + len(new_bin))
    out += struct.pack("<II", len(jb), 0x4E4F534A) + jb + struct.pack("<II", len(new_bin), 0x004E4942) + bytes(new_bin)
    gz = gzip.compress(out, 9)
    open(out_gz, "wb").write(gz)
    log(f"slim {os.path.basename(path)}: {os.path.getsize(path)/1e6:.1f} MB → {len(out)/1e6:.1f} MB → gz {len(gz)/1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help='JSON {"tag": ["url", "prompt"], ...} to add/override IMAGES')
    ap.add_argument("--skip-trellis", action="store_true")
    ap.add_argument("--skip-hy", action="store_true")
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--faces", type=int, default=120000)
    a = ap.parse_args()
    images = dict(IMAGES)
    if a.images:
        images.update({k: tuple(v) for k, v in json.loads(a.images).items()})
    for tag, (url, _) in images.items():
        fetch_image(tag, url)

    stats = {tag: {"prompt": p, "image_url": url} for tag, (url, p) in images.items()}
    if not a.skip_trellis:
        for tag, (url, _) in images.items():
            _, secs = run_trellis(tag, url)
            if secs is not None:
                stats[tag]["trellis_s"] = round(secs)
    if not a.skip_hy:
        hy = run_hy_all(list(images), a.views, a.res, a.faces)
        for tag, d in hy.items():
            stats[tag].update(d)
    for tag in images:
        for engine in ("trellis", "hy"):
            p = f"{OUT}/{tag}_{engine}.glb"
            if os.path.exists(p):
                stats[tag][engine] = mesh_stats(p)
                slim(p, f"{OUT}/s_{tag}_{engine}.glb.gz")
    json.dump(stats, open(f"{OUT}/bake_stats.json", "w"), indent=2)
    log("DONE\n" + json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
