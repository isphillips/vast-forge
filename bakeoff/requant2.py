"""Viewer-only shrink: KHR_mesh_quantization (int16 positions, uint16 UVs) on the slim GLBs + source thumbnails."""
import glob, gzip, io, json, os, struct
import numpy as np
from PIL import Image

CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

def chunks(b):
    total = struct.unpack("<I", b[8:12])[0]; off, out = 12, []
    while off < total:
        ln, typ = struct.unpack("<II", b[off:off + 8]); out.append((typ, b[off + 8:off + 8 + ln])); off += 8 + ln
    return out

def node_matrix(n):
    if "matrix" in n:
        return np.array(n["matrix"], dtype=np.float64).reshape(4, 4).T  # glTF column-major -> row-major
    T = np.eye(4); T[:3, 3] = n.get("translation", [0, 0, 0])
    x, y, z, w = n.get("rotation", [0, 0, 0, 1])
    R = np.eye(4); R[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    Sm = np.diag(list(n.get("scale", [1, 1, 1])) + [1])
    return T @ R @ Sm

for path in sorted(glob.glob("/tmp/bake2/s_*.glb.gz")):
    b = gzip.decompress(open(path, "rb").read()); ch = chunks(b); js = json.loads(ch[0][1]); bin_ = ch[1][1]
    def arr(ai):
        a = js["accessors"][ai]; bv = js["bufferViews"][a["bufferView"]]; n = NC[a["type"]]; dt = CT[a["componentType"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        assert "byteStride" not in bv or bv["byteStride"] == n * np.dtype(dt).itemsize
        return np.frombuffer(bin_, dtype=dt, count=a["count"] * n, offset=off).reshape(a["count"], n)
    # per-mesh dequantization transform, composed into every node that uses the mesh
    new_acc = {}  # accessor index -> (array, componentType, normalized)
    for mi, mesh in enumerate(js["meshes"]):
        pos_all = np.concatenate([arr(p["attributes"]["POSITION"]) for p in mesh["primitives"]]).astype(np.float64)
        lo, hi = pos_all.min(0), pos_all.max(0); c = (lo + hi) / 2; h = np.maximum((hi - lo) / 2, 1e-9)
        for p in mesh["primitives"]:
            pa = p["attributes"]["POSITION"]; q = np.clip(np.round((arr(pa) - c) / h * 32767), -32767, 32767).astype(np.int16)
            new_acc[pa] = (q, 5122, True)
            if "TEXCOORD_0" in p["attributes"]:
                ua = p["attributes"]["TEXCOORD_0"]; uv = arr(ua)
                if uv.min() >= -1e-4 and uv.max() <= 1 + 1e-4:
                    new_acc[ua] = (np.clip(np.round(uv * 65535), 0, 65535).astype(np.uint16), 5123, True)
        D = np.eye(4); D[:3, 3] = c; D[:3, :3] = np.diag(h)
        for n in js["nodes"]:
            if n.get("mesh") == mi:
                M = node_matrix(n) @ D
                for k in ("translation", "rotation", "scale"): n.pop(k, None)
                n["matrix"] = [float(v) for v in M.T.reshape(-1)]
    # orphan images (extra PBR maps the slim step dropped from the buffer) -> 1x1 data URI so any loader is happy
    PX = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    for img in js.get("images", []):
        if "bufferView" in img:
            img.clear(); img["uri"] = PX; img["mimeType"] = "image/png"
    # rebuild the binary: one tightly packed bufferView per accessor
    nb = bytearray(); bvs = []
    for ai, a in enumerate(js["accessors"]):
        if ai in new_acc:
            data, ct, norm = new_acc[ai]
            a["componentType"] = ct; a["normalized"] = norm
        else:
            data = arr(ai)
        if "min" in a or a.get("type") == "VEC3" and ai in new_acc:
            a["min"] = [float(v) for v in data.min(0)]; a["max"] = [float(v) for v in data.max(0)]
        while len(nb) % 4: nb.append(0)
        raw = np.ascontiguousarray(data).tobytes()
        bv = {"buffer": 0, "byteOffset": len(nb), "byteLength": len(raw)}
        if ai in new_acc: bv["byteStride"] = data.shape[1] * data.dtype.itemsize if data.shape[1] * data.dtype.itemsize % 4 == 0 else None
        if bv.get("byteStride") is None: bv.pop("byteStride", None)
        bvs.append(bv); a["bufferView"] = len(bvs) - 1; a.pop("byteOffset", None); nb += raw
    js["bufferViews"] = bvs; js["buffers"] = [{"byteLength": len(nb)}]
    js["extensionsUsed"] = sorted(set(js.get("extensionsUsed", [])) | {"KHR_mesh_quantization"})
    js["extensionsRequired"] = sorted(set(js.get("extensionsRequired", [])) | {"KHR_mesh_quantization"})
    while len(nb) % 4: nb.append(0)
    jb = json.dumps(js, separators=(",", ":")).encode()
    while len(jb) % 4: jb += b" "
    out = b"glTF" + struct.pack("<II", 2, 12 + 8 + len(jb) + 8 + len(nb)) + struct.pack("<II", len(jb), 0x4E4F534A) + jb + struct.pack("<II", len(nb), 0x004E4942) + bytes(nb)
    gz = gzip.compress(out, 9); op = path.replace("/s_", "/q_"); open(op, "wb").write(gz)
    print(f"{os.path.basename(path)}: {len(b)/1e6:.2f} MB raw -> {len(out)/1e6:.2f} MB quantized, gz {len(gz)/1e6:.2f} MB (was {os.path.getsize(path)/1e6:.2f})")
for tag in ("mustache", "car", "turtle"):
    im = Image.open(f"/tmp/bake2/{tag}.png").convert("RGB"); im.thumbnail((512, 512)); im.save(f"/tmp/bake2/thumb_{tag}.jpg", "JPEG", quality=82)
    print(f"thumb_{tag}.jpg {os.path.getsize(f'/tmp/bake2/thumb_{tag}.jpg')/1e3:.0f} KB {im.size}")
