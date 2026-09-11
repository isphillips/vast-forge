"""Solidify hair-like geometry BEFORE export.

TRELLIS generates hair, fur, bristles and manes as thin, torn sheets one voxel thick. The exporter's decimation
cannot afford them at a mobile triangle budget, and its hole fill cannot close them (their rims are junctions of
crossing sheets, not simple loops) — that is the eyebrow and mustache holes. Instead of patching the surface, work on
the model's own occupancy: a morphological closing on the 1024³ voxel grid (dilate by r, erode by r) merges strands
that lie within ~2r voxels of each other and fills gaps narrower than that, while flat surfaces are left exactly
where they were. Marching cubes then rebuilds one clean surface, and the exporter bakes it with the model's ORIGINAL
per-voxel colours; voxels the closing added inherit the colour of their nearest original voxel, so the new surface
does not sample empty space (which the app draws as grey).

Measured on the mustache-and-glasses facet (same generation, 120k faces): remesh export kept 92% of the raw surface
within two voxels in 5,808 pieces; solidify r=2 kept 97% in 332 pieces, with 0% low-alpha texels, in half the export
time. r=2 is the default (gaps up to ~4 voxels = 0.4% of the object close); r=3 starts to blunt fine detail.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F


def solidify(mesh, radius: int = 2, res: int = 1024, pad: int = 8, log=print):
    """mesh: the pipeline's MeshWithVoxel (coords N×3 int voxel indices on a `res` grid, voxel_size, attrs).
    Returns (vertices float32 cuda, faces int32 cuda, closed_coords int64 cuda) — the closed volume's occupied voxels
    are handed back so the caller can extend the attribute volume (see extend_attrs)."""
    from skimage import measure

    t0 = time.time()
    c = mesh.coords[:, -3:].long()                                   # drop a leading batch column if present
    lo = (c.min(dim=0).values - pad).clamp(min=0)
    hi = (c.max(dim=0).values + pad).clamp(max=res - 1)
    shape = (hi - lo + 1).tolist()
    vol = torch.zeros(shape, dtype=torch.float16, device=c.device)
    idx = c - lo
    vol[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    occ0 = int(torch.count_nonzero(vol).item())                      # (a fp16 .sum() overflows past 65,504)
    k = 2 * radius + 1
    dil = F.max_pool3d(vol[None, None], k, stride=1, padding=radius)            # dilate
    closed = (1 - F.max_pool3d(1 - dil, k, stride=1, padding=radius))[0, 0]     # erode → closing
    del dil, vol
    occ1 = int(torch.count_nonzero(closed).item())
    log(f"[solidify] r={radius} box={shape} occupied {occ0} -> {occ1} voxels (+{(occ1 - occ0) / max(occ0, 1) * 100:.0f}%) "
        f"gpu {time.time() - t0:.1f}s")

    t1 = time.time()
    field = np.pad(closed.float().cpu().numpy(), 2)                  # empty border → closed where the object meets the cube edge
    verts, faces, _, _ = measure.marching_cubes(field, level=0.5)   # verts in voxel units of the padded, cropped box
    verts = verts - 2.0
    # voxel units → model space: the unit cube [-0.5, 0.5], voxel centres at (i + 0.5) * voxel_size - 0.5
    vs = float(mesh.voxel_size) if not torch.is_tensor(mesh.voxel_size) else float(mesh.voxel_size.flatten()[0])
    verts_ms = (verts + lo.cpu().numpy()[None, :] + 0.5) * vs - 0.5
    log(f"[solidify] marching cubes: {len(faces)} faces in {time.time() - t1:.0f}s (voxel_size {vs:.6f})")
    closed_coords = torch.nonzero(closed > 0.5) + lo[None, :]
    del closed, field
    return (
        torch.from_numpy(verts_ms.astype(np.float32)).cuda(),
        torch.from_numpy(faces.astype(np.int32)).cuda(),
        closed_coords,
    )


def extend_attrs(mesh, closed_coords, log=print):
    """Give every voxel the closing added the attributes (colour, alpha, ...) of its nearest ORIGINAL voxel, so the
    texture bake samples real colour on the new surface instead of empty space (0% low-alpha texels, vs 3.5%
    without this). Returns (attr_volume, coords) to pass to to_glb in place of mesh.attrs / mesh.coords."""
    from scipy.spatial import cKDTree

    t = time.time()
    coords = mesh.coords
    orig = coords[:, -3:].long().cpu().numpy()
    new = closed_coords.cpu().numpy()
    # only voxels that were not occupied before need a colour
    key_o = orig[:, 0].astype(np.int64) * 1048576 + orig[:, 1].astype(np.int64) * 1024 + orig[:, 2]
    key_n = new[:, 0].astype(np.int64) * 1048576 + new[:, 1].astype(np.int64) * 1024 + new[:, 2]
    new = new[~np.isin(key_n, key_o)]
    if new.shape[0] == 0:
        return mesh.attrs, coords
    _, nn = cKDTree(orig).query(new, k=1, workers=-1)
    nn = torch.from_numpy(nn.astype(np.int64)).to(coords.device)
    new_coords = torch.zeros((new.shape[0], coords.shape[1]), dtype=coords.dtype, device=coords.device)
    new_coords[:, -3:] = torch.from_numpy(new.astype(np.int64)).to(coords.device).to(coords.dtype)
    if coords.shape[1] > 3:
        new_coords[:, :-3] = coords[0, :-3]                          # keep the batch column the pipeline uses
    attrs = torch.cat([mesh.attrs, mesh.attrs[nn]], 0)
    coords_out = torch.cat([coords, new_coords], 0)
    log(f"[solidify] attrs extended: {mesh.attrs.shape[0]} + {new.shape[0]} voxels inherit their nearest colour "
        f"({time.time() - t:.1f}s)")
    return attrs, coords_out
