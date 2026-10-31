#!/usr/bin/env python
"""Label-free risk map for a spiral fit: where does the fitted sheet run along the radial direction?

The spiral model assumes each winding sits further out than the last along rays from the umbilicus. Where the sheet
runs nearly parallel to those rays (crushed folds), the fit tends to slip a wrap. On PHerc. Paris 4 z10000-12000
(a tracks + winding-inference fit, 6,699 truth bins) the off-surface rate was 26% for |n.radial| < 0.5 and 4% for > 0.95 (AUC 0.68), the same whether
the normal came from the published surface or from the fit itself.
Per (z bin, angle bin) of each winding: median |surface normal . radial unit vector| (1 = across the ray, safe;
0 = along the ray, risky). Output: npz (windings, zmin, zbin, abin, radiality[W, nz, na]) and an optional summary PNG.

usage: radiality_map.py RUN --umb UMB.json [--z0 --z1] [--zbin 100] [--abin 3] --npz out.npz [--png out.png]
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import tifffile

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run", help="fit_spiral output dir (contains */meshes/fitted/wNNN)"); ap.add_argument("--umb", required=True, help="umbilicus json")
ap.add_argument("--z0", type=float, default=-1e9); ap.add_argument("--z1", type=float, default=1e9)
ap.add_argument("--zbin", type=float, default=100); ap.add_argument("--abin", type=float, default=3)
ap.add_argument("--npz", required=True, help="output risk map (input to fold_repair.py --risk)"); ap.add_argument("--png", help="optional summary image (windings x angle; needs Pillow)")
a = ap.parse_args()

cp = json.load(open(a.umb))["control_points"]
uz, ux, uy = (np.array([p[k] for p in cp], float) for k in "zxy"); o = np.argsort(uz); uz, ux, uy = uz[o], ux[o], uy[o]
mesh = next(Path(a.run).glob("*/meshes/fitted"))
wdirs = sorted((d for d in mesh.iterdir() if re.fullmatch(r"w\d+", d.name)), key=lambda d: int(d.name[1:]))
per = []
for d in wdirs:
    X, Y, Z = (tifffile.imread(d / f"{c}.tif").astype(np.float64) for c in "xyz")
    v = (X > 0) & (Y > 0) & (Z > 0)
    P = np.stack([X, Y, Z], -1); du = np.zeros_like(P); dv = np.zeros_like(P)
    du[1:-1] = P[2:] - P[:-2]; dv[:, 1:-1] = P[:, 2:] - P[:, :-2]
    ok = v.copy(); ok[1:-1] &= v[2:] & v[:-2]; ok[:, 1:-1] &= v[:, 2:] & v[:, :-2]; ok[0] = ok[-1] = False; ok[:, 0] = ok[:, -1] = False
    ok &= (Z >= a.z0) & (Z < a.z1)
    N = np.cross(du, dv); N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-9)
    p, n = P[ok], N[ok]
    cx, cy = np.interp(p[:, 2], uz, ux), np.interp(p[:, 2], uz, uy)
    rad = np.stack([p[:, 0] - cx, p[:, 1] - cy, np.zeros(len(p))], 1); rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-9)
    per.append((int(d.name[1:]), p[:, 2], np.degrees(np.arctan2(p[:, 1] - cy, p[:, 0] - cx)) % 360, np.abs((n * rad).sum(1)),
                np.hypot(p[:, 0] - cx, p[:, 1] - cy)))
zmin = min(x[1].min() for x in per if len(x[1])); zmax = max(x[1].max() for x in per if len(x[1]))
nz = int(np.ceil((zmax - zmin + 1) / a.zbin)); na = int(np.ceil(360 / a.abin))
Rm = np.full((len(per), nz, na), np.nan, np.float32); Rr = np.full_like(Rm, np.nan)   # radiality, median radius
for i, (w, z, th, c, rr) in enumerate(per):
    if not len(z): continue
    key = ((z - zmin) // a.zbin).astype(int) * na + (th // a.abin).astype(int) % na
    order = np.argsort(key); key, c, rr = key[order], c[order], rr[order]
    u, start = np.unique(key, return_index=True)
    Rm[i].flat[u] = [np.median(s) for s in np.split(c, start[1:])]
    Rr[i].flat[u] = [np.median(s) for s in np.split(rr, start[1:])]
np.savez_compressed(a.npz, windings=np.array([x[0] for x in per]), zmin=zmin, zbin=a.zbin, abin=a.abin, radiality=Rm, radius=Rr)
risky = np.nanmean(Rm < 0.7)
print(f"{a.run}: {np.isfinite(Rm).sum()} bins, risky (|n.radial| < 0.7) {risky:.1%}")
if a.png:
    from PIL import Image
    m = np.nanmean(Rm, axis=1)                                   # windings x angle
    img = np.where(np.isfinite(m), np.clip(m, 0, 1) * 255, 40).astype(np.uint8)
    Image.fromarray(img).resize((na * 4, len(per) * 4), Image.NEAREST).save(a.png)
