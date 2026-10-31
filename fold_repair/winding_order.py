#!/usr/bin/env python
"""Ray-ordering check of fitted windings (idea from villa issue #2167 / kit stackcheck).

Along every radial ray from the umbilicus, winding w must lie outside w-1 and inside w+1.
For each (z bin, angle bin) take the median radius of each winding; then for each consecutive
pair report
  inverted : r[w+1] <= r[w]                      (windings cross: the fit switched sheets)
  squeezed : gap < 0.4 x local median gap         (two windings on the same sheet)
  stretched: gap > 1.8 x local median gap         (a sheet skipped between them)
A winding's 'bad' fraction = share of its bins touching any of these.

usage: winding_order.py RUN --umb UMB.json [--z0 --z1] [--zbin 100] [--abin 3] [--npz flags.npz] [--png out.png] [--json out.json]
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import tifffile

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run", help="fit_spiral output dir (contains */meshes/fitted/wNNN)")
ap.add_argument("--zbin", type=float, default=100)
ap.add_argument("--abin", type=float, default=3)
ap.add_argument("--umb", required=True, help="umbilicus json")
ap.add_argument("--z0", type=float, default=-1e9, help="only use mesh points with z >= z0")
ap.add_argument("--z1", type=float, default=1e9, help="only use mesh points with z < z1")
ap.add_argument("--png")
ap.add_argument("--json")
ap.add_argument("--npz", help="save per-bin bad flags: windings, zmin, zbin, abin, bad[W, nz, na], has[W, nz, na]")
a = ap.parse_args()

mesh = next(Path(a.run).glob("*/meshes/fitted"))
cp = json.load(open(a.umb))["control_points"]
uz, ux, uy = (np.array([p[k] for p in cp], float) for k in "zxy")
o = np.argsort(uz); uz, ux, uy = uz[o], ux[o], uy[o]

wdirs = sorted((d for d in mesh.iterdir() if re.fullmatch(r"w\d+", d.name)), key=lambda d: int(d.name[1:]))
ws = [int(d.name[1:]) for d in wdirs]
zmin = zmax = None
pts = {}
for w, d in zip(ws, wdirs):
    x, y, z = (tifffile.imread(d / f"{c}.tif").ravel() for c in "xyz")
    m = (x > 0) & (y > 0) & (z > 0) & (z >= a.z0) & (z < a.z1)
    x, y, z = x[m], y[m], z[m]
    cx, cy = np.interp(z, uz, ux), np.interp(z, uz, uy)
    pts[w] = (z, np.degrees(np.arctan2(y - cy, x - cx)) % 360, np.hypot(x - cx, y - cy))
    zmin = z.min() if zmin is None else min(zmin, z.min()); zmax = z.max() if zmax is None else max(zmax, z.max())

nz = int(np.ceil((zmax - zmin + 1) / a.zbin)); na = int(np.ceil(360 / a.abin))
R = np.full((len(ws), nz, na), np.nan, np.float32)
for i, w in enumerate(ws):
    z, th, r = pts[w]
    zi = ((z - zmin) // a.zbin).astype(int); ai = (th // a.abin).astype(int) % na
    key = zi * na + ai
    order = np.argsort(key); key, r = key[order], r[order]
    u, start = np.unique(key, return_index=True)
    med = np.array([np.median(s) for s in np.split(r, start[1:])])
    R[i].flat[u] = med

gap = R[1:] - R[:-1]                       # (W-1, nz, na)
both = np.isfinite(gap)
loc = np.nanmedian(np.where(gap > 0, gap, np.nan), axis=0)   # per-bin median spacing across windings
inv = both & (gap <= 0)
sq = both & ~inv & (gap < 0.4 * loc)
st = both & ~inv & (gap > 1.8 * loc)
bad_pair = inv | sq | st
bad_w = np.zeros(R.shape, bool); bad_w[1:] |= bad_pair; bad_w[:-1] |= bad_pair
has = np.isfinite(R)

res = {"run": a.run, "pairs": int(both.sum()), "inverted": round(float(inv.sum() / max(both.sum(), 1)), 4),
       "squeezed": round(float(sq.sum() / max(both.sum(), 1)), 4), "stretched": round(float(st.sum() / max(both.sum(), 1)), 4),
       "median_gap_vx": round(float(np.nanmedian(loc)), 1), "windings": {}}
for i, w in enumerate(ws):
    n = int(has[i].sum())
    if n: res["windings"][f"w{w:03d}"] = {"bins": n, "bad": round(float((bad_w[i] & has[i]).sum() / n), 3)}
print(f"{a.run}: {res['pairs']} pair-bins  inverted {res['inverted']:.1%}  squeezed {res['squeezed']:.1%}  "
      f"stretched {res['stretched']:.1%}  median gap {res['median_gap_vx']} vx")
bad = sorted(res["windings"].items(), key=lambda kv: kv[1]["bad"])
print("cleanest:", ", ".join(f"{k} {v['bad']:.0%}" for k, v in bad[:8]))
print("worst:   ", ", ".join(f"{k} {v['bad']:.0%}" for k, v in bad[-8:]))
if a.json: Path(a.json).write_text(json.dumps(res, indent=1))
if a.npz:
    np.savez_compressed(a.npz, windings=np.array(ws), zmin=zmin, zbin=a.zbin, abin=a.abin, bad=bad_w & has, has=has)
if a.png:
    from PIL import Image
    # rows = windings, cols = angle; value = share of z bins bad (white = bad, black = clean, grey = no data)
    frac = np.where(has.any(1), (bad_w & has).sum(1) / np.maximum(has.sum(1), 1), np.nan)
    img = np.where(np.isfinite(frac), frac * 255, 60).astype(np.uint8)
    Image.fromarray(img).resize((na * 4, len(ws) * 4), Image.NEAREST).save(a.png)
