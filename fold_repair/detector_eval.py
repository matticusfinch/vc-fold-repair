#!/usr/bin/env python
"""Does a label-free per-bin error flag find the places where a fit is off a published surface?

For each reference surface (published GP segment), take the fitted winding that matches it best, bin that winding's
mesh points by (z, angle) like winding_order.py, and call a bin 'off' when its median distance to the reference is
above --off vox. Then compare against the flags in a winding_order.py --npz file: precision/recall of the flag and
the share of off bins it catches.

With --risk (radiality_map.py npz of RUN) it also splits the off rate into risky (folded, radiality < 0.7) and safe
bins: the "risky off" / "safe off" numbers in the README.

usage: detector_eval.py RUN FLAGS.npz REF_TIFXYZ [REF ...] --umb UMB [--ref-scale 0.25] [--z0 --z1] [--off 8] [--risk R.npz]
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import tifffile
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run", help="fit_spiral output dir (contains */meshes/fitted/wNNN)")
ap.add_argument("flags", help="winding_order.py --npz output for RUN (defines the z/angle bins)")
ap.add_argument("refs", nargs="+", help="reference tifxyz segment dirs (x.tif, y.tif, z.tif)")
ap.add_argument("--ref-scale", type=float, default=0.25, help="multiply reference coordinates by this to reach the fit frame (Paris 4: 2.4 um segments -> L2 fit = 0.25)")
ap.add_argument("--umb", required=True, help="umbilicus json")
ap.add_argument("--z0", type=float, default=-1e9); ap.add_argument("--z1", type=float, default=1e9)
ap.add_argument("--off", type=float, default=8.0, help="a bin is off when its median distance to the reference exceeds this (vox)")
ap.add_argument("--risk", help="radiality_map.py npz of the same run: also report off rate in risky (< 0.7) vs safe bins")
ap.add_argument("--dump", help="write per-bin rows (ref, winding, zbin, abin, median_dist, flag) to this CSV")
a = ap.parse_args()

F = np.load(a.flags); ws = list(F["windings"]); zmin, zbin, abin = float(F["zmin"]), float(F["zbin"]), float(F["abin"])
bad, has = F["bad"], F["has"]; nz, na = bad.shape[1:]
cp = json.load(open(a.umb))["control_points"]
uz, ux, uy = (np.array([p[k] for p in cp], float) for k in "zxy"); o = np.argsort(uz); uz, ux, uy = uz[o], ux[o], uy[o]
mesh = next(Path(a.run).glob("*/meshes/fitted"))

def load(d, scale=1.0):
    x, y, z = (tifffile.imread(Path(d) / f"{c}.tif").astype(np.float64).ravel() for c in "xyz")
    m = (x > 0) & (y > 0) & (z > 0)
    P = np.stack([x[m], y[m], z[m]], 1) * scale
    return P[(P[:, 2] >= a.z0) & (P[:, 2] < a.z1)]

tot = {"off": 0, "on": 0, "flag_off": 0, "flag_on": 0}
rows_out = []
for ref in a.refs:
    R = load(ref, a.ref_scale); rt = cKDTree(R)
    sub = R[np.random.default_rng(0).choice(len(R), min(len(R), 20000), replace=False)]
    best = None
    for w in ws:  # winding that best matches the reference (median ref->mesh distance)
        Pm = load(mesh / f"w{w:03d}")
        if len(Pm) == 0: continue
        d, _ = cKDTree(Pm).query(sub, distance_upper_bound=200)
        if np.isfinite(d).mean() < 0.5: continue
        med = np.median(np.where(np.isfinite(d), d, 200))
        if best is None or med < best[1]: best = (w, med, Pm)
    if best is None:
        print(f"{Path(ref).parent.name}: no matching winding"); continue
    w, med, Pm = best; i = ws.index(w)
    cx, cy = np.interp(Pm[:, 2], uz, ux), np.interp(Pm[:, 2], uz, uy)
    th = np.degrees(np.arctan2(Pm[:, 1] - cy, Pm[:, 0] - cx)) % 360
    zi = ((Pm[:, 2] - zmin) // zbin).astype(int); ai = (th // abin).astype(int) % na
    dist, _ = rt.query(Pm, distance_upper_bound=200); dist = np.where(np.isfinite(dist), dist, 200)
    key = zi * na + ai; ok = (zi >= 0) & (zi < nz)
    # only bins the reference actually covers (reference within 40 vox of most of the bin's points)
    stats = {}
    for k in np.unique(key[ok]):
        sel = key == k; dd = dist[sel]
        if (dd < 40).mean() < 0.5 and np.median(dd) >= 40: continue
        stats[k] = np.median(dd)
    off = np.array([v > a.off for v in stats.values()]); fl = np.array([bad[i].flat[k] for k in stats])
    r = {"ref": Path(ref).parent.name, "winding": f"w{w:03d}", "match_median": round(float(med), 1), "bins": len(stats),
         "off_frac": round(float(off.mean()), 3) if len(off) else None,
         "recall": round(float(fl[off].mean()), 3) if off.any() else None,
         "precision": round(float(off[fl].mean()), 3) if fl.any() else None,
         "flag_rate_on": round(float(fl[~off].mean()), 3) if (~off).any() else None}
    print(r)
    rows_out += [(Path(ref).parent.name, w, int(k // na), int(k % na), float(v), int(bad[i].flat[k])) for k, v in stats.items()]
    tot["off"] += int(off.sum()); tot["on"] += int((~off).sum()); tot["flag_off"] += int(fl[off].sum()); tot["flag_on"] += int(fl[~off].sum())
if tot["off"] and tot["on"]:
    print({"all_bins": tot["off"] + tot["on"], "off_frac": round(tot["off"] / (tot["off"] + tot["on"]), 3),
           "recall": round(tot["flag_off"] / tot["off"], 3), "false_flag_rate": round(tot["flag_on"] / tot["on"], 3),
           "precision": round(tot["flag_off"] / max(tot["flag_off"] + tot["flag_on"], 1), 3)})
if a.dump:
    import csv
    with open(a.dump, "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["ref", "winding", "zbin", "abin", "median_dist", "flag"]); wr.writerows(rows_out)
if a.risk and rows_out:
    Rk = np.load(a.risk); rws = list(Rk["windings"]); rad = Rk["radiality"]
    if float(Rk["zbin"]) != zbin or float(Rk["abin"]) != abin:
        print("!! risk map bin sizes differ from flags; skipping risk breakdown")
    else:
        risky = []; safe = []; rz0 = float(Rk["zmin"])
        for ref_, w_, zb_, ab_, d_, _ in rows_out:
            if w_ not in rws: continue
            rzb = int((zmin + (zb_ + 0.5) * zbin - rz0) // zbin)   # same z, the risk map's own bin origin
            if not 0 <= rzb < rad.shape[1]: continue
            v_ = rad[rws.index(w_), rzb, ab_]
            if np.isfinite(v_): (risky if v_ < 0.7 else safe).append(d_ > a.off)
        print({"risky_bins": len(risky), "risky_off": round(float(np.mean(risky)), 3) if risky else None,
               "safe_bins": len(safe), "safe_off": round(float(np.mean(safe)), 3) if safe else None})
