#!/usr/bin/env python
"""Post-fit fold repair: move fitted vertices in folded regions onto the sheet line chosen by winding counting.

Forcing counted sheet positions into the spiral fit itself breaks the spiral (a crushed fold cannot be expressed by
it), so this repairs the finished mesh instead. For every winding of RUN and every vertex whose angle lies in --wedge
and whose z lies in the m7 surface map: candidates = m7 skeleton crossings on the vertex's own ray within +-band of it;
references = the same winding's fitted points at --ref-angles in that slice (safe side); count = median line integral
of Lasagna grad_mag x normal (normal flipped away from the umbilicus) from the references to each candidate, scaled per
slice so neighbouring windings are one wrap apart. A confident candidate (|count| < --acc, runner-up >= --margin
further) replaces the vertex's x, y (z kept). Unconfident vertices keep the fit. Writes a copy of the run with the
repaired windings (same layout, so winding_order.py / detector_eval.py score it like a fit).

With --risk (a radiality_map.py npz of RUN) only vertices in risky (folded) cells of their own winding are touched:
the label-free gate that keeps the repair from degrading unfolded regions.

Coordinates: RUN, SURFACE_ZARR and the umbilicus share one voxel frame (Paris 4: L2, 9.6 um); the Lasagna respool
sidecars are read 4x coarser in z, y and x (fit_spiral's group-4 pools).

usage: fold_repair.py RUN SURFACE_ZARR OUT_RUN --dataset SPIRAL_DATASET [--z0 10000 --z1 10890] [--wedge 0 55]
       [--risk RUN_radiality.npz] [--acc 0.5 --margin 0 --fill 6 --distinct 4]
"""
import argparse, json, shutil, time
from pathlib import Path
import numpy as np, tifffile, zarr
from scipy.ndimage import map_coordinates
from skimage.morphology import skeletonize

import sys; sys.path.insert(0, str(Path(__file__).parent))
from winding_pool import Pool

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run", help="fit_spiral output dir (contains */meshes/fitted/wNNN/{x,y,z}.tif)")
ap.add_argument("surface", help="m7 surface prediction array (an OME-Zarr level, e.g. MAP.zarr/0), u8 0/255, same voxel frame as the fit")
ap.add_argument("out", help="output run dir (copy of RUN's fitted meshes with repaired windings + fold_repair.json)")
ap.add_argument("--dataset", default="data/PHercParis4/spiral_dataset", help="fit_spiral dataset dir; supplies the default --umb, --las-grad, --las-normals (default: %(default)s)")
ap.add_argument("--z0", type=int, default=10000); ap.add_argument("--z1", type=int, default=10890)
ap.add_argument("--wedge", type=float, nargs=2, default=[0, 55], help="angle range (deg around the umbilicus) of vertices to repair (default: Paris 4 fold wedge %(default)s)")
ap.add_argument("--ref-angles", type=float, nargs="+", default=[320, 330, 340], help="safe-side angles (deg) whose fitted radius anchors the count (default: %(default)s)")
ap.add_argument("--box", type=int, nargs=4, default=[1500, 6200, 4880, 7700], metavar=("Y0", "Y1", "X0", "X1"), help="fixed slice region read from the map and pools (default: Paris 4 z10000-12000 region %(default)s)")
ap.add_argument("--umb", help="umbilicus json with control_points [{x,y,z}] (default: DATASET/umbilicus.json)"); ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--band", type=float, default=60.0); ap.add_argument("--acc", type=float, default=0.4); ap.add_argument("--margin", type=float, default=0.6)
ap.add_argument("--fill", type=int, default=0, help="also shift unmoved wedge vertices radially by the shift interpolated between moved vertices of the same row at most this many columns apart (0 = off)")
ap.add_argument("--distinct", type=float, default=0.0, help="drop the less confident of two different windings' moves that land within this many vox in one slice (0 = off)")
ap.add_argument("--chain", type=int, default=0, help="count from this many nearest trusted points behind each vertex, including vertices repaired earlier in the slice (0 = only the safe-side references)")
ap.add_argument("--keep-ok", action="store_true", help="leave a vertex alone when its own fitted position already counts as the winding (|count| < --acc)")
ap.add_argument("--auto-box", type=int, default=0, help="per slice, use y = umbilicus +- this, x = umbilicus - 200 .. + this instead of --box (0 = fixed --box)")
ap.add_argument("--path-row", action="store_true", help="count along the fitted row from the reference columns (not a straight chord), then radially to the candidate")
ap.add_argument("--debug-w", type=int, default=-1)
ap.add_argument("--risk", help="radiality_map.py npz of RUN: repair only vertices in risky cells of their own winding")
ap.add_argument("--risk-thr", type=float, default=0.7)
ap.add_argument("--slab-gate", type=float, default=0.0, help="keep moves only in 100-slice slabs whose median move <= this x the local sheet gap (0 = off; ~1.2 suggested)")
ap.add_argument("--slab-max-move", type=float, default=0.0, help="keep moves only in 100-slice slabs whose median move <= this many vox (0 = off)")
ap.add_argument("--las-grad", help="grad_mag respool sidecar (default: DATASET/lasagna_inputs/las_008_grad_mag.ome.zarr.respool_g4)")
ap.add_argument("--las-normals", help="nx/ny pair respool sidecar (default: DATASET/lasagna_inputs/las_008_nx.ome.zarr.respool_g4_pair)")
ap.add_argument("--zq", type=int, default=2, help="vertices are evaluated on the map slice rounded to this many voxels")
a = ap.parse_args()
t0 = time.time()
y0, y1, x0, x1 = a.box
LAS = Path(a.dataset) / "lasagna_inputs"
cp = json.load(open(a.umb or Path(a.dataset) / "umbilicus.json"))["control_points"]
uz = np.array([p["z"] for p in cp], float); o = np.argsort(uz)
UX = np.array([p["x"] for p in cp], float)[o]; UY = np.array([p["y"] for p in cp], float)[o]; uz = uz[o]
src_mesh = next(Path(a.run).glob("*/meshes/fitted")); rel = src_mesh.relative_to(a.run)
out_mesh = Path(a.out) / rel
if not out_mesh.exists(): shutil.copytree(src_mesh, out_mesh)
W = {}
for d in sorted(src_mesh.glob("w[0-9][0-9][0-9]")):
    W[int(d.name[1:])] = [tifffile.imread(d / f"{c}.tif").astype(np.float64) for c in "xyz"]
S = zarr.open(a.surface, mode="r")
GM = Pool(a.las_grad or LAS / "las_008_grad_mag.ome.zarr.respool_g4"); NP = Pool(a.las_normals or LAS / "las_008_nx.ome.zarr.respool_g4_pair")
L0, L1, M0, M1 = y0 // 4, y1 // 4 + 1, x0 // 4, x1 // 4 + 1
wspan = (a.wedge[1] - a.wedge[0]) % 360

RK = np.load(a.risk) if a.risk else None
if RK is not None:
    RQ = RK["radiality"]; RW = {int(w): i for i, w in enumerate(RK["windings"])}; RZ0, RZB, RAB = float(RK["zmin"]), float(RK["zbin"]), float(RK["abin"])
# vertices to repair, grouped by evaluation slice
todo = {}
for w, (X, Y, Z) in W.items():
    v = (X > 0) & (Y > 0) & (Z >= a.z0) & (Z <= a.z1)
    for r, c in zip(*np.nonzero(v)):
        z = Z[r, c]; cx, cy = np.interp(z, uz, UX), np.interp(z, uz, UY)
        th = np.degrees(np.arctan2(Y[r, c] - cy, X[r, c] - cx)) % 360
        if (th - a.wedge[0]) % 360 > wspan: continue
        if RK is not None:   # only where this winding's sheet runs radially (label-free fold flag, radiality_map.py)
            if w not in RW: continue
            zi = int((z - RZ0) // RZB); ai = int(th // RAB) % RQ.shape[2]
            if not (0 <= zi < RQ.shape[1]) or not (np.isfinite(RQ[RW[w], zi, ai]) and RQ[RW[w], zi, ai] < a.risk_thr): continue
        todo.setdefault(int(round(z / a.zq) * a.zq), []).append((w, r, c, th))
stats = {"vertices": sum(len(v) for v in todo.values()), "moved": 0, "no_candidate": 0, "unconfident": 0, "slices": len(todo)}
moves = []; slice_gap = {}
for z, verts in sorted(todo.items()):
    cx, cy = np.interp(z, uz, UX), np.interp(z, uz, UY); zl = int(round(z / 4))
    if a.auto_box:   # box follows the umbilicus (it drifts ~600 vox in x over Paris 4 z10000-15000); wedge and refs lie right of it
        y0, y1 = int(max(0, cy - a.auto_box)), int(min(S.shape[1], cy + a.auto_box)); x0, x1 = int(max(0, cx - 200)), int(min(S.shape[2], cx + a.auto_box))
        L0, L1, M0, M1 = y0 // 4, y1 // 4 + 1, x0 // 4, x1 // 4 + 1
    g = GM.slice(zl, L0, L1, M0, M1).astype(np.float32) / 1000.0
    nx = (NP.slice(zl, L0, L1, M0, M1, 0).astype(np.float32) - 128) / 127; ny = (NP.slice(zl, L0, L1, M0, M1, 1).astype(np.float32) - 128) / 127
    yy, xx = np.mgrid[L0:L1, M0:M1].astype(np.float32)
    sgn = np.where(nx * (xx * 4 - cx) + ny * (yy * 4 - cy) < 0, -1.0, 1.0); Fx, Fy = g * nx * sgn, g * ny * sgn

    def integ(p, q):
        n = max(8, int(np.hypot(*(q - p)) / 2)); t = np.linspace(0, 1, n)[:, None]
        P = p * (1 - t) + q * t; cc = np.stack([P[:, 1] / 4 - L0, P[:, 0] / 4 - M0]); dl = (q - p) / (n - 1)
        return float(np.sum(map_coordinates(Fx, cc, order=1) * dl[0] + map_coordinates(Fy, cc, order=1) * dl[1]))

    sk = skeletonize(np.asarray(S[z, y0:y1, x0:x1]).astype(np.float32) / 255.0 > a.thr)
    if not sk.any(): continue
    # each winding's fitted radius at the reference angles in this slice (vertices within 10 slices)
    refs = {}
    for w, (X, Y, Z) in W.items():
        m = (X > 0) & (np.abs(Z - z) < 10)
        if not m.any(): continue
        th = np.degrees(np.arctan2(Y[m] - cy, X[m] - cx)) % 360; rr = np.hypot(X[m] - cx, Y[m] - cy)
        for ang in a.ref_angles:
            k = np.abs(((th - ang + 180) % 360) - 180) < 3
            if k.any(): refs.setdefault(w, []).append((ang, float(np.median(rr[k]))))
    pt = lambda r, ang: np.array([cx + r * np.cos(np.radians(ang)), cy + r * np.sin(np.radians(ang))])
    cal = np.array([integ(pt(r1, ang), pt(r2, ang2)) for w in refs if w + 1 in refs
                    for ang, r1 in refs[w] for ang2, r2 in refs[w + 1] if ang2 == ang])
    cal = cal[cal > 0]
    if len(cal) < 5: continue
    scale = 1.0 / np.median(cal)
    gaps_r = [r2 - r1 for w in refs if w + 1 in refs for ang, r1 in refs[w] for ang2, r2 in refs[w + 1] if ang2 == ang and r2 > r1]
    if gaps_r: slice_gap[z] = float(np.median(gaps_r))   # local sheet spacing from the fit's neighbouring windings
    # trusted points per winding: the safe-side references, plus (with --chain) every vertex repaired so far in this
    # slice; vertices are processed in wedge order so each one is counted from the nearest trusted points behind it
    trusted = {w: [(ang, pt(rr_, ang)) for ang, rr_ in v] for w, v in refs.items()}
    rowcum = {}   # --path-row: (w, r) -> (valid columns, cumulative winding integral along the fitted row)

    def row_count(w, r, c):
        """Winding count from the row's reference columns to column c, integrated along the fitted row itself."""
        if (w, r) not in rowcum:
            X, Y, _ = W[w]; cols = np.flatnonzero(X[r] > 0)
            P = np.stack([X[r, cols], Y[r, cols]], 1); seg = [integ(P[i], P[i + 1]) for i in range(len(P) - 1)]
            th_r = np.degrees(np.arctan2(P[:, 1] - cy, P[:, 0] - cx)) % 360
            refc = np.flatnonzero((th_r - min(a.ref_angles)) % 360 <= (max(a.ref_angles) - min(a.ref_angles)) % 360)
            rowcum[(w, r)] = (cols, np.concatenate([[0.0], np.cumsum(seg)]), refc)
        cols, cum, refc = rowcum[(w, r)]
        if not len(refc): return None
        i = int(np.searchsorted(cols, c)); return float(np.median(cum[i] - cum[refc]))
    for w, r, c, th in sorted(verts, key=lambda t: (t[3] - a.wedge[0]) % 360):
        if len(refs.get(w, [])) < 2: stats["unconfident"] += 1; continue
        if a.chain > 0:   # the --chain nearest trusted points behind this vertex (smallest forward angle gap)
            tp = sorted(trusted[w], key=lambda t: (th - t[0]) % 360)[:a.chain]
        else:
            tp = trusted[w][:len(refs[w])]
        X, Y, Z = W[w]; r_fit = float(np.hypot(X[r, c] - cx, Y[r, c] - cy)); d = np.array([np.cos(np.radians(th)), np.sin(np.radians(th))])
        rs = np.arange(max(1.0, r_fit - a.band), r_fit + a.band, 0.5)
        px = cx + rs * d[0] - x0; py = cy + rs * d[1] - y0
        ok = (px >= 1) & (px < sk.shape[1] - 1) & (py >= 1) & (py < sk.shape[0] - 1)
        hit = np.zeros(len(rs), bool); ix, iy = px[ok].astype(int), py[ok].astype(int)
        hit[ok] = sk[iy, ix] | sk[iy + 1, ix] | sk[iy, ix + 1] | sk[iy - 1, ix] | sk[iy, ix - 1]
        cross, run = [], []
        for rr_, h in zip(rs, hit):
            if h: run.append(rr_)
            elif run: cross.append(float(np.mean(run))); run = []
        if run: cross.append(float(np.mean(run)))
        if not cross: stats["no_candidate"] += 1; continue
        C = [pt(q, th) for q in cross]
        if a.path_row:   # along the fitted row to this vertex, then a short radial step to each candidate
            base = row_count(w, r, c)
            if base is None: stats["unconfident"] += 1; continue
            here = np.array([X[r, c], Y[r, c]]); dw = np.abs([(base + integ(here, cnd)) * scale for cnd in C]); fit_dw = abs(base * scale)
            if a.debug_w == w: stats.setdefault('debug', []).append([int(r), int(c), round(float(th), 1), round(base * scale, 2), int(np.searchsorted(rowcum[(w, r)][0], c)), [int(x) for x in rowcum[(w, r)][2][:3]], len(rowcum[(w, r)][0])])
        else:
            dw = np.abs([np.median([integ(q, cnd) for _, q in tp]) * scale for cnd in C]); fit_dw = None
        oo = np.argsort(dw)
        if dw[oo[0]] >= a.acc or (len(oo) > 1 and dw[oo[1]] < dw[oo[0]] + a.margin): stats["unconfident"] += 1; continue
        if a.keep_ok and (fit_dw if fit_dw is not None else abs(np.median([integ(q, np.array([X[r, c], Y[r, c]])) for _, q in tp]) * scale)) < a.acc:
            stats["kept_ok"] = stats.get("kept_ok", 0) + 1; continue   # the fit's own position already counts as this winding
        moves.append((w, r, c, C[oo[0]][0], C[oo[0]][1], float(np.hypot(*(C[oo[0]] - [X[r, c], Y[r, c]]))), float(dw[oo[0]]), z, th))
        stats["moved"] += 1
        if a.chain > 0: trusted[w].append((th, C[oo[0]]))
if (a.slab_gate > 0 or a.slab_max_move > 0) and moves:   # label-free go/no-go: keep a z slab's moves only if its median move is ~one sheet
    slab = lambda z: int(z // 100)
    by_slab = {}
    for m in moves: by_slab.setdefault(slab(m[7]), []).append(m)
    keep_slabs = set(); stats["slabs"] = {}
    for k, ms in by_slab.items():
        g = [slice_gap[z] for z in slice_gap if slab(z) == k]
        if not g: continue
        med = float(np.median([m[5] for m in ms])); gap = float(np.median(g)); ok = (med <= a.slab_gate * gap) if a.slab_gate > 0 else (med <= a.slab_max_move)
        stats["slabs"][k * 100] = [round(med, 1), round(gap, 1), bool(ok)]
        if ok: keep_slabs.add(k)
    moves = [m for m in moves if slab(m[7]) in keep_slabs]; stats["moved_after_slab_gate"] = len(moves)
if a.distinct > 0:   # neighbouring windings may not claim the same sheet line: keep the more confident claim
    from scipy.spatial import cKDTree
    keep = np.ones(len(moves), bool); by_z = {}
    for i, m in enumerate(moves): by_z.setdefault(m[7], []).append(i)
    for z, idx in by_z.items():
        P = np.array([[moves[i][3], moves[i][4]] for i in idx])
        for i, j in cKDTree(P).query_pairs(a.distinct):
            mi, mj = moves[idx[i]], moves[idx[j]]
            if mi[0] == mj[0]: continue
            loser = idx[i] if mi[6] > mj[6] else idx[j]; keep[loser] = False
    stats["dropped_conflicts"] = int((~keep).sum()); stats["moved"] = int(keep.sum())
    moves = [m for m, k in zip(moves, keep) if k]
wedge_by_w = {}
for verts in todo.values():
    for ww, r, c, _ in verts: wedge_by_w.setdefault(ww, set()).add((r, c))
for w in W:
    mv = [m for m in moves if m[0] == w]
    if not mv: continue
    X, Y, _ = W[w]; X, Y = X.copy(), Y.copy()
    for _, r, c, nx_, ny_, *_ in mv: X[r, c], Y[r, c] = nx_, ny_
    if a.fill > 0:   # radial shift of moved vertices, interpolated along each row across gaps of <= a.fill columns
        X0, Y0, Z0 = W[w]; wedge_v = wedge_by_w.get(w, set())
        for r in sorted({m[1] for m in mv}):
            cols = sorted((m[2], m) for m in mv if m[1] == r)
            for (c1, m1), (c2, m2) in zip(cols, cols[1:]):
                if c2 - c1 < 2 or c2 - c1 > a.fill: continue
                def dr(m):
                    cx, cy = np.interp(Z0[r, m[2]], uz, UX), np.interp(Z0[r, m[2]], uz, UY)
                    return np.hypot(m[3] - cx, m[4] - cy) - np.hypot(X0[r, m[2]] - cx, Y0[r, m[2]] - cy)
                d1, d2 = dr(m1), dr(m2)
                for c in range(c1 + 1, c2):
                    if (r, c) not in wedge_v or X0[r, c] <= 0: continue
                    cx, cy = np.interp(Z0[r, c], uz, UX), np.interp(Z0[r, c], uz, UY)
                    t = (c - c1) / (c2 - c1); dd = d1 * (1 - t) + d2 * t; rr = np.hypot(X0[r, c] - cx, Y0[r, c] - cy)
                    X[r, c] = cx + (X0[r, c] - cx) * (rr + dd) / rr; Y[r, c] = cy + (Y0[r, c] - cy) * (rr + dd) / rr
                    stats["filled"] = stats.get("filled", 0) + 1
    tifffile.imwrite(out_mesh / f"w{w:03d}" / "x.tif", X.astype(np.float32)); tifffile.imwrite(out_mesh / f"w{w:03d}" / "y.tif", Y.astype(np.float32))
shift = np.array([m[5] for m in moves]) if moves else np.zeros(1)
stats.update({"median_shift_vox": round(float(np.median(shift)), 1), "share_shift_gt_8": round(float((shift > 8).mean()), 3), "seconds": round(time.time() - t0)})
(Path(a.out) / "fold_repair.json").write_text(json.dumps(stats))
print(json.dumps(stats))
