#!/usr/bin/env python
"""Fetch a z/y/x region of a published m7 surface prediction (OME-Zarr v2, '/' chunk keys) into a local zarr.

Only the chunks that cover the region are downloaded (missing chunks read as 0 = no surface), together with the
group and array metadata, so fold_repair.py can open OUT/0 as the full-size array in the scroll's own voxel frame.
Existing chunks are skipped, so the command is resumable. Standard library only.

Default source: PHerc. Paris 4 m7 surface prediction at L2 (9.6 um, 18946 x 8174 x 8174, u8 0/255 at threshold 0.2,
192^3 chunks) from the Vesuvius Challenge open-data bucket.

usage: fetch_m7_map.py OUT.zarr --z0 10000 --z1 12000 [--y0 600 --y1 7400 --x0 3900 --x1 7700] [--url URL] [--jobs 16]
"""
import argparse, json, os, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PARIS4_M7_L2 = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/representations/predictions/"
                "surfaces/20260411134726-surface-20260413222639-surface-m7-L2-th0.2.zarr")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("out", help="local zarr directory to create/extend")
ap.add_argument("--url", default=PARIS4_M7_L2, help="published zarr root (default: Paris 4 m7 L2)")
ap.add_argument("--level", default="0", help="multiscale level to fetch (default: %(default)s)")
ap.add_argument("--z0", type=int, required=True); ap.add_argument("--z1", type=int, required=True)
ap.add_argument("--y0", type=int, default=600); ap.add_argument("--y1", type=int, default=7400)
ap.add_argument("--x0", type=int, default=3900); ap.add_argument("--x1", type=int, default=7700)
ap.add_argument("--jobs", type=int, default=16, help="parallel downloads (default: %(default)s)")
a = ap.parse_args()


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


out = Path(a.out); arr = out / a.level; arr.mkdir(parents=True, exist_ok=True)
for name in (".zgroup", ".zattrs"):
    try:
        (out / name).write_bytes(get(f"{a.url}/{name}"))
    except urllib.error.HTTPError:
        pass
zarray = get(f"{a.url}/{a.level}/.zarray"); (arr / ".zarray").write_bytes(zarray)
meta = json.loads(zarray); cz, cy, cx = meta["chunks"]; sep = meta.get("dimension_separator", ".")
keys = [sep.join(map(str, (z, y, x)))
        for z in range(a.z0 // cz, a.z1 // cz + 1) for y in range(a.y0 // cy, a.y1 // cy + 1) for x in range(a.x0 // cx, a.x1 // cx + 1)]


def fetch(k):
    dst = arr / k
    if dst.exists() and dst.stat().st_size > 0: return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True); tmp = dst.with_name(dst.name + ".part")
    try:
        tmp.write_bytes(get(f"{a.url}/{a.level}/{k}")); os.replace(tmp, dst); return "ok"
    except urllib.error.HTTPError as e:   # 404 = empty chunk (not stored), reads as fill_value
        tmp.unlink(missing_ok=True); return "missing" if e.code in (403, 404) else f"http {e.code}"


with ThreadPoolExecutor(a.jobs) as ex:
    res = list(ex.map(fetch, keys))
print(json.dumps({"chunks": len(keys), **{s: res.count(s) for s in sorted(set(res))}}))
