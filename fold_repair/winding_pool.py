"""Read 2-D slices from fit_spiral's respool sidecars (villa/spiral-fitting/pack_resident_pools.py: bricks of
brick_shape^3 voxels in zyx order, indexed by table.npy; table row 0 = empty brick)."""
import json
from pathlib import Path
import numpy as np


class Pool:
    def __init__(self, d):
        d = Path(d); self.meta = json.load(open(d / "meta.json")); self.table = np.load(d / "table.npy")
        self.b = self.meta["brick_shape"][0]; nb = self.b ** 3
        self.ch = [np.memmap(d / f"channel_{i}.u8", np.uint8, "r").reshape(-1, nb) for i in range(len(self.meta["channels"]))]

    def slice(self, z, y0, y1, x0, x1, c=0):
        b = self.b; out = np.zeros((y1 - y0, x1 - x0), np.uint8); gz, zz = divmod(z, b)
        for gy in range(y0 // b, (y1 - 1) // b + 1):
            for gx in range(x0 // b, (x1 - 1) // b + 1):
                row = self.table[gz, gy, gx]
                if row == 0: continue
                brick = self.ch[c][row].reshape(b, b, b)[zz]
                ys, xs = max(y0, gy * b), max(x0, gx * b); ye, xe = min(y1, gy * b + b), min(x1, gx * b + b)
                out[ys - y0:ye - y0, xs - x0:xe - x0] = brick[ys - gy * b:ye - gy * b, xs - gx * b:xe - gx * b]
        return out
