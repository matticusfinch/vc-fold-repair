# vc-fold-repair

Fixes spiral-fit meshes of Herculaneum scrolls where the papyrus is folded. It runs after the fit and needs no labels: it counts windings along the sheet crossings of the m7 surface prediction and moves fitted vertices onto the right sheet. A label-free risk gate limits the repair to folded parts of the fit.

Submitted to the Vesuvius Challenge progress prizes. Validated on PHerc. Paris 4 only (see [Limitations](#limitations)).

## What it does

A spiral fit (villa `spiral-fitting/fit_spiral.py`) assumes each winding lies further out than the last along every ray from the umbilicus. In crushed folds the sheet runs nearly *along* those rays, and the fit slips onto a neighbouring wrap. Adding the counted positions to the fit as constraints breaks the spiral, because the fold cannot be expressed by it. So the repair works on the finished mesh:

1. **Risk map (`radiality_map.py`, label-free).** For each winding, z bin (100 slices) and angle bin (3 deg), take the median `|surface normal . radial unit vector|` of the fitted mesh. Near 1 means the sheet crosses the ray, which is safe. Near 0 means the sheet runs along the ray, which marks a fold. Cells below 0.7 are *risky*. On Paris 4 z10000-12000 (6,699 truth bins) the off-surface rate was 26% for radiality < 0.5 and 4% for radiality > 0.95 (AUC 0.68).
2. **Fold repair (`fold_repair.py`).** For each vertex in a risky cell of its own winding and inside the angle wedge:
   - **Candidates:** the m7 surface skeleton crossings on the vertex's ray from the umbilicus, within +-60 vox of the vertex.
   - **References:** the same winding's fitted points at safe-side angles (320/330/340 deg) in that slice.
   - **Winding count:** the median line integral of Lasagna `grad_mag x normal` (normal oriented away from the umbilicus) from the references to each candidate. It is scaled per slice so that neighbouring windings are exactly one wrap apart.
   - **Move:** the vertex's x, y move to the candidate whose count is closest to 0, if that count is confidently close (`--acc`, `--margin`). z is kept.
   - **Clean-up:** short gaps along a mesh row get the interpolated radial shift (`--fill`). When two windings claim the same sheet line, only the more confident claim is kept (`--distinct`).

   Vertices without a confident candidate keep the fitted position. The output is a copy of the run with the same layout, so any tool that reads a fit_spiral run reads it too.
3. **Scoring (`winding_order.py` + `detector_eval.py`, needs truth).** Each published reference segment is matched to its best fitted winding. Mesh points are binned by (z, angle), and a bin is *off* when its median distance to the reference is above 8 vox. The off rate is reported overall and split into risky and safe bins using the risk map.

## Inputs and outputs

All geometry is in one voxel frame. For Paris 4 that is the L2 volume (9.6 um, z 0-18946, y/x 0-8174).

| Input | What | Where to get it |
|---|---|---|
| `RUN` | A fit_spiral output directory: `RUN/<run-name>/meshes/fitted/wNNN/{x,y,z}.tif` (tifxyz, one per winding) | `villa/spiral-fitting/fit_spiral.py` with `FIT_SPIRAL_OUT_DIR=RUN` |
| m7 surface map | Published m7 surface prediction, OME-Zarr v2, u8 0/255 | Open-data bucket, for example Paris 4 L2: `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/representations/predictions/surfaces/20260411134726-surface-20260413222639-surface-m7-L2-th0.2.zarr`. `fold_repair/fetch_m7_map.py` downloads only the chunks a band needs (z10000-12000: about 4,600 chunks, about 2 GB). |
| `umbilicus.json` | `{"control_points": [{"x","y","z"}, ...]}` | fit_spiral dataset, for example `https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/umbilicus.json` |
| Lasagna sidecars | `las_008_grad_mag.ome.zarr.respool_g4` (grad_mag) and `las_008_nx.ome.zarr.respool_g4_pair` (nx, ny), packed respool bricks at 1/4 of the fit frame | The same dataset under `lasagna_inputs/`. They are also the fit's own inputs. For a scroll without them, pack the published Lasagna `grad_mag`/`nx`/`ny` group 4 with `villa/spiral-fitting/pack_resident_pools.py --what normals,grad_mag`. |
| Truth (scoring only) | Published reference segments (tifxyz) | Paris 4 published segments `20260602225659-5753_0`, `20260603005223-5753_-1`, `20260603024952-5753_-2` (2.4 um frame, so `--ref-scale 0.25`) |

`fold_repair.py --dataset DIR` points at a fit_spiral dataset directory and fills in the defaults `DIR/umbilicus.json`, `DIR/lasagna_inputs/las_008_grad_mag.ome.zarr.respool_g4` and `DIR/lasagna_inputs/las_008_nx.ome.zarr.respool_g4_pair`. Any of them can be overridden with `--umb`, `--las-grad` or `--las-normals`.

Outputs:

- `radiality_map.py --npz R.npz` writes the risk map (`windings, zmin, zbin, abin, radiality[W, nz, na], radius[W, nz, na]`) and prints the risky share.
- `fold_repair.py ... OUT` writes:
  - `OUT/<run-name>/meshes/fitted/wNNN/{x,y,z}.tif`: all windings copied, with repaired windings' `x.tif`/`y.tif` rewritten.
  - `OUT/fold_repair.json`: vertices considered, moved, filled, conflicts dropped, median shift.

### Getting the Paris 4 inputs

```bash
D=https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4
O=data/PHercParis4/spiral_dataset; mkdir -p $O && cd $O
wget -c $D/umbilicus.json
for d in lasagna_inputs/las_008_nx.ome.zarr.respool_g4_pair/ lasagna_inputs/las_008_grad_mag.ome.zarr.respool_g4/; do
  wget -c -r -np -nH --cut-dirs=3 -R 'index.html*' $D/$d; done
cd -
# (the fit itself also needs the tracks, and optionally winding_inference/ and outer_shell/, from the same directory)

python fold_repair/fetch_m7_map.py maps/p4_m7_L2.zarr --z0 10000 --z1 12000
```

## Install

Python 3.10 or newer, CPU only (no GPU used).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.14.8, numpy 2.5.4, scipy 1.18.1, scikit-image 0.26.0, tifffile 2026.10.15, zarr 3.4.1, and Pillow 11.3.0 (Pillow is only needed for the optional `--png`).

## Worked example: Paris 4, z10000-12000

This is the standard post-fit step used for all the results below. `RUN` is a finished fit_spiral run of the band.

```bash
RUN=runs/fit_paris4_z10000-12000            # fit_spiral output (FIT_SPIRAL_OUT_DIR)
DS=data/PHercParis4/spiral_dataset
MAP=maps/p4_m7_L2.zarr                      # from fetch_m7_map.py

# 1. label-free fold flag
python fold_repair/radiality_map.py $RUN --umb $DS/umbilicus.json --z0 10000 --z1 12000 --npz $RUN.radiality.npz

# 2. risk-gated fold repair (CPU, ~25-35 min per 2,000 slices)
nice python fold_repair/fold_repair.py $RUN $MAP/0 ${RUN}_repaired --dataset $DS --z0 10000 --z1 12000 \
    --acc 0.5 --margin 0 --fill 6 --distinct 4 --risk $RUN.radiality.npz

# 3. (optional) truth score of fit and repaired fit against the published segments
for R in $RUN ${RUN}_repaired; do
  python fold_repair/winding_order.py $R --umb $DS/umbilicus.json --z0 10000 --z1 12000 --npz $R.order.npz
  python fold_repair/radiality_map.py $R --umb $DS/umbilicus.json --z0 10000 --z1 12000 --npz $R.radiality.npz
  python fold_repair/detector_eval.py $R $R.order.npz gp/*5753_0/*.tifxyz gp/*5753_-1/*.tifxyz gp/*5753_-2/*.tifxyz \
      --umb $DS/umbilicus.json --z0 10000 --z1 12000 --ref-scale 0.25 --risk $R.radiality.npz
done
```

`detector_eval.py` prints one line per reference, an `all_bins ... off_frac` line (the *off* number below), and a `risky_off` / `safe_off` line. The risky and safe split uses the risk map of the run being scored. The repair moves vertices, so the repaired run's own map can classify a few bins differently from the fit's.

The other `fold_repair.py` defaults are the values used for Paris 4 and are listed in `--help`:

| Option | Default | Meaning |
|---|---|---|
| `--wedge` | `0 55` | Fold wedge, in degrees around the umbilicus |
| `--ref-angles` | `320 330 340` | Safe-side reference angles, in degrees |
| `--box` | `1500 6200 4880 7700` | Region read from the map, as Y0 Y1 X0 X1 |
| `--band` | `60` | Candidate search range along the ray, +- vox |
| `--risk-thr` | `0.7` | Radiality below this marks a risky cell |

For a different band or scroll, set `--wedge`, `--ref-angles` and `--box` (or `--auto-box R`, a box that follows the umbilicus) from that scroll's geometry.

## Plugging into villa's fit_spiral

No change to fit_spiral is needed. Run the fit as usual with `FIT_SPIRAL_OUT_DIR=RUN`. Then run steps 1-2 above on `RUN` and use `RUN_repaired` wherever you would use `RUN`, for example in `vc_render_tifxyz --segmentation RUN_repaired/<run-name>/meshes/fitted/wNNN`. The windings keep the fit's grid, numbering and z, so renders, ink detection and scoring tools work unchanged. Windings with no moved vertices are byte-identical copies.

## Results (PHerc. Paris 4, truth = published segments 5753_0/-1/-2)

Off is the share of (z 100 x angle 3 deg) bins whose median distance to the matched reference segment is above 8 vox (L2). Risky bins have radiality < 0.7 (folded) and safe bins >= 0.7. The repair is the gated standard step above.

| Band, fit | Fit: off / risky / safe | Fit + gated repair: off / risky / safe |
|---|---|---|
| z10000-12000, fit from tracks + official winding inference + outer shell | 10.8% / **21.5%** / 5.9% | **8.2%** / **13.0%** / 5.9% |
| z8800-10800 (unseen band), same recipe | 18.7% / **33.3%** / 12.0% | **13.9%** / **19.6%** / 10.9% |
| z8800-10800, full recipe (refit with risky tracks masked, then gated repair) | | **12.4%** off |
| z10000-12000, tracks only (the inputs PHerc. 0358 has) | 11.6% / 21.8% / 6.8% | **8.9%** / 14.0% / 6.3% |
| z13000-15000, light fold | 4.4% off | 4.4% off (risky 7.0%, safe 4.0%), unchanged |

In bands that need it, the repair cuts error in folded bins by about 40% (21.5 to 13.0 and 33.3 to 19.6) and leaves safe bins unchanged or better. Without the gate (repairing every vertex in the wedge), z10000-12000 reached 6.5% off but z13000-15000 got worse (4.4 to 4.9%). The gate keeps the fold gain and removes that regression. Runtime is about 25-35 min of CPU per 2,000-slice band; the fit itself takes about 3 h on GPU.

## Limitations

- **Validated on Paris 4 only.** The wedge, reference angles and box defaults come from Paris 4's geometry. Other scrolls need their own values, and no truth was available there to check them.
- **Hurts above about z12000 on Paris 4.** On z11500-13500 the gated repair made things worse (off 6.1 to 6.5%, risky 14.5 to 16.3%). A box that follows the umbilicus (`--auto-box`) made it worse still. The winding counts become path-inconsistent in the crushed core and across fitted-row seams, and the cause is not understood.
- **No label-free go/no-go beyond the risky-share rule.** The only rule is to run the repair when the fit's risky share (`radiality_map.py`'s printout) is at least 15%. Three per-slab gates were tried: median move against local sheet gap (`--slab-gate`), an absolute move cap (`--slab-max-move`), and a centred box. None separated helpful from harmful bands. A median vertex move well above one sheet gap (about 15-16 vox on Paris 4; 25+ vox where the repair hurt) is a warning sign, not a validated rule. On scrolls without truth, treat the repaired mesh as a variant to inspect, not a replacement.
- Needs a published m7 surface prediction and Lasagna grad_mag/normal sidecars in the fit's frame.
- Moves are in-slice only (x, y). Inversions between windings can rise slightly (1.8% to 2.7% for the ungated repair on z10000-12000).

## Files

```
fold_repair/
  fold_repair.py     post-fit repair by winding counting (main tool)
  radiality_map.py   label-free fold/risk map of a fit (input to --risk)
  winding_pool.py    reader for fit_spiral respool sidecars
  fetch_m7_map.py    download the m7 surface chunks for a band
  winding_order.py   ray-ordering check, defines the scoring bins
  detector_eval.py   truth scorer: off / risky off / safe off against reference segments
```

Self-test (no data needed): `python -m py_compile fold_repair/*.py && for f in fold_repair/*.py; do [ $f = fold_repair/winding_pool.py ] || python $f --help > /dev/null; done`

## Credit

Author: Matthew (matticusfinch). Developed with Claude (Anthropic) as an AI assistant.

Builds on villa's `spiral-fitting` (fit_spiral, respool packing), the Vesuvius Challenge open data (m7 surface predictions, Lasagna, spiral datasets, published segments), and the published Paris 4 segments used as truth.

## License

MIT, see [LICENSE](LICENSE).
