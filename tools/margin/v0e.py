"""Lie Detector v0-E: edition-defined intercolumn placebo arm, PHerc.1667 w018 + w023. Scalars only, no images.

Frozen rule: prereg/lie_detector_v0e.md (= the v0-E block of the frozen-criteria history at 7538c31, SHA-256 a3dcaea6...).
It is run exactly; every deviation is logged in the results. Reuses the Amendment D measures and recalibrated
tripwire (tools/margin/deep_ld.py, forensics_analyze.py) and the edition-lattice candidates + official-to-HF canvas
map (tools/margin/lattice.py, results at 1b84e89). Never reads or transcribes glyphs; never writes an image.

    .venv/bin/python -m tools.margin.v0e regions            # guarded placebo masks on the HF canvas (reads no pred)
    .venv/bin/python -m tools.margin.v0e synth              # synthetic harness checks (no scroll data)
    .venv/bin/python -m tools.margin.v0e plumb              # plumbing check, label file as a stand-in pred (no pred)
    .venv/bin/python -m tools.margin.v0e reduce --seg w018  # one pred at a time: download, reduce, delete raw
    .venv/bin/python -m tools.margin.v0e reduce --seg w023
    .venv/bin/python -m tools.margin.v0e analyze            # pair ratios, Spearman, sanity, verdict (JSON)
    .venv/bin/python -m tools.margin.v0e report             # results/lie_detector_v0e.md

Order (pre-registered): `regions` writes results/lie_detector_v0e_regions.json (masks, hashes, tile blocks,
analysis plan); it is committed and pushed before the first pred is downloaded. A tripwire firing in a placebo
region writes data/v0e/HARD_STOP.json and exits with code 3; nothing further is computed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.stats import spearmanr

from tools.harness.fetch import RAW, download, free_bytes, is_reverse, list_remote, sha256
from tools.margin import lattice as LAT
from tools.margin.common import REPO, TILE_PX, Lazy2D, full_to_grid, grid_factors, meta, seg, voxel_um
from tools.margin.deep_ld import blob_null, letter_S, shuffle_tiles
from tools.margin.forensics import CELL
from tools.margin.forensics_analyze import FULL, TILE, Profile, periodicity
from tools.margin.reduce_pred import eval_mask_files

# ------------------------------------------------------------------------------------------------ constants
SEGS = ("w018", "w023")                      # processing order (pre-registered)
REGION = {"w018": "ic_201", "w023": "ic_132"}
FROZEN_EXTENT = {"w018": {"x_mm": [195.4, 205.8], "z_mm": [8.5, 68.3]},
                 "w023": {"x_mm": [131.5, 144.5], "z_mm": [9.1, 76.7]}}
FROZEN_N_PREDS = {"w018": 11, "w023": 8}
PREREG = REPO / "vault" / "prereg" / "lie_detector_v0e.md"
PREREG_SHA = "a3dcaea682f49e561dca9bcfeac6eb23a9b34e6f87efdfe2a4f611a24fc54010"
GUARD_X_MM, EDGE_MM, Z_MAX_MM, MIN_CORE_MM = 1.5, 2.0, 81.0, 5.0
LATD = REPO / "data" / "edition_lattice"
OUT = REPO / "data" / "v0e"
REG_JSON = REPO / "vault" / "results" / "lie_detector_v0e_regions.json"
RES_JSON = REPO / "vault" / "results" / "lie_detector_v0e.json"
RES_MD = REPO / "vault" / "results" / "lie_detector_v0e.md"
STOP_FILE = OUT / "HARD_STOP.json"
SEED = 20260925
B = 2000
SUB = 512                                    # streamed rows per strip (2 x the 256 px label tiles)
SH = CELL * TILE                             # 416 px = 1 mm shuffle tile
MIN_FREE_GB = 6.0
FAMILY_KEYS = ("canonical", "new_canon", "autoresearch", "hecate", "resnet152", "resnet-152", "resnet_152")

PLAN = {
    "status": "Fixed and committed BEFORE any v0-E pred is downloaded or read. Where the frozen text "
              "(prereg/lie_detector_v0e.md) leaves a choice, the choice is made here; each is repeated in "
              "the results. Anything else that changes later is a logged deviation.",
    "regions": {
        "source": "edition-lattice candidate masks ic_201 (w018) and ic_132 (w023) on the official label canvas "
                  "(data/edition_lattice/<seg>/candidates_grid.npz, tools/margin/lattice.py, results 1b84e89): the two "
                  "intercolumns marked reliable that passed the tripwire. Their unguarded extents are checked "
                  "against the frozen text (x 195.4-205.8 mm, z 8.5-68.3 mm; x 131.5-144.5 mm, z 9.1-76.7 mm).",
        "x_guard": "1.5 mm on both x sides, applied on the official canvas where x (along the lines) is defined: "
                   "32 grid cells of 20 px = 1.535 mm, the smallest whole-cell guard >= 1.5 mm.",
        "edge_guard": ">= 2 mm from any mesh edge, on the official tifxyz (as in the candidate) AND on the HF "
                      "tifxyz the preds live on (Euclidean distance transform on the grid; the grid border counts "
                      "as an edge).",
        "z": "official tifxyz z < 81 mm AND HF tifxyz z < 81 mm.",
        "drop_rule": "a region whose guarded official core (x extent after the guard) is < 5 mm wide is dropped.",
        "canvas_map": "official -> HF canvas through the saved 3D nearest-neighbour map of lattice.canvas_map "
                      "(official tifxyz point -> nearest HF tifxyz point; matches > 2 mm off the unguarded "
                      "region's median offset rejected; 3x3 closing; HF-valid). The map is rebuilt from the saved "
                      "inputs and must reproduce the saved HF mask (candidates_hf_grid.npz) bit for bit; the "
                      "guarded region is that same map restricted to the guarded official cells. Never a fixed "
                      "shift.",
        "full_res": "HF full-res pixel (r, c) is in the region iff its nearest HF tifxyz grid point "
                    "(round(r*sy), round(c*sx)) is (same rule as the margin bands).",
        "hash_recipe": "sha256 over int64 shape bytes, then np.packbits(mask, axis=1) bytes row by row "
                       "(full-res: the crop given by crop_rows/crop_cols).",
    },
    "models": {
        "set": "the non-reverse preds named in this file (bucket listing: w018 11, w023 8 = the frozen counts). "
               "A pred missing at download time is logged and dropped; a pred added to the bucket later is not "
               "used.",
        "merged": "merged_* count as ONE model (first merged_* in sorted order; the others are compared by their "
                  "placebo p > 0.5 map hash and reported as duplicates). None exist on w018/w023 in the listing, "
                  "so the rule is vacuous unless the listing changes.",
        "ink_canonical_2um_family": "a pred is identified as ink_canonical_2um family if its file name or its "
                                    "TIFF metadata (Software, ImageDescription, other ASCII tags) contains one of "
                                    + ", ".join(FAMILY_KEYS) + " (case-insensitive). Identified preds are "
                                    "reported in a separate row, and left out of the pair ratios, the Spearman "
                                    "and the 'every model < 0.1%' clause. The tripwire applies to them. A pred "
                                    "with unknown provenance (1667_2um_pred.tif) is not identifiable and stays "
                                    "in the main set, with its provenance stated.",
        "odd_shape": "shape equal to the labels -> as is; an exact integer downsample f (each axis within f px) "
                     "-> nearest-neighbour upsample, logged; anything else -> excluded, logged (Amendment D plan).",
        "registration": "pixel AUROC of the pred value, inklabels_v2 > 0 vs == 0 inside val_v2 | sup_v2 | sup_v1 "
                        "(exact, from value histograms). AUROC < 0.75 -> excluded from ratios and Spearman, "
                        "listed. If more than half of a segment's preds fail -> harness bug, no verdict.",
        "threshold": "p > 0.5: uint8 > 127; float > 0.5 (0-1 range checked); uint16 > 32767.",
    },
    "measures": {
        "placebo_fp_area": "placebo px with p > 0.5 / placebo px, per model per segment, whole guarded region.",
        "letter_blob_density": "8-connected full-res components of (p > 0.5) & region with area 1-4 mm2, per "
                               "cm2 of region.",
        "reference_fp": "same model, p > 0.5 on in-mask unlabeled text-block px (h_neg above threshold / h_neg); "
                        "reported, never a PASS input.",
    },
    "tripwire": {
        "rule": "Amendment D recalibrated flag, implemented as in tools/margin/deep_ld.py (plan in "
                "results/deep_bands.json): periodicity p <= 0.01 from the 1 mm tile-shuffle null (200 "
                "surrogates, seeds 20260925+k) on the pooled 0.25 mm row profile, T1 = max power over 2.5-8 mm / "
                "median power over 0.5-8 mm; OR letter-blob rows: S = max letter-blob (1-4 mm2) centroids in a "
                "1 mm row window, flag iff S >= 3 AND S > 99th pct of S on the tile-shuffled region. No FP-area "
                "gate.",
        "scope": "every pred on its segment's placebo region (including preds that fail registration), then the "
                 "model-mean consensus of the used preds (periodicity only), per segment.",
        "action": "any flag -> HARD STOP: stop computing and downloading, commit scalars only, report. No images.",
    },
    "statistics": {
        "bootstrap_units": "HF-canvas 5 mm tiles (2084 px grid anchored at (0,0), partial trailing tiles kept) "
                           "holding >= 1 px of the guarded region; a unit's value = its region FP px. Every region "
                           "px is in a unit, so the point estimate is the whole-region FP ratio. (Deviation from the "
                           "Amendment D plan's '>= 50% of the tile in the band': the placebo strips are 7-10 mm "
                           "wide, and a 50% rule would drop region px from the measured FP area.)",
        "bootstrap": "B = 2000, seed 20260925 (numpy default_rng, one stream per segment: each resample draws n "
                     "units with replacement), paired: the same resample for every model of the segment; "
                     "percentile 95% CI (linear interpolation, inf-safe).",
        "pair_ratio": "sum of region FP px, model A / model B, A = larger point value; B = 0 & A > 0 -> inf; both "
                      "0 -> 1.0. CI excludes 1 iff its lower bound > 1. All pairs within a segment.",
        "spearman": "Spearman rho of placebo FP area across the main-set models present and used on both "
                    "segments (matched by model name up to '_w0'); fewer than 3 shared or a constant ranking -> "
                    "rho undefined, counted as < 0.5.",
    },
    "verdict": {
        "sanity_first": "registration per pred (exclusion), planted blobs recovered (ratio >= 1.3), self-pair "
                        "ratio = 1.00 exactly. A planted or self-pair failure = harness bug -> NO VERDICT (fix, "
                        "rerun).",
        "KILL": "all pair CIs include 1; OR every (model, segment) placebo FP area < 0.1%; OR Spearman < 0.5.",
        "PASS": ">= 1 pair (either segment) with ratio >= 1.5 AND CI excluding 1, AND Spearman >= 0.5, AND no "
                "KILL clause holds. (Precedence fixed here: the frozen KILL line lists 'every model's placebo FP < "
                "0.1%' as a KILL regardless of ratios, so PASS needs no KILL clause to hold.)",
        "else": "INCONCLUSIVE = KILL (no retry).",
        "margins_row": "NOT run: conditional on the org confirming the column foot at high z; not confirmed.",
    },
    "planted_sanity": "2 mm2 disks set to p > 0.5 at seeded (20260925) random positions inside the region (disk "
                      "cut to the region), until planted px >= max(0.5 x the pred's region FP px, 0.1% of region "
                      "px); FP recounted through the same tile-count code, ratio planted/original through the same "
                      "bootstrap. Pred: the first used pred in processing order (w018 first, sorted names) whose "
                      "placebo FP px > 0. Recovered iff ratio >= 1.3 (frozen); the CI is reported.",
    "self_pair": "every used model against itself through the pair code: ratio 1.00 exactly, CI [1, 1].",
    "processing": "w018 then w023, preds in sorted name order, one at a time in data/raw/: download, sha256, "
                  "stream in 512-row strips (never whole), reduce, delete raw. Stop if free disk < 6 GB. "
                  "Registration histograms read an ev/ink bit cache built once per segment from the label files.",
}


# ------------------------------------------------------------------------------------------------ helpers
def jdump(o, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(o, indent=1, default=LAT._js))


def sha_mask(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(a.shape, np.int64).tobytes())
    h.update(np.packbits(np.asarray(a, bool), axis=1).tobytes())
    return h.hexdigest()


def rle_rows(m: np.ndarray) -> list:
    """'row:c0-c1[,c0-c1...]' per grid row with region cells (half-open column runs on the HF tifxyz grid)."""
    out = []
    for i in np.flatnonzero(m.any(1)):
        d = np.diff(np.r_[0, m[i].astype(np.int8), 0])
        s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        out.append(f"{int(i)}:" + ",".join(f"{int(a)}-{int(b)}" for a, b in zip(s, e)))
    return out


def bin_matrix(g: np.ndarray, n_grid: int, lo: int, hi: int, step: int, n_bins: int | None = None) -> np.ndarray:
    """M[k, i] = number of full-res indices r in [lo, hi) with (r - lo) // step == k and g[r] == i."""
    r = np.arange(lo, hi)
    k = (r - lo) // step
    M = np.zeros((int(k.max()) + 1 if n_bins is None else n_bins, n_grid))
    np.add.at(M, (k, g[lo:hi]), 1.0)
    return M


def pool(a: np.ndarray, c: int = CELL) -> np.ndarray:
    """Sum over c x c cells anchored at (0, 0); a trailing partial cell is kept (partial)."""
    rs = np.add.reduceat(a, np.arange(0, a.shape[0], c), axis=0, dtype=np.int64)
    return np.add.reduceat(rs, np.arange(0, a.shape[1], c), axis=1).astype(np.float64)


def tile_sum(a: np.ndarray, R0: int, C0: int, TY: int, TX: int) -> np.ndarray:
    """Per 2084 px canvas tile (anchored (0,0)) sum of a crop whose origin is (R0, C0)."""
    out = np.zeros((TY, TX), np.int64)
    rt = (R0 + np.arange(a.shape[0])) // TILE_PX
    ct = (C0 + np.arange(a.shape[1])) // TILE_PX
    rb = np.flatnonzero(np.r_[True, np.diff(rt) != 0])
    cb = np.flatnonzero(np.r_[True, np.diff(ct) != 0])
    for k, r in enumerate(rb):
        r_end = rb[k + 1] if k + 1 < len(rb) else a.shape[0]
        colsum = a[r:r_end].sum(0, dtype=np.int64)
        out[rt[r], ct[cb]] += np.add.reduceat(colsum, cb)
    return out


def pct(x, q: float) -> float:
    """numpy 'linear' percentile, safe with +inf."""
    x = np.sort(np.asarray(x, float))
    pos = (len(x) - 1) * q / 100.0
    lo, hi = int(np.floor(pos)), int(np.ceil(pos))
    a, b, f = x[lo], x[hi], pos - lo
    if a == b or f == 0:
        return float(a)
    if np.isinf(b):
        return float("inf")
    return float(a + (b - a) * f)


def ci95(x) -> list:
    return [pct(x, 2.5), pct(x, 97.5)]


def ratio(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.where(a > 0, np.inf, 1.0))


def boot_weights(n_units: int) -> np.ndarray:
    """(B, n_units) resample counts; one rng stream (seed 20260925) per segment (as deep_ld)."""
    rng = np.random.default_rng(SEED)
    W = np.zeros((B, n_units))
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, n_units, n_units), minlength=n_units)
    return W


def model_key(rel: str) -> str:
    n = Path(rel).name
    return n.split("_w0")[0] if "_w0" in n else n.replace(".tif", "")


def check_disk(need: int = 0) -> float:
    f = free_bytes(REPO)
    if (f - need) / 2**30 < MIN_FREE_GB:
        raise SystemExit(f"STOP: free disk {f / 2**30:.1f} GB, need {need / 2**30:.2f} GB + {MIN_FREE_GB} GB floor")
    return f / 2**30


def git_head():
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:           # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------------ regions
def hf_geometry(short):
    s = seg(short)
    m = meta(short)
    Ph = np.stack([np.squeeze(tifffile.imread(s.small / f"{c}.tif")).astype(np.float32) for c in "xyz"], -1)
    okh = (Ph[..., 0] != -1) & (Ph[..., 2] > 0)
    lab = Lazy2D(s.small / f"{s.seg_id}_inklabels_v2.tif")
    H, W = lab.shape
    lab.close()
    sy, sx = grid_factors(m, (H, W), okh.shape)
    return {"P": Ph, "ok": okh, "H": H, "W": W, "sy": sy, "sx": sx, "vox": voxel_um(m), "scale_sx_sy": m["scale"]}


def build_region(short):
    from scipy.spatial import cKDTree
    name = REGION[short]
    ed = json.loads((REPO / "vault" / "results" / "edition_lattice.json").read_text())["segments"][short]
    cand = next(c for c in ed["candidates"] if c["name"] == name)
    with np.load(LATD / short / "candidates_grid.npz") as f:
        m = f[name].copy()
    with np.load(LATD / short / "candidates_hf_grid.npz") as f:
        hm_saved = f[name].copy()
    pooled = LAT.load_pooled(short)
    gh, gw = pooled["mean"].shape
    del pooled
    geo = LAT.load_grid(short, gh, gw)
    Z, valid = geo["Z"], geo["valid"]
    ii, jj = np.nonzero(m)
    j0, j1 = int(jj.min()), int(jj.max()) + 1
    check = {"n_cells": int(ii.size) == cand["n_cells"],
             "x_mm": [round(LAT.mm(j0), 2), round(LAT.mm(j1), 2)] == cand["x_mm"],
             "x_mm_matches_frozen_0.1mm": bool(abs(LAT.mm(j0) - FROZEN_EXTENT[short]["x_mm"][0]) < 0.06 and
                                              abs(LAT.mm(j1) - FROZEN_EXTENT[short]["x_mm"][1]) < 0.06),
             "z_mm_matches_frozen_0.1mm": bool(abs(float(Z[m].min()) - FROZEN_EXTENT[short]["z_mm"][0]) < 0.06 and
                                              abs(float(Z[m].max()) - FROZEN_EXTENT[short]["z_mm"][1]) < 0.06),
             "reliable_in_lattice": bool(cand["reliable"]),
             "tripwire_passed_on_render": not cand["tripwire"]["text_like_flag"]}
    if not all(check.values()):
        raise SystemExit(f"{short} {name}: source candidate does not match the frozen text / lattice: {check}")
    # ---- guards on the official canvas
    g = int(np.ceil(GUARD_X_MM / LAT.CELL_MM - 1e-9))
    core_w = (j1 - j0 - 2 * g) * LAT.CELL_MM
    colsel = np.zeros(gw, bool)
    colsel[j0 + g:j1 - g] = True
    far_o = LAT.edt_cells(valid) >= int(np.ceil(EDGE_MM / LAT.CELL_MM - 1e-9))
    z_ok = np.isfinite(Z) & (Z < Z_MAX_MM)
    mg = m & colsel[None, :] & far_o & z_ok
    removed = {"x_guard_cells": int((m & ~colsel[None, :]).sum()),
               "official_edge_guard_cells_extra": int((m & colsel[None, :] & ~far_o).sum()),
               "official_z_cells_extra": int((m & colsel[None, :] & far_o & ~z_ok).sum())}
    rows_o = np.flatnonzero(mg.any(1))
    full_w = [(int(np.flatnonzero(mg[i])[0]), int(np.flatnonzero(mg[i])[-1]) + 1) for i in rows_o]
    frac_full = float(np.mean([a == j0 + g and b == j1 - g for a, b in full_w])) if full_w else 0.0
    dropped = bool(core_w < MIN_CORE_MM)
    # ---- canvas map (rebuilt from the saved inputs, as lattice.canvas_map)
    hf = hf_geometry(short)
    Ph, okh = hf["P"], hf["ok"]
    hm_meta = meta(short)
    sxh, syh = float(hm_meta["scale"][0]), float(hm_meta["scale"][1])
    ih, jh = np.nonzero(okh)
    tree = cKDTree(Ph[ih, jh])
    dist, idx = tree.query(geo["P"][ii, jj])
    del tree
    oy = ih[idx] / syh - ii * LAT.G
    ox = jh[idx] / sxh - jj * LAT.G
    outl = 2.0 / (LAT.UM / 1000)
    keep = (np.abs(oy - np.median(oy)) <= outl) & (np.abs(ox - np.median(ox)) <= outl)
    st = np.ones((3, 3), bool)
    rebuilt = np.zeros(okh.shape, bool)
    rebuilt[ih[idx[keep]], jh[idx[keep]]] = True
    rebuilt = ndi.binary_closing(rebuilt, st) & okh
    map_ok = bool(np.array_equal(rebuilt, hm_saved))
    if not map_ok:
        raise SystemExit(f"{short}: rebuilt canvas map differs from the saved HF mask in {int((rebuilt ^ hm_saved).sum())} cells")
    sel = mg[ii, jj]
    hmg = np.zeros(okh.shape, bool)
    hmg[ih[idx[keep & sel]], jh[idx[keep & sel]]] = True
    hmg = ndi.binary_closing(hmg, st) & okh
    # ---- guards on the HF canvas
    cell_mm_h = LAT.UM / 1000.0 / max(sxh, syh)                  # smaller mm per cell (conservative)
    edt_h = ndi.distance_transform_edt(np.pad(okh, 1))[1:-1, 1:-1] * cell_mm_h
    zh = np.where(okh, Ph[..., 2] * LAT.UM / 1000.0, np.nan)
    far_h = edt_h >= EDGE_MM
    zok_h = np.isfinite(zh) & (zh < Z_MAX_MM)
    hm = hmg & far_h & zok_h
    removed.update({"hf_edge_guard_cells": int((hmg & ~far_h).sum()), "hf_z_cells": int((hmg & far_h & ~zok_h).sum())})
    assert not (hm & ~hm_saved).any(), "guarded HF mask is not inside the saved unguarded HF mask"
    # HF width per HF grid row (mm), rows vs z
    rows_h = np.flatnonzero(hm.any(1))
    wid = hm[rows_h].sum(1) * (LAT.UM / 1000.0 / sxh)
    zr = np.array([np.nanmedian(zh[i][hm[i]]) for i in rows_h])
    rho_rz = float(spearmanr(rows_h, zr)[0]) if len(rows_h) > 2 else float("nan")
    # ---- full res on the HF canvas
    H, W, sy, sx = hf["H"], hf["W"], hf["sy"], hf["sx"]
    GH, GW = okh.shape
    gy, gx = full_to_grid(H, sy, GH), full_to_grid(W, sx, GW)
    TY, TX = -(-H // TILE_PX), -(-W // TILE_PX)
    hmf = hm.astype(np.float64)
    CY = bin_matrix(gy, GH, 0, H, TILE_PX, TY)
    CX = bin_matrix(gx, GW, 0, W, TILE_PX, TX)
    T = np.rint(CY @ hmf @ CX.T).astype(np.int64)
    del CY, CX
    rows_any = hm.any(1)[gy]
    cols_any = hm.any(0)[gx]
    rr, cc = np.flatnonzero(rows_any), np.flatnonzero(cols_any)
    r_lo, r_hi, c_lo, c_hi = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
    R0, C0 = (r_lo // SH) * SH, (c_lo // SH) * SH
    R1, C1 = min(H, -(-r_hi // SH) * SH), min(W, -(-c_hi // SH) * SH)
    CYc = bin_matrix(gy, GH, R0, R1, CELL)
    CXc = bin_matrix(gx, GW, C0, C1, CELL)
    bcells = CYc @ hmf @ CXc.T
    del CYc, CXc, hmf
    # full-res crop: hash + exact px count + pooled check (streamed in row strips)
    h = hashlib.sha256()
    h.update(np.asarray([R1 - R0, C1 - C0], np.int64).tobytes())
    npx = 0
    bchk = np.zeros_like(bcells)
    for r0 in range(R0, R1, 4 * SH):
        r1 = min(R1, r0 + 4 * SH)
        blk = hm[gy[r0:r1]][:, gx[C0:C1]]
        h.update(np.packbits(blk, axis=1).tobytes())
        npx += int(blk.sum())
        pb = pool(blk)
        bchk[(r0 - R0) // CELL:(r0 - R0) // CELL + pb.shape[0]] += pb
        del blk
    assert npx == int(T.sum()) and np.array_equal(bchk, bcells), "full-res region px mismatch"
    tiles_mm1 = shuffle_tiles(bcells)
    units = [[int(a), int(b), int(T[a, b])] for a, b in zip(*np.nonzero(T))]
    full_tiles = int(sum(1 for a, b, n in units if n == TILE_PX * TILE_PX))
    px_mm2 = (hf["vox"] / 1000.0) ** 2
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"regions_{short}.npz", official_guarded=mg, hf_grid=hm, tile_px=T, bcells=bcells,
                        crop=np.array([R0, R1, C0, C1]))
    return {
        "name": name, "source_candidate": {k: cand[k] for k in ("x_mm", "z_mm", "area_mm2", "canvas_px_rows",
                                                                "canvas_px_cols", "n_cells", "reliable")},
        "source_checks": check,
        "guard": {"x_guard_cells": g, "x_guard_mm": round(g * LAT.CELL_MM, 4), "edge_guard_mm": EDGE_MM,
                  "z_max_mm": Z_MAX_MM, "cells_removed": removed},
        "official": {"x_mm": [round(LAT.mm(j0 + g), 2), round(LAT.mm(j1 - g), 2)], "core_width_mm": round(core_w, 3),
                     "dropped_core_lt_5mm": dropped, "n_cells": int(mg.sum()),
                     "area_mm2": round(float(mg.sum()) * LAT.CELL_MM ** 2, 1),
                     "z_mm": [round(float(Z[mg].min()), 2), round(float(Z[mg].max()), 2)],
                     "canvas_px_rows": [int(rows_o.min() * LAT.G), int((rows_o.max() + 1) * LAT.G)],
                     "canvas_px_cols": [int((j0 + g) * LAT.G), int((j1 - g) * LAT.G)],
                     "rows_at_full_guarded_width_frac": round(frac_full, 4),
                     "min_edge_distance_mm": round(float(LAT.edt_cells(valid)[mg].min() * LAT.CELL_MM), 2),
                     "mask_sha256": sha_mask(mg), "grid_shape": list(mg.shape)},
        "canvas_map": {"rebuilt_equals_saved_hf_mask": map_ok, "saved_hf_cells_unguarded": int(hm_saved.sum()),
                       "nn_dist_um_median": round(float(np.median(dist) * LAT.UM), 1),
                       "nn_dist_um_p95": round(float(np.percentile(dist, 95) * LAT.UM), 1),
                       "rejected_frac_offset_gt_2mm": round(float(1 - keep.mean()), 6),
                       "median_offset_px_rows_cols": [round(float(np.median(oy)), 1), round(float(np.median(ox)), 1)],
                       "offset_px_rows_p5_p95": [round(float(np.percentile(oy, 5)), 1), round(float(np.percentile(oy, 95)), 1)],
                       "offset_px_cols_p5_p95": [round(float(np.percentile(ox, 5)), 1), round(float(np.percentile(ox, 95)), 1)]},
        "hf": {"grid_shape": [GH, GW], "scale_sx_sy": [sxh, syh], "canvas_hw": [H, W], "grid_factor_sy_sx": [sy, sx],
               "n_cells": int(hm.sum()), "mask_sha256": sha_mask(hm), "rle_rows": rle_rows(hm),
               "width_mm_per_row": {"median": round(float(np.median(wid)), 3), "p5": round(float(np.percentile(wid, 5)), 3),
                                    "p95": round(float(np.percentile(wid, 95)), 3), "min": round(float(wid.min()), 3),
                                    "max": round(float(wid.max()), 3)},
               "z_mm": [round(float(np.nanmin(zh[hm])), 2), round(float(np.nanmax(zh[hm])), 2)],
               "rows_vs_z_spearman": round(rho_rz, 4),
               "min_edge_distance_mm": round(float(edt_h[hm].min()), 2)},
        "full_res": {"px": npx, "area_mm2": round(npx * px_mm2, 1), "px_um": hf["vox"],
                     "bbox_rows": [r_lo, r_hi], "bbox_cols": [c_lo, c_hi],
                     "crop_rows": [R0, R1], "crop_cols": [C0, C1], "crop_mask_sha256": h.hexdigest(),
                     "tile_grid_TY_TX": [TY, TX], "tile_blocks": units, "n_tile_blocks": len(units),
                     "tiles_5mm_100pct_inside": full_tiles, "n_1mm_shuffle_tiles": len(tiles_mm1)},
    }


def pred_set(short):
    """Non-reverse preds from the recorded bucket listing (the frozen counts)."""
    listing = json.loads((seg(short).data / "listing.json").read_text())
    preds = sorted(k for k in listing if k.startswith("preds/") and not is_reverse(k))
    return [{"rel": k, "bytes": listing[k], "model": model_key(k)} for k in preds]


def regions():
    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA:
        raise SystemExit(f"frozen text hash mismatch: {got}")
    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code": "tools/margin/v0e.py",
           "git_head_at_build": git_head(),
           "frozen_rule": {"file": "prereg/lie_detector_v0e.md", "sha256": got, "matches_HASHES_json": True,
                           "ideas_commit": "7538c31"},
           "note": "Placebo regions fixed from edition-lattice geometry, official + HF tifxyz and the saved canvas "
                   "map ONLY. No v0-E pred had been downloaded or read when this file was written and committed. "
                   "No images.",
           "analysis_plan_fixed_before_reading_preds": PLAN, "segments": {}}
    tot = 0.0
    for short in SEGS:
        t0 = time.time()
        r = build_region(short)
        r["models"] = pred_set(short)
        r["n_models_listed"] = len(r["models"])
        r["n_models_frozen"] = FROZEN_N_PREDS[short]
        r["build_s"] = round(time.time() - t0, 1)
        res["segments"][short] = r
        tot += r["full_res"]["area_mm2"] if not r["official"]["dropped_core_lt_5mm"] else 0.0
        print(short, r["name"], "official", r["official"]["x_mm"], r["official"]["core_width_mm"], "mm core,",
              r["official"]["area_mm2"], "mm2; HF", r["full_res"]["area_mm2"], "mm2,", r["full_res"]["n_tile_blocks"],
              "tile blocks; width", r["hf"]["width_mm_per_row"], "; models", r["n_models_listed"], flush=True)
    res["total_placebo_area_mm2_hf"] = round(tot, 1)
    res["frozen_unguarded_total_mm2"] = 1782
    sj = OUT / "synth.json"
    res["synthetic_harness_checks_before_any_pred"] = json.loads(sj.read_text()) if sj.exists() else None
    pj = OUT / "plumb.json"
    res["plumbing_check_labels_as_standin_before_any_pred"] = json.loads(pj.read_text()) if pj.exists() else None
    jdump(res, REG_JSON)
    print("wrote", REG_JSON, "total", round(tot, 1), "mm2", flush=True)


# ------------------------------------------------------------------------------------------------ tripwire
def tripwire(hi: np.ndarray, hcells: np.ndarray, bcells: np.ndarray, vox: float, label: str,
             stop_on_first: bool = True) -> dict:
    """Amendment D recalibrated text-like flag on one region crop (origin on the 416 px grid). In the real run
    (stop_on_first) the blob-row null is not computed once the periodicity flag has fired (stop computing); the
    synthetic checks run both parts."""
    per = periodicity(Profile(hcells, bcells), label)
    per_keep = {k: per[k] for k in ("n_rows", "length_mm", "T1_ratio_pooled", "T1_peak_period_mm", "p_tile_T1",
                                    "tile_null_T1_median", "tile_null_T1_p95", "shuffle_cells_frac", "mean")}
    per_flag = bool(per["p_tile_T1"] <= 0.01)
    S, n_letter, area = letter_S(hi, vox)
    blob = {"n_components": int(len(area)), "letter_blobs": int(n_letter), "S_max_letter_blobs_1mm_row": int(S),
            "largest_component_mm2": float(area.max()) if len(area) else 0.0}
    if stop_on_first and per_flag and S >= 3:
        blob.update({"null_S_p99": None, "null_note": "periodicity flag already fired: HARD STOP, blob-row null not "
                                                      "computed (stop computing); observed S reported"})
        blob_flag = False
    elif S >= 3:
        tiles = shuffle_tiles(bcells)
        nullS = blob_null(hi, tiles, vox)
        blob.update({"null_S_p99": float(np.percentile(nullS, 99)), "null_S_median": float(np.median(nullS)),
                     "null_S_max": int(max(nullS)), "n_shuffle_tiles": len(tiles)})
        blob_flag = bool(S > blob["null_S_p99"])
    else:
        blob.update({"null_S_p99": None, "null_note": "S < 3: flag cannot fire, null not computed (pre-registered)"})
        blob_flag = False
    return {"periodicity": per_keep, "periodicity_flag": per_flag, "blobs": blob, "blob_flag": blob_flag,
            "text_like_flag": bool(per_flag or blob_flag)}


# ------------------------------------------------------------------------------------------------ synthetic
def _disk(rad):
    R = int(np.ceil(rad))
    dy, dx = np.mgrid[-R:R + 1, -R:R + 1]
    return dy ** 2 + dx ** 2 <= rad ** 2, R


def _stamp(a, r, c, d, R, inside=None):
    y0, y1, x0, x1 = max(0, r - R), min(a.shape[0], r + R + 1), max(0, c - R), min(a.shape[1], c + R + 1)
    dd = d[y0 - r + R:y1 - r + R, x0 - c + R:x1 - c + R]
    if inside is not None:
        dd = dd & inside[y0:y1, x0:x1]
    new = dd & ~a[y0:y1, x0:x1]
    a[y0:y1, x0:x1] |= dd
    return int(new.sum())


def plant(hi: np.ndarray, mpx: np.ndarray, fp_px: int, region_px: int, vox: float) -> tuple[np.ndarray, dict]:
    """Copy of hi with 2 mm2 disks (cut to the region) at seeded random region positions."""
    rng = np.random.default_rng(SEED)
    target = max(0.5 * fp_px, 0.001 * region_px)
    d, R = _disk(np.sqrt(2.0 / np.pi) * 1000.0 / vox)
    hp = hi.copy()
    tot = n = tries = 0
    while tot < target and tries < 100000:
        tries += 1
        r, c = int(rng.integers(0, hp.shape[0])), int(rng.integers(0, hp.shape[1]))
        if not mpx[r, c]:
            continue
        add = _stamp(hp, r, c, d, R, mpx)
        if add:
            tot += add
            n += 1
    return hp, {"target_px": float(target), "planted_px": int(tot), "n_disks": int(n), "disk_mm2": 2.0,
                "expected_ratio": (fp_px + tot) / fp_px if fp_px else None}


def pair_table(fp_units: np.ndarray, names: list, Wb: np.ndarray) -> list:
    S0 = fp_units.sum(1)
    Sb = Wb @ fp_units.T
    out = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            hi_, lo_ = (a, b) if S0[a] >= S0[b] else (b, a)
            pt = float(ratio(S0[hi_], S0[lo_]))
            c = ci95(ratio(Sb[:, hi_], Sb[:, lo_]))
            out.append({"num": names[hi_], "den": names[lo_], "num_fp_px": int(S0[hi_]), "den_fp_px": int(S0[lo_]),
                        "ratio": pt, "ci95": c, "ci_excludes_1": bool(c[0] > 1),
                        "pass_pair": bool(pt >= 1.5 and c[0] > 1)})
    return out


def self_pairs(fp_units: np.ndarray, names: list, Wb: np.ndarray) -> dict:
    Sb = Wb @ fp_units.T
    S0 = fp_units.sum(1)
    res = {}
    for a, n in enumerate(names):
        pt = float(ratio(S0[a], S0[a]))
        rb = ratio(Sb[:, a], Sb[:, a])
        res[n] = {"ratio": pt, "ci95": ci95(rb), "exact_1": bool(pt == 1.0 and np.all(rb == 1.0))}
    return res


def _strip(rng, h_mm, w_mm=7.4, ragged=True):
    ppm = 1000.0 / 2.399
    w, h = int(w_mm * ppm), int(h_mm * ppm)
    Hc, Wc = -(-(h + 600) // SH) * SH, -(-(w + 600) // SH) * SH
    mpx = np.zeros((Hc, Wc), bool)
    r0, c0 = 300, 250
    mpx[r0:r0 + h, c0:c0 + w] = True
    if ragged:                                  # ragged left/right edges (+-0.3 mm), like a mapped strip
        for rr in range(r0, r0 + h, 208):
            k = int(rng.integers(-125, 126))
            if k > 0:
                mpx[rr:rr + 208, c0:c0 + k] = False
            k = int(rng.integers(-125, 126))
            if k > 0:
                mpx[rr:rr + 208, c0 + w - k:c0 + w] = False
    return mpx, (r0, c0, h, w)


def _blobs(rng, shape, box, k, amin, amax):
    ppm = 1000.0 / 2.399
    r0, c0, h, w = box
    A = np.zeros(shape, bool)
    for _ in range(k):
        d, R = _disk(np.sqrt(rng.uniform(amin, amax) / np.pi) * ppm)
        _stamp(A, int(rng.integers(r0, r0 + h)), int(rng.integers(c0, c0 + w)), d, R)
    return A


def synth():
    """Harness checks on synthetic maps (no scroll data), plus a characterization of the frozen tripwire.
    checks (pass/fail = harness correctness): letter rows fire; periodicity null calibrated on structureless pixel
    noise; planted blobs recovered through tile counts + bootstrap; a true 2x pair passes; self-pair exact; tile sums.
    characterization (a property of the frozen Amendment D flag, reported, never pass/fail): how often the flag fires
    on NON-periodic FP (scattered letter-sized blobs, sub-tile blobs) in a 7.4 x 60 mm strip."""
    rng = np.random.default_rng(SEED)
    vox = 2.399
    ppm = 1000.0 / vox
    mpx, box = _strip(rng, 40.0)
    r0, c0, h, w = box
    bcells = pool(mpx)
    checks, charac = {}, {}
    # letter rows at 6.0 mm pitch, 4 letters per row -> must fire
    Bm = np.zeros_like(mpx)
    for k in range(6):
        rc = r0 + int((3.0 + 6.0 * k) * ppm)
        for xc in np.linspace(c0 + 0.9 * ppm, c0 + w - 0.9 * ppm, 4):
            d, R = _disk(np.sqrt(rng.uniform(1.5, 2.5) / np.pi) * ppm)
            _stamp(Bm, rc + int(rng.integers(-60, 61)), int(xc) + int(rng.integers(-40, 41)), d, R)
    Bm &= mpx
    tB = tripwire(Bm, pool(Bm), bcells, vox, "synth_letter_rows", stop_on_first=False)
    checks["letter_rows_fire"] = {"pass": tB["text_like_flag"], "periodicity_flag": tB["periodicity_flag"],
                                  "blob_flag": tB["blob_flag"], "p_tile_T1": tB["periodicity"]["p_tile_T1"],
                                  "S": tB["blobs"]["S_max_letter_blobs_1mm_row"], "null_S_p99": tB["blobs"]["null_S_p99"]}
    # scattered letter-sized blobs + speckle, full tripwire (characterization)
    A = (_blobs(rng, mpx.shape, box, 20, 1.2, 3.5) | _blobs(rng, mpx.shape, box, 400, 0.01, 0.2)) & mpx
    tA = tripwire(A, pool(A), bcells, vox, "synth_scattered", stop_on_first=False)
    charac["scattered_20_letter_blobs_plus_speckle_full_tripwire"] = {
        "fp_area": float(A.sum() / mpx.sum()), "fires": tA["text_like_flag"], "periodicity_flag": tA["periodicity_flag"],
        "blob_flag": tA["blob_flag"], "p_tile_T1": tA["periodicity"]["p_tile_T1"],
        "S": tA["blobs"]["S_max_letter_blobs_1mm_row"], "null_S_p99": tA["blobs"]["null_S_p99"]}
    # planted recovery through tile counts + bootstrap
    R0s, C0s = 5 * TILE_PX - 700, 9 * TILE_PX - 1000          # crop straddles tile borders
    TY, TX = (R0s + mpx.shape[0]) // TILE_PX + 1, (C0s + mpx.shape[1]) // TILE_PX + 1
    T = tile_sum(mpx, R0s, C0s, TY, TX)
    u = np.array(list(zip(*np.nonzero(T))))
    fpA = int(A.sum())
    Ap, pl = plant(A, mpx, fpA, int(mpx.sum()), vox)
    tA_units, tP_units = tile_sum(A, R0s, C0s, TY, TX), tile_sum(Ap, R0s, C0s, TY, TX)
    fa, fpl = tA_units[u[:, 0], u[:, 1]], tP_units[u[:, 0], u[:, 1]]
    Wb = boot_weights(len(u))
    pr = pair_table(np.stack([fpl, fa]), ["planted", "original"], Wb)[0]
    checks["planted_recovered"] = {"pass": bool(pr["ratio"] >= 1.3), "ratio": pr["ratio"], "ci95": pr["ci95"],
                                   "expected": pl["expected_ratio"], "n_disks": pl["n_disks"], "n_units": len(u)}
    base = rng.poisson(2000, len(u)).astype(float)
    pr2 = pair_table(np.stack([2 * base + rng.poisson(50, len(u)), base]), ["x2", "x1"], Wb)[0]
    checks["true_2x_pair_passes"] = {"pass": pr2["pass_pair"], "ratio": pr2["ratio"], "ci95": pr2["ci95"]}
    sp = self_pairs(np.stack([fa, np.zeros_like(fa)]), ["A", "zero"], Wb)
    checks["self_pair_exact"] = {"pass": all(v["exact_1"] for v in sp.values()), **sp}
    checks["tile_sum_consistent"] = {"pass": bool(T.sum() == mpx.sum() and tA_units.sum() == fpA)}
    del A, Ap, Bm
    # periodicity flag on non-periodic maps, 7.4 x 60 mm strip, seeds 5000+k (periodicity part only: cheap)
    m60, box60 = _strip(np.random.default_rng(SEED + 1), 60.0, ragged=False)
    b60 = pool(m60)

    def rate(gen, n):
        ps, fps = [], []
        for k in range(n):
            g = np.random.default_rng(5000 + k)
            X = gen(g) & m60
            ps.append(periodicity(Profile(pool(X), b60), "char")["p_tile_T1"])
            fps.append(float(X.sum() / m60.sum()))
        ps = np.array(ps)
        return {"n_seeds": n, "fp_area_median": float(np.median(fps)), "frac_p_le_0.01": float((ps <= 0.01).mean()),
                "frac_p_le_0.05": float((ps <= 0.05).mean()), "median_p": float(np.median(ps))}

    pix = rate(lambda g: g.random(m60.shape) < 0.02, 50)
    checks["periodicity_null_calibrated_pixel_noise"] = {"pass": bool(pix["frac_p_le_0.01"] <= 0.06), **pix}
    for lab, k, a0, a1 in (("1_letter_blob", 1, 1.2, 3.5), ("4_letter_blobs", 4, 1.2, 3.5),
                           ("40_subtile_blobs_0.05-0.2mm2", 40, 0.05, 0.2)):
        charac[f"periodicity_rate_{lab}"] = rate(lambda g, k=k, a0=a0, a1=a1: _blobs(g, m60.shape, box60, k, a0, a1), 20)
    ok = all(v["pass"] for v in checks.values())
    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "all_harness_checks_pass": ok,
           "checks": checks, "tripwire_characterization": charac,
           "reading": "The periodicity part of the frozen Amendment D flag compares the observed row profile with a "
                      "1 mm tile-shuffle null. The null is exact for structureless pixel noise, but shuffling 1 mm "
                      "tiles also destroys any FP structure coherent over > 1 mm, so isolated letter-sized blobs (no "
                      "rows) push p to its minimum. A firing therefore means 'FP structure coherent over > 1 mm in the "
                      "placebo', not specifically 'rows'. Under the frozen rule it is still a HARD STOP and the authors "
                      "decides. The blob-row part needs S >= 3 blobs in one 1 mm row, above the shuffled p99."}
    jdump(res, OUT / "synth.json")
    print(json.dumps(res, indent=1, default=LAT._js))
    return res


# ------------------------------------------------------------------------------------------------ reduce
class Pred:
    """Row/column window reads of a pred; nearest-neighbour upsampling if it is an exact integer downsample."""

    def __init__(self, path: Path, H: int, W: int):
        self.lz = Lazy2D(path)
        h, w = self.lz.shape
        self.f, self.note, self.H, self.W = 1, None, H, W
        if (h, w) != (H, W):
            f = int(round(H / h))
            if f >= 2 and abs(h * f - H) <= f and abs(w * f - W) <= f and int(round(W / w)) == f:
                self.f = f
                self.note = f"shape {h}x{w} = labels / {f}; nearest-neighbour upsampled x{f}"
            else:
                self.note = f"shape {h}x{w} not an integer downsample of labels {H}x{W}"

    def rows(self, r0, r1, c0, c1):
        if self.f == 1:
            return self.lz.rows(r0, r1, c0, c1)
        f = self.f
        a0, a1 = r0 // f, min(self.lz.shape[0], -(-r1 // f))
        b0, b1 = c0 // f, min(self.lz.shape[1], -(-c1 // f))
        a = self.lz.rows(a0, a1, b0, b1)
        a = np.repeat(np.repeat(a, f, axis=0), f, axis=1)
        a = a[r0 - a0 * f:r0 - a0 * f + (r1 - r0), c0 - b0 * f:c0 - b0 * f + (c1 - c0)]
        if a.shape != (r1 - r0, c1 - c0):
            a = np.pad(a, ((0, r1 - r0 - a.shape[0]), (0, c1 - c0 - a.shape[1])), mode="edge")
        return a


def tiff_meta(path: Path) -> tuple[dict, str]:
    """ASCII tags of page 0 (stored truncated to 600 chars) and the full concatenated text (keyword search)."""
    out, full = {}, []
    with tifffile.TiffFile(path) as tf:
        p = tf.pages[0]
        for t in p.tags.values():
            if isinstance(t.value, (str, bytes)):
                v = t.value.decode("latin-1", "replace") if isinstance(t.value, bytes) else t.value
                out[t.name] = v[:600]
                full.append(v)
        out["_n_pages"] = len(tf.pages)
        out["_is_ome"] = bool(getattr(tf, "is_ome", False))
    return out, " ".join(full)


def family_hits(rel: str, md_text: str) -> list:
    txt = (Path(rel).name + " " + md_text).lower()
    return [k for k in FAMILY_KEYS if k in txt]


def ev_cache(short):
    """Bit-packed evaluated-mask (val_v2 | sup_v2 | sup_v1) and labeled-ink-in-mask planes over the mask bbox,
    built once per segment from the label files (labels only), for the registration histograms."""
    d = OUT / short
    d.mkdir(parents=True, exist_ok=True)
    f_meta = d / "evcache.json"
    if f_meta.exists():
        cm = json.loads(f_meta.read_text())
        return cm, np.load(d / "evcache_ev.npy", mmap_mode="r"), np.load(d / "evcache_pos.npy", mmap_mode="r")
    s = seg(short)
    lab = Lazy2D(s.small / f"{s.seg_id}_inklabels_v2.tif")
    ms = [Lazy2D(f) for f in eval_mask_files(s)]
    H, W = lab.shape
    rows = np.zeros(H, bool)
    cols = np.zeros(W, bool)
    for r0 in range(0, H, 1024):
        r1 = min(H, r0 + 1024)
        ev = ms[0].rows(r0, r1) > 0
        for m in ms[1:]:
            ev |= m.rows(r0, r1) > 0
        rows[r0:r1] = ev.any(1)
        cols |= ev.any(0)
    rr, cc = np.flatnonzero(rows), np.flatnonzero(cols)
    er0, er1, ec0, ec1 = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
    nb = -(-(ec1 - ec0) // 8)
    EV = np.lib.format.open_memmap(d / "evcache_ev.npy", mode="w+", dtype=np.uint8, shape=(er1 - er0, nb))
    PO = np.lib.format.open_memmap(d / "evcache_pos.npy", mode="w+", dtype=np.uint8, shape=(er1 - er0, nb))
    n_ev = n_pos = 0
    for r0 in range(er0, er1, 1024):
        r1 = min(er1, r0 + 1024)
        ev = ms[0].rows(r0, r1, ec0, ec1) > 0
        for m in ms[1:]:
            ev |= m.rows(r0, r1, ec0, ec1) > 0
        pos = ev & (lab.rows(r0, r1, ec0, ec1) > 0)
        EV[r0 - er0:r1 - er0] = np.packbits(ev, axis=1)
        PO[r0 - er0:r1 - er0] = np.packbits(pos, axis=1)
        n_ev += int(ev.sum())
        n_pos += int(pos.sum())
    EV.flush(); PO.flush()
    del EV, PO
    lab.close(); [m.close() for m in ms]
    cm = {"files": [f.name for f in eval_mask_files(s)] + [f"{s.seg_id}_inklabels_v2.tif"], "canvas_hw": [H, W],
          "ev_bbox_rows": [er0, er1], "ev_bbox_cols": [ec0, ec1], "n_ev_px": n_ev, "n_pos_px": n_pos,
          "n_neg_px": n_ev - n_pos}
    jdump(cm, f_meta)
    return cm, np.load(d / "evcache_ev.npy", mmap_mode="r"), np.load(d / "evcache_pos.npy", mmap_mode="r")


def value_scale(dt):
    dt = np.dtype(dt)
    if dt == np.uint8:
        return 127, 256, 1
    if dt == np.uint16:
        return 32767, 65536, 1
    if dt.kind == "f":
        return 0.5, 1001, 1000
    raise SystemExit(f"unexpected pred dtype {dt}")


def reduce_one(short, raw: Path, rel: str, R: dict, hm: np.ndarray, cm: dict, EVp, POp) -> tuple[dict, dict, np.ndarray]:
    s = seg(short)
    H, W = R["hf"]["canvas_hw"]
    sy, sx = R["hf"]["grid_factor_sy_sx"]
    GH, GW = hm.shape
    gy, gx = full_to_grid(H, sy, GH), full_to_grid(W, sx, GW)
    R0, R1 = R["full_res"]["crop_rows"]
    C0, C1 = R["full_res"]["crop_cols"]
    er0, er1 = cm["ev_bbox_rows"]
    ec0, ec1 = cm["ev_bbox_cols"]
    P = Pred(raw, H, W)
    info = dict(P.lz.info)
    if P.f == 1 and P.note:
        P.lz.close()
        return {"pred": rel, "file_info": info, "excluded": P.note}, None, None
    dt = np.dtype(P.lz.dtype)
    thr, nb, qf = value_scale(dt)
    h_pos = np.zeros(nb, np.int64)
    h_neg = np.zeros(nb, np.int64)
    vmin, vmax = np.inf, -np.inf
    hi = np.zeros((R1 - R0, C1 - C0), bool)
    rlo, rhi = min(er0, R0), max(er1, R1)
    clo, chi = min(ec0, C0), max(ec1, C1)
    nbits = ec1 - ec0
    for r0 in range((rlo // SUB) * SUB, rhi, SUB):
        r0c, r1 = max(r0, rlo), min(rhi, r0 + SUB)
        a = P.rows(r0c, r1, clo, chi)
        vmin, vmax = min(vmin, float(a.min())), max(vmax, float(a.max()))
        o0, o1 = max(r0c, er0), min(r1, er1)
        if o0 < o1:
            av = a[o0 - r0c:o1 - r0c, ec0 - clo:ec1 - clo]
            ev = np.unpackbits(EVp[o0 - er0:o1 - er0], axis=1, count=nbits).view(bool)
            pos = np.unpackbits(POp[o0 - er0:o1 - er0], axis=1, count=nbits).view(bool)
            for k0 in range(0, o1 - o0, 128):                 # small chunks: bincount copies to int64
                qa = av[k0:k0 + 128]
                q = qa if dt.kind != "f" else np.clip(np.rint(qa * qf), 0, nb - 1).astype(np.int64)
                pk, ek = pos[k0:k0 + 128], ev[k0:k0 + 128]
                h_pos += np.bincount(q[pk].ravel(), minlength=nb)[:nb]
                h_neg += np.bincount(q[ek & ~pk].ravel(), minlength=nb)[:nb]
            del av, ev, pos, q, qa
        o0, o1 = max(r0c, R0), min(r1, R1)
        if o0 < o1:
            mpx = hm[gy[o0:o1]][:, gx[C0:C1]]
            hi[o0 - R0:o1 - R0] = (a[o0 - r0c:o1 - r0c, C0 - clo:C1 - clo] > thr) & mpx
            del mpx
        del a
    P.lz.close()
    cn = np.cumsum(h_neg) - h_neg
    npos, nneg = int(h_pos.sum()), int(h_neg.sum())
    auc = float((h_pos * (cn + 0.5 * h_neg)).sum() / (npos * nneg)) if npos and nneg else float("nan")
    hb = np.arange(nb) > thr * qf
    out = {"pred": rel, "model": model_key(rel), "file_info": info, "upsample_note": P.note, "upsample_factor": P.f,
           "dtype": str(dt), "value_min_window": vmin, "value_max_window": vmax, "p05_threshold_raw": thr,
           "range_ok": bool((dt == np.uint8) or (dt.kind == "f" and vmin >= -1e-6 and vmax <= 1 + 1e-6) or dt == np.uint16),
           "registration_auroc": auc, "registration_ok": bool(auc >= 0.75),
           "recall_p05_labeled_ink": float(h_pos[hb].sum() / npos) if npos else None,
           "reference_fp_p05_inmask_unlabeled": float(h_neg[hb].sum() / nneg) if nneg else None,
           "n_pos_px": npos, "n_neg_px": nneg}
    return out, {"h_pos": h_pos, "h_neg": h_neg}, hi


def placebo_stats(hi: np.ndarray, R: dict, bcells: np.ndarray, vox: float, label: str) -> tuple[dict, dict]:
    R0, _ = R["full_res"]["crop_rows"]
    C0, _ = R["full_res"]["crop_cols"]
    TY, TX = R["full_res"]["tile_grid_TY_TX"]
    region_px = R["full_res"]["px"]
    px_mm2 = (vox / 1000.0) ** 2
    fp_px = int(hi.sum())
    tfp = tile_sum(hi, R0, C0, TY, TX)
    hcells = pool(hi)
    tw = tripwire(hi, hcells, bcells, vox, label)
    b = tw["blobs"]
    out = {"region_px": region_px, "fp_px": fp_px, "fp_area_frac": fp_px / region_px,
           "letter_blobs": b["letter_blobs"], "letter_blob_per_cm2": b["letter_blobs"] / (region_px * px_mm2 / 100.0),
           "hi_map_sha256": sha_mask(hi), "tripwire": tw}
    return out, {"tile_fp": tfp, "hcells": hcells}


def planted_done() -> bool:
    return any("planted" in json.loads(f.read_text()) for s in SEGS for f in (OUT / s).glob("*.tif.json"))


def do_reduce(short):
    if STOP_FILE.exists():
        raise SystemExit(f"HARD STOP recorded ({STOP_FILE}); nothing more is computed")
    reg = json.loads(REG_JSON.read_text())
    R = reg["segments"][short]
    if R["official"]["dropped_core_lt_5mm"]:
        print(short, "region dropped (core < 5 mm): nothing to do")
        return
    with np.load(OUT / f"regions_{short}.npz") as z:
        hm, bcells = z["hf_grid"].copy(), z["bcells"].copy()
    assert sha_mask(hm) == R["hf"]["mask_sha256"], "HF region mask changed since the committed regions file"
    vox = R["full_res"]["px_um"]
    cm, EVp, POp = ev_cache(short)
    s = seg(short)
    remote = list_remote(s)
    od = OUT / short
    od.mkdir(parents=True, exist_ok=True)
    print(f"{short}: {len(R['models'])} preds; ev bbox rows {cm['ev_bbox_rows']} cols {cm['ev_bbox_cols']}", flush=True)
    for mdl in R["models"]:
        rel = mdl["rel"]
        name = Path(rel).name
        if (od / f"{name}.json").exists():
            print("  skip", name, flush=True)
            continue
        if rel not in remote:
            jdump({"pred": rel, "model": mdl["model"], "excluded": "missing from the bucket at download time"},
                  od / f"{name}.json")
            print("  MISSING", name, flush=True)
            continue
        free = check_disk(remote[rel].size)
        raw = RAW / name
        t0 = time.time()
        print(f"  pull {name} ({remote[rel].size / 2**30:.2f} GiB), free {free:.1f} GB", flush=True)
        for k, wait in enumerate((2, 4, 8, 16, None)):
            try:
                if not (raw.exists() and raw.stat().st_size == remote[rel].size):
                    download(remote[rel], raw)
                break
            except SystemExit:
                raise
            except Exception as e:          # noqa: BLE001  (network)
                raw.unlink(missing_ok=True)
                if wait is None:
                    raise
                print(f"   download error ({type(e).__name__}: {e}); retry in {wait} s", flush=True)
                time.sleep(wait)
        td = time.time() - t0
        digest = sha256(raw)
        md, md_text = tiff_meta(raw)
        out, hist, hi = reduce_one(short, raw, rel, R, hm, cm, EVp, POp)
        raw.unlink()
        out.update({"sha256": digest, "bytes": int(remote[rel].size), "tiff_metadata": md,
                    "family_keyword_hits": family_hits(rel, md_text), "download_s": round(td, 1),
                    "free_disk_gb_after_delete": round(free_bytes(REPO) / 2**30, 1)})
        if hi is None:
            out["runtime_s"] = round(time.time() - t0, 1)
            jdump(out, od / f"{name}.json")
            print("   EXCLUDED", out["excluded"], flush=True)
            continue
        pl, arrays = placebo_stats(hi, R, bcells, vox, f"{short}:{mdl['model']}")
        out["placebo"] = pl
        arrays.update(hist)
        tw = pl["tripwire"]
        # planted sanity: first USED pred in processing order (w018 first) with placebo FP px > 0
        if (not tw["text_like_flag"] and not planted_done() and out["registration_ok"]
                and not out["family_keyword_hits"] and pl["fp_px"] > 0):
            R0, R1 = R["full_res"]["crop_rows"]
            C0, C1 = R["full_res"]["crop_cols"]
            GH, GW = hm.shape
            H, W = R["hf"]["canvas_hw"]
            sy, sx = R["hf"]["grid_factor_sy_sx"]
            gy, gx = full_to_grid(H, sy, GH), full_to_grid(W, sx, GW)
            mpx = np.zeros_like(hi)
            for r0 in range(R0, R1, 4 * SH):
                r1 = min(R1, r0 + 4 * SH)
                mpx[r0 - R0:r1 - R0] = hm[gy[r0:r1]][:, gx[C0:C1]]
            hp, info = plant(hi, mpx, pl["fp_px"], pl["region_px"], vox)
            del mpx
            TY, TX = R["full_res"]["tile_grid_TY_TX"]
            arrays["tile_fp_planted"] = tile_sum(hp, R0, C0, TY, TX)
            info["letter_blobs_after"] = int(letter_S(hp, vox)[1])
            info["letter_blobs_before"] = pl["letter_blobs"]
            out["planted"] = info
            del hp
        del hi
        out["runtime_s"] = round(time.time() - t0, 1)
        if tw["text_like_flag"]:            # stop: keep scalars + 5 mm tile counts only, not the 0.25 mm FP map
            arrays.pop("hcells", None)
        np.savez_compressed(od / f"{name}.npz", **arrays)
        jdump(out, od / f"{name}.json")
        p = tw["periodicity"]
        b = tw["blobs"]
        print(f"   {out['dtype']} x{out['upsample_factor']} AUROC {out['registration_auroc']:.3f} "
              f"FP {pl['fp_area_frac']:.4%} refFP {out['reference_fp_p05_inmask_unlabeled']:.3%} "
              f"letters {b['letter_blobs']} S {b['S_max_letter_blobs_1mm_row']} (p99 {b['null_S_p99']}) "
              f"T1 {p['T1_ratio_pooled']:.1f} p {p['p_tile_T1']:.3f} flag {tw['text_like_flag']} "
              f"fam {out['family_keyword_hits']} planted {'planted' in out} (dl {td:.0f} s, {out['runtime_s']} s)",
              flush=True)
        if tw["text_like_flag"]:
            jdump({"segment": short, "region": R["name"], "pred": rel, "utc": time.strftime("%FT%TZ", time.gmtime()),
                   "periodicity": p, "blobs": b, "fp_area_frac": pl["fp_area_frac"]}, STOP_FILE)
            print("HARD STOP: recalibrated text-like flag fired inside a placebo region.", flush=True)
            raise SystemExit(3)
    consensus(short, R, bcells)


def used_records(short, R):
    recs = []
    for mdl in R["models"]:
        f = OUT / short / f"{Path(mdl['rel']).name}.json"
        if f.exists():
            recs.append(json.loads(f.read_text()))
    return recs


def consensus(short, R, bcells):
    od = OUT / short
    recs = [r for r in used_records(short, R) if "excluded" not in r and r["registration_ok"]]
    fr = []
    for r in recs:
        with np.load(od / f"{Path(r['pred']).name}.npz") as z:
            fr.append(z["hcells"] / np.maximum(bcells, 1))
    if not fr:
        return None
    cons = periodicity(Profile(np.mean(fr, 0) * bcells, bcells), f"{short}:consensus")
    c = {k: cons[k] for k in ("n_rows", "length_mm", "T1_ratio_pooled", "T1_peak_period_mm", "p_tile_T1",
                              "tile_null_T1_median", "tile_null_T1_p95")}
    c["flag"] = bool(cons["p_tile_T1"] <= 0.01)
    c["n_models"] = len(recs)
    jdump(c, od / "consensus.json")
    print("  consensus", c, flush=True)
    if c["flag"]:
        jdump({"segment": short, "region": R["name"], "pred": "consensus", "periodicity": c}, STOP_FILE)
        print("HARD STOP: consensus periodicity flag fired inside a placebo region.", flush=True)
        raise SystemExit(3)
    return c


def plumb():
    """End-to-end plumbing check with NO pred: the inklabels_v2 file itself (0/255) is fed through reduce_one and
    placebo_stats as a stand-in pred. Expected: AUROC 1, recall 1, reference FP 0, px counts = cache counts,
    per-tile region px = the committed tile blocks. Also reports labeled-ink px inside each placebo region."""
    reg = json.loads(REG_JSON.read_text())
    res = {}
    for short in SEGS:
        t0 = time.time()
        R = reg["segments"][short]
        with np.load(OUT / f"regions_{short}.npz") as z:
            hm, bcells, T = z["hf_grid"].copy(), z["bcells"].copy(), z["tile_px"].copy()
        cm, EVp, POp = ev_cache(short)
        s = seg(short)
        out, _, hi = reduce_one(short, s.small / f"{s.seg_id}_inklabels_v2.tif", "STANDIN/inklabels_v2.tif", R, hm,
                                cm, EVp, POp)
        R0, R1 = R["full_res"]["crop_rows"]
        C0, C1 = R["full_res"]["crop_cols"]
        H, W = R["hf"]["canvas_hw"]
        sy, sx = R["hf"]["grid_factor_sy_sx"]
        gy, gx = full_to_grid(H, sy, hm.shape[0]), full_to_grid(W, sx, hm.shape[1])
        TY, TX = R["full_res"]["tile_grid_TY_TX"]
        m = hm[gy[R0:R1]][:, gx[C0:C1]]
        tiles_equal = bool(np.array_equal(tile_sum(m, R0, C0, TY, TX), T))
        del m
        pl, _ = placebo_stats(hi, R, bcells, R["full_res"]["px_um"], f"{short}:standin")
        chk = {"auroc_1": out["registration_auroc"] == 1.0, "recall_1": out["recall_p05_labeled_ink"] == 1.0,
               "reference_fp_0": out["reference_fp_p05_inmask_unlabeled"] == 0.0,
               "counts_equal_cache": out["n_pos_px"] == cm["n_pos_px"] and out["n_neg_px"] == cm["n_neg_px"],
               "tile_px_equal_committed": tiles_equal}
        res[short] = {"pass": all(chk.values()), "checks": chk, "labeled_ink_px_in_placebo": pl["fp_px"],
                      "ev_cache": cm, "stream_s": round(time.time() - t0, 1)}
        print(short, res[short], flush=True)
    jdump(res, OUT / "plumb.json")
    return res


# ------------------------------------------------------------------------------------------------ analysis
def seg_analysis(short, R, stats=True):
    od = OUT / short
    recs = used_records(short, R)
    excluded, family, dup, main = [], [], [], []
    merged_rep = None
    for r in recs:
        if "excluded" in r:
            excluded.append({"pred": r["pred"], "why": r["excluded"]})
        elif not r["registration_ok"]:
            excluded.append({"pred": r["pred"], "why": f"registration AUROC {r['registration_auroc']:.3f} < 0.75"})
        elif r["family_keyword_hits"]:
            family.append(r)
        elif r["model"].startswith("merged_"):
            if merged_rep is None:
                merged_rep = r
                main.append(r)
            else:
                dup.append({"pred": r["pred"], "same_p05_map_as": merged_rep["pred"],
                            "identical": r["placebo"]["hi_map_sha256"] == merged_rep["placebo"]["hi_map_sha256"]})
        else:
            main.append(r)
    units = np.array([[u[0], u[1]] for u in R["full_res"]["tile_blocks"]])
    upx = np.array([u[2] for u in R["full_res"]["tile_blocks"]])

    def unit_fp(r, key="tile_fp"):
        with np.load(od / f"{Path(r['pred']).name}.npz") as z:
            return z[key][units[:, 0], units[:, 1]].astype(float)

    fp = np.stack([unit_fp(r) for r in main]) if main else np.zeros((0, len(units)))
    names = [r["model"] for r in main]
    Wb = boot_weights(len(units)) if stats else None
    pairs = pair_table(fp, names, Wb) if (stats and len(main) > 1) else []
    selfp = self_pairs(fp, names, Wb) if (stats and main) else {}
    rows = []
    for r in recs:
        if "excluded" in r:
            continue
        p = r["placebo"]
        tw = p["tripwire"]
        rows.append({"model": r["model"], "pred": r["pred"],
                     "role": ("family (separate row)" if r["family_keyword_hits"] else
                              "excluded: registration" if not r["registration_ok"] else
                              "duplicate merged_*" if any(d["pred"] == r["pred"] for d in dup) else "main"),
                     "upsample_factor": r["upsample_factor"], "registration_auroc": r["registration_auroc"],
                     "recall_p05": r["recall_p05_labeled_ink"], "placebo_fp_area": p["fp_area_frac"],
                     "placebo_fp_px": p["fp_px"], "reference_fp": r["reference_fp_p05_inmask_unlabeled"],
                     "placebo_over_reference": (p["fp_area_frac"] / r["reference_fp_p05_inmask_unlabeled"]
                                                if r["reference_fp_p05_inmask_unlabeled"] else None),
                     "letter_blobs": p["letter_blobs"], "letter_blob_per_cm2": p["letter_blob_per_cm2"],
                     "S_1mm_row": tw["blobs"]["S_max_letter_blobs_1mm_row"], "S_null_p99": tw["blobs"]["null_S_p99"],
                     "T1": tw["periodicity"]["T1_ratio_pooled"], "T1_peak_mm": tw["periodicity"]["T1_peak_period_mm"],
                     "p_tile_T1": tw["periodicity"]["p_tile_T1"], "periodicity_flag": tw["periodicity_flag"],
                     "blob_flag": tw["blob_flag"], "text_like_flag": tw["text_like_flag"],
                     "family_keyword_hits": r["family_keyword_hits"], "sha256": r["sha256"],
                     "download_s": r.get("download_s"), "runtime_s": r.get("runtime_s")})
    out = {"region": R["name"], "region_px": R["full_res"]["px"], "region_area_mm2": R["full_res"]["area_mm2"],
           "n_tile_blocks": len(units), "tile_block_px_min_median_max": [int(upx.min()), int(np.median(upx)), int(upx.max())],
           "n_listed": len(R["models"]), "n_reduced": len(recs), "n_main": len(main), "excluded": excluded,
           "family_rows": [r["pred"] for r in family], "merged_duplicates": dup,
           "registration_fail_frac": (sum(1 for e in excluded if e["why"].startswith("registration")) / len(recs)
                                      if recs else None),
           "models": rows, "pairs": pairs, "self_pairs": selfp,
           "consensus": json.loads((od / "consensus.json").read_text()) if (od / "consensus.json").exists() else None}
    for r in recs:
        if stats and "planted" in r:
            fpo = unit_fp(r)
            fpp = unit_fp(r, "tile_fp_planted")
            pt = pair_table(np.stack([fpp, fpo]), ["planted", "original"], Wb)[0]
            out["planted"] = {**r["planted"], "model": r["model"], "ratio": pt["ratio"], "ci95": pt["ci95"],
                              "recovered": bool(pt["ratio"] >= 1.3), "ci_excludes_1": pt["ci_excludes_1"]}
    return out


def analyze():
    reg = json.loads(REG_JSON.read_text())
    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code": "tools/margin/v0e.py",
           "git_head": git_head(), "frozen_rule": reg["frozen_rule"],
           "regions_file": "results/lie_detector_v0e_regions.json",
           "hard_stop": json.loads(STOP_FILE.read_text()) if STOP_FILE.exists() else None,
           "synthetic_checks": json.loads((OUT / "synth.json").read_text()) if (OUT / "synth.json").exists() else None,
           "segments": {}}
    if res["hard_stop"]:
        # HARD STOP: compile the scalars already computed; no ratios, no Spearman, no verdict (stop computing)
        for s in SEGS:
            if (OUT / s).exists():
                res["segments"][s] = seg_analysis(s, reg["segments"][s], stats=False)
        res["verdict"], res["verdict_rule"] = "HARD STOP", ("recalibrated tripwire fired inside a placebo region; "
                                                            "nothing further computed; The authors decide")
        jdump(res, RES_JSON)
        print(json.dumps({"verdict": res["verdict"], "hard_stop": res["hard_stop"]}, indent=1, default=LAT._js))
        return res
    for s in SEGS:
        res["segments"][s] = seg_analysis(s, reg["segments"][s])
    m = {s: {r["model"]: r["placebo_fp_area"] for r in res["segments"][s]["models"] if r["role"] == "main"}
         for s in SEGS}
    common = sorted(set(m["w018"]) & set(m["w023"]))
    a, b = [m["w018"][k] for k in common], [m["w023"][k] for k in common]
    rho = float(spearmanr(a, b)[0]) if len(common) >= 3 and len(set(a)) > 1 and len(set(b)) > 1 else float("nan")
    res["spearman"] = {"models": common, "rho": rho, "n": len(common), "w018": a, "w023": b}
    allp = [p for s in SEGS for p in res["segments"][s]["pairs"]]
    allm = [r for s in SEGS for r in res["segments"][s]["models"] if r["role"] == "main"]
    planted = next((res["segments"][s]["planted"] for s in SEGS if "planted" in res["segments"][s]), None)
    sanity = {"registration_used_all_ge_0.75": all(r["registration_auroc"] >= 0.75 for r in allm),
              "registration_fail_le_half_each_segment": all((res["segments"][s]["registration_fail_frac"] or 0) <= 0.5
                                                            for s in SEGS),
              "planted_recovered": bool(planted and planted["recovered"]),
              "self_pair_exact_all": all(v["exact_1"] for s in SEGS for v in res["segments"][s]["self_pairs"].values())}
    res["sanity"] = sanity
    any_pass = any(p["pass_pair"] for p in allp)
    all_incl = all(not p["ci_excludes_1"] for p in allp)
    all_clean = all(r["placebo_fp_area"] < 0.001 for r in allm)
    rho_lt = not (rho >= 0.5)
    res["counts"] = {"pairs": len(allp), "pairs_ci_excl_1": sum(p["ci_excludes_1"] for p in allp),
                     "pairs_ratio_ge_1.5": sum(p["ratio"] >= 1.5 for p in allp),
                     "pairs_pass": sum(p["pass_pair"] for p in allp)}
    kill = []
    if all_incl:
        kill.append("All pair CIs include 1")
    if all_clean:
        kill.append("every model's placebo FP < 0.1%")
    if rho_lt:
        kill.append(f"Spearman < 0.5 (rho = {rho:.3f})" if np.isfinite(rho) else "Spearman undefined (counted as < 0.5)")
    if res["hard_stop"]:
        v, why = "HARD STOP", "tripwire fired in a placebo region; no verdict (The authors decide)"
    elif not (sanity["planted_recovered"] and sanity["self_pair_exact_all"] and
              sanity["registration_fail_le_half_each_segment"]):
        v, why = "NO VERDICT", "sanity failure = harness bug"
    elif kill:
        v, why = "KILL", "KILL: " + "; ".join(kill)
    elif any_pass and rho >= 0.5:
        v, why = "PASS", ">= 1 model pair with placebo FP ratio >= 1.5 and 95% CI excluding 1, AND Spearman >= 0.5"
    else:
        v, why = "KILL", "INCONCLUSIVE = KILL (no retry)"
    res["verdict"], res["verdict_rule"] = v, why
    res["verdict_context"] = {"any_pass_pair": any_pass, "all_ci_include_1": all_incl,
                              "all_fp_below_0.1pct": all_clean, "spearman": rho, "kill_clauses_true": kill}
    res["margins_row"] = "NOT RUN: conditional on the org confirming the column foot at high z (not confirmed)."
    jdump(res, RES_JSON)
    print(json.dumps({k: res[k] for k in ("verdict", "verdict_rule", "sanity", "counts", "spearman")}, indent=1,
                     default=LAT._js))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["regions", "synth", "plumb", "reduce", "analyze", "report"])
    ap.add_argument("--seg")
    a = ap.parse_args()
    import resource
    t0 = time.time()
    try:
        if a.cmd == "regions":
            regions()
        elif a.cmd == "synth":
            synth()
        elif a.cmd == "plumb":
            plumb()
        elif a.cmd == "reduce":
            do_reduce(a.seg)
        elif a.cmd == "analyze":
            analyze()
        else:
            from tools.margin.v0e_report import report
            report()
    finally:
        print(f"[{a.cmd}] wall {time.time() - t0:.0f} s, peak RSS "
              f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
