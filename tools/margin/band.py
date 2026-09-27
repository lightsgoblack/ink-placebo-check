"""Fix the upper-margin band from labels + geometry ONLY (Amendment C / Lie Detector v0-M).

Band (the frozen-criteria history Amendment C, frozen): per segment, surface with
    z > (99.5th pct of labeled-ink z) + 2 mm guard,  >= 2 mm from any segment edge,  supervision = 0.

Implementation (all on the tifxyz grid, 1 cell = 1/scale ~ 20 full-res px, then mapped x20):
  * labeled-ink z: inklabels_v2 > 0 counted per grid cell (exact, nearest-grid-point cells, streamed
    in row strips); percentile weighted by ink px count.
  * segment edge: grid points with z <= 0 (tifxyz invalid = -1) plus the image border; distance
    = Euclidean distance transform on the grid x cell size (mm).
  * supervision = 0: cell has zero px of supervision_mask_v2 AND zero px of supervision_mask (v1).
  * full res: pixel (r, c) is in the band iff its nearest grid cell is.
Reads no render and no pred. Writes data/margin_bands.json, results/margin_bands.json and
data/margin_band_<seg>.npz (band on the grid). No images.

    .venv/bin/python -m tools.margin.band
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import tifffile
from scipy import ndimage as ndi

from tools.margin.common import (BAND_JSON, BAND_JSON_VAULT, REPO, SEGMENTS, TILE_PX, Lazy2D, full_to_grid,
                                 grid_factors, meta, pool_count_to_grid, seg, tile_grid, voxel_um)

INK_PCT = 99.5
GUARD_MM = 2.0
EDGE_MM = 2.0

# Fixed BEFORE any pred or render is read (committed with the band). Interpretations of the frozen
# criteria where the text leaves a choice; each is listed again in the results deviations table.
ANALYSIS_PLAN = {
    "tile_grid": "5 mm = 2084 px full res, anchored (0,0), trailing partial tiles dropped (stage-1 code)",
    "blank_tile": "100% of tile px in the band (Amendment C)",
    "inked_tile": ">= 50% in validation_mask_v2 OR supervision_mask_v2 OR supervision_mask(v1), "
                  "inklabels_v2 fraction >= 2% (stage 1 rule and code)",
    "text_block_tiles_for_validity_check": "the inked tiles above (per segment)",
    "validity_check": "per segment, mean over the segment's non-reverse preds of the tile mean pred value; "
                      "margin (blank) tiles below the 10th pct of that segment's text-block tiles in >= 90% "
                      "of margin tiles. Evaluated on all 4 band segments; failing on ANY segment = hard stop.",
    "pred_threshold": "p > 0.5: value > 127 if uint8 0-255, > 0.5 if float 0-1 (dtype/range verified per file)",
    "registration_auroc": "pixel AUROC of pred value, inklabels_v2 > 0 vs label == 0, within the evaluated "
                          "mask (val_v2 | sup_v2 | sup_v1); exact from value histograms; must be >= 0.75",
    "recall_ref": "fraction of labeled-ink px (inside evaluated mask) with p > 0.5",
    "ld_band_region": "band px only; bootstrap units = 5 mm tiles with >= 50% of px in band (v0 CI rule); "
                      "tile blocks = the 5 mm tiles themselves (B=2000, seed 20260925, paired across models)",
    "ld_fp_area": "band px with p > 0.5 / band px, per model per segment",
    "ld_blob": "8-connected components of (p > 0.5) & band, area 1 to 4 mm2 (173,760 to 695,041 px at "
               "2.399 um), counted per cm2 of band",
    "ld_ratio": "pairwise ratio of band FP px sums over resampled tiles, larger/smaller model; PASS pair = "
                "ratio >= 1.5 with 95% CI excluding 1 on any band segment",
    "ld_spearman": "Spearman of band FP area fraction across the models present on both w018 and w023",
    "ld_planted": "w023 first pred: plant disks of 2 mm2 (value 255) at seeded random band positions until "
                  "planted area = max(0.5 x its band FP px, 0.1% of band px); recovered if planted/original "
                  "ratio >= 1.3 with CI excluding 1",
    "text_like_T1_row_periodicity": "per model and for the model mean: band row profile = mean over band px "
                                    "of (p > 0.5) per image row, linearly detrended, periodogram; flag if the "
                                    "model's band FP area >= 0.1% AND max power over periods 2.5-8 mm >= 10x "
                                    "median power over periods 0.5-8 mm. Same statistic reported on the "
                                    "text block rows as a positive control.",
    "text_like_T2_blobs_in_rows": "per model: >= 5 letter-sized (1-4 mm2) blobs in the band AND >= 3 of them "
                                  "with centroid rows within +-1 mm of each other = aligned in a row -> flag",
    "text_like_action": "any T1 or T2 flag, or a failed validity check = HARD STOP (compute nothing further, "
                        "no images, report scalars only)",
    "md_position_baseline_P": "tile mean surface z distance below the segment's top edge (max valid z), mm; "
                              "higher = further from top; AUROC like other features; PASS needs "
                              "AUROC(F1) - AUROC(P) >= 0.05 with CI excluding 0",
    "md_render": "full-res max_22_42; features over valid render px (render > 0) as stage-1 code; "
                 "DoG via FFT Gaussian (truncation >= 4 sigma, same as scipy default) on 4x4-tile windows "
                 "with 1700 px context",
}


def weighted_pct(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    o = np.argsort(values)
    v, w = values[o], weights[o].astype(np.float64)
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, q / 100.0 * c[-1])])


def sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def band_for(short: str, guard_mm: float = GUARD_MM, inclusive: bool = False, npz_stem: str = "margin_band") -> dict:
    """guard_mm / inclusive (z >= instead of z >) / npz_stem let Amendment D reuse this; defaults = Amendment C."""
    t0 = time.time()
    s = seg(short)
    m = meta(short)
    vox = voxel_um(m)
    z = np.squeeze(tifffile.imread(s.small / "z.tif")).astype(np.float64)
    GH, GW = z.shape
    lab = Lazy2D(s.small / f"{s.seg_id}_inklabels_v2.tif")
    H, W = lab.shape
    sy, sx = grid_factors(m, (H, W), (GH, GW))
    cell_mm = vox / 1000.0 * 0.5 * (1 / sy + 1 / sx)
    valid = z > 0
    zmm = z * vox / 1000.0

    ink = pool_count_to_grid(lab, sy, sx, (GH, GW)); lab.close()
    sup = np.zeros((GH, GW), np.int64)
    sup_files = []
    for nm in ("supervision_mask_v2", "supervision_mask"):
        f = s.small / f"{s.seg_id}_{nm}.tif"
        if f.exists():
            lz = Lazy2D(f); sup += pool_count_to_grid(lz, sy, sx, (GH, GW)); lz.close()
            sup_files.append(f.name)
    lz = Lazy2D(s.small / f"{s.seg_id}_validation_mask_v2.tif")
    val = pool_count_to_grid(lz, sy, sx, (GH, GW)); lz.close()

    iw = (ink > 0) & valid
    ink_z = zmm[iw]
    thr_ink = weighted_pct(ink_z, ink[iw], INK_PCT)
    z_lo = weighted_pct(ink_z, ink[iw], 0.5)
    edge = ndi.distance_transform_edt(np.pad(valid, 1))[1:-1, 1:-1] * cell_mm
    zabove = (zmm >= thr_ink + guard_mm) if inclusive else (zmm > thr_ink + guard_mm)
    band = valid & zabove & (edge >= EDGE_MM) & (sup == 0)

    # full-res mapping
    gy = full_to_grid(H, sy, GH)
    gx = full_to_grid(W, sx, GW)
    rows_per = np.bincount(gy, minlength=GH)
    cols_per = np.bincount(gx, minlength=GW)
    band_px = int(rows_per @ band.astype(np.int64) @ cols_per)
    px_mm2 = (vox / 1000.0) ** 2
    br = np.flatnonzero(band.any(1)); bc = np.flatnonzero(band.any(0))
    row_lo = int(np.searchsorted(gy, br.min(), "left")); row_hi = int(np.searchsorted(gy, br.max(), "right"))
    col_lo = int(np.searchsorted(gx, bc.min(), "left")); col_hi = int(np.searchsorted(gx, bc.max(), "right"))

    # tiles
    ty, tx = tile_grid((H, W))
    frac = np.zeros((ty, tx))
    for i in range(ty):
        yy = gy[i * TILE_PX:(i + 1) * TILE_PX]
        if not band[yy[0]:yy[-1] + 1].any():
            continue
        for j in range(tx):
            xx = gx[j * TILE_PX:(j + 1) * TILE_PX]
            sub = band[yy[0]:yy[-1] + 1, xx[0]:xx[-1] + 1]
            if not sub.any():
                continue
            frac[i, j] = (rows_per_sub(yy) @ sub.astype(np.float64) @ rows_per_sub(xx)) / (TILE_PX * TILE_PX)
    full = np.argwhere(frac >= 1.0 - 1e-12)
    half = np.argwhere(frac >= 0.5)

    ncomp = ndi.label(band)[1]
    colh = band.sum(0)[bc] * cell_mm
    zdir = np.corrcoef(np.arange(GH)[valid.any(1)], np.nanmean(np.where(valid, zmm, np.nan), 1)[valid.any(1)])[0, 1]
    npz = REPO / "data" / f"{npz_stem}_{short}.npz"
    np.savez_compressed(npz, band_grid=band, sy=sy, sx=sx)
    out = {
        "segment": SEGMENTS[short], "voxel_um": vox, "full_shape": [H, W], "grid_shape": [GH, GW],
        "grid_factor_sy_sx": [sy, sx], "grid_cell_mm": cell_mm,
        "labeled_ink_px_total": int(ink.sum()),
        "ink_z_mm_p0.5": z_lo, "ink_z_mm_p99.5": thr_ink, "band_z_threshold_mm": thr_ink + guard_mm,
        "surface_z_mm_range": [float(zmm[valid].min()), float(zmm[valid].max())],
        "z_top_mm_max_valid": float(zmm[valid].max()),
        "corr_row_vs_z": float(zdir),
        "supervision_files": sup_files,
        "band_grid_cells": int(band.sum()),
        "band_full_res_px": band_px, "band_area_mm2": band_px * px_mm2,
        "band_row_range_full_res": [row_lo, row_hi], "band_col_range_full_res": [col_lo, col_hi],
        "band_components_grid_8conn": int(ndi.label(band, np.ones((3, 3)))[1]), "band_components_grid_4conn": ncomp,
        "band_height_mm_per_column": {"median": float(np.median(colh)), "p10": float(np.percentile(colh, 10)),
                                      "p90": float(np.percentile(colh, 90))},
        "band_overlap": {"labeled_ink_px": int(ink[band].sum()), "validation_v2_px": int(val[band].sum()),
                         "supervision_px": int(sup[band].sum())},
        "tile_px": TILE_PX, "tile_grid": [ty, tx],
        "tiles_100pct_in_band": int(len(full)), "tiles_ge50pct_in_band": int(len(half)),
        "tiles_100pct_ij": full.tolist(), "tiles_ge50pct_ij": half.tolist(),
        "tiles_ge50pct_band_frac": [float(frac[i, j]) for i, j in half],
        "band_grid_npz": str(npz.relative_to(REPO)), "band_grid_npz_sha256": sha256(npz),
        "runtime_s": round(time.time() - t0, 1),
    }
    return out


def rows_per_sub(idx: np.ndarray) -> np.ndarray:
    """Count of full-res indices per grid index within a tile, over the tile's grid index span."""
    return np.bincount(idx - idx[0]).astype(np.float64)


def main():
    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "note": "Fixed from labels + tifxyz geometry only. No render or pred had been read when this was "
                   "written (see git commit time vs first pred pull).",
           "definition": {"ink_pct": INK_PCT, "guard_mm": GUARD_MM, "edge_mm": EDGE_MM,
                          "labels": "inklabels_v2", "supervision": "supervision_mask_v2 OR supervision_mask(v1) == 0",
                          "segment_edge": "tifxyz z <= 0 (invalid) or grid border",
                          "full_res_mapping": "nearest tifxyz grid point, index = round(px * scale)"},
           "analysis_plan_fixed_before_reading_preds": ANALYSIS_PLAN,
           "segments": {}}
    for short in SEGMENTS:
        r = band_for(short)
        res["segments"][short] = r
        print(short, {k: r[k] for k in ("ink_z_mm_p99.5", "band_area_mm2", "tiles_100pct_in_band",
                                        "tiles_ge50pct_in_band", "band_row_range_full_res", "band_overlap",
                                        "corr_row_vs_z", "runtime_s")}, flush=True)
    for p in (BAND_JSON, BAND_JSON_VAULT):
        p.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
