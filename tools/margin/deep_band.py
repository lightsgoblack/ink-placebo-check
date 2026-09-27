"""Amendment D, step 1: the "deep margin" band on w023 + w018, from labels + tifxyz geometry ONLY.

Deep band (the frozen-criteria history Amendment D, frozen at a9eac12): per segment, surface with
    z >= (99.5th pct labeled-ink z) + 13 mm,  >= 2 mm from any segment edge,  supervision = 0.
Same machinery as band.py (Amendment C band), only the guard (13 mm) and ">=" change.

Also written here, BEFORE any w023/w018 pred or render is read:
  * 5 mm tile counts (100% in band = Metal Detector blank class; >= 50% = Lie Detector bootstrap units);
  * pooled 0.25 mm band cell counts (for the 1 mm tile-shuffle null);
  * a labels-only power check of the recalibrated periodicity flag (descriptive, not a gate): the same
    tile-shuffle test applied to the inklabels_v2 row profile inside the evaluated mask, in windows as tall
    as the deep-band profile, and on the full text block;
  * the analysis plan (interpretation choices) for Lie Detector v0-M and Metal Detector stage 1-M on the band.
Writes data/deep_band_<seg>.npz, data/deep_bands.json, results/deep_bands.json. No images.

    .venv/bin/python -m tools.margin.deep_band
"""
from __future__ import annotations

import json
import time

import numpy as np

from tools.margin.band import EDGE_MM, INK_PCT, band_for
from tools.margin.common import REPO, Lazy2D, full_to_grid, seg, tile_grid, TILE_PX
from tools.margin.forensics import CELL
from tools.margin.forensics_analyze import FULL, TILE, Profile, periodicity
from tools.margin.reduce_pred import eval_mask_files

GUARD_MM = 13.0
SEGS = ("w023", "w018")
DEEP_JSON = REPO / "data" / "deep_bands.json"
DEEP_JSON_VAULT = REPO / "vault" / "results" / "deep_bands.json"
POOL_OUT = REPO / "data" / "margin_deep"

PLAN = {
    "status": "Fixed and committed BEFORE any w023/w018 pred or render is read (Amendment D order). "
              "Where the frozen text leaves a choice, the choice is made here; every item is repeated in the "
              "results deviations/interpretations table.",
    "models": {
        "set": "all non-reverse preds per segment from the bucket listing (w023: 8 ps*; w018: 10 ps* + "
               "1667_2um_pred.tif). No merged_* preds exist on w023/w018, so the 'merged_* = one model' rule "
               "is vacuous here.",
        "odd_pred_1667_2um_pred": "shape/dtype checked on read. Same shape as labels -> used as is. Shape an "
                                  "exact integer downsample f of the labels (each axis within f px) -> nearest-"
                                  "neighbour upsample (pixel (r,c) takes pred[r//f, c//f]) and logged as a "
                                  "deviation. Anything else -> excluded (deviation). Registration applies as to "
                                  "every pred.",
        "registration": "pixel AUROC of pred value, inklabels_v2 > 0 vs == 0 inside val_v2 | sup_v2 | sup_v1 "
                        "(exact from histograms). A pred with AUROC < 0.75 is excluded from ratios, Spearman "
                        "and the validity mean, and listed; if more than half of a segment's preds fail, it is "
                        "treated as a harness bug (no verdict).",
        "threshold": "p > 0.5 (uint8 > 127; float > 0.5; uint16 > 32767)",
    },
    "lie_detector_v0m_deep": {
        "fp_area": "deep-band px with p > 0.5 / deep-band px, per model per segment (full band)",
        "blob_density": "8-connected components of (p > 0.5) & deep band at full res, area 1-4 mm2, per cm2 "
                        "of deep band",
        "bootstrap_units": "5 mm tiles (2084 px grid, anchored (0,0)) with >= 50% of px in the deep band; "
                           "B = 2000, seed 20260925, paired (same tile resample for all models of a segment); "
                           "no minimum tile count is stated in v0-M, the count is reported",
        "pair_ratio": "sum of FP px over the bootstrap tiles, model A / model B with A = larger point value; "
                      "CI = 2.5/97.5 pct of the resampled A/B; B sum 0 & A > 0 -> inf, both 0 -> 1.0; CI "
                      "excludes 1 iff lower bound > 1",
        "pairs": "all pairs within a segment (w023 28, w018 up to 55)",
        "spearman": "Spearman rho of full-band FP area across the models present on both segments (8 ps* "
                    "names shared by w023 and w018)",
        "PASS": ">= 1 pair (either segment) with ratio >= 1.5 AND CI excluding 1, AND Spearman >= 0.5",
        "KILL": "all pairwise CIs (both segments) include 1, OR every model's full-band FP area < 0.1% on both "
                "segments, OR Spearman < 0.5",
        "else": "INCONCLUSIVE = KILL (Amendment D: no retry)",
        "extra_baseline": "same model's FP at p > 0.5 on in-mask unlabeled text-block px (h_neg above threshold "
                          "/ h_neg), reported next to band FP; not a PASS input",
        "sanity": {
            "registration": "each used pred AUROC >= 0.75 (above)",
            "planted": "w023 first pred (sorted listing order): 2 mm2 disks (value = on) planted at seeded "
                       "(20260925) random deep-band positions until planted px >= max(0.5 x band FP px, 0.1% "
                       "of band px); recovered iff planted/original ratio >= 1.3 with CI lower > 1",
            "self_pair": "ratio of a model with itself through the same code = 1.00 exactly, CI [1, 1]",
        },
    },
    "recalibrated_text_like_flag": {
        "periodicity": "pooled 0.25 mm cells (104 px) of (p > 0.5) & deep band and deep-band px; row profile = "
                       "sum hi / sum band per pooled row over rows with band px >= 10% of the max row "
                       "(forensics_analyze.Profile); T1 ratio = max periodogram power over periods 2.5-8 mm / "
                       "median power over 0.5-8 mm (linear detrend). Null: 1 mm tiles (4x4 cells, anchored at "
                       "absolute multiples of 416 px) wholly inside the band permuted among themselves, partial "
                       "tiles stay; 200 surrogates, seeds 20260925+k; p = (1 + #{null >= obs}) / 201. FLAG iff "
                       "p <= 0.01 (T1 over 2.5-8 mm is the 'at a period in 2.5 to 8 mm' statistic, so the "
                       "window maximum is inside the null). No FP-area gate (conservative for a stop).",
        "blob_rows": "letter blobs = 8-connected full-res components of (p > 0.5) & deep band, 1-4 mm2. "
                     "Statistic S = max number of letter-blob centroids inside any 1 mm tall row window "
                     "[r, r + 1 mm). Null: same 1 mm (416 px) tile shuffle applied to the full-res (p > 0.5) & "
                     "band map, components recomputed, 200 surrogates (seeds 20260925+k). FLAG iff S >= 3 AND "
                     "S > 99th pct of null S. If S < 3 the flag cannot fire and the null is not computed.",
        "scope": "each used pred, and the model-mean consensus (mean over used preds of the pooled hi "
                 "fraction; periodicity only), per segment",
        "action": "any flag -> HARD STOP (text rule a/b): stop computing, commit scalars only, report",
        "positive_control_labels_only": "descriptive, not a gate: the periodicity test on the inklabels_v2 "
                                        "profile inside the evaluated mask, in windows of the deep-band profile "
                                        "height (step half a window) and on the full text block. Computed in "
                                        "step 1 from labels only.",
    },
    "metal_detector_stage1m_deep": {
        "blank_tiles": "5 mm tiles 100% inside the deep band; pooled w023 + w018; if < 15 -> NOT EVALUABLE "
                       "(stage 1-M part only)",
        "inked_tiles": ">= 50% in val_v2 | sup_v2 | sup_v1 and inklabels_v2 fraction >= 2% (stage 1 rule)",
        "validity_check": "per segment, mean over the used preds of the tile mean pred value (0-1): deep-band "
                          "blank tiles below the 10th pct of that segment's inked tiles in >= 90% of blank "
                          "tiles; failing on either segment = HARD STOP (Amendment C: possible unlabeled text)",
        "render": "full-res max_22_42 via the harness bucket, one file at a time, deleted after reduction",
        "features_stats": "frozen stage 1 (tools/metal_detector_v0/stage1.py): per-segment 1-99 pct linear "
                          "normalization over valid px (render > 0), B0, F1 (p99.5), F2/F3 (DoG 0.2/1.0 mm, "
                          "normalized convolution; FFT Gaussian on 2x2-tile windows with 1700 px context), Q, "
                          "4x4-tile bootstrap blocks never spanning segments, B = 2000, seed 20260925, 200 "
                          "permutations, 10 mm shift",
        "position_baseline_P": "tile mean z distance below the segment's max valid z (mm); PASS also needs "
                               "AUROC(F1) - AUROC(P) >= 0.05 with CI excluding 0",
        "verdict": "stage 1 PASS/KILL/INCONCLUSIVE lines; PASS also needs P beaten and validity passed; "
                   "INCONCLUSIVE = KILL",
    },
}


def pooled_band(short: str, grid: np.ndarray, sy: float, sx: float) -> dict:
    """Pooled 104 px cell counts of deep-band px over the band rows; also eval-mask and label cells for the
    labels-only positive control (whole image)."""
    s = seg(short)
    lab = Lazy2D(s.small / f"{s.seg_id}_inklabels_v2.tif")
    masks = [Lazy2D(f) for f in eval_mask_files(s)]
    H, W = lab.shape
    GH, GW = grid.shape
    gy, gx = full_to_grid(H, sy, GH), full_to_grid(W, sx, GW)
    nr, nc = H // CELL, W // CELL
    n_band = np.zeros((nr, nc)); n_ev = np.zeros((nr, nc)); n_evink = np.zeros((nr, nc))
    step = CELL * 5
    for r0 in range(0, nr * CELL, step):
        r1 = min(nr * CELL, r0 + step)
        k = (r1 - r0) // CELL
        i0 = r0 // CELL
        ink = lab.rows(r0, r1)[:, :nc * CELL] > 0
        ev = masks[0].rows(r0, r1)[:, :nc * CELL] > 0
        for mm in masks[1:]:
            ev |= mm.rows(r0, r1)[:, :nc * CELL] > 0
        bnd = grid[gy[r0:r1]][:, gx[:nc * CELL]]
        def ps(a):
            return a.reshape(k, CELL, nc, CELL).sum(axis=(1, 3), dtype=np.float64)
        n_band[i0:i0 + k] = ps(bnd); n_ev[i0:i0 + k] = ps(ev); n_evink[i0:i0 + k] = ps(ev & ink)
        del ink, ev, bnd
    lab.close(); [m.close() for m in masks]
    return {"n_band": n_band, "n_ev": n_ev, "n_evink": n_evink}


def power_check(pc: dict) -> dict:
    """Labels-only: can the recalibrated periodicity flag fire on real labeled text at deep-band height?"""
    band = Profile(pc["n_band"], pc["n_band"])
    h = len(band.rows)
    txt = Profile(pc["n_evink"], pc["n_ev"])
    full = periodicity(txt, "labels_text_block_full")
    wins = []
    r = txt.rows[0]
    while r + h <= txt.rows[-1] + 1:
        rm = np.zeros(pc["n_ev"].shape[0], bool); rm[r:r + h] = True
        try:
            pr = Profile(pc["n_evink"], pc["n_ev"], rowmask=rm)
        except ValueError:
            r += max(1, h // 2); continue
        if len(pr.rows) == h and len(pr.tiles()) >= 8:
            o = periodicity(pr, f"rows {r}-{r + h}")
            wins.append({"row0_cell": int(r), "p_tile_T1": o["p_tile_T1"], "T1": o["T1_ratio_pooled"],
                         "peak_mm": o["T1_peak_period_mm"], "n_tiles": len(pr.tiles())})
        r += max(1, h // 2)
    ps = np.array([w["p_tile_T1"] for w in wins]) if wins else np.array([])
    return {"deep_profile_rows_cells": int(h), "deep_profile_height_mm": float(h * CELL * 2.399e-3),
            "deep_band_shuffle_tiles": len(band.tiles()),
            "text_block_full": {k: full[k] for k in ("n_rows", "length_mm", "T1_ratio_pooled", "T1_peak_period_mm",
                                                     "p_tile_T1", "tile_null_T1_median")},
            "windows_n": len(wins), "windows_frac_p_le_0.01": float((ps <= 0.01).mean()) if len(ps) else None,
            "windows_median_p": float(np.median(ps)) if len(ps) else None, "windows": wins}


def main():
    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "note": "Amendment D deep band, fixed from labels + tifxyz geometry only. No w023/w018 pred or render "
                   "had been read when this was written and committed.",
           "definition": {"ink_pct": INK_PCT, "guard_mm": GUARD_MM, "edge_mm": EDGE_MM, "z_rule": ">=",
                          "labels": "inklabels_v2", "supervision": "supervision_mask_v2 OR supervision_mask(v1) == 0",
                          "segment_edge": "tifxyz z <= 0 (invalid) or grid border",
                          "full_res_mapping": "nearest tifxyz grid point, index = round(px * scale)",
                          "guard_source": "w031 margin forensics (results/margin_forensics_w031.md); never tuned "
                                          "on w023/w018"},
           "analysis_plan_fixed_before_reading_preds": PLAN, "segments": {}}
    POOL_OUT.mkdir(parents=True, exist_ok=True)
    for short in SEGS:
        r = band_for(short, guard_mm=GUARD_MM, inclusive=True, npz_stem="deep_band")
        with np.load(REPO / r["band_grid_npz"]) as z:
            grid = z["band_grid"]
        pc = pooled_band(short, grid, *r["grid_factor_sy_sx"])
        np.savez_compressed(POOL_OUT / f"{short}_static.npz", **pc)
        r["pooled_band_px_check"] = int(pc["n_band"].sum())
        r["labels_only_power_check"] = power_check(pc)
        res["segments"][short] = r
        print(short, {k: r[k] for k in ("ink_z_mm_p99.5", "band_z_threshold_mm", "z_top_mm_max_valid",
                                        "band_area_mm2", "tiles_100pct_in_band", "tiles_ge50pct_in_band",
                                        "band_row_range_full_res", "band_overlap", "band_height_mm_per_column")},
              flush=True)
        pw = r["labels_only_power_check"]
        print("  power", {k: pw[k] for k in ("deep_profile_height_mm", "deep_band_shuffle_tiles", "windows_n",
                                             "windows_frac_p_le_0.01", "windows_median_p")}, pw["text_block_full"],
              flush=True)
    n_blank = sum(res["segments"][s]["tiles_100pct_in_band"] for s in SEGS)
    res["pooled_blank_tiles"] = n_blank
    res["metal_detector_evaluable"] = bool(n_blank >= 15)
    for p in (DEEP_JSON, DEEP_JSON_VAULT):
        p.write_text(json.dumps(res, indent=1))
    print("pooled blank tiles", n_blank, "MD evaluable", n_blank >= 15)


if __name__ == "__main__":
    main()
