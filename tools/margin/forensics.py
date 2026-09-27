"""Forensic margin check on PHerc.1667 w031 (the authors, the settled-rules file 2026-09-26: "go margin check THOROUGHLY").

Decides among H1-H6 for the text-like signal in the w031 upper-margin band. Scalars / profiles only:
NO images, no pixel arrays in internal/ or git. Pooled 0.25 mm maps and full-res row profiles live in
data/margin_forensics/ (gitignored) only.

Steps (one big file at a time in data/raw/, reduced, deleted):
    .venv/bin/python -m tools.margin.forensics prereg            # write + commit thresholds FIRST
    .venv/bin/python -m tools.margin.forensics static --seg w031 # labels/masks/band pooled (model-free)
    .venv/bin/python -m tools.margin.forensics preds --seg w031  # every pred incl. reverse, one at a time
    .venv/bin/python -m tools.margin.forensics render --seg w031 # max_22_42, no model
    .venv/bin/python -m tools.margin.forensics analyze           # checks 1-7 -> results/*.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

from tools.harness.fetch import RAW, download, is_reverse, list_remote, sha256
from tools.margin.common import (BAND_JSON, REPO, Lazy2D, full_to_grid, meta, seg, voxel_um)
from tools.margin.reduce_pred import check_disk, eval_mask_files, label_components, periodogram_stat

OUT = REPO / "data" / "margin_forensics"
RES_JSON = REPO / "vault" / "results" / "margin_forensics_w031.json"
CELL = 104                    # pooled cell, px (0.2495 mm at 2.399 um)
STRIP_F = CELL * 20           # 2080-row streaming strips (whole cells)
SEED0 = 20260925
N_SURR = 200
P0_MM = 6.42
# Lower control fired the text-like rule on the first pred (2026-09-26): new-region trigger. That line of work
# is STOPPED: no lower-region metrics are computed for any further file, and lower-control cells/rows are
# blanked in every stored array. Only the first pred's lower scalars exist (recorded in the results JSON).
LOWER_STOPPED = True

PREREG = {
    "written_utc": None,
    "status": "PRE-REGISTERED before any check in this run is computed (checks 1-6). Only inputs read so far: "
              "labels/masks/tifxyz (as in band.py) and the already-committed first-pred reduction (the stop).",
    "constants": {
        "P0_mm": P0_MM,
        "P0_source": "label line pitch quoted in the hard-stop record. Label-only re-estimates in this run are "
                     "unstable (2D spectral peak 6.39 mm in x-strips 4-6; whole-profile ACF peak 6.88 mm; the "
                     "main labeled column 10-12 has its 2D peak along x), so P0 is fixed at 6.42 mm and every "
                     "phase result is repeated at 0.95*P0 and 1.05*P0 (sensitivity).",
        "pooled_cell_px": CELL, "pooled_cell_mm": CELL * 2.399e-3, "shuffle_tile_cells": 4,
        "shuffle_tile_mm": 4 * CELL * 2.399e-3,
        "surrogates": N_SURR, "surrogate_seeds": f"{SEED0}+k, k=0..{N_SURR - 1}",
        "p_value": "(1 + #{surrogate stat >= observed}) / (1 + 200); alpha = 0.01",
        "threshold": "p > 0.5 (uint8 > 127; float > 0.5), as the frozen v0-M plan",
        "text_like_rule": "unchanged v0-M: band FP >= 0.1% AND (T1 ratio >= 10 OR T2 flag)",
    },
    "null_models": {
        "tile_shuffle": "1 mm tiles (4x4 pooled cells) that lie wholly in the region are permuted among "
                        "themselves (internal layout kept); partial tiles stay. Null for periodicity STRENGTH "
                        "(T1 ratio on the pooled row profile; power at P0).",
        "phase_randomized": "row profile detrended, FFT phases randomized (DC/Nyquist kept). Preserves the power "
                            "spectrum exactly, so it is NOT a null for strength; it is the null for PHASE/alignment "
                            "statistics: cross-model phase agreement, render-vs-model alignment, phase continuity "
                            "across the labeled edge.",
    },
    "phase_definition": "C = sum_r (y_r - mean) exp(-2 pi i x_r / P0), x_r = image row in mm (absolute, row 0 = 0); "
                        "line-centre phase phi = -arg C (profile peak at x = phi P0 / 2pi mod P0).",
    "regions": {
        "band": "frozen v0-M band (data/margin_bands.json, commit 98c4339)",
        "text_block": "evaluated mask (val_v2 | sup_v2 | sup_v1), as v0-M",
        "lower_control": "valid surface, z < (0.5th pct labeled-ink z) - 2 mm, >= 2 mm from segment edge, "
                         "supervision = 0 (mirror of the band rule at the bottom). Control region: text-like = "
                         "NEW-REGION trigger, record scalars and stop that line.",
        "edge": "labeled-ink edge = 99.5th pct of labeled-ink z (global, 67.94 mm); local edge per 5 mm x-strip "
                "= 99.5th pct of labeled-ink z in that strip (strips with >= 10k ink px).",
    },
    "checks": {
        "C1_all_models": {
            "registration": "AUROC >= 0.75 on text block, else model excluded from consensus/phase stats",
            "fires": "text_like_rule",
            "periodic_sig": "tile-shuffle p <= 0.01 for the pooled band T1 ratio",
            "shared_phase": "circular resultant R of phase at P0 over registered models with periodic_sig: "
                            "R >= 0.8 AND phase-randomized p <= 0.01 -> 'same physical line positions'; "
                            "R < 0.5 or p > 0.05 -> 'model-specific' (supports H1a, per-model hallucination)",
            "map_corr": "median pairwise Pearson r of 0.25 mm band maps (cells wholly in band) reported next to "
                        "the same on unlabeled text-block cells (descriptive)",
            "receptive_field_note": "model patches 48 to 640 px (0.12 to 1.54 mm) are smaller than P0; a phase "
                                    "shared across models at P0 must come from the input, not from per-model "
                                    "letterform memory",
        },
        "C2_reverse": {
            "H4_support": "reverse periodic_sig AND (reverse band T1 ratio > max forward ratio OR |phase(reverse) - "
                          "forward consensus phase| >= pi/3)",
            "H4_no_support": "reverse not periodic_sig, OR same phase (<= pi/4) with ratio <= forward median",
            "caveat": "the reverse pred is named differently and is likely a different model (scout); weight low",
        },
        "C3_render": {
            "valid": "render > 0 on >= 95% of band px; band mean within the [p1, p99] of 1 mm text-block cell "
                     "means; band/text-block within-cell std ratio in [0.5, 2]. Failing = H6 support",
            "sign": "sign of the band-passed (2.5-8 mm) correlation of render and label-ink text-block row "
                    "profiles; |r| < 0.2 -> sign undetermined, phases compared modulo pi",
            "sensitive": "render text-block row profile: T1 ratio >= 10 AND tile-shuffle p <= 0.01. If not "
                         "sensitive, a null render result in the band cannot contradict physical hypotheses",
            "render_periodic_band": "band T1 ratio >= 10 AND tile-shuffle p <= 0.01",
            "render_matches_models": "band-passed correlation (sign-calibrated) of render and model-consensus band "
                                     "profiles >= 0.5 AND phase-randomized p <= 0.01 AND |dphi at P0| <= pi/4",
        },
        "C4_layout": {
            "near_far_ratio": "model-consensus FP fraction in cells [2, 5) mm above the global edge / in cells "
                              ">= 10 mm above it (edge distance >= 2 mm)",
            "lattice_continuity": "|phase(top 10 mm of text block: [-10, 0) mm) - phase(first 10 mm of band: "
                                  "[2, 12) mm)| <= pi/4 for the model consensus, phase-randomized p <= 0.01 "
                                  "(null: band window profile randomized), same verdict at 0.95/1.05 P0",
            "text_block_lattice_ok": "labels and model consensus agree in [-10, 0) mm: |dphi| <= pi/4; else "
                                     "continuity is 'undetermined'",
            "H2_support": "lattice_continuity holds (H2-partial if near_far >= 2, H2-full if near_far < 2 and "
                          "the band's far part [10, top) is itself periodic_sig)",
            "H2_contradict": "text_block_lattice_ok AND |dphi| >= pi/2 AND near_far < 1.5",
            "spread_evenly": "near_far in [0.67, 1.5] -> consistent with H1/H3/H6 (not H2-partial)",
            "H3_localized": "top 20% of band x-strips (5 mm, by band px) hold >= 50% of band FP px AND "
                            "no lattice_continuity",
        },
        "C5_geometry": {
            "radius": "per tifxyz row (z-slice) algebraic circle fit to (x, y) -> centre, 5 mm running median; "
                      "r = distance to that centre",
            "wrap_spacing_s": "slope of radius vs wrap number over w018/w023/w029/w031 at matched z (30-65 mm) and "
                              "angle (5 deg bins), assuming the 'wNNN' name is the wrap number (checked: radius "
                              "must increase monotonically with it)",
            "edge_jump": "per column: [median r over +2..+7 mm] - [median r over -7..-2 mm] minus the text-block "
                         "linear trend of r(z) for that column; baseline = same statistic for 9 mm-apart window "
                         "pairs inside the text block",
            "H5_support": "median |edge_jump| >= 0.5 s AND > 95th pct of baseline; OR |change in r31 - r29 "
                          "between band z and text-block z| >= 0.5 s",
            "H5_contradict": "median |edge_jump| < 0.2 s AND |change in r31 - r29| < 0.2 s",
            "H6_support": "band median area-stretch differs from text block by > 10%, OR FP fraction in band "
                          "cells 2-4 mm from segment edge >= 2x that in cells >= 6 mm, OR render invalid",
            "H4_plausible": "s <= 0.26 mm (Dreamskid reading window) -> plausible; s >= 0.6 mm -> implausible "
                            "for mean spacing (local contacts still possible)",
        },
        "C6_controls": {
            "lower": "same metrics; text-like (v0-M rule) = new-region trigger -> scalars only, stop that line",
            "w029": "band with merged_confidence_vote_betti_ema_640_forward first; if it fires (v0-M rule) -> "
                    "text-rule trigger for a new region: record scalars, do NOT pull the w029 render, stop w029",
        },
        "C7_scoring": "each H: supported / contradicted / undetermined from the rules above; overall call = "
                      "hypotheses supported and not contradicted; confidence stated with the weakest link",
    },
    "deviations": [],
}


# ------------------------------------------------------------------------------------------ utils
def seg_dir(short):
    d = OUT / short
    d.mkdir(parents=True, exist_ok=True)
    return d


def pooled_shape(H, W):
    return H // CELL, W // CELL


def pool_sum(a, nr, nc):
    """Sum over CELL x CELL blocks of a (rows multiple of CELL), trailing cols dropped."""
    return a[:, :nc * CELL].reshape(a.shape[0] // CELL, CELL, nc, CELL).sum(axis=(1, 3), dtype=np.float64)


def band_grid(short):
    bj = json.loads(BAND_JSON.read_text())["segments"][short]
    with np.load(REPO / bj["band_grid_npz"]) as z:
        return bj, z["band_grid"]


def lower_grid(short):
    """Mirror of band.py at the bottom (pre-registered). Returns (dict, bool grid)."""
    from tools.margin.band import weighted_pct
    from tools.margin.common import pool_count_to_grid, grid_factors
    s = seg(short)
    m = meta(short)
    vox = voxel_um(m)
    z = np.squeeze(tifffile.imread(s.small / "z.tif")).astype(np.float64)
    GH, GW = z.shape
    valid = z > 0
    zmm = z * vox / 1000
    lab = Lazy2D(s.small / f"{s.seg_id}_inklabels_v2.tif")
    sy, sx = grid_factors(m, lab.shape, (GH, GW))
    cell_mm = vox / 1000.0 * 0.5 * (1 / sy + 1 / sx)
    ink = pool_count_to_grid(lab, sy, sx, (GH, GW)); lab.close()
    sup = np.zeros((GH, GW), np.int64)
    for nm in ("supervision_mask_v2", "supervision_mask"):
        f = s.small / f"{s.seg_id}_{nm}.tif"
        if f.exists():
            lz = Lazy2D(f); sup += pool_count_to_grid(lz, sy, sx, (GH, GW)); lz.close()
    iw = (ink > 0) & valid
    z_lo = weighted_pct(zmm[iw], ink[iw], 0.5)
    edge = ndi.distance_transform_edt(np.pad(valid, 1))[1:-1, 1:-1] * cell_mm
    low = valid & (zmm < z_lo - 2.0) & (edge >= 2.0) & (sup == 0)
    return {"ink_z_p0.5_mm": z_lo, "lower_z_threshold_mm": z_lo - 2.0, "lower_grid_cells": int(low.sum()),
            "overlap_ink_cells": int((ink[low] > 0).sum())}, low, edge, zmm, valid


# ------------------------------------------------------------------------------------------ static
def do_static(short):
    s = seg(short)
    m = meta(short)
    vox = voxel_um(m)
    bj, bgrid = band_grid(short)
    linfo, lgrid, edge, zmm, valid = lower_grid(short)
    lab = Lazy2D(s.small / f"{s.seg_id}_inklabels_v2.tif")
    masks = [Lazy2D(f) for f in eval_mask_files(s)]
    H, W = lab.shape
    GH, GW = bgrid.shape
    sy, sx = bj["grid_factor_sy_sx"]
    gy, gx = full_to_grid(H, sy, GH), full_to_grid(W, sx, GW)
    nr, nc = pooled_shape(H, W)
    A = {k: np.zeros((nr, nc)) for k in ("n_band", "n_low", "n_ev", "n_ink", "n_evink")}
    rows = {k: np.zeros(H, np.int64) for k in ("band_n", "low_n", "ev_n", "evink_n", "ink_n")}
    for r0 in range(0, H, STRIP_F):
        r1 = min(H, r0 + STRIP_F)
        ink = lab.rows(r0, r1) > 0
        ev = masks[0].rows(r0, r1) > 0
        for mm in masks[1:]:
            ev |= mm.rows(r0, r1) > 0
        bnd = bgrid[gy[r0:r1]][:, gx]
        low = lgrid[gy[r0:r1]][:, gx]
        rows["band_n"][r0:r1] = bnd.sum(1); rows["low_n"][r0:r1] = low.sum(1)
        rows["ev_n"][r0:r1] = ev.sum(1); rows["evink_n"][r0:r1] = (ev & ink).sum(1)
        rows["ink_n"][r0:r1] = ink.sum(1)
        k = (r1 - r0) // CELL
        if k:
            rr = slice(0, k * CELL); i0 = r0 // CELL
            for key, arr in (("n_band", bnd), ("n_low", low), ("n_ev", ev), ("n_ink", ink), ("n_evink", ev & ink)):
                A[key][i0:i0 + k] = pool_sum(arr[rr], k, nc)
        del ink, ev, bnd, low
    lab.close()
    for x in masks:
        x.close()
    # cell-centre geometry
    cr = full_to_grid(H, sy, GH)[np.arange(nr) * CELL + CELL // 2]
    cc = full_to_grid(W, sx, GW)[np.arange(nc) * CELL + CELL // 2]
    np.savez_compressed(seg_dir(short) / "static.npz", **A, **{f"row_{k}": v for k, v in rows.items()},
                        z_mm=zmm[np.ix_(cr, cc)], edge_mm=edge[np.ix_(cr, cc)], valid=valid[np.ix_(cr, cc)])
    info = {"segment": short, "H": H, "W": W, "pooled_shape": [nr, nc], "voxel_um": vox,
            "band_px": int(rows["band_n"].sum()), "lower": linfo, "lower_px": int(rows["low_n"].sum()),
            "lower_row_range": [int(np.flatnonzero(rows["low_n"]).min()), int(np.flatnonzero(rows["low_n"]).max()) + 1]}
    (seg_dir(short) / "static.json").write_text(json.dumps(info, indent=1))
    np.savez_compressed(seg_dir(short) / "lower_grid.npz", lower_grid=lgrid)
    print(json.dumps(info), flush=True)


# ------------------------------------------------------------------------------------------ preds
def region_rows(rowcount):
    nz = np.flatnonzero(rowcount)
    return int(nz.min()), int(nz.max()) + 1


def blob_stats(hi, r_lo, c_lo, vox):
    px_mm2 = (vox / 1000) ** 2
    comps = label_components(hi, px_mm2)
    area = comps["area_px"] * px_mm2
    letter = (area >= 1.0) & (area <= 4.0)
    cr = comps["cent_r"][letter] + r_lo
    cc = comps["cent_c"][letter] + c_lo
    one = 1000.0 / vox
    srt = np.sort(cr)
    mx = max([int(np.sum(np.abs(srt - v) <= one)) for v in srt], default=0)
    return {"n_components": int(len(area)), "letter_blobs": int(letter.sum()),
            "max_letter_blobs_in_one_row": mx, "T2_flag": bool(letter.sum() >= 5 and mx >= 3),
            "largest_mm2": float(area.max()) if len(area) else 0.0,
            "letter_cent_row": cr.tolist(), "letter_cent_col": cc.tolist(), "letter_area_mm2": area[letter].tolist()}


def reduce_file(short, raw, rel, kind):
    """kind = 'pred' or 'render'."""
    s = seg(short)
    vox = voxel_um(meta(short))
    bj, bgrid = band_grid(short)
    with np.load(seg_dir(short) / "lower_grid.npz") as z:
        lgrid = z["lower_grid"]
    st = np.load(seg_dir(short) / "static.npz")
    P = Lazy2D(raw)
    lab = Lazy2D(s.small / f"{s.seg_id}_inklabels_v2.tif")
    masks = [Lazy2D(f) for f in eval_mask_files(s)]
    H, W = lab.shape
    if tuple(P.shape) != (H, W):
        P.close(); lab.close(); [x.close() for x in masks]
        return {"pred": rel, "excluded": f"shape {P.shape} != {(H, W)}", "file_info": P.info}, None
    dt = np.dtype(P.dtype)
    if kind == "pred":
        if dt == np.uint8:
            thr, scale, nb = 127, 255.0, 256
        elif dt.kind == "f":
            thr, scale, nb = 0.5, 1.0, 1001
        else:
            raise SystemExit(f"unexpected pred dtype {dt}")
    else:
        thr, scale, nb = None, 1.0, (256 if dt == np.uint8 else 65536)
    GH, GW = bgrid.shape
    sy, sx = bj["grid_factor_sy_sx"]
    gy, gx = full_to_grid(H, sy, GH), full_to_grid(W, sx, GW)
    nr, nc = pooled_shape(H, W)
    cellkeys = ["val_all", "sq_all", "nvalid_all", "val_band", "val_low", "val_noink", "nvalid_band"]
    if kind == "pred":
        cellkeys += ["hi_all", "hi_band", "hi_low", "hi_noink"]
    C = {k: np.zeros((nr, nc)) for k in cellkeys}
    R = {k: np.zeros(H) for k in ("band_val", "band_hi", "band_nvalid", "ev_val", "ev_hi", "evneg_hi", "evneg_n",
                                   "evneg_val", "low_val", "low_hi", "all_hi", "all_val", "all_nvalid")}
    h_ink = np.zeros(nb, np.int64); h_neg = np.zeros(nb, np.int64)
    h_band = np.zeros(nb, np.int64)
    b_r0, b_r1 = bj["band_row_range_full_res"]; b_c0, b_c1 = bj["band_col_range_full_res"]
    l_r0, l_r1 = region_rows(st["row_low_n"])
    lc = np.flatnonzero(lgrid.any(0)); l_c0 = int(np.searchsorted(gx, lc.min(), "left")); l_c1 = int(np.searchsorted(gx, lc.max(), "right"))
    if LOWER_STOPPED:          # do not compute anything on the lower control any more
        lgrid = np.zeros_like(lgrid)
    band_hi = np.zeros((b_r1 - b_r0, b_c1 - b_c0), bool) if kind == "pred" else None
    low_hi = np.zeros((l_r1 - l_r0, l_c1 - l_c0), bool) if (kind == "pred" and not LOWER_STOPPED) else None
    vmin, vmax = np.inf, -np.inf
    for r0 in range(0, H, STRIP_F):
        r1 = min(H, r0 + STRIP_F)
        a = P.rows(r0, r1)
        vmin, vmax = min(vmin, float(a.min())), max(vmax, float(a.max()))
        v = a.astype(np.float32) / np.float32(scale)
        valid = a > 0 if kind == "render" else np.ones(a.shape, bool)
        ink = lab.rows(r0, r1) > 0
        ev = masks[0].rows(r0, r1) > 0
        for mm in masks[1:]:
            ev |= mm.rows(r0, r1) > 0
        bnd = bgrid[gy[r0:r1]][:, gx]
        low = lgrid[gy[r0:r1]][:, gx]
        q = a if dt.kind != "f" else np.rint(a * 1000).astype(np.int64)
        h_ink += np.bincount(q[ink & ev].ravel(), minlength=nb)[:nb]
        h_neg += np.bincount(q[~ink & ev].ravel(), minlength=nb)[:nb]
        h_band += np.bincount(q[bnd].ravel(), minlength=nb)[:nb]
        vb = v * bnd
        R["band_val"][r0:r1] = vb.sum(1); R["band_nvalid"][r0:r1] = (valid & bnd).sum(1)
        R["ev_val"][r0:r1] = (v * ev).sum(1)
        neg = ev & ~ink
        R["evneg_n"][r0:r1] = (neg & valid).sum(1); R["evneg_val"][r0:r1] = (v * neg).sum(1)
        R["low_val"][r0:r1] = (v * low).sum(1)
        R["all_val"][r0:r1] = v.sum(1); R["all_nvalid"][r0:r1] = valid.sum(1)
        if kind == "pred":
            hi = a > thr
            R["band_hi"][r0:r1] = (hi & bnd).sum(1); R["ev_hi"][r0:r1] = (hi & ev).sum(1)
            R["evneg_hi"][r0:r1] = (hi & neg).sum(1); R["low_hi"][r0:r1] = (hi & low).sum(1)
            R["all_hi"][r0:r1] = hi.sum(1)
            o0, o1 = max(r0, b_r0), min(r1, b_r1)
            if o0 < o1:
                band_hi[o0 - b_r0:o1 - b_r0] = (hi & bnd)[o0 - r0:o1 - r0, b_c0:b_c1]
            o0, o1 = max(r0, l_r0), min(r1, l_r1)
            if o0 < o1 and low_hi is not None:
                low_hi[o0 - l_r0:o1 - l_r0] = (hi & low)[o0 - r0:o1 - r0, l_c0:l_c1]
        k = (r1 - r0) // CELL
        if k:
            rr = slice(0, k * CELL); i0 = r0 // CELL
            C["val_all"][i0:i0 + k] = pool_sum(v[rr], k, nc)
            C["sq_all"][i0:i0 + k] = pool_sum((v * v)[rr], k, nc)
            C["nvalid_all"][i0:i0 + k] = pool_sum(valid[rr], k, nc)
            C["nvalid_band"][i0:i0 + k] = pool_sum((valid & bnd)[rr], k, nc)
            C["val_band"][i0:i0 + k] = pool_sum(vb[rr], k, nc)
            C["val_low"][i0:i0 + k] = pool_sum((v * low)[rr], k, nc)
            C["val_noink"][i0:i0 + k] = pool_sum((v * ~ink)[rr], k, nc)
            if kind == "pred":
                C["hi_all"][i0:i0 + k] = pool_sum(hi[rr], k, nc)
                C["hi_band"][i0:i0 + k] = pool_sum((hi & bnd)[rr], k, nc)
                C["hi_low"][i0:i0 + k] = pool_sum((hi & low)[rr], k, nc)
                C["hi_noink"][i0:i0 + k] = pool_sum((hi & ~ink)[rr], k, nc)
        del a, v, valid, ink, ev, bnd, low, q, vb, neg
    P.close(); lab.close(); [x.close() for x in masks]
    cn = np.cumsum(h_neg) - h_neg
    npos, nneg = h_ink.sum(), h_neg.sum()
    auc = float((h_ink * (cn + 0.5 * h_neg)).sum() / (npos * nneg)) if npos and nneg else float("nan")
    out = {"file": rel, "kind": kind, "file_info": P.info, "dtype": str(dt), "value_min": vmin, "value_max": vmax,
           "registration_auroc": auc}
    if kind == "pred":
        hb = np.arange(nb) > (thr * (1000 if dt.kind == "f" else 1))
        out["recall_p05"] = float(h_ink[hb].sum() / npos)
        out["fpr_p05_text_unlabeled"] = float(h_neg[hb].sum() / nneg)
        out["band_fp_frac"] = float(R["band_hi"].sum() / st["row_band_n"].sum())
        out["band_blobs"] = blob_stats(band_hi, b_r0, b_c0, vox)
        if not LOWER_STOPPED:
            out["lower_fp_frac"] = float(R["low_hi"].sum() / max(st["row_low_n"].sum(), 1))
            out["lower_blobs"] = blob_stats(low_hi, l_r0, l_c0, vox)
        del band_hi, low_hi
    if LOWER_STOPPED:          # blank the lower control everywhere (new-region stop)
        lowcell = st["n_low"] > 0
        for k in list(C):
            C[k][lowcell] = np.nan
        for k in ("low_val", "low_hi"):
            R.pop(k)
        lr0, lr1 = region_rows(st["row_low_n"])
        for k in ("all_hi", "all_val", "all_nvalid"):
            R[k][lr0:lr1] = np.nan
        low_hi = None
    arrays = {**{f"cell_{k}": v.astype(np.float32) for k, v in C.items()}, **{f"row_{k}": v for k, v in R.items()},
              "h_ink": h_ink, "h_neg": h_neg, "h_band": h_band}
    return out, arrays


def pull_reduce(short, rel, kind, remote):
    name = Path(rel).name
    od = seg_dir(short)
    if (od / f"{name}.json").exists():
        print("  skip", name, flush=True)
        return json.loads((od / f"{name}.json").read_text())
    free = check_disk()
    raw = RAW / name
    t0 = time.time()
    print(f"  pull {name} ({remote[rel].size / 2**30:.2f} GiB), free {free:.1f} GB", flush=True)
    for attempt in range(4):
        try:
            if not (raw.exists() and raw.stat().st_size == remote[rel].size):
                download(remote[rel], raw)
            break
        except SystemExit:
            raise
        except Exception as e:          # network
            print("   download retry", attempt, e, flush=True)
            if raw.exists():
                raw.unlink()
            time.sleep(2 ** (attempt + 1))
    digest = sha256(raw)
    try:
        out, arrays = reduce_file(short, raw, rel, kind)
    finally:
        raw.unlink()
    out["sha256"] = digest
    out["runtime_s"] = round(time.time() - t0, 1)
    if arrays is not None:
        np.savez_compressed(od / f"{name}.npz", **arrays)
    (od / f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
    if kind == "pred" and "band_blobs" in out:
        print(f"   AUROC {out['registration_auroc']:.3f} bandFP {out['band_fp_frac']:.3%} blobs {out['band_blobs']['letter_blobs']} "
              f"row {out['band_blobs']['max_letter_blobs_in_one_row']} ({out['runtime_s']} s)", flush=True)
    else:
        print(f"   {out.get('dtype')} [{out.get('value_min')},{out.get('value_max')}] ({out['runtime_s']} s)", flush=True)
    return out


def do_preds(short, only=None):
    s = seg(short)
    remote = list_remote(s)
    preds = sorted(k for k in remote if k.startswith("preds/"))
    if only:
        preds = [p for p in preds if only in p]
    # forward first, reverse last
    preds = [p for p in preds if not is_reverse(p)] + [p for p in preds if is_reverse(p)]
    print(f"{short}: {len(preds)} preds", flush=True)
    for rel in preds:
        pull_reduce(short, rel, "pred", remote)


def do_render(short):
    s = seg(short)
    remote = list_remote(s)
    pull_reduce(short, s.max_render, "render", remote)


def do_prereg():
    res = json.loads(RES_JSON.read_text()) if RES_JSON.exists() else {}
    if "preregistration" in res:
        raise SystemExit("prereg already written; log changes as deviations")
    p = dict(PREREG)
    p["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    RES_JSON.write_text(json.dumps({"preregistration": p}, indent=1))
    print("wrote", RES_JSON)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prereg", "static", "preds", "render", "analyze"])
    ap.add_argument("--seg", default="w031")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    if a.step == "prereg":
        do_prereg()
    elif a.step == "static":
        do_static(a.seg)
    elif a.step == "preds":
        do_preds(a.seg, a.only)
    elif a.step == "render":
        do_render(a.seg)
    elif a.step == "analyze":
        from tools.margin.forensics_analyze import main as amain
        amain()
