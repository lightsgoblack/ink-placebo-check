"""Amendment D, step 2: Lie Detector v0-M on the deep-margin band (w023, w018). Scalars only, no images.

    .venv/bin/python -m tools.margin.deep_ld reduce --seg w023 --plant   # one pred at a time, raw deleted
    .venv/bin/python -m tools.margin.deep_ld reduce --seg w018
    .venv/bin/python -m tools.margin.deep_ld analyze                     # ratios, Spearman, sanity, verdict

Per pred (streamed in 521-row strips = 1/4 tile row, never loaded whole):
  * registration AUROC / recall / in-mask unlabeled FPR from value histograms (as reduce_pred);
  * per 5 mm tile: sum of pred value (validity check), deep-band px, deep-band FP px, label / mask px;
  * deep-band crop of (p > 0.5) at full res (crop aligned to 416 px = 1 mm shuffle tiles):
      - pooled 104 px cells -> recalibrated periodicity flag (forensics_analyze.periodicity, 200 surrogates);
      - letter blobs (1-4 mm2, 8-conn) -> density, S = max centroids in a 1 mm row window, and when S >= 3
        the full-res 1 mm tile-shuffle null for S (200 surrogates);
  * planted-blob sanity on the first w023 pred.
A recalibrated flag firing = HARD STOP: the scalars so far are written, the run exits with code 3.
Plan: results/deep_bands.json["analysis_plan_fixed_before_reading_preds"] (commit 2e7d835).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from tools.harness.fetch import RAW, download, is_reverse, list_remote, sha256
from tools.margin.common import REPO, TILE_PX, Lazy2D, full_to_grid, meta, seg, tile_grid, voxel_um
from tools.margin.deep_band import DEEP_JSON, POOL_OUT
from tools.margin.forensics import CELL
from tools.margin.forensics_analyze import FULL, TILE, Profile, periodicity
from tools.margin.reduce_pred import check_disk, eval_mask_files, label_components, plant_blobs

SEED = 20260925
N_SURR = 200
B = 2000
SUB = TILE_PX // 4          # 521-row strips
SH = CELL * TILE            # 416 px = 1 mm shuffle tile
STOP_FILE = POOL_OUT / "HARD_STOP.json"
RES_JSON = REPO / "vault" / "results" / "lie_detector_v0m_deep.json"


class Pred:
    """Row access to a pred; nearest-neighbour upsampling if it is an exact integer downsample."""

    def __init__(self, path: Path, H: int, W: int):
        self.lz = Lazy2D(path)
        h, w = self.lz.shape
        self.f = 1
        self.note = None
        if (h, w) != (H, W):
            f = int(round(H / h))
            if f >= 2 and abs(h * f - H) <= f and abs(w * f - W) <= f and int(round(W / w)) == f:
                self.f = f
                self.note = f"shape {h}x{w} = labels / {f}; nearest-neighbour upsampled x{f}"
            else:
                self.note = f"shape {h}x{w} not an integer downsample of labels {H}x{W}"
        self.H, self.W = H, W

    def rows(self, r0, r1):
        if self.f == 1:
            return self.lz.rows(r0, r1)
        f = self.f
        a0, a1 = r0 // f, min(self.lz.shape[0], -(-r1 // f))
        a = self.lz.rows(a0, a1)
        a = np.repeat(np.repeat(a, f, axis=0), f, axis=1)
        a = a[r0 - a0 * f:r0 - a0 * f + (r1 - r0), :self.W]
        if a.shape[0] < r1 - r0 or a.shape[1] < self.W:      # edge padding (last partial px)
            a = np.pad(a, ((0, r1 - r0 - a.shape[0]), (0, self.W - a.shape[1])), mode="edge")
        return a


def load_deep(short):
    bj = json.loads(DEEP_JSON.read_text())["segments"][short]
    with np.load(REPO / bj["band_grid_npz"]) as z:
        return bj, z["band_grid"]


def row_window_max(cent_rows: np.ndarray, win_px: float) -> int:
    """Max number of centroids in any half-open row window [r, r + win)."""
    c = np.sort(np.asarray(cent_rows, float))
    if not len(c):
        return 0
    return int(np.max(np.searchsorted(c, c + win_px, side="left") - np.arange(len(c))))


def letter_S(hi: np.ndarray, vox: float) -> tuple[int, int, np.ndarray]:
    px_mm2 = (vox / 1000) ** 2
    comps = label_components(hi, px_mm2)
    area = comps["area_px"] * px_mm2
    let = (area >= 1.0) & (area <= 4.0)
    cr = comps["cent_r"][let]
    return row_window_max(cr, 1000.0 / vox), int(let.sum()), area


def shuffle_tiles(bcells: np.ndarray) -> list:
    """1 mm tiles (4x4 cells) wholly in band, crop-local cell coords (crop origin is a multiple of 4 cells)."""
    full = bcells >= FULL - 0.5
    nr, nc = full.shape
    return [(i, j) for i in range(0, nr - nr % TILE, TILE) for j in range(0, nc - nc % TILE, TILE)
            if full[i:i + TILE, j:j + TILE].all()]


def blob_null(hi: np.ndarray, tiles: list, vox: float) -> list:
    out = []
    for k in range(N_SURR):
        rng = np.random.default_rng(SEED + k)
        perm = rng.permutation(len(tiles))
        blocks = [hi[i * CELL:i * CELL + SH, j * CELL:j * CELL + SH].copy() for i, j in tiles]
        h2 = hi.copy()
        for (i, j), p in zip(tiles, perm):
            h2[i * CELL:i * CELL + SH, j * CELL:j * CELL + SH] = blocks[p]
        del blocks
        out.append(letter_S(h2, vox)[0])
        del h2
    return out


def reduce_one(short, raw, rel, grid, bj, plant):
    s = seg(short)
    vox = voxel_um(meta(short))
    px_mm2 = (vox / 1000) ** 2
    lab = Lazy2D(s.small / f"{s.seg_id}_inklabels_v2.tif")
    H, W = lab.shape
    P = Pred(raw, H, W)
    info = dict(P.lz.info)
    if P.f == 1 and P.note:
        P.lz.close(); lab.close()
        return {"pred": rel, "file_info": info, "excluded": P.note}, None
    masks = [Lazy2D(f) for f in eval_mask_files(s)]
    dt = np.dtype(P.lz.dtype)
    if dt == np.uint8:
        thr, scale, nb, qf = 127, 255.0, 256, 1
    elif dt.kind == "f":
        thr, scale, nb, qf = 0.5, 1.0, 1001, 1000
    elif dt == np.uint16:
        thr, scale, nb, qf = 32767, 65535.0, 65536, 1
    else:
        raise SystemExit(f"unexpected pred dtype {dt}")
    GH, GW = grid.shape
    sy, sx = bj["grid_factor_sy_sx"]
    gy, gx = full_to_grid(H, sy, GH), full_to_grid(W, sx, GW)
    ty, tx = tile_grid((H, W))
    tw = tx * TILE_PX
    r_lo, r_hi = bj["band_row_range_full_res"]
    c_lo, c_hi = bj["band_col_range_full_res"]
    R0, C0 = (r_lo // SH) * SH, (c_lo // SH) * SH
    R1, C1 = min(H, -(-r_hi // SH) * SH), min(W, -(-c_hi // SH) * SH)
    T = {k: np.zeros((ty, tx)) for k in ("sum_val", "band_n", "band_fp", "lab_n", "ev_n")}
    h_ink = np.zeros(nb, np.int64); h_neg = np.zeros(nb, np.int64)
    vmin, vmax = np.inf, -np.inf
    hi_c = np.zeros((R1 - R0, C1 - C0), bool)
    for r0 in range(0, H, SUB):
        r1 = min(H, r0 + SUB)
        a = P.rows(r0, r1)
        vmin, vmax = min(vmin, float(a.min())), max(vmax, float(a.max()))
        hi = a > thr
        ink = lab.rows(r0, r1) > 0
        ev = masks[0].rows(r0, r1) > 0
        for mm in masks[1:]:
            ev |= mm.rows(r0, r1) > 0
        q = np.rint(a * qf).astype(np.int64) if dt.kind == "f" else a
        h_ink += np.bincount(q[ink & ev].ravel(), minlength=nb)[:nb]
        h_neg += np.bincount(q[ev & ~ink].ravel(), minlength=nb)[:nb]
        del q
        bnd = grid[gy[r0:r1]][:, gx]
        i = r0 // TILE_PX
        if i < ty:
            n = r1 - r0
            T["sum_val"][i] += a[:, :tw].reshape(n, tx, TILE_PX).sum(axis=(0, 2), dtype=np.float64) / scale
            T["band_n"][i] += bnd[:, :tw].reshape(n, tx, TILE_PX).sum(axis=(0, 2))
            T["band_fp"][i] += (hi & bnd)[:, :tw].reshape(n, tx, TILE_PX).sum(axis=(0, 2))
            T["lab_n"][i] += ink[:, :tw].reshape(n, tx, TILE_PX).sum(axis=(0, 2))
            T["ev_n"][i] += ev[:, :tw].reshape(n, tx, TILE_PX).sum(axis=(0, 2))
        o0, o1 = max(r0, R0), min(r1, R1)
        if o0 < o1:
            hi_c[o0 - R0:o1 - R0] = (hi & bnd)[o0 - r0:o1 - r0, C0:C1]
        del a, hi, ink, ev, bnd
    P.lz.close(); lab.close(); [m.close() for m in masks]

    # ---- band scalars
    band_c = grid[gy[R0:R1]][:, gx[C0:C1]]
    band_px = int(band_c.sum()); fp_px = int(hi_c.sum())
    nr, nc = (R1 - R0) // CELL, (C1 - C0) // CELL
    def ps(x):
        return x[:nr * CELL, :nc * CELL].reshape(nr, CELL, nc, CELL).sum(axis=(1, 3), dtype=np.float64)
    bcells, hcells = ps(band_c), ps(hi_c)
    del band_c
    per = periodicity(Profile(hcells, bcells), rel)
    S, n_letter, area = letter_S(hi_c, vox)
    blob = {"n_components": int(len(area)), "letter_blobs": n_letter,
            "letter_blob_per_cm2": float(n_letter / (band_px * px_mm2 / 100.0)),
            "largest_component_mm2": float(area.max()) if len(area) else 0.0, "S_max_letter_blobs_1mm_row": S}
    tiles = shuffle_tiles(bcells)
    if S >= 3:
        nullS = blob_null(hi_c, tiles, vox)
        blob["null_S_p99"] = float(np.percentile(nullS, 99)); blob["null_S_median"] = float(np.median(nullS))
        blob["null_S_max"] = int(max(nullS))
        blob["flag"] = bool(S > blob["null_S_p99"])
    else:
        blob["null_S_p99"] = None
        blob["flag"] = False
        blob["null_note"] = "S < 3: flag cannot fire, null not computed (pre-registered)"
    # ---- registration
    cn = np.cumsum(h_neg) - h_neg
    npos, nneg = h_ink.sum(), h_neg.sum()
    auc = float((h_ink * (cn + 0.5 * h_neg)).sum() / (npos * nneg)) if npos and nneg else float("nan")
    hb = np.arange(nb) > thr * qf
    out = {"pred": rel, "file_info": info, "upsample_note": P.note, "upsample_factor": P.f, "dtype": str(dt),
           "value_min": vmin, "value_max": vmax, "p05_threshold_raw": thr,
           "band_px": band_px, "band_px_in_full_tiles": int(T["band_n"].sum()), "band_fp_px": fp_px, "band_fp_area_frac": fp_px / band_px,
           "blobs": blob, "periodicity": {k: per[k] for k in ("n_rows", "length_mm", "T1_ratio_pooled",
                                                               "T1_peak_period_mm", "p_tile_T1",
                                                               "tile_null_T1_median", "tile_null_T1_p95",
                                                               "shuffle_cells_frac", "mean")},
           "periodicity_flag": bool(per["p_tile_T1"] <= 0.01), "n_shuffle_tiles": len(tiles),
           "registration_auroc": auc, "registration_ok": bool(auc >= 0.75),
           "recall_p05_labeled_ink": float(h_ink[hb].sum() / npos),
           "fpr_p05_textblock_unlabeled": float(h_neg[hb].sum() / nneg)}
    out["text_like_flag"] = bool(out["periodicity_flag"] or blob["flag"])
    arrays = {**{f"tile_{k}": v for k, v in T.items()}, "h_ink": h_ink, "h_neg": h_neg,
              "hcells": hcells, "bcells": bcells, "crop_origin": np.array([R0, C0])}
    if plant and not out["text_like_flag"]:
        pl, arrays["tile_band_fp_planted"] = plant_blobs(hi_c, grid, bj, gy, gx, R0, C0, T["band_fp"].astype(np.int64),
                                                         fp_px, band_px, vox)
        out["planted"] = pl
    del hi_c
    return out, arrays


def do_reduce(short, plant):
    if STOP_FILE.exists():
        raise SystemExit(f"HARD STOP recorded ({STOP_FILE}); nothing more is computed")
    s = seg(short)
    bj, grid = load_deep(short)
    remote = list_remote(s)
    preds = sorted(k for k in remote if k.startswith("preds/") and not is_reverse(k))
    od = POOL_OUT / short
    od.mkdir(parents=True, exist_ok=True)
    print(f"{short}: {len(preds)} non-reverse preds", flush=True)
    for n, rel in enumerate(preds):
        name = Path(rel).name
        if (od / f"{name}.json").exists():
            print("  skip", name, flush=True); continue
        free = check_disk()
        raw = RAW / name
        t0 = time.time()
        print(f"  pull {name} ({remote[rel].size / 2**30:.2f} GiB), free {free:.1f} GB", flush=True)
        if not (raw.exists() and raw.stat().st_size == remote[rel].size):
            download(remote[rel], raw)
        digest = sha256(raw)
        td = time.time() - t0
        out, arrays = reduce_one(short, raw, rel, grid, bj, plant=(plant and n == 0))
        raw.unlink()
        out["sha256"] = digest
        out["download_s"] = round(td, 1); out["runtime_s"] = round(time.time() - t0, 1)
        if arrays is not None:
            np.savez_compressed(od / f"{name}.npz", **arrays)
        (od / f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
        if "excluded" in out:
            print("   EXCLUDED", out["excluded"], flush=True); continue
        b = out["blobs"]; p = out["periodicity"]
        print(f"   {out['dtype']} x{out['upsample_factor']} AUROC {out['registration_auroc']:.3f} "
              f"FP {out['band_fp_area_frac']:.3%} txtFPR {out['fpr_p05_textblock_unlabeled']:.3%} "
              f"letter {b['letter_blobs']} S {b['S_max_letter_blobs_1mm_row']} (null p99 {b['null_S_p99']}) "
              f"T1 {p['T1_ratio_pooled']:.1f} p {p['p_tile_T1']:.3f} flag {out['text_like_flag']} "
              f"({out['runtime_s']} s)", flush=True)
        if out["text_like_flag"]:
            STOP_FILE.write_text(json.dumps({"segment": short, "pred": rel, "utc": time.strftime("%FT%TZ", time.gmtime()),
                                             "periodicity": p, "blobs": b}, indent=1, default=float))
            print("HARD STOP: recalibrated text-like flag fired in the deep band.", flush=True)
            raise SystemExit(3)
    # consensus periodicity over registered preds
    used = [json.loads(f.read_text()) for f in sorted(od.glob("*.json"))]
    used = [u for u in used if "excluded" not in u and u["registration_ok"]]
    fr = []
    for u in used:
        with np.load(od / f"{Path(u['pred']).name}.npz") as z:
            fr.append(z["hcells"] / np.maximum(z["bcells"], 1)); bc = z["bcells"]
    cons = periodicity(Profile(np.mean(fr, 0) * bc, bc), "consensus")
    c = {k: cons[k] for k in ("n_rows", "T1_ratio_pooled", "T1_peak_period_mm", "p_tile_T1", "tile_null_T1_median")}
    c["flag"] = bool(cons["p_tile_T1"] <= 0.01); c["n_models"] = len(used)
    (od / "consensus.json").write_text(json.dumps(c, indent=1, default=float))
    print("  consensus", c, flush=True)
    if c["flag"]:
        STOP_FILE.write_text(json.dumps({"segment": short, "pred": "consensus", "periodicity": c}, indent=1, default=float))
        print("HARD STOP: consensus periodicity flag fired.", flush=True)
        raise SystemExit(3)


# ------------------------------------------------------------------------------------------ analysis
def model_key(rel: str) -> str:
    n = Path(rel).name
    return n.split("_w0")[0] if "_w0" in n else n.replace(".tif", "")


def boot_weights(n_units: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    W = np.zeros((B, n_units))
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, n_units, n_units), minlength=n_units)
    return W


def ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(b > 0, a / np.where(b > 0, b, 1), np.where(a > 0, np.inf, 1.0))
    return r


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def seg_analysis(short):
    bj = json.loads(DEEP_JSON.read_text())["segments"][short]
    od = POOL_OUT / short
    recs = [json.loads(f.read_text()) for f in sorted(od.glob("*.tif.json"))]
    models, excluded = [], []
    for r in recs:
        if "excluded" in r:
            excluded.append({"pred": r["pred"], "why": r["excluded"]})
        elif not r["registration_ok"]:
            excluded.append({"pred": r["pred"], "why": f"registration AUROC {r['registration_auroc']:.3f} < 0.75"})
        else:
            models.append(r)
    units = [tuple(t) for t in bj["tiles_ge50pct_ij"]]
    ii = np.array([u[0] for u in units]); jj = np.array([u[1] for u in units])
    fp = []
    for r in models:
        with np.load(od / f"{Path(r['pred']).name}.npz") as z:
            fp.append(z["tile_band_fp"][ii, jj]); bn = z["tile_band_n"][ii, jj]
    fp = np.array(fp)                       # (models, units)
    Wb = boot_weights(len(units))
    S0 = fp.sum(1); Sb = Wb @ fp.T          # (B, models)
    pairs = []
    for a in range(len(models)):
        for b in range(a + 1, len(models)):
            hi_, lo_ = (a, b) if S0[a] >= S0[b] else (b, a)
            pt = float(ratio(np.array(S0[hi_]), np.array(S0[lo_])))
            c = ci(ratio(Sb[:, hi_], Sb[:, lo_]))
            pairs.append({"num": model_key(models[hi_]["pred"]), "den": model_key(models[lo_]["pred"]),
                          "ratio": pt, "ci95": c, "ci_excludes_1": bool(c[0] > 1),
                          "pass_pair": bool(pt >= 1.5 and c[0] > 1)})
    self_r = ratio(Sb[:, 0], Sb[:, 0]); self_pt = float(ratio(np.array(S0[0]), np.array(S0[0])))
    out = {"n_models_used": len(models), "excluded": excluded, "n_bootstrap_tiles": len(units),
           "bootstrap_band_px": int(bn.sum()), "band_px": int(models[0]["band_px"]) if models else None,
           "models": [{"model": model_key(r["pred"]), "pred": r["pred"], "upsample_factor": r["upsample_factor"],
                       "registration_auroc": r["registration_auroc"], "recall_p05": r["recall_p05_labeled_ink"],
                       "band_fp_area": r["band_fp_area_frac"],
                       "textblock_unlabeled_fpr": r["fpr_p05_textblock_unlabeled"],
                       "band_over_text_unlabeled": r["band_fp_area_frac"] / r["fpr_p05_textblock_unlabeled"]
                       if r["fpr_p05_textblock_unlabeled"] > 0 else None,
                       "letter_blobs": r["blobs"]["letter_blobs"], "letter_blob_per_cm2": r["blobs"]["letter_blob_per_cm2"],
                       "S_1mm_row": r["blobs"]["S_max_letter_blobs_1mm_row"], "S_null_p99": r["blobs"]["null_S_p99"],
                       "blob_flag": r["blobs"]["flag"], "T1": r["periodicity"]["T1_ratio_pooled"],
                       "T1_peak_mm": r["periodicity"]["T1_peak_period_mm"], "p_tile_T1": r["periodicity"]["p_tile_T1"],
                       "tile_null_T1_median": r["periodicity"]["tile_null_T1_median"],
                       "periodicity_flag": r["periodicity_flag"]} for r in models],
           "pairs": pairs, "self_pair": {"ratio": self_pt, "ci95": ci(self_r), "exact_1": bool(self_pt == 1.0 and
                                                                                           np.all(self_r == 1.0))}}
    cons = od / "consensus.json"
    out["consensus"] = json.loads(cons.read_text()) if cons.exists() else None
    for r in models:
        if "planted" in r:
            with np.load(od / f"{Path(r['pred']).name}.npz") as z:
                fpp = z["tile_band_fp_planted"][ii, jj]; fpo = z["tile_band_fp"][ii, jj]
            pb = ratio(Wb @ fpp, Wb @ fpo)
            pt = float(fpp.sum() / fpo.sum()) if fpo.sum() else float("inf")
            out["planted"] = {**r["planted"], "model": model_key(r["pred"]), "ratio": pt, "ci95": ci(pb),
                              "recovered": bool(pt >= 1.3 and ci(pb)[0] > 1)}
    return out


def analyze():
    res = {"created_utc": time.strftime("%FT%TZ", time.gmtime()), "plan_commit": "2e7d835",
           "hard_stop": json.loads(STOP_FILE.read_text()) if STOP_FILE.exists() else None, "segments": {}}
    for s in ("w023", "w018"):
        res["segments"][s] = seg_analysis(s)
    m23 = {m["model"]: m["band_fp_area"] for m in res["segments"]["w023"]["models"]}
    m18 = {m["model"]: m["band_fp_area"] for m in res["segments"]["w018"]["models"]}
    common = sorted(set(m23) & set(m18))
    rho = float(spearmanr([m23[k] for k in common], [m18[k] for k in common])[0]) if len(common) >= 3 else float("nan")
    res["spearman"] = {"models": common, "rho": rho, "w023": [m23[k] for k in common], "w018": [m18[k] for k in common]}
    allp = [p for s in res["segments"].values() for p in s["pairs"]]
    allm = [m for s in res["segments"].values() for m in s["models"]]
    any_pass = any(p["pass_pair"] for p in allp)
    all_incl = all(not p["ci_excludes_1"] for p in allp)
    all_clean = all(m["band_fp_area"] < 0.001 for m in allm)
    sanity = {"registration_all_used_ok": all(m["registration_auroc"] >= 0.75 for m in allm),
              "planted_recovered": res["segments"]["w023"].get("planted", {}).get("recovered"),
              "self_pair_exact": all(s["self_pair"]["exact_1"] for s in res["segments"].values())}
    res["sanity"] = sanity
    res["counts"] = {"pairs": len(allp), "pairs_ci_excl_1": sum(p["ci_excludes_1"] for p in allp),
                     "pairs_pass": sum(p["pass_pair"] for p in allp)}
    if not all(v for v in sanity.values()):
        v, why = "NO VERDICT", "sanity failure = harness bug"
    elif any_pass and rho >= 0.5:
        v, why = "PASS", "PASS: >= 1 pair with ratio >= 1.5 and 95% CI excluding 1, AND Spearman >= 0.5"
    elif all_incl:
        v, why = "KILL", "KILL: 'All pairwise CIs include 1'"
    elif all_clean:
        v, why = "KILL", "KILL: 'every model's band FP area < 0.1%'"
    elif rho < 0.5:
        v, why = "KILL", f"KILL: 'ranking unstable (Spearman < 0.5)': rho = {rho:.3f}"
    else:
        v, why = "KILL", "INCONCLUSIVE = KILL (Amendment D, no retry)"
    res["verdict"], res["verdict_rule"] = v, why
    res["verdict_context"] = {"any_pass_pair": any_pass, "all_ci_include_1": all_incl, "all_fp_below_0.1pct": all_clean,
                              "spearman": rho}
    RES_JSON.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("verdict", "verdict_rule", "sanity", "counts", "spearman")}, indent=1,
                     default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["reduce", "analyze"])
    ap.add_argument("--seg")
    ap.add_argument("--plant", action="store_true")
    a = ap.parse_args()
    if a.cmd == "reduce":
        do_reduce(a.seg, a.plant)
    else:
        analyze()


if __name__ == "__main__":
    main()
