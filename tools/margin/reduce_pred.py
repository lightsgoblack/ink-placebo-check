"""Pull each non-reverse pred of a segment ONE AT A TIME, reduce to scalars, delete the raw file.

Per pred (streamed in 2084-row strips, never loaded whole):
  * tile mean pred value (0-1 scale) for every 5 mm tile  -> Amendment C validity check
  * value histograms of labeled ink / unlabeled px inside the evaluated mask -> registration AUROC, recall
  * band: per-tile band px and band FP px (p > 0.5); per-row band px / FP px (T1 row profile)
  * text block: per-row evaluated-mask px / FP px (T1 positive control)
  * band blobs: 8-connected components of (p > 0.5) & band (column-chunked labeling, merged across
    chunk borders with union-find): area, centroid -> letter-sized blob density and T2
Planted-blob sanity (one pred, pre-registered) runs on the in-memory band array.
Only scalars are written: data/margin_reduced/<seg>/<pred>.npz + .json. No images.

    .venv/bin/python -m tools.margin.reduce_pred --seg w023
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from tools.harness.fetch import RAW, download, free_bytes, is_reverse, list_remote, sha256
from tools.margin.common import (BAND_JSON, REPO, TILE_PX, STRIP, Lazy2D, full_to_grid, seg, tile_grid,
                                 voxel_um, meta)

MIN_FREE_GB = 6.0
OUT = REPO / "data" / "margin_reduced"
CHUNK_W = 16384
SEED = 20260925


class HardStop(Exception):
    pass


def check_disk():
    f = free_bytes(REPO) / 2**30
    if f < MIN_FREE_GB:
        raise SystemExit(f"STOP: free disk {f:.1f} GB < {MIN_FREE_GB} GB")
    return f


def load_band(short: str):
    bj = json.loads(BAND_JSON.read_text())["segments"][short]
    with np.load(REPO / bj["band_grid_npz"]) as z:
        return bj, z["band_grid"]


def eval_mask_files(s):
    names = ["validation_mask_v2", "supervision_mask_v2", "supervision_mask"]
    return [s.small / f"{s.seg_id}_{n}.tif" for n in names if (s.small / f"{s.seg_id}_{n}.tif").exists()]


def label_components(hi: np.ndarray, px_mm2: float) -> dict:
    """8-connected components of a bool array, column-chunked; returns per-component area (px), centroid
    row/col (array coordinates)."""
    H, W = hi.shape
    comps = []          # per chunk: arrays of (area, sum_r, sum_c)
    offsets = []
    borders = []        # (left labels col, right labels col) per chunk boundary
    base = 0
    prev_last = None
    st8 = np.ones((3, 3), bool)
    for c0 in range(0, W, CHUNK_W):
        c1 = min(W, c0 + CHUNK_W)
        lab, n = ndi.label(hi[:, c0:c1], st8)
        if n:
            area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
            rr, cc = np.nonzero(lab)
            ll = lab[rr, cc]
            sr = np.bincount(ll, weights=rr, minlength=n + 1)[1:]
            sc = np.bincount(ll, weights=cc + c0, minlength=n + 1)[1:]
            del rr, cc, ll
        else:
            area = sr = sc = np.zeros(0)
        comps.append(np.stack([area, sr, sc], 1) if n else np.zeros((0, 3)))
        first = lab[:, 0].astype(np.int64)
        if prev_last is not None:
            borders.append((prev_last, first, offsets[-1], base))
        offsets.append(base)
        prev_last = lab[:, -1].astype(np.int64)
        base += n
        del lab
    allc = np.concatenate(comps, 0) if comps else np.zeros((0, 3))
    parent = np.arange(len(allc))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for left, right, lo, ro in borders:
        for dr in (-1, 0, 1):          # right pixel row = left pixel row + dr (8-connectivity)
            if dr < 0:
                l, r = left[-dr:], right[:H + dr]
            elif dr > 0:
                l, r = left[:H - dr], right[dr:]
            else:
                l, r = left, right
            m = (l > 0) & (r > 0)
            for a, b in set(zip((l[m] - 1 + lo).tolist(), (r[m] - 1 + ro).tolist())):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    roots = np.array([find(i) for i in range(len(allc))], dtype=np.int64)
    if len(allc):
        u, inv = np.unique(roots, return_inverse=True)
        agg = np.zeros((len(u), 3))
        np.add.at(agg, inv, allc)
    else:
        agg = np.zeros((0, 3))
    area = agg[:, 0]
    return {"area_px": area, "cent_r": agg[:, 1] / np.maximum(area, 1), "cent_c": agg[:, 2] / np.maximum(area, 1)}


def periodogram_stat(profile: np.ndarray, um_per_px: float) -> dict:
    """Linear-detrended periodogram; max power over periods 2.5-8 mm / median over 0.5-8 mm."""
    y = np.asarray(profile, np.float64)
    n = len(y)
    if n < 16 or not np.isfinite(y).all():
        return {"n": n, "ratio": None}
    x = np.arange(n)
    y = y - np.polyval(np.polyfit(x, y, 1), x)
    p = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(n, d=um_per_px / 1000.0)   # cycles per mm
    with np.errstate(divide="ignore"):
        per = np.where(f > 0, 1.0 / f, np.inf)
    win = (per >= 2.5) & (per <= 8.0)
    ref = (per >= 0.5) & (per <= 8.0)
    if not win.any() or ref.sum() < 3:
        return {"n": n, "ratio": None, "length_mm": n * um_per_px / 1000}
    med = float(np.median(p[ref]))
    k = int(np.argmax(np.where(win, p, -1)))
    return {"n": n, "length_mm": n * um_per_px / 1000, "peak_period_mm": float(per[k]),
            "peak_power": float(p[k]), "median_power_0.5_8mm": med,
            "ratio": float(p[k] / med) if med > 0 else (float("inf") if p[k] > 0 else 0.0),
            "n_freq_window": int(win.sum()), "n_freq_ref": int(ref.sum())}


def reduce_one(short: str, raw: Path, rel: str, band_grid: np.ndarray, bj: dict, plant: bool) -> dict:
    s = seg(short)
    m = meta(short)
    vox = voxel_um(m)
    px_mm2 = (vox / 1000) ** 2
    P = Lazy2D(raw)
    lab = Lazy2D(s.small / f"{s.seg_id}_inklabels_v2.tif")
    masks = [Lazy2D(f) for f in eval_mask_files(s)]
    H, W = lab.shape
    info = dict(P.info)
    if tuple(P.shape) != (H, W):
        for x in [P, lab, *masks]:
            x.close()
        return {"pred": rel, "file_info": info, "excluded": f"shape {P.shape} != labels {(H, W)}"}
    dt = np.dtype(P.dtype)
    if dt == np.uint8:
        thr, scale, nb = 127, 255.0, 256
    elif dt.kind == "f":
        thr, scale, nb = 0.5, 1.0, 1001
    elif dt == np.uint16:
        thr, scale, nb = 32767, 65535.0, 65536
    else:
        raise SystemExit(f"unexpected pred dtype {dt}")
    GH, GW = band_grid.shape
    sy, sx = bj["grid_factor_sy_sx"]
    gy, gx = full_to_grid(H, sy, GH), full_to_grid(W, sx, GW)
    ty, tx = tile_grid((H, W))
    r_lo, r_hi = bj["band_row_range_full_res"]
    c_lo, c_hi = bj["band_col_range_full_res"]
    tile_mean = np.full((ty, tx), np.nan)
    band_n = np.zeros((ty, tx), np.int64)
    band_fp = np.zeros((ty, tx), np.int64)
    row_band_n = np.zeros(H, np.int64); row_band_fp = np.zeros(H, np.int64)
    row_ev_n = np.zeros(H, np.int64); row_ev_fp = np.zeros(H, np.int64)
    h_ink = np.zeros(nb, np.int64); h_neg = np.zeros(nb, np.int64)
    vmin, vmax = np.inf, -np.inf
    band_hi = np.zeros((r_hi - r_lo, c_hi - c_lo), bool)
    for r0 in range(0, H, STRIP):
        r1 = min(H, r0 + STRIP)
        a = P.rows(r0, r1)
        vmin, vmax = min(vmin, float(a.min())), max(vmax, float(a.max()))
        hi = a > thr
        ink = lab.rows(r0, r1) > 0
        ev = masks[0].rows(r0, r1) > 0
        for mm in masks[1:]:
            ev |= mm.rows(r0, r1) > 0
        q = np.rint(a * (1000 if dt.kind == "f" else 1)).astype(np.int64) if dt.kind == "f" else a
        h_ink += np.bincount(q[ink & ev].ravel(), minlength=nb)[:nb]
        h_neg += np.bincount(q[~ink & ev].ravel(), minlength=nb)[:nb]
        row_ev_n[r0:r1] = ev.sum(1); row_ev_fp[r0:r1] = (hi & ev).sum(1)
        del ink, ev, q
        bnd = band_grid[gy[r0:r1]][:, gx]
        row_band_n[r0:r1] = bnd.sum(1); row_band_fp[r0:r1] = (hi & bnd).sum(1)
        i = r0 // TILE_PX
        if r1 - r0 == TILE_PX and i < ty:
            w = tx * TILE_PX
            tile_mean[i] = a[:, :w].reshape(TILE_PX, tx, TILE_PX).mean(axis=(0, 2), dtype=np.float64) / scale
            band_n[i] = bnd[:, :w].reshape(TILE_PX, tx, TILE_PX).sum(axis=(0, 2))
            band_fp[i] = (hi & bnd)[:, :w].reshape(TILE_PX, tx, TILE_PX).sum(axis=(0, 2))
        o0, o1 = max(r0, r_lo), min(r1, r_hi)
        if o0 < o1:
            band_hi[o0 - r_lo:o1 - r_lo] = (hi & bnd)[o0 - r0:o1 - r0, c_lo:c_hi]
        del a, hi, bnd
    for x in [lab, *masks]:
        x.close()
    P.close()

    comps = label_components(band_hi, px_mm2)
    area_mm2 = comps["area_px"] * px_mm2
    letter = (area_mm2 >= 1.0) & (area_mm2 <= 4.0)
    L = {"area_mm2": area_mm2[letter].tolist(), "cent_row": (comps["cent_r"][letter] + r_lo).tolist(),
         "cent_col": (comps["cent_c"][letter] + c_lo).tolist()}
    # T2: >= 5 letter blobs and >= 3 with centroid rows within +-1 mm
    one_mm = 1000.0 / vox
    cr = np.sort(np.array(L["cent_row"]))
    max_in_row = 0
    for k in range(len(cr)):
        max_in_row = max(max_in_row, int(np.sum(np.abs(cr - cr[k]) <= one_mm)))
    band_px = int(row_band_n.sum())
    fp_px = int(row_band_fp.sum())
    fp_frac = fp_px / band_px if band_px else float("nan")
    # T1 row profile over band rows with >= 10% of max band row count
    keep = row_band_n >= 0.1 * row_band_n.max()
    rows = np.flatnonzero(keep)
    rows = np.arange(rows.min(), rows.max() + 1)
    prof = row_band_fp[rows] / np.maximum(row_band_n[rows], 1)
    t1 = periodogram_stat(prof, vox)
    evk = row_ev_n >= 0.1 * row_ev_n.max()
    er = np.flatnonzero(evk)
    er = np.arange(er.min(), er.max() + 1)
    t1_text = periodogram_stat(row_ev_fp[er] / np.maximum(row_ev_n[er], 1), vox)
    # registration AUROC from histograms (ties half) and recall
    cn = np.cumsum(h_neg) - h_neg          # negatives strictly below each bin
    n_pos, n_neg = h_ink.sum(), h_neg.sum()
    auc = float((h_ink * (cn + 0.5 * h_neg)).sum() / (n_pos * n_neg)) if n_pos and n_neg else float("nan")
    hi_bins = np.arange(nb) > (thr * (1000 if dt.kind == "f" else 1))
    recall = float(h_ink[hi_bins].sum() / n_pos) if n_pos else float("nan")
    fpr_text = float(h_neg[hi_bins].sum() / n_neg) if n_neg else float("nan")

    out = {"pred": rel, "file_info": info, "dtype": str(dt), "value_min": vmin, "value_max": vmax,
           "p05_threshold_raw": thr, "band_px": band_px, "band_fp_px": fp_px, "band_fp_area_frac": fp_frac,
           "band_n_components": int(len(area_mm2)), "band_letter_blobs": int(letter.sum()),
           "band_letter_blob_per_cm2": float(letter.sum() / (band_px * px_mm2 / 100.0)),
           "band_largest_component_mm2": float(area_mm2.max()) if len(area_mm2) else 0.0,
           "letter_blobs": L, "T2_max_letter_blobs_in_one_row": max_in_row,
           "T2_flag": bool(letter.sum() >= 5 and max_in_row >= 3),
           "T1_band": t1, "T1_flag": bool(fp_frac >= 0.001 and (t1.get("ratio") or 0) >= 10),
           "T1_textblock_positive_control": t1_text,
           "registration_auroc": auc, "registration_ok": bool(auc >= 0.75),
           "recall_p05_labeled_ink": recall, "fpr_p05_textblock_unlabeled": fpr_text,
           "n_ink_px_eval": int(n_pos), "n_neg_px_eval": int(n_neg)}
    arrays = {"tile_mean": tile_mean, "band_n": band_n, "band_fp": band_fp,
              "row_band_n": row_band_n, "row_band_fp": row_band_fp, "row_ev_n": row_ev_n, "row_ev_fp": row_ev_fp,
              "h_ink": h_ink, "h_neg": h_neg}
    if plant:
        out["planted"], arrays["band_fp_planted"] = plant_blobs(band_hi, band_grid, bj, gy, gx, r_lo, c_lo,
                                                                band_fp, fp_px, band_px, vox)
    return out, arrays


def plant_blobs(band_hi, band_grid, bj, gy, gx, r_lo, c_lo, band_fp, fp_px, band_px, vox):
    """Plant 2 mm2 disks at seeded random band positions until planted px >= max(0.5*FP, 0.1% band)."""
    rng = np.random.default_rng(SEED)
    target = max(0.5 * fp_px, 0.001 * band_px)
    rad = np.sqrt(2.0 / np.pi) * 1000.0 / vox
    R = int(np.ceil(rad))
    dy, dx = np.mgrid[-R:R + 1, -R:R + 1]
    disk = dy ** 2 + dx ** 2 <= rad ** 2
    Hh, Ww = band_hi.shape
    added = np.zeros_like(band_fp)
    tot, n_disks, tries = 0, 0, 0
    while tot < target and tries < 100000:
        tries += 1
        r, c = int(rng.integers(0, Hh)), int(rng.integers(0, Ww))
        R0, C0 = r + r_lo, c + c_lo
        if not band_grid[gy[R0], gx[C0]]:
            continue
        y0, y1, x0, x1 = max(0, r - R), min(Hh, r + R + 1), max(0, c - R), min(Ww, c + R + 1)
        d = disk[y0 - r + R:y1 - r + R, x0 - c + R:x1 - c + R]
        inb = band_grid[gy[y0 + r_lo:y1 + r_lo]][:, gx[x0 + c_lo:x1 + c_lo]]
        new = d & inb & ~band_hi[y0:y1, x0:x1]
        if not new.any():
            continue
        band_hi[y0:y1, x0:x1] |= new
        yy, xx = np.nonzero(new)
        ti, tj = (yy + y0 + r_lo) // TILE_PX, (xx + x0 + c_lo) // TILE_PX
        ok = (ti < added.shape[0]) & (tj < added.shape[1])
        np.add.at(added, (ti[ok], tj[ok]), 1)
        tot += int(new.sum()); n_disks += 1
    return ({"target_px": float(target), "planted_px": tot, "n_disks": n_disks, "disk_mm2": 2.0,
             "expected_ratio": (fp_px + tot) / fp_px if fp_px else None}, band_fp + added)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--plant", action="store_true", help="planted-blob sanity on the first pred")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    s = seg(a.seg)
    bj, band_grid = load_band(a.seg)
    remote = list_remote(s)
    preds = sorted(k for k in remote if k.startswith("preds/") and not is_reverse(k))
    od = OUT / a.seg
    od.mkdir(parents=True, exist_ok=True)
    print(f"{a.seg}: {len(preds)} non-reverse preds", flush=True)
    for n, rel in enumerate(preds):
        if a.limit and n >= a.limit:
            break
        name = Path(rel).name
        if (od / f"{name}.json").exists():
            print("  skip", name, flush=True)
            continue
        free = check_disk()
        raw = RAW / name
        t0 = time.time()
        print(f"  pull {name} ({remote[rel].size / 2**30:.2f} GiB), free {free:.1f} GB", flush=True)
        if not (raw.exists() and raw.stat().st_size == remote[rel].size):
            download(remote[rel], raw)
        digest = sha256(raw)
        res = reduce_one(a.seg, raw, rel, band_grid, bj, plant=(a.plant and n == 0))
        raw.unlink()
        if isinstance(res, dict):       # excluded
            res["sha256"] = digest
            (od / f"{name}.json").write_text(json.dumps(res, indent=1))
            print("   EXCLUDED", res["excluded"], flush=True)
            continue
        out, arrays = res
        out["sha256"] = digest
        out["runtime_s"] = round(time.time() - t0, 1)
        np.savez_compressed(od / f"{name}.npz", **arrays)
        (od / f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
        print(f"   {out['dtype']} [{out['value_min']:.3g},{out['value_max']:.3g}] AUROC {out['registration_auroc']:.3f} "
              f"recall {out['recall_p05_labeled_ink']:.3f} bandFP {out['band_fp_area_frac']:.2e} "
              f"letter {out['band_letter_blobs']} T1 {out['T1_band'].get('ratio')} T1flag {out['T1_flag']} "
              f"T2flag {out['T2_flag']} ({out['runtime_s']} s)", flush=True)
        if out["T1_flag"] or out["T2_flag"]:
            print("HARD STOP: text-like flag in band. Stopping.", flush=True)
            raise SystemExit(3)


if __name__ == "__main__":
    main()
