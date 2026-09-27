"""Checks 1-7 of the w031 margin forensics from the reduced arrays (data/margin_forensics/, gitignored).
Pre-registered thresholds: results/margin_forensics_w031.json["preregistration"]. Scalars only, no images.

    .venv/bin/python -m tools.margin.forensics analyze
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

from tools.margin.common import REPO, BAND_JSON, seg, meta, voxel_um, pool_count_to_grid, grid_factors, Lazy2D
from tools.margin.forensics import CELL, N_SURR, OUT, P0_MM, RES_JSON, SEED0
from tools.margin.reduce_pred import eval_mask_files, periodogram_stat

VOX_MM = 2.399e-3
CMM = CELL * VOX_MM                   # pooled cell size, mm
FULL = CELL * CELL
TILE = 4                              # cells per 1 mm shuffle tile
LO_BP, HI_BP = 2.5, 8.0               # band-pass (periods, mm) for alignment correlations


# ------------------------------------------------------------------------------ signal helpers
def detrend(y):
    x = np.arange(len(y))
    return y - np.polyval(np.polyfit(x, y, 1), x)


def phase_at(y, x_mm, P=P0_MM):
    """Line-centre phase (rad) of profile y sampled at absolute positions x_mm; amplitude-normalized power."""
    yd = detrend(np.asarray(y, float))
    C = np.sum(yd * np.exp(-2j * np.pi * x_mm / P))
    return float(-np.angle(C)), float(np.abs(C) ** 2)


def wrap(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def phase_rand(y, rng):
    yd = detrend(np.asarray(y, float))
    F = np.fft.rfft(yd)
    ph = rng.uniform(0, 2 * np.pi, len(F))
    ph[0] = 0
    if len(yd) % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), len(yd))


def bandpass(y):
    yd = detrend(np.asarray(y, float))
    F = np.fft.rfft(yd)
    f = np.fft.rfftfreq(len(yd), CMM)
    with np.errstate(divide="ignore"):
        per = np.where(f > 0, 1 / f, np.inf)
    F[(per < LO_BP) | (per > HI_BP)] = 0
    return np.fft.irfft(F, len(yd))


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def p_ge(obs, surr):
    surr = np.asarray(surr)
    return float((1 + np.sum(surr >= obs)) / (1 + len(surr)))


def p0_ratio(y):
    """Power at P0 / median periodogram power over 0.5-8 mm (same normalization as T1)."""
    yd = detrend(np.asarray(y, float))
    p = np.abs(np.fft.rfft(yd)) ** 2
    f = np.fft.rfftfreq(len(yd), CMM)
    with np.errstate(divide="ignore"):
        per = np.where(f > 0, 1 / f, np.inf)
    med = np.median(p[(per >= 0.5) & (per <= 8)])
    _, pw = phase_at(yd, np.arange(len(yd)) * CMM)
    return float(pw / med) if med > 0 else float("nan")


def fine_peak(y, lo=LO_BP, hi=HI_BP):
    yd = detrend(np.asarray(y, float)); x = np.arange(len(yd)) * CMM
    per = np.linspace(lo, hi, 1101)
    pw = np.array([np.abs(np.sum(yd * np.exp(-2j * np.pi * x / p))) ** 2 for p in per])
    return float(per[int(np.argmax(pw))])


class Profile:
    """Row profile over a region of the pooled grid: sum(num)/sum(den) per pooled row, rows with den >= 10% max."""

    def __init__(self, num, den, rowmask=None):
        self.num, self.den = np.nan_to_num(num), np.nan_to_num(den)
        dr = self.den.sum(1)
        keep = dr >= 0.1 * dr.max() if dr.max() > 0 else dr > 0
        if rowmask is not None:
            keep &= rowmask
        r = np.flatnonzero(keep)
        self.rows = np.arange(r.min(), r.max() + 1)
        self.x_mm = (self.rows * CELL + CELL / 2) * VOX_MM

    def prof(self, num=None):
        num = self.num if num is None else num
        return num[self.rows].sum(1) / np.maximum(self.den[self.rows].sum(1), 1e-9)

    def tiles(self):
        """Eligible 1 mm tiles: all 16 cells wholly in region (den == FULL)."""
        H, W = self.den.shape
        full = self.den >= FULL - 0.5
        out = []
        for i in range(self.rows[0] - self.rows[0] % TILE, self.rows[-1] + 1, TILE):
            for j in range(0, W - W % TILE, TILE):
                if i + TILE <= H and full[i:i + TILE, j:j + TILE].all():
                    out.append((i, j))
        return out

    def shuffled(self, rng, tiles):
        num = self.num.copy()
        perm = rng.permutation(len(tiles))
        blocks = [self.num[i:i + TILE, j:j + TILE] for i, j in tiles]
        for (i, j), k in zip(tiles, perm):
            num[i:i + TILE, j:j + TILE] = blocks[k]
        return num


def periodicity(pr: Profile, label):
    """T1 ratio (pooled), P0 ratio, phase at P0 (and +-5%), tile-shuffle p for both strengths."""
    y = pr.prof()
    t1 = periodogram_stat(y, CMM * 1000)
    ph = {f"{s:.2f}": phase_at(y, pr.x_mm, P0_MM * s)[0] for s in (0.95, 1.0, 1.05)}
    obs_r, obs_p0 = t1["ratio"], p0_ratio(y)
    tiles = pr.tiles()
    sr, sp = [], []
    for k in range(N_SURR):
        rng = np.random.default_rng(SEED0 + k)
        ys = pr.prof(pr.shuffled(rng, tiles))
        sr.append(periodogram_stat(ys, CMM * 1000)["ratio"]); sp.append(p0_ratio(ys))
    n_cells_tiles = len(tiles) * TILE * TILE
    return {"label": label, "n_rows": int(len(y)), "length_mm": float(len(y) * CMM),
            "T1_ratio_pooled": obs_r, "T1_peak_period_mm": t1.get("peak_period_mm"),
            "fine_peak_period_mm": fine_peak(y), "P0_ratio": obs_p0,
            "phase_P0": ph["1.00"], "phase_0.95P0": ph["0.95"], "phase_1.05P0": ph["1.05"],
            "p_tile_T1": p_ge(obs_r, sr), "p_tile_P0": p_ge(obs_p0, sp),
            "tile_null_T1_median": float(np.median(sr)), "tile_null_T1_p95": float(np.percentile(sr, 95)),
            "shuffle_cells_frac": float(n_cells_tiles / max((pr.den[pr.rows] > 0).sum(), 1)),
            "mean": float(np.mean(y))}


def circ_R(phases):
    return float(np.abs(np.mean(np.exp(1j * np.asarray(phases))))) if len(phases) else float("nan")


def circ_mean(phases):
    return float(np.angle(np.mean(np.exp(1j * np.asarray(phases)))))


# ------------------------------------------------------------------------------ loading
def load(short):
    d = OUT / short
    st = dict(np.load(d / "static.npz"))
    info = json.loads((d / "static.json").read_text())
    mods = {}
    for f in sorted(d.glob("*.tif.json")):
        o = json.loads(f.read_text())
        if "excluded" in o:
            mods[f.name[:-5]] = (o, None)
            continue
        mods[f.name[:-5]] = (o, dict(np.load(d / (f.name[:-5] + ".npz"))))
    return st, info, mods


def fullres_T1(o, a, st):
    n = st["row_band_n"]
    r = np.flatnonzero(n >= 0.1 * n.max()); r = np.arange(r.min(), r.max() + 1)
    return periodogram_stat(a["row_band_hi"][r] / np.maximum(n[r], 1), 2.399)


# ------------------------------------------------------------------------------ geometry (check 5)
def geometry():
    segs = ["w018", "w023", "w029", "w031"]
    G = {}
    for s in segs:
        d = REPO / "data" / f"1667_{s}" / "small"
        x, y, z = [np.squeeze(tifffile.imread(d / f"{c}.tif")).astype(np.float64) for c in "xyz"]
        G[s] = (x * VOX_MM, y * VOX_MM, z * VOX_MM, z > 0)

    def kasa(x, y):
        A = np.c_[x, y, np.ones_like(x)]
        D, E, F = np.linalg.lstsq(A, -(x * x + y * y), rcond=None)[0]
        return -D / 2, -E / 2

    # scroll-centre proxy c(z): circle fits per tifxyz row of w018 (5 wraps, spiral-centred), 5 mm running median
    x, y, z, v = G["w018"]
    zc, cxs, cys = [], [], []
    for i in range(z.shape[0]):
        m = v[i]
        if m.sum() < 200:
            continue
        cx, cy = kasa(x[i, m], y[i, m]); zc.append(z[i, m].mean()); cxs.append(cx); cys.append(cy)
    zc, cxs, cys = map(np.array, (zc, cxs, cys))
    o = np.argsort(zc); zc, cxs, cys = zc[o], cxs[o], cys[o]
    step = np.median(np.diff(zc)); k = max(3, int(round(5.0 / step)) | 1)
    cxs = ndi.median_filter(cxs, k, mode="nearest"); cys = ndi.median_filter(cys, k, mode="nearest")
    centre = lambda zz: (np.interp(zz, zc, cxs), np.interp(zz, zc, cys))

    polar = {}
    for s in segs:
        x, y, z, v = G[s]
        cx, cy = centre(z)
        r = np.hypot(x - cx, y - cy)
        th = np.arctan2(y - cy, x - cx)
        # unwrap along each row (columns = along the sheet)
        thu = np.where(v, th, np.nan)
        for i in range(thu.shape[0]):
            m = v[i]
            if m.sum() > 1:
                thu[i, m] = np.unwrap(th[i, m])
        polar[s] = (r, thu, z, v)

    # wrap spacing from 2pi self-overlap inside each segment: r(theta + 2pi) - r(theta), same row
    spacing = {}
    for s in segs:
        r, thu, z, v = polar[s]
        vals, zs = [], []
        for i in range(0, r.shape[0], 5):
            m = v[i]
            if m.sum() < 50:
                continue
            t, rr = thu[i, m], r[i, m]
            if not (np.all(np.diff(t) > 0) or np.all(np.diff(t) < 0)):
                o = np.argsort(t); t, rr = t[o], rr[o]
            elif t[0] > t[-1]:
                t, rr = t[::-1], rr[::-1]
            sel = t + 2 * np.pi <= t[-1]
            if sel.sum() < 5:
                continue
            ds = np.interp(t[sel] + 2 * np.pi, t, rr) - rr[sel]
            vals.append(ds); zs.append(np.full(len(ds), z[i, m].mean()))
        if vals:
            ds = np.concatenate(vals); zz = np.concatenate(zs)
            ab = np.abs(ds)
            spacing[s] = {"n": int(len(ds)), "median_abs_mm": float(np.median(ab)), "p10": float(np.percentile(ab, 10)),
                          "p90": float(np.percentile(ab, 90)), "median_signed_mm": float(np.median(ds)),
                          "median_abs_z_ge_70mm": float(np.median(ab[zz >= 70])) if (zz >= 70).any() else None,
                          "median_abs_z_30_65mm": float(np.median(ab[(zz >= 30) & (zz < 65)])) if ((zz >= 30) & (zz < 65)).any() else None}
    s_all = [spacing[s]["median_abs_mm"] for s in ("w018", "w023") if s in spacing]
    s_est = float(np.median(s_all)) if s_all else float(np.median([v["median_abs_mm"] for v in spacing.values()]))

    # w031: per-column radius continuity across the labeled edge
    bj = json.loads(BAND_JSON.read_text())["segments"]["w031"]
    edge_z = bj["ink_z_mm_p99.5"]
    r, thu, z, v = polar["w031"]
    jumps, base = [], []
    for j in range(r.shape[1]):
        m = v[:, j]
        if m.sum() < 100:
            continue
        zz, rr = z[m, j], r[m, j]
        tb = (zz >= edge_z - 30) & (zz < edge_z - 2)
        if tb.sum() < 50:
            continue
        a, b = np.polyfit(zz[tb], rr[tb], 1)
        res = rr - (a * zz + b)
        up = (zz >= edge_z + 2) & (zz < edge_z + 7)
        dn = (zz >= edge_z - 7) & (zz < edge_z - 2)
        if up.sum() > 5 and dn.sum() > 5:
            jumps.append(np.median(res[up]) - np.median(res[dn]))
        for c0 in np.arange(edge_z - 30, edge_z - 16, 2.0):   # text-block 9 mm-apart window pairs
            w1 = (zz >= c0) & (zz < c0 + 5); w2 = (zz >= c0 + 9) & (zz < c0 + 14)
            if w1.sum() > 5 and w2.sum() > 5:
                base.append(np.median(res[w2]) - np.median(res[w1]))
    jumps, base = np.abs(np.array(jumps)), np.abs(np.array(base))
    # band departure: residual in the whole band vs text-block trend
    # w031 - w029 radius difference at matched (z 1 mm, theta 5 deg) bins, overlap angles excluded
    def binned(s):
        r, thu, z, v = polar[s]
        th = np.mod(thu, 2 * np.pi)
        span_ok = np.zeros_like(v)
        for i in range(r.shape[0]):
            m = v[i]
            if m.sum() < 2:
                continue
            t = thu[i, m]; lo, hi = np.nanmin(t), np.nanmax(t)
            ok = (t - lo >= (hi - lo - 2 * np.pi)) & (hi - t >= (hi - lo - 2 * np.pi))   # angles covered once
            if hi - lo <= 2 * np.pi:
                ok[:] = True
            span_ok[i, np.flatnonzero(m)] = ok
        mm = v & span_ok
        zb = np.floor(z[mm]).astype(int); tb = np.floor(np.degrees(th[mm]) / 5).astype(int)
        key = zb * 100 + tb
        u, inv = np.unique(key, return_inverse=True)
        med = np.array([np.median(r[mm][inv == k]) for k in range(len(u))])
        return dict(zip(u.tolist(), med.tolist()))
    b31, b29 = binned("w031"), binned("w029")
    common = sorted(set(b31) & set(b29))
    dz = np.array([k // 100 for k in common]); dd = np.array([b31[k] - b29[k] for k in common])
    dth = np.array([k % 100 for k in common])
    band_sel = dz >= edge_z + 2; text_sel = (dz >= edge_z - 12) & (dz < edge_z - 2)
    per_th = []
    for t in np.unique(dth):
        a1 = dd[band_sel & (dth == t)]; a0 = dd[text_sel & (dth == t)]
        if len(a1) >= 3 and len(a0) >= 3:
            per_th.append(np.median(a1) - np.median(a0))
    per_th = np.array(per_th)

    # mesh stretch / area distortion on the w031 grid, band vs text block
    x, y, z, v = G["w031"]
    m31 = meta("w031"); sx, sy = m31["scale"]
    du = np.stack([np.diff(c, axis=1)[:-1] for c in (x, y, z)], -1); dv = np.stack([np.diff(c, axis=0)[:, :-1] for c in (x, y, z)], -1)
    nom_u, nom_v = VOX_MM / sx, VOX_MM / sy
    vv = v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:]
    area = np.linalg.norm(np.cross(du, dv), axis=-1) / (nom_u * nom_v)
    su = np.linalg.norm(du, axis=-1) / nom_u; sv = np.linalg.norm(dv, axis=-1) / nom_v
    with np.load(REPO / bj["band_grid_npz"]) as zz:
        band = zz["band_grid"][:-1, :-1] & vv
    zmm = z[:-1, :-1]
    s = seg("w031")
    lz = Lazy2D(s.small / f"{s.seg_id}_validation_mask_v2.tif")
    evg = pool_count_to_grid(lz, *grid_factors(m31, lz.shape, z.shape), z.shape); lz.close()
    for f in eval_mask_files(s)[1:]:
        lz = Lazy2D(f); evg += pool_count_to_grid(lz, *grid_factors(m31, lz.shape, z.shape), z.shape); lz.close()
    text = (evg[:-1, :-1] > 0) & vv
    near_edge = vv & (zmm >= edge_z - 3) & (zmm < edge_z + 5)

    def q(a, m):
        return {"median": float(np.median(a[m])), "p5": float(np.percentile(a[m], 5)), "p95": float(np.percentile(a[m], 95))}

    # distance from segment edge inside the band
    edge_d = ndi.distance_transform_edt(np.pad(v, 1))[1:-1, 1:-1] * float(bj["grid_cell_mm"])
    bfull = np.zeros_like(v); bfull[:-1, :-1] = band
    return {
        "centre_proxy": "circle fits per tifxyz row of w018 (5 wraps), 5 mm running median, interpolated in z",
        "wrap_spacing_self_overlap_mm": spacing, "wrap_spacing_estimate_s_mm": s_est,
        "wrap_spacing_source": "median |r(theta+2pi) - r(theta)| within w018/w023 (multi-wrap segments)",
        "w031_edge_jump_mm": {"n_columns": int(len(jumps)), "median_abs": float(np.median(jumps)),
                              "p90_abs": float(np.percentile(jumps, 90)),
                              "frac_ge_half_s": float(np.mean(jumps >= 0.5 * s_est)),
                              "baseline_text_pairs_median_abs": float(np.median(base)),
                              "baseline_text_pairs_p95_abs": float(np.percentile(base, 95)),
                              "baseline_frac_ge_half_s": float(np.mean(base >= 0.5 * s_est))},
        "w031_minus_w029": {"n_theta_bins": int(len(per_th)),
                            "median_change_band_vs_text_mm": float(np.median(per_th)) if len(per_th) else None,
                            "median_abs_change_mm": float(np.median(np.abs(per_th))) if len(per_th) else None,
                            "median_diff_text_mm": float(np.median(dd[text_sel])) if text_sel.any() else None,
                            "median_diff_band_mm": float(np.median(dd[band_sel])) if band_sel.any() else None},
        "mesh_area_ratio": {"band": q(area, band), "text_block": q(area, text), "near_edge_-3_+5mm": q(area, near_edge)},
        "mesh_stretch_u": {"band": q(su, band), "text_block": q(su, text)},
        "mesh_stretch_v": {"band": q(sv, band), "text_block": q(sv, text)},
        "band_distance_from_segment_edge_mm": q(edge_d, bfull),
    }


# ------------------------------------------------------------------------------ main analysis
def analyze_w031():
    st, info, mods = load("w031")
    bj = json.loads(BAND_JSON.read_text())["segments"]["w031"]
    edge_z = bj["ink_z_mm_p99.5"]
    pred_names = [k for k in mods if not k.endswith("max_22_42.tif")]
    fwd = [k for k in pred_names if "reverse" not in k]
    rev = [k for k in pred_names if "reverse" in k]
    rend = [k for k in mods if k.endswith("max_22_42.tif")]
    nb = st["n_band"]
    band_full = nb >= FULL - 0.5
    text_unlab = (st["n_ev"] >= FULL - 0.5) & (st["n_ink"] == 0)
    text_full = st["n_ev"] >= FULL - 0.5
    out = {"models": {}, "excluded": {}}
    maps, maps_t, profs = {}, {}, {}
    band_pr = None
    for k in pred_names:
        o, a = mods[k]
        if a is None:
            out["excluded"][k] = o["excluded"]; continue
        t1f = fullres_T1(o, a, st)
        bb = o["band_blobs"]
        pr = Profile(a["cell_hi_band"], nb)
        band_pr = pr
        per = periodicity(pr, k)
        fires = bool(o["band_fp_frac"] >= 0.001 and ((t1f.get("ratio") or 0) >= 10 or bb["T2_flag"]))
        # text-block (cells wholly in eval mask) profile, model and continuous value
        out["models"][k] = {
            "reverse": "reverse" in k, "registration_auroc": o["registration_auroc"],
            "registration_ok": bool(o["registration_auroc"] >= 0.75), "recall_p05": o["recall_p05"],
            "fpr_p05_text_unlabeled": o["fpr_p05_text_unlabeled"], "band_fp_frac": o["band_fp_frac"],
            "band_fp_over_text_unlabeled_fpr": o["band_fp_frac"] / max(o["fpr_p05_text_unlabeled"], 1e-12),
            "letter_blobs": bb["letter_blobs"], "letter_blobs_per_cm2": bb["letter_blobs"] / (info["band_px"] * (VOX_MM ** 2) / 100),
            "max_letter_blobs_in_one_row": bb["max_letter_blobs_in_one_row"], "T2_flag": bb["T2_flag"],
            "T1_fullres": {"ratio": t1f.get("ratio"), "peak_period_mm": t1f.get("peak_period_mm")},
            "fires_v0M_rule": fires, "band_periodicity": per,
            "periodic_sig": bool(per["p_tile_T1"] <= 0.01),
            "band_mean_value": float(np.nansum(a["cell_val_band"]) / info["band_px"]),
        }
        m = np.where(band_full, a["cell_hi_band"] / FULL, np.nan)
        maps[k] = m[band_full]
        maps_t[k] = (a["cell_hi_all"] / FULL)[text_unlab]
        profs[k] = pr.prof()
    reg_fwd = [k for k in fwd if k in out["models"] and out["models"][k]["registration_ok"]]
    # ---- check 1 cross-model
    ks = [k for k in pred_names if k in maps]
    C = np.full((len(ks), len(ks)), np.nan); Ct = C.copy()
    for i, a in enumerate(ks):
        for j, b in enumerate(ks):
            C[i, j] = corr(maps[a], maps[b]); Ct[i, j] = corr(maps_t[a], maps_t[b])
    iu = np.triu_indices(len(ks), 1)
    fi = [ks.index(k) for k in reg_fwd]
    sub = C[np.ix_(fi, fi)][np.triu_indices(len(fi), 1)]; subt = Ct[np.ix_(fi, fi)][np.triu_indices(len(fi), 1)]
    per_sig = [k for k in reg_fwd if out["models"][k]["periodic_sig"]]
    ph = [out["models"][k]["band_periodicity"]["phase_P0"] for k in per_sig]
    R = circ_R(ph)
    Rs = []
    for s in range(N_SURR):
        rng = np.random.default_rng(SEED0 + s)
        Rs.append(circ_R([phase_at(phase_rand(profs[k], rng), band_pr.x_mm)[0] for k in per_sig]))
    ph_all = [out["models"][k]["band_periodicity"]["phase_P0"] for k in reg_fwd]
    indep = [k for k in reg_fwd if k.startswith("ps")]      # merged_* are ensembles (identical at p > 0.5)
    ph_ind = [out["models"][k]["band_periodicity"]["phase_P0"] for k in indep]
    R_ind = circ_R(ph_ind); Rs_ind = []
    for s_ in range(N_SURR):
        rng = np.random.default_rng(SEED0 + s_)
        Rs_ind.append(circ_R([phase_at(phase_rand(profs[k], rng), band_pr.x_mm)[0] for k in indep]))
    ii = [ks.index(k) for k in indep]
    sub_i = C[np.ix_(ii, ii)][np.triu_indices(len(ii), 1)]; subt_i = Ct[np.ix_(ii, ii)][np.triu_indices(len(ii), 1)]
    cons = np.mean([profs[k] for k in reg_fwd], 0)
    cons_ph = phase_at(cons, band_pr.x_mm)[0]
    check1 = {
        "n_forward": len(fwd), "n_registered_forward": len(reg_fwd),
        "n_fire_v0M_rule": int(sum(out["models"][k]["fires_v0M_rule"] for k in reg_fwd)),
        "frac_fire": float(np.mean([out["models"][k]["fires_v0M_rule"] for k in reg_fwd])) if reg_fwd else None,
        "n_periodic_sig": len(per_sig),
        "band_map_corr_median_forward_pairs": float(np.median(sub)) if len(sub) else None,
        "band_map_corr_range": [float(np.min(sub)), float(np.max(sub))] if len(sub) else None,
        "text_unlabeled_map_corr_median_forward_pairs": float(np.median(subt)) if len(subt) else None,
        "map_models_order": ks, "band_map_corr_matrix": np.round(C, 3).tolist(),
        "phase_R_periodic_sig_models": R, "phase_R_null_p": p_ge(R, Rs), "phase_R_null_median": float(np.median(Rs)),
        "phase_R_all_registered": circ_R(ph_all),
        "independent_models_ps": indep, "phase_R_independent": R_ind, "phase_R_independent_null_p": p_ge(R_ind, Rs_ind),
        "phase_R_independent_null_median": float(np.median(Rs_ind)),
        "phase_R_independent_at_0.95_1.05P0": [circ_R([out["models"][k]["band_periodicity"]["phase_0.95P0"] for k in indep]),
                                               circ_R([out["models"][k]["band_periodicity"]["phase_1.05P0"] for k in indep])],
        "band_map_corr_median_independent": float(np.median(sub_i)) if len(sub_i) else None,
        "band_map_corr_range_independent": [float(np.min(sub_i)), float(np.max(sub_i))] if len(sub_i) else None,
        "text_unlabeled_map_corr_median_independent": float(np.median(subt_i)) if len(subt_i) else None,
        "n_fire_independent": int(sum(out["models"][k]["fires_v0M_rule"] for k in indep)),
        "n_periodic_sig_independent": int(sum(out["models"][k]["periodic_sig"] for k in indep)),
        "phases_rad": {k: out["models"][k]["band_periodicity"]["phase_P0"] for k in pred_names if k in out["models"]},
        "consensus_phase_P0_rad": cons_ph,
        "consensus_line_centre_mod_P0_mm": float((cons_ph / (2 * np.pi) * P0_MM) % P0_MM),
    }
    if not per_sig:     # deviation D1: no model passes periodic_sig -> apply the R rule to the independent set
        R, pR = R_ind, check1["phase_R_independent_null_p"]
        check1["shared_phase_rule_applied_to"] = "independent ps* models (no model periodic_sig) - deviation D1"
    else:
        pR = check1["phase_R_null_p"]
    check1["shared_phase_verdict"] = ("same physical line positions" if (R >= 0.8 and pR <= 0.01)
                                      else "model-specific" if (R < 0.5 or pR > 0.05) else "intermediate")
    # consensus periodicity with the null
    cons_num = np.mean([mods[k][1]["cell_hi_band"] for k in reg_fwd], 0)
    check1["consensus_band_periodicity"] = periodicity(Profile(cons_num, nb), "consensus_forward_registered")
    # ---- check 2 reverse
    check2 = {}
    for k in rev:
        if k not in out["models"]:
            continue
        mr = out["models"][k]
        rph = mr["band_periodicity"]["phase_P0"]
        fr = [out["models"][f]["band_periodicity"]["T1_ratio_pooled"] for f in reg_fwd]
        d = wrap(rph - cons_ph)
        rev_sig = mr["periodic_sig"]
        H4s = bool(rev_sig and (mr["band_periodicity"]["T1_ratio_pooled"] > max(fr) or abs(d) >= np.pi / 3))
        H4n = bool((not rev_sig) or (abs(d) <= np.pi / 4 and mr["band_periodicity"]["T1_ratio_pooled"] <= np.median(fr)))
        check2 = {"model": k, "registration_auroc": mr["registration_auroc"], "band_fp_frac": mr["band_fp_frac"],
                  "fwd_band_fp_median": float(np.median([out["models"][f]["band_fp_frac"] for f in reg_fwd])),
                  "T1_ratio_pooled": mr["band_periodicity"]["T1_ratio_pooled"], "fwd_T1_max": float(max(fr)),
                  "fwd_T1_median": float(np.median(fr)), "p_tile_T1": mr["band_periodicity"]["p_tile_T1"],
                  "phase_minus_forward_consensus_rad": d, "band_map_corr_with_forward_median":
                      float(np.median([C[ks.index(k), ks.index(f)] for f in reg_fwd])),
                  "H4_support": H4s, "H4_no_support": H4n}
    # ---- check 3 render
    check3 = {}
    if rend:
        o, a = mods[rend[0]]
        prr = Profile(a["cell_val_band"], a["cell_nvalid_band"])
        per_r = periodicity(prr, "render_band")
        cov = float(np.nansum(a["cell_nvalid_band"]) / info["band_px"])
        # 1 mm (4x4 cell) text-block tile means, text block = cells wholly in eval mask and valid
        mean_c = a["cell_val_all"] / np.maximum(a["cell_nvalid_all"], 1)
        sd_c = np.sqrt(np.maximum(a["cell_sq_all"] / np.maximum(a["cell_nvalid_all"], 1) - mean_c ** 2, 0))
        tv = text_full & (a["cell_nvalid_all"] >= FULL - 0.5)
        bv = band_full & (a["cell_nvalid_all"] >= FULL - 0.5)
        H, W = mean_c.shape
        tm = []
        for i in range(0, H - 3, 4):
            for j in range(0, W - 3, 4):
                if tv[i:i + 4, j:j + 4].all():
                    tm.append(mean_c[i:i + 4, j:j + 4].mean())
        band_mean = float(np.nansum(a["cell_val_band"]) / np.nansum(a["cell_nvalid_band"]))
        p1, p99 = np.percentile(tm, [1, 99])
        sd_ratio = float(np.median(sd_c[bv]) / np.median(sd_c[tv]))
        valid_ok = bool(cov >= 0.95 and p1 <= band_mean <= p99 and 0.5 <= sd_ratio <= 2)
        # text block profiles: render, labels, model consensus (cells wholly in eval mask)
        prt = Profile(np.where(tv, a["cell_val_all"], 0), np.where(tv, a["cell_nvalid_all"], 0))
        per_rt = periodicity(prt, "render_text_block")
        lab_t = prt.prof(np.where(tv, st["n_ink"], 0) * 1.0)
        sign_r = corr(bandpass(prt.prof()), bandpass(lab_t))
        sgn = np.sign(sign_r) if abs(sign_r) >= 0.2 else 0
        # render vs model consensus in the band (same rows)
        yr = prr.prof(); rows_common = np.intersect1d(prr.rows, band_pr.rows)
        yr_c = yr[np.isin(prr.rows, rows_common)]; yc_c = cons[np.isin(band_pr.rows, rows_common)]
        rc = corr(bandpass(yr_c), bandpass(yc_c))
        stat = (sgn * rc) if sgn != 0 else abs(rc)
        null = []
        for s in range(N_SURR):
            rng = np.random.default_rng(SEED0 + s)
            q = corr(bandpass(phase_rand(yr_c, rng)), bandpass(yc_c))
            null.append(sgn * q if sgn != 0 else abs(q))
        xr = (rows_common * CELL + CELL / 2) * VOX_MM
        dph = wrap(phase_at(yr_c, xr)[0] + (np.pi if sgn < 0 else 0) - phase_at(yc_c, xr)[0])
        if sgn == 0:
            dph = min(abs(dph), np.pi - abs(dph))
        check3 = {"file": rend[0], "dtype": o["dtype"], "value_range": [o["value_min"], o["value_max"]],
                  "valid_coverage_band": cov, "band_mean_intensity": band_mean,
                  "text_1mm_tile_mean_p1_p99": [float(p1), float(p99)], "band_over_text_within_cell_sd": sd_ratio,
                  "render_valid_in_band": valid_ok,
                  "text_block_periodicity": per_rt,
                  "render_sensitive": bool(per_rt["T1_ratio_pooled"] >= 10 and per_rt["p_tile_T1"] <= 0.01),
                  "render_vs_labels_text_bandpassed_r": sign_r, "render_sign": int(sgn),
                  "band_periodicity": per_r,
                  "render_periodic_band": bool(per_r["T1_ratio_pooled"] >= 10 and per_r["p_tile_T1"] <= 0.01),
                  "render_vs_consensus_band_bandpassed_r_signed": float(stat), "p_phase_random": p_ge(stat, null),
                  "dphase_render_vs_consensus_rad": float(dph)}
        check3["render_matches_models"] = bool(stat >= 0.5 and check3["p_phase_random"] <= 0.01 and abs(dph) <= np.pi / 4)
    # ---- check 4 layout
    zc = st["z_mm"]; ed = st["edge_mm"]; valid = st["valid"].astype(bool)
    dist = zc - edge_z
    ok = valid & (ed >= 2.0) & np.isfinite(mods[reg_fwd[0]][1]["cell_hi_all"])
    hi_cons = np.mean([mods[k][1]["cell_hi_all"] for k in reg_fwd], 0) / FULL
    hi_cons_noink = np.mean([mods[k][1]["cell_hi_noink"] for k in reg_fwd], 0) / np.maximum(FULL - st["n_ink"], 1)
    lab_frac = st["n_ink"] / FULL
    bins = np.arange(-15, np.ceil(np.nanmax(dist[ok])) + 1, 1.0)
    prof4 = []
    for b0 in bins[:-1]:
        m = ok & (dist >= b0) & (dist < b0 + 1)
        if m.sum() < 20:
            continue
        mb = m & band_full
        prof4.append({"d_mm": float(b0), "n_cells": int(m.sum()), "model_hi_frac": float(np.nanmean(hi_cons[m])),
                      "model_hi_frac_unlabeled_px": float(np.nanmean(hi_cons_noink[m])),
                      "label_ink_frac": float(lab_frac[m].mean()),
                      "band_cells": int(mb.sum()), "model_hi_frac_band_cells": float(np.nanmean(hi_cons[mb])) if mb.any() else None})
    near = band_full & ok & (dist >= 2) & (dist < 5); far = band_full & ok & (dist >= 10)
    nf = float(np.nanmean(hi_cons[near]) / np.nanmean(hi_cons[far]))

    def win_profile(num, lo, hi, extra=None):
        m = ok & (dist >= lo) & (dist < hi)
        if extra is not None:
            m &= extra
        return Profile(np.where(m, num * FULL, 0), np.where(m, FULL, 0))

    lat = {}
    for s in (0.95, 1.0, 1.05):
        P = P0_MM * s
        pt = win_profile(hi_cons, -10, 0); pb = win_profile(hi_cons, 2, 12, band_full)
        pl = win_profile(lab_frac, -10, 0)
        phi_t = phase_at(pt.prof(), pt.x_mm, P)[0]; phi_b = phase_at(pb.prof(), pb.x_mm, P)[0]
        phi_l = phase_at(pl.prof(), pl.x_mm, P)[0]
        d = wrap(phi_b - phi_t)
        null = []
        for k in range(N_SURR):
            rng = np.random.default_rng(SEED0 + k)
            null.append(-abs(wrap(phase_at(phase_rand(pb.prof(), rng), pb.x_mm, P)[0] - phi_t)))
        lat[f"{s:.2f}P0"] = {"P_mm": P, "phase_text_top10_models": phi_t, "phase_text_top10_labels": phi_l,
                             "phase_band_first10_models": phi_b, "dphi_band_minus_text_models": d,
                             "dphi_models_minus_labels_text": wrap(phi_t - phi_l),
                             "p_phase_random": p_ge(-abs(d), null),
                             "text_lattice_ok": bool(abs(wrap(phi_t - phi_l)) <= np.pi / 4),
                             "continuity": bool(abs(d) <= np.pi / 4 and p_ge(-abs(d), null) <= 0.01),
                             "T1_text_top10": periodogram_stat(pt.prof(), CMM * 1000).get("ratio"),
                             "T1_band_first10": periodogram_stat(pb.prof(), CMM * 1000).get("ratio"),
                             "line_centre_text_mod_P_mm": float((phi_t / (2 * np.pi) * P) % P),
                             "line_centre_band_mod_P_mm": float((phi_b / (2 * np.pi) * P) % P)}
    # far part of the band periodic?
    pfar = win_profile(np.mean([mods[k][1]["cell_hi_band"] for k in reg_fwd], 0), 10, 99, band_full)
    far_per = periodicity(Profile(np.where(ok & (dist >= 10) & band_full, cons_num, 0),
                                  np.where(ok & (dist >= 10) & band_full, nb, 0)), "consensus_band_far_ge10mm")
    # sliding windows (10 mm, step 2.5 mm) of consensus periodicity
    slide = []
    for lo in np.arange(-20, np.nanmax(dist[ok]) - 5, 2.5):
        pw = win_profile(hi_cons, lo, lo + 10, None if lo + 10 <= 0 else (band_full | (dist < 0)))
        y = pw.prof()
        if len(y) < 16:
            continue
        slide.append({"window_mm": [float(lo), float(lo + 10)], "T1_ratio": periodogram_stat(y, CMM * 1000).get("ratio"),
                      "P0_ratio": p0_ratio(y), "phase_P0": phase_at(y, pw.x_mm)[0], "mean_hi": float(np.mean(y))})
    # along x: 5 mm strips (20 cells)
    strips = []
    fp_cells = np.where(band_full, cons_num, 0)
    tot_fp = fp_cells.sum()
    for j0 in range(0, zc.shape[1], 20):
        sl = slice(j0, j0 + 20)
        n_b = nb[:, sl].sum(); fp = np.nansum(cons_num[:, sl])
        ink_w = st["n_ink"][:, sl]
        rec = {"x_mm": [j0 * CMM, (j0 + 20) * CMM], "band_px": float(n_b), "band_fp_frac": float(fp / n_b) if n_b else None,
               "ink_px": float(ink_w.sum())}
        if ink_w.sum() >= 10000:
            zz = zc[:, sl][ink_w > 0]; ww = ink_w[ink_w > 0]
            o_ = np.argsort(zz); cw = np.cumsum(ww[o_])
            rec["local_edge_z_mm"] = float(zz[o_][np.searchsorted(cw, 0.995 * cw[-1])])
            d_loc = zc[:, sl] - rec["local_edge_z_mm"]
            okk = ok[:, sl]; bf = band_full[:, sl]
            hc = hi_cons[:, sl]
            m_near = okk & bf & (d_loc >= 2) & (d_loc < 5); m_far = okk & bf & (d_loc >= 10)
            rec["near_far_local"] = float(np.nanmean(hc[m_near]) / np.nanmean(hc[m_far])) if m_near.any() and m_far.any() else None
            mt = okk & (d_loc >= -10) & (d_loc < 0); mb = okk & bf & (d_loc >= 2) & (d_loc < 12)
            if mt.sum() > 50 and mb.sum() > 50:
                pt = Profile(np.where(mt, hc, 0), np.where(mt, 1.0, 0)); pb = Profile(np.where(mb, hc, 0), np.where(mb, 1.0, 0))
                if len(pt.rows) >= 16 and len(pb.rows) >= 16:
                    rec["dphi_band_minus_text_local"] = wrap(phase_at(pb.prof(), pb.x_mm)[0] - phase_at(pt.prof(), pt.x_mm)[0])
        strips.append(rec)
    # EXPLORATORY (not pre-registered): FP vs distance above each strip's LOCAL labeled edge, pooled over strips
    dloc = np.full(zc.shape, np.nan)
    for rec, j0 in zip(strips, range(0, zc.shape[1], 20)):
        if "local_edge_z_mm" in rec:
            dloc[:, j0:j0 + 20] = zc[:, j0:j0 + 20] - rec["local_edge_z_mm"]
    loc_prof = []
    for b0 in np.arange(-10, 26, 1.0):
        m = ok & (dloc >= b0) & (dloc < b0 + 1) & ((dloc < 0) | band_full)
        if m.sum() >= 20:
            loc_prof.append({"d_local_mm": float(b0), "n_cells": int(m.sum()), "model_hi_frac": float(np.nanmean(hi_cons[m])),
                             "model_hi_frac_unlabeled_px": float(np.nanmean(hi_cons_noink[m])),
                             "label_ink_frac": float(lab_frac[m].mean())})
    # EXPLORATORY comparator: text heights (labeled-ink z 0.5-99.5 pct, 2 mm inset), outside any eval/supervision mask
    zlo = float(json.loads((OUT / "w031" / "static.json").read_text())["lower"]["ink_z_p0.5_mm"])
    th = ok & (zc >= zlo + 2) & (zc < edge_z - 2)
    out_mask = th & (st["n_ev"] == 0) & (st["n_ink"] == 0)
    in_mask_unl = th & (st["n_ev"] >= FULL - 0.5) & (st["n_ink"] == 0)
    comparator = {"text_heights_outside_masks_cells": int(out_mask.sum()),
                  "model_hi_frac_text_heights_outside_masks": float(np.nanmean(hi_cons[out_mask])) if out_mask.any() else None,
                  "model_hi_frac_text_heights_inside_mask_unlabeled_cells": float(np.nanmean(hi_cons[in_mask_unl])) if in_mask_unl.any() else None,
                  "model_hi_frac_band_cells": float(np.nanmean(hi_cons[band_full & ok])),
                  "model_hi_frac_band_ge13mm_above_global_edge": float(np.nanmean(hi_cons[band_full & ok & (dist >= 13)]))}
    sb = sorted([s for s in strips if s["band_px"] > 0], key=lambda s: -(s["band_fp_frac"] or 0) * s["band_px"])
    bp = np.array([s["band_px"] for s in sb]); fpp = np.array([(s["band_fp_frac"] or 0) * s["band_px"] for s in sb])
    # top 20% of strips BY BAND PX COUNT (i.e., area): smallest set of highest-FP strips covering 20% of band area
    cum_area = np.cumsum(bp) / bp.sum()
    k20 = int(np.searchsorted(cum_area, 0.2)) + 1
    share20 = float(fpp[:k20].sum() / fpp.sum())
    dl = [s["dphi_band_minus_text_local"] for s in strips if "dphi_band_minus_text_local" in s]
    check4 = {"global_edge_z_mm": edge_z, "profile_by_distance_above_edge": prof4,
              "near_far_ratio_2to5_over_ge10": nf, "lattice": lat, "far_band_periodicity": far_per,
              "sliding_10mm_windows": slide, "x_strips_5mm": strips,
              "fp_share_in_top20pct_band_area_strips": share20,
              "EXPLORATORY_profile_by_distance_above_LOCAL_edge": loc_prof,
              "EXPLORATORY_comparator_unlabeled_text_heights": comparator,
              "corr_strip_band_fp_vs_local_edge": corr(np.array([r["band_fp_frac"] for r in strips if r.get("local_edge_z_mm") and r["band_px"] > 1e6]),
                                                     np.array([r["local_edge_z_mm"] for r in strips if r.get("local_edge_z_mm") and r["band_px"] > 1e6])),
              "local_dphi_strips": {"n": len(dl), "values": dl, "R": circ_R(dl), "mean": circ_mean(dl) if dl else None}}
    L = lat["1.00P0"]
    cont_all = all(lat[s]["continuity"] for s in lat)
    check4["verdicts"] = {
        "lattice_continuity_all_P": cont_all,
        "text_block_lattice_ok": L["text_lattice_ok"],
        "H2_support": bool(cont_all), "H2_partial": bool(cont_all and nf >= 2),
        "H2_full": bool(cont_all and nf < 2 and far_per["p_tile_T1"] <= 0.01),
        "H2_contradict": bool(L["text_lattice_ok"] and abs(L["dphi_band_minus_text_models"]) >= np.pi / 2 and nf < 1.5),
        "spread_evenly": bool(0.67 <= nf <= 1.5),
        "H3_localized": bool(share20 >= 0.5 and not cont_all)}
    # H6 edge gradient
    e_near = band_full & ok & (ed >= 2) & (ed < 4); e_far = band_full & ok & (ed >= 6)
    check4["fp_vs_segment_edge"] = {"band_2to4mm": float(np.nanmean(hi_cons[e_near])) if e_near.any() else None,
                                    "band_ge6mm": float(np.nanmean(hi_cons[e_far])) if e_far.any() else None}
    return out, check1, check2, check3, check4


def main():
    t0 = time.time()
    res = json.loads(RES_JSON.read_text())
    models, c1, c2, c3, c4 = analyze_w031()
    geo = geometry()
    res.update({"computed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "per_model": models, "check1_all_models": c1, "check2_reverse": c2, "check3_render": c3,
                "check4_layout": c4, "check5_geometry": geo})
    w029 = OUT / "w029"
    if (w029 / "static.json").exists():
        from tools.margin.forensics_w029 import summarize
        res["check6b_w029"] = summarize()
    RES_JSON.write_text(json.dumps(res, indent=1, default=float))
    print("analysis written", round(time.time() - t0), "s")
