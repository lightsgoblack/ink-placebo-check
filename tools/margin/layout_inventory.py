"""Label-only layout inventory for the PHerc.1667 ink segments (no preds, no renders, no images).

Two questions, both answered from inklabels_v2 + masks + tifxyz geometry only:

1. Intercolumn gaps. Along-line (x) profile of labeled ink inside the text height range; runs of
   near-zero ink bounded by labeled column blocks on both sides are candidate intercolumns (blank by
   scribal convention, same height as the text). Reports width, supervision / validation cover, and
   how many 5 mm tiles would fit.
2. Column-foot orientation. Greek bookhand letters descend below the line (rho, upsilon, phi, psi,
   chi) far more than they ascend, so the average line profile of ink vs scroll height z has its
   tail toward the column foot. Lines are folded at the measured pitch per 10 mm x-strip, phase
   aligned, averaged; the sign of the third moment says which z direction is "down" in the writing.
   Bootstrap over strips (B = 2000, seed 20260925). Also reports how straight each labeled edge is.

Decision rule (fixed before the first run on real data, 2026-09-26): primary statistic = skew of the
folded mean line profile. Foot at HIGH z if the skew 95% CI is > 0 on >= 3 of the 4 segments and no
segment's CI is < 0; foot at LOW z if the mirror holds; otherwise UNDETERMINED. tail_asym is secondary
(it carries a ~ -0.007 bias on symmetric synthetic lines). Synthetic check: planted +z descenders give
skew +0.186 [0.175, 0.196], mirrored -0.187, no descenders -0.002 [-0.009, 0.005].
Intercolumn candidates: gaps 8 to 30 mm wide between labeled blocks >= 5 mm, relative threshold 0.2
of the 95th pct of the 2 mm-smoothed x-profile; wider gaps are reported but treated as possible
unlabeled columns, not intercolumns.

Labels are pooled to a 20 x 20 px grid; each pooled cell takes z from the tifxyz grid at the
matching position (index ratio = grid shape / image shape).

    .venv/bin/python -m tools.margin.layout_inventory --out results/layout_inventory.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import zarr

REPO = Path(__file__).resolve().parents[2]
SEGS = ["w018_20240304144031", "w023_20240304161941", "w029_20251212185248662", "w031_2025122323"]
G = 20                      # pooling factor (px)
UM = 2.399                  # voxel size
CELL_MM = G * UM / 1000.0   # 0.04798 mm per pooled cell
SEED = 20260925


def pooled(path: Path) -> np.ndarray:
    """Fraction of nonzero px per G x G block, read in row strips via the zarr view of the TIFF."""
    store = tifffile.imread(path, aszarr=True)
    try:
        z = zarr.open(store, mode="r")
        H, W = z.shape
        ny, nx = H // G, W // G
        out = np.zeros((ny, nx), np.float32)
        step = 128
        for i in range(0, ny, step):
            j = min(ny, i + step)
            blk = np.asarray(z[i * G:j * G, :nx * G]) != 0
            out[i:j] = blk.reshape(j - i, G, nx, G).mean(axis=(1, 3))
        return out
    finally:
        store.close()


def load(seg: str) -> dict:
    d = REPO / "data" / f"1667_{seg[:4]}" / "small"
    L = pooled(d / f"{seg}_inklabels_v2.tif")
    S = np.maximum(pooled(d / f"{seg}_supervision_mask_v2.tif"), pooled(d / f"{seg}_supervision_mask.tif"))
    V = pooled(d / f"{seg}_validation_mask_v2.tif")
    zt = np.squeeze(tifffile.imread(d / "z.tif")).astype(np.float32)
    xt = np.squeeze(tifffile.imread(d / "x.tif"))
    gh, gw = zt.shape
    with tifffile.TiffFile(d / f"{seg}_inklabels_v2.tif") as t:
        H, W = t.pages[0].shape
    ny, nx = L.shape
    ri = np.clip(np.rint((np.arange(ny) * G + G / 2) * gh / H).astype(int), 0, gh - 1)
    ci = np.clip(np.rint((np.arange(nx) * G + G / 2) * gw / W).astype(int), 0, gw - 1)
    Z = zt[ri][:, ci] * UM / 1000.0
    valid = (xt[ri][:, ci] != -1) & (zt[ri][:, ci] > 0)
    return {"L": L, "S": S, "V": V, "Z": Z, "valid": valid}


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[False], mask, [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def intercolumns(a: dict) -> dict:
    L, S, V, Z, valid = a["L"], a["S"], a["V"], a["Z"], a["valid"]
    ink = valid & (L > 0.02)
    z_lo, z_hi = np.percentile(Z[ink], [2, 98])
    band = valid & (Z >= z_lo) & (Z <= z_hi)
    cnt = band.sum(0)
    prof = np.where(cnt > 0, (L * band).sum(0) / np.maximum(cnt, 1), 0.0)
    k = int(round(2.0 / CELL_MM))
    sm = np.convolve(prof, np.ones(k) / k, mode="same")
    thr = 0.2 * np.percentile(sm[cnt > 0], 95)
    blocks = [(s, e) for s, e in runs(sm > thr) if (e - s) * CELL_MM >= 5.0]
    gaps = []
    for (s0, e0), (s1, e1) in zip(blocks[:-1], blocks[1:]):
        g0, g1 = e0, s1
        sub = band[:, g0:g1]
        w = (g1 - g0) * CELL_MM
        h = (z_hi - z_lo)
        gaps.append({
            "x_mm": [round(g0 * CELL_MM, 1), round(g1 * CELL_MM, 1)], "width_mm": round(w, 1),
            "valid_frac_in_text_height": round(float(sub.mean()), 3) if sub.size else 0.0,
            "label_frac_mean": round(float(L[:, g0:g1][sub].mean()), 5) if sub.any() else None,
            "cells_with_label_frac": round(float((L[:, g0:g1][sub] > 0).mean()), 4) if sub.any() else None,
            "supervised_frac": round(float((S[:, g0:g1][sub] > 0.5).mean()), 3) if sub.any() else None,
            "validation_frac": round(float((V[:, g0:g1][sub] > 0.5).mean()), 3) if sub.any() else None,
            "tiles_5mm_fit_1mm_guard": int(max(0, (w - 2.0) // 5) * max(0, h // 5)),
        })
    centers = [((s + e) / 2) * CELL_MM for s, e in blocks]
    return {"text_height_z_mm": [round(z_lo, 1), round(z_hi, 1)], "segment_x_mm": round(len(prof) * CELL_MM, 1),
            "blocks_x_mm": [[round(s * CELL_MM, 1), round(e * CELL_MM, 1)] for s, e in blocks],
            "block_widths_mm": [round((e - s) * CELL_MM, 1) for s, e in blocks],
            "block_center_spacing_mm": [round(b - a_, 1) for a_, b in zip(centers[:-1], centers[1:])],
            "gaps": gaps, "threshold_rel": 0.2}


def orientation(a: dict, rng: np.random.Generator) -> dict:
    L, Z, valid = a["L"], a["Z"], a["valid"]
    ink = valid & (L > 0.02)
    z_lo, z_hi = np.percentile(Z[ink], [2, 98])
    sw = int(round(10.0 / CELL_MM))
    dz = 0.05
    edges = np.arange(z_lo, z_hi + dz, dz)
    cen = (edges[:-1] + edges[1:]) / 2
    strips, tops, bots = [], [], []
    for x0 in range(0, L.shape[1] - sw + 1, sw):
        m = valid[:, x0:x0 + sw]
        if (L[:, x0:x0 + sw][m] > 0.02).mean() < 0.02:
            continue
        zz, ll = Z[:, x0:x0 + sw][m], L[:, x0:x0 + sw][m]
        num, _ = np.histogram(zz, edges, weights=ll)
        den, _ = np.histogram(zz, edges)
        ok = den > 20
        if ok.mean() < 0.8:
            continue
        p = np.where(ok, num / np.maximum(den, 1), np.nan)
        p = np.interp(cen, cen[ok], p[ok])
        strips.append(p - p.mean())
        zi = zz[ll > 0.02]
        tops.append(np.percentile(zi, 99.5)); bots.append(np.percentile(zi, 0.5))
    if len(strips) < 5:
        return {"n_strips": len(strips), "note": "too few labeled strips"}
    P_grid = np.arange(3.0, 9.0, 0.01)
    def power(p):
        return np.array([abs(np.sum(p * np.exp(-2j * np.pi * cen / P))) for P in P_grid])
    pw = np.mean([power(p) for p in strips], axis=0)
    P = float(P_grid[np.argmax(pw)])
    nb = int(round(P / dz))
    th = np.linspace(-P / 2, P / 2, nb, endpoint=False)
    folded = []
    for p in strips:
        ph = np.angle(np.sum(p * np.exp(-2j * np.pi * cen / P)))
        z0 = (-ph / (2 * np.pi)) * P              # z of the fundamental's maximum (mod P)
        rel = ((cen - z0 + P / 2) % P) - P / 2
        f, _ = np.histogram(rel, np.linspace(-P / 2, P / 2, nb + 1), weights=p)
        c, _ = np.histogram(rel, np.linspace(-P / 2, P / 2, nb + 1))
        folded.append(f / np.maximum(c, 1))
    folded = np.array(folded)

    def skew(f):
        w = f - f.min()
        w = w / w.sum()
        mu = np.sum(w * th)
        sd = np.sqrt(np.sum(w * (th - mu) ** 2))
        return float(np.sum(w * (th - mu) ** 3) / sd ** 3)

    def tail_asym(f):
        w = f - f.min()
        core = w >= 0.5 * w.max()
        idx = np.flatnonzero(core)
        up = w[idx.max() + 1:].sum(); dn = w[:idx.min()].sum()
        return float((up - dn) / w.sum())

    mean_f = folded.mean(0)
    B = 2000
    boot_s, boot_a = np.empty(B), np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(folded), len(folded))
        fb = folded[pick].mean(0)
        boot_s[b], boot_a[b] = skew(fb), tail_asym(fb)
    per_strip_sign = np.sign([skew(f) for f in folded])
    return {
        "n_strips": len(strips), "pitch_mm": round(P, 2),
        "skew_about_line": round(skew(mean_f), 3), "skew_ci95": [round(float(np.quantile(boot_s, q)), 3) for q in (0.025, 0.975)],
        "tail_asym_plusz_minus_minusz": round(tail_asym(mean_f), 4),
        "tail_asym_ci95": [round(float(np.quantile(boot_a, q)), 4) for q in (0.025, 0.975)],
        "frac_strips_skew_positive": round(float((per_strip_sign > 0).mean()), 3),
        "reading": "positive = ink tail toward +z = descenders toward +z = column foot at HIGH z; negative = foot at LOW z",
        "labeled_top_edge_z_mm": {"median": round(float(np.median(tops)), 2), "sd_across_strips": round(float(np.std(tops)), 2)},
        "labeled_bottom_edge_z_mm": {"median": round(float(np.median(bots)), 2), "sd_across_strips": round(float(np.std(bots)), 2)},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "cell_mm": CELL_MM, "inputs": "inklabels_v2, supervision_mask_v2 | supervision_mask, validation_mask_v2, tifxyz x/z; no preds, no renders, no images", "segments": {}}
    for seg in SEGS:
        d = load(seg)
        res["segments"][seg[:4]] = {"intercolumns": intercolumns(d), "orientation": orientation(d, rng)}
        print(seg[:4], json.dumps(res["segments"][seg[:4]])[:1500], flush=True)
        del d
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
