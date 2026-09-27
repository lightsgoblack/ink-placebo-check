"""Edition route (the settled-rules file 2026-09-26): column lattice of PHerc.1667, fitted NUMBERS-ONLY on the org's own
2.4 um ink render (model 20260417190342 `new_canon_autoresearch_recipe`), so that intercolumns and margins can
later be called blank by papyrology (the edition's layout), not by the ink models under audit.

No images anywhere. Never read or transcribe glyphs. Reduced arrays live only in data/edition_lattice/ (gitignored).
One render at a time in data/raw/: reduce, run the tripwire, delete. Pre-registered plan (committed before any
render was downloaded): results/edition_lattice_plan.json. Everything below implements that plan.

    .venv/bin/python -m tools.margin.lattice synth              # synthetic checks (no scroll data)
    .venv/bin/python -m tools.margin.lattice geom --seg w018    # official tifxyz + official label zarrs (small files)
    .venv/bin/python -m tools.margin.lattice reduce --seg w018  # pull the render; pass 1: pooled arrays + registration
    .venv/bin/python -m tools.margin.lattice layout --seg w018  # lattice, z extents, candidate blank regions
    .venv/bin/python -m tools.margin.lattice trip --seg w018    # pass 2: full-res text-like tripwire; delete the raw
    .venv/bin/python -m tools.margin.lattice orient             # pre-set descender-skew rule over all segments
    .venv/bin/python -m tools.margin.lattice report             # results/edition_lattice.{json,md}

Coordinates: the render, the official labels and the official tifxyz share one canvas (checked in `reduce`).
Grid cell (i, j) = canvas px rows [20i, 20i+20), cols [20j, 20j+20) = tifxyz grid point (i, j) (scale 0.05).
x = canvas column (along the text lines, unrolled length); z = tifxyz z (scroll height) in mm.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.stats import spearmanr

from tools.harness.fetch import RAW, free_bytes, sha256
from tools.margin.common import REPO, Lazy2D

# ----------------------------------------------------------------------------------------------- constants
SEG_ORDER = ["w018", "w023", "w029", "w031"]          # = tools/margin/layout_inventory.SEGS order
SEGS = {"w018": "20240304144031-w018_20240304144031_flatboi",
        "w023": "20240304161941-w023_20240304161941_flatboi",
        "w029": "20251212185248-w029_20251212185248662_flatboi",
        "w031": "20251223230000-w031_2025122323_flatboi"}
HF_SEG = {"w018": "w018_20240304144031", "w023": "w023_20240304161941",
          "w029": "w029_20251212185248662", "w031": "w031_2025122323"}
S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1667/segments"
MODEL = "20260417190342-new_canon_autoresearch_recipe"
RENDER = ("ink-detection/PHerc1667-{ts}-2.399um-0.22m-78keV-volume-20251217075048-" + MODEL
          + "-tile256-stride128.tif")
MESH = "mesh/{ts}-on-20251217075048-2.399um.tifxyz/{f}"
LABELS = "ink-labels/2.399um-volume-20251217075048/20260918/{name}.zarr"

G = 20                              # px per grid cell (tifxyz scale 0.05)
UM = 2.399
CELL_MM = G * UM / 1000.0           # 0.04798 mm
STRIP = 2080                        # rows per streamed strip (= 104 cells = 5 x 416 px)
SEED = 20260925
MIN_FREE_GB = 6.0

OUT = REPO / "data" / "edition_lattice"
PLAN_JSON = REPO / "vault" / "results" / "edition_lattice_plan.json"
RES_JSON = REPO / "vault" / "results" / "edition_lattice.json"
RES_MD = REPO / "vault" / "results" / "edition_lattice.md"

# lattice (plan section 3)
SMOOTH_MM = 3.0
T_BLOCK, T_HALF, T_CORE = 0.3, 0.5, 0.1
MERGE_GAP_MM, MIN_BLOCK_MM = 4.0, 5.0
GAP_MIN_MM, GAP_MAX_MM = 4.0, 30.0
LONG_MM = 150.0
P_LO, P_HI, P_STEP = 40.0, 110.0, 0.05
RES_TOL = 0.2
N_MC, MIN_SEP_MM = 2000, 9.0
BLOCK_SHUF_MM, N_SURR = 16.0, 200
EDITION_P, DIEGO_P, ROLL_MM = 64.0, 75.0, 1400.0
# z extents (plan section 4)
DZ = 0.1
PLINE = np.arange(3.0, 9.0, 0.01)
MIN_COL_MM = 10.0
COL_GUARD_MM = 1.0
SUB_MM = 5.0
MIN_CONTRAST = 0.25
# candidates (plan section 6)
CORE_SHRINK_MM, CORE_MIN_MM = 0.5, 2.0
MARGIN_GUARD_MM, EDGE_GUARD_MM = 2.0, 2.0
TAPER_Z_MM = 81.0
TILE_PX = 2084


def mm(n_cells):
    return float(n_cells) * CELL_MM


def cells(n_mm):
    return int(round(n_mm / CELL_MM))


def jdump(o, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(o, indent=1, default=_js))


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def r(x, n=2):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


# ----------------------------------------------------------------------------------------------- downloads
def http_get(url: str, dest: Path, expect: int | None = None) -> Path:
    """Single-file HTTPS download with resume; retries after 2/4/8/16 s. Verifies the byte count."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expect is None or dest.stat().st_size == expect):
        return dest
    part = dest.with_name(dest.name + ".part")
    last = None
    for wait in (0, 2, 4, 8, 16):
        time.sleep(wait)
        try:
            have = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=120) as resp:
                mode = "ab" if (have and resp.status == 206) else "wb"
                with open(part, mode) as f:
                    while True:
                        b = resp.read(1 << 22)
                        if not b:
                            break
                        f.write(b)
            got = part.stat().st_size
            if expect is not None and got != expect:
                raise IOError(f"got {got} bytes, expected {expect}")
            part.rename(dest)
            return dest
        except Exception as e:          # noqa: BLE001 (network: retry)
            last = e
            print(f"  retry after error: {e}", flush=True)
    raise SystemExit(f"download failed: {url}: {last}")


def head_size(url: str) -> int:
    for wait in (0, 2, 4, 8, 16):
        time.sleep(wait)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as resp:
                return int(resp.headers["Content-Length"])
        except Exception as e:          # noqa: BLE001
            last = e
    raise SystemExit(f"HEAD failed: {url}: {last}")


def check_disk(need_bytes: int = 0) -> float:
    free = free_bytes(REPO)
    if free - need_bytes < MIN_FREE_GB * 2**30:
        raise SystemExit(f"STOP: {free / 2**30:.1f} GB free; need {need_bytes / 2**30:.2f} GB + {MIN_FREE_GB} GB reserve")
    return free / 2**30


def render_url(short):
    ts = SEGS[short].split("-")[0]
    return f"{S3}/{SEGS[short]}/" + RENDER.format(ts=ts)


def raw_path(short):
    return RAW / Path(RENDER.format(ts=SEGS[short].split("-")[0])).name


# ----------------------------------------------------------------------------------------------- geometry
def geom(short):
    d = OUT / short / "official"
    ts = SEGS[short].split("-")[0]
    base = f"{S3}/{SEGS[short]}/"
    for f in ("meta.json", "x.tif", "y.tif", "z.tif"):
        http_get(base + MESH.format(ts=ts, f=f), d / "tifxyz" / f)
    for name in ("inklabels", "supervision", "validation"):
        b = base + LABELS.format(name=name)
        http_get(f"{b}/zarr.json", d / f"{name}.zarr" / "zarr.json")
        http_get(f"{b}/0/zarr.json", d / f"{name}.zarr" / "0" / "zarr.json")
        http_get(f"{b}/0/c/0/0", d / f"{name}.zarr" / "0" / "c" / "0" / "0")
    attrs = {n: json.loads((d / f"{n}.zarr" / "zarr.json").read_text())["attributes"].get("ink_republish")
             for n in ("inklabels", "supervision", "validation")}
    jdump(attrs, d / "republish_attrs.json")
    print(short, "geometry + labels ok", {k: (v or {}).get("source_raster") for k, v in attrs.items()})


def label_array(short, name):
    import zarr
    return zarr.open_array(str(OUT / short / "official" / f"{name}.zarr" / "0"), mode="r")


def load_grid(short, gh, gw):
    """Official tifxyz on the pooled grid: z (mm), 3D points (voxels), valid mask. Cropped / padded to (gh, gw)."""
    d = OUT / short / "official" / "tifxyz"
    arr = {c: np.squeeze(tifffile.imread(d / f"{c}.tif")).astype(np.float32) for c in "xyz"}
    h, w = arr["z"].shape
    note = None if (h, w) == (gh, gw) else f"tifxyz grid {h}x{w} vs pooled {gh}x{gw}: cropped/padded"
    out = {}
    for c, a in arr.items():
        b = np.full((gh, gw), -1.0, np.float32)
        b[:min(h, gh), :min(w, gw)] = a[:min(h, gh), :min(w, gw)]
        out[c] = b
    valid = (out["x"] != -1) & (out["z"] > 0) & np.isfinite(out["z"])
    Z = np.where(valid, out["z"] * UM / 1000.0, np.nan).astype(np.float32)
    return {"Z": Z, "valid": valid, "P": np.stack([out["x"], out["y"], out["z"]], -1), "shape_tifxyz": [h, w],
            "note": note}


# ----------------------------------------------------------------------------------------------- pass 1
def value_scale(dt):
    dt = np.dtype(dt)
    if dt == np.uint8:
        return 127, 255.0, 256
    if dt == np.uint16:
        return 32767, 65535.0, 65536
    if dt.kind == "f":
        return 0.5, 1.0, 1001
    raise SystemExit(f"unexpected render dtype {dt}")


def reduce(short):
    d = OUT / short
    url = render_url(short)
    size = head_size(url)
    free = check_disk(size)
    raw = raw_path(short)
    t0 = time.time()
    print(f"{short}: pulling render {size / 2**20:.0f} MB ({free:.1f} GB free)", flush=True)
    http_get(url, raw, size)
    dl_s = time.time() - t0
    digest = sha256(raw)
    lz = Lazy2D(raw)
    lab, sup, val = (label_array(short, n) for n in ("inklabels", "supervision", "validation"))
    H, W = lab.shape
    info = dict(lz.info)
    if tuple(lz.shape) != (H, W):
        lz.close()
        res = {"render_shape": list(lz.shape), "label_canvas": [H, W], "excluded": "render is not on the label canvas;"
               " plan: map via tifxyz. Not implemented for this case -> segment stopped, reported."}
        jdump(res, d / "reduce.json")
        raise SystemExit(json.dumps(res))
    thr, scale, nb = value_scale(lz.dtype)
    gh, gw = H // G, W // G
    Hc, Wc = gh * G, gw * G
    S = np.zeros((gh, gw), np.float64)          # sum of values per cell
    nhi = np.zeros((gh, gw), np.uint16)
    nlab = np.zeros((gh, gw), np.uint16)
    nev = np.zeros((gh, gw), np.uint16)
    h_pos = np.zeros(nb, np.int64)
    h_neg = np.zeros(nb, np.int64)
    vmin, vmax = np.inf, -np.inf
    for r0 in range(0, Hc, STRIP):
        r1 = min(Hc, r0 + STRIP)
        k = (r1 - r0) // G
        i0 = r0 // G
        a = lz.rows(r0, r1)[:, :Wc]
        vmin, vmax = min(vmin, float(a.min())), max(vmax, float(a.max()))
        S[i0:i0 + k] = a.reshape(k, G, gw, G).sum(axis=(1, 3), dtype=np.float64)
        hi = a > thr
        nhi[i0:i0 + k] = hi.reshape(k, G, gw, G).sum(axis=(1, 3))
        del hi
        L = np.asarray(lab[r0:r1, :Wc]) > 0
        E = (np.asarray(sup[r0:r1, :Wc]) > 0) | (np.asarray(val[r0:r1, :Wc]) > 0)
        nlab[i0:i0 + k] = L.reshape(k, G, gw, G).sum(axis=(1, 3))
        nev[i0:i0 + k] = E.reshape(k, G, gw, G).sum(axis=(1, 3))
        q = a if np.dtype(lz.dtype).kind != "f" else np.clip(np.rint(a * 1000), 0, 1000).astype(np.int64)
        h_pos += np.bincount(q[L & E].ravel(), minlength=nb)[:nb]
        h_neg += np.bincount(q[~L & E].ravel(), minlength=nb)[:nb]
        del a, L, E, q
        print(f"  pass1 rows {r1}/{Hc} {time.time() - t0:.0f}s", flush=True)
    lz.close()
    mean = (S / (G * G * scale)).astype(np.float32)
    np.savez_compressed(d / "pooled.npz", mean=mean, nhi=nhi, nlab=nlab, nev=nev, h_pos=h_pos, h_neg=h_neg)
    # registration (plan section 1)
    cn = np.cumsum(h_neg) - h_neg
    n_pos, n_neg = int(h_pos.sum()), int(h_neg.sum())
    auc = float((h_pos * (cn + 0.5 * h_neg)).sum() / (n_pos * n_neg)) if n_pos and n_neg else float("nan")
    hib = np.arange(nb) > (thr * (1000 if np.dtype(lz.dtype).kind == "f" else 1))
    hf_lab = REPO / "data" / f"1667_{short}" / "small" / f"{HF_SEG[short]}_inklabels_v2.tif"
    hf_count = None
    if hf_lab.exists():
        hl = Lazy2D(hf_lab)
        hf_count = 0
        for r0 in range(0, hl.shape[0], 4096):
            hf_count += int((hl.rows(r0, min(hl.shape[0], r0 + 4096)) > 0).sum())
        hf_shape = list(hl.shape)
        hl.close()
    res = {"segment": short, "render_url": url, "render_bytes": size, "render_sha256": digest,
           "download_s": round(dl_s, 1), "file_info": info, "dtype": str(np.dtype(lz.dtype)),
           "value_min": vmin, "value_max": vmax, "p05_threshold_raw": thr,
           "canvas_hw": [H, W], "grid_hw": [gh, gw], "render_shape_equals_label_canvas": True,
           "official_label_px": int(nlab.sum(dtype=np.int64)), "hf_inklabels_v2_px": hf_count,
           "hf_inklabels_v2_shape": hf_shape if hf_lab.exists() else None,
           "registration_auroc": auc, "registration_ok": bool(auc >= 0.75),
           "recall_p05_labeled_ink": float(h_pos[hib].sum() / n_pos) if n_pos else None,
           "fpr_p05_unlabeled_in_mask": float(h_neg[hib].sum() / n_neg) if n_neg else None,
           "n_pos_px": n_pos, "n_neg_px": n_neg, "pass1_s": round(time.time() - t0, 1)}
    jdump(res, d / "reduce.json")
    print(json.dumps({k: res[k] for k in ("registration_auroc", "registration_ok", "official_label_px",
                                          "hf_inklabels_v2_px", "canvas_hw", "dtype")}), flush=True)
    return res


def shift_diag(mean, nlab, nev, span=10):
    """Pearson r (render cell mean vs label fraction) on fully evaluated cells, over integer cell shifts."""
    ev = nev >= G * G
    rows, cols = np.nonzero(ev)
    i0, i1 = max(0, rows.min() - span), min(ev.shape[0], rows.max() + span + 1)
    j0, j1 = max(0, cols.min() - span), min(ev.shape[1], cols.max() + span + 1)
    X = mean[i0:i1, j0:j1].astype(np.float64)
    Y = nlab[i0:i1, j0:j1].astype(np.float64) / (G * G)
    M = ev[i0:i1, j0:j1]
    best, grid = None, {}
    for dy in range(-span, span + 1):
        for dx in range(-span, span + 1):
            ys = slice(max(0, dy), Y.shape[0] + min(0, dy)); xs = slice(max(0, -dy), X.shape[0] + min(0, -dy))
            yc = slice(max(0, dx), Y.shape[1] + min(0, dx)); xc = slice(max(0, -dx), X.shape[1] + min(0, -dx))
            m = M[ys, yc] & M[xs, xc]
            a, b = X[xs, xc][m], Y[ys, yc][m]
            rr = float(np.corrcoef(a, b)[0, 1]) if a.size > 100 else float("nan")
            grid[(dy, dx)] = rr
            if best is None or rr > best[0]:
                best = (rr, dy, dx)
    return {"r_at_0": grid[(0, 0)], "r_best": best[0], "best_shift_cells_dy_dx": [best[1], best[2]],
            "best_shift_mm": [round(best[1] * CELL_MM, 3), round(best[2] * CELL_MM, 3)], "search_cells": span,
            "note": "shift applied to labels; label cell (i+dy, j+dx) vs render cell (i, j)"}


# ----------------------------------------------------------------------------------------------- 1-D helpers
def nan_boxcar(y, w):
    w = max(1, int(w))
    ok = np.isfinite(y)
    v = np.where(ok, y, 0.0)
    k = np.ones(w)
    num = np.convolve(v, k, mode="same")
    den = np.convolve(ok.astype(float), k, mode="same")
    out = num / np.maximum(den, 1e-9)
    out[(den < 0.5 * w) | ~ok] = np.nan
    return out


def rle(lab):
    idx = np.flatnonzero(np.diff(lab)) + 1
    st = np.r_[0, idx]
    en = np.r_[idx, len(lab)]
    return [[int(lab[a]), int(a), int(b)] for a, b in zip(st, en)]


def merge_runs(runs):
    out = []
    for t, a, b in runs:
        if out and out[-1][0] == t:
            out[-1][2] = b
        else:
            out.append([t, a, b])
    return out


NAN, LO, HI = 0, 1, 2


def blocks_and_gaps(s, t):
    """Runs of s >= t (HI), s < t (LO), NaN; cleaned per plan 3 (short gaps merged, short islands dissolved)."""
    lab = np.where(np.isnan(s), NAN, np.where(s >= t, HI, LO))
    runs = rle(lab)
    for i in range(1, len(runs) - 1):
        t_, a, b = runs[i]
        if t_ == LO and runs[i - 1][0] == HI and runs[i + 1][0] == HI and mm(b - a) < MERGE_GAP_MM:
            runs[i][0] = HI
    runs = merge_runs(runs)
    islands = []
    for i in range(1, len(runs) - 1):
        t_, a, b = runs[i]
        if t_ == HI and runs[i - 1][0] == LO and runs[i + 1][0] == LO and mm(b - a) < MIN_BLOCK_MM:
            runs[i][0] = LO
            islands.append([round(mm(a), 1), round(mm(b), 1)])
    return merge_runs(runs), islands


def rayleigh(c, Pg):
    return np.abs(np.exp(2j * np.pi * np.asarray(c)[None, :] / Pg[:, None]).mean(1))


def hardcore_uniform(rng, n, x0, x1, sep, tries=10000):
    for _ in range(tries):
        u = np.sort(rng.uniform(x0, x1, n))
        if n < 2 or np.diff(u).min() >= sep:
            return u
    return np.sort(rng.uniform(x0, x1, n))


def lattice_fit(c, x0, x1, rng):
    """Rayleigh scan + least-squares lattice on gap centres (plan 3)."""
    c = np.asarray(sorted(c), float)
    Pg = np.arange(P_LO, P_HI + 1e-9, P_STEP)
    R = rayleigh(c, Pg)
    kR = int(np.argmax(R))
    P_R = float(Pg[kR])
    phiR = float(np.angle(np.exp(2j * np.pi * c / P_R).mean()) / (2 * np.pi) * P_R)
    nodes = np.round((c - phiR) / P_R).astype(int)

    def ls(cc, kk):
        X = np.c_[np.ones(len(kk)), kk]
        beta, *_ = np.linalg.lstsq(X, cc, rcond=None)
        res = cc - X @ beta
        se = None
        if len(set(kk.tolist())) >= 3 and len(cc) > 2:
            s2 = (res ** 2).sum() / (len(cc) - 2)
            cov = s2 * np.linalg.inv(X.T @ X)
            se = float(np.sqrt(cov[1, 1]))
        return beta, res, se

    beta, res, se = ls(c, nodes)
    P1 = beta[1] if len(set(nodes.tolist())) >= 2 else P_R
    consistent = np.abs(res) <= RES_TOL * P1
    for kk in set(nodes.tolist()):                        # duplicate node: keep the smaller residual
        idx = np.flatnonzero(nodes == kk)
        if len(idx) > 1:
            keep = idx[np.argmin(np.abs(res[idx]))]
            consistent[idx] = False
            consistent[keep] = abs(res[keep]) <= RES_TOL * P1
    if consistent.sum() >= 2 and len(set(nodes[consistent].tolist())) >= 2:
        beta, res_c, se = ls(c[consistent], nodes[consistent])
        res = c - (beta[0] + beta[1] * nodes)
    n = len(c)
    rmax = []
    for _ in range(N_MC):
        u = hardcore_uniform(rng, n, x0, x1, MIN_SEP_MM)
        rmax.append(rayleigh(u, Pg).max())
    p = float((1 + np.sum(np.array(rmax) >= R[kR])) / (1 + N_MC))
    ncons = int(consistent.sum())
    return {"n_gaps": n, "centres_mm": c.round(2).tolist(), "P_rayleigh_mm": round(P_R, 2),
            "R_max": round(float(R[kR]), 4), "p_R_hardcore_null": round(p, 4), "null_draws": N_MC,
            "node_index": nodes.tolist(), "consistent": consistent.tolist(), "n_consistent": ncons,
            "P_fit_mm": round(float(beta[1]), 2) if ncons >= 2 else None, "P_fit_se_mm": r(se, 2) if se else None,
            "phase_mm": round(float(beta[0]), 2) if ncons >= 2 else None,
            "residuals_mm": res.round(2).tolist(),
            "resid_rms_consistent_mm": r(np.sqrt(np.mean(res[consistent] ** 2)), 2) if ncons else None,
            "lattice_detected": bool(ncons >= 3 and p <= 0.05)}


def periodogram_x(s, x_mm, rng):
    ok = np.isfinite(s)
    y = s[ok] - s[ok].mean()
    xx = x_mm[ok]
    Pg = np.arange(P_LO, P_HI + 1e-9, 0.1)
    Emat = np.exp(-2j * np.pi * xx[None, :] / Pg[:, None])
    pw = np.abs(Emat @ y) ** 2
    k = int(np.argmax(pw))
    bl = cells(BLOCK_SHUF_MM)
    nblk = len(y) // bl
    blocks = [y[i * bl:(i + 1) * bl] for i in range(nblk)]
    tail = y[nblk * bl:]
    null = []
    for s_ in range(N_SURR):
        rg = np.random.default_rng(SEED + s_)
        perm = rg.permutation(nblk)
        ys = np.concatenate([blocks[q] for q in perm] + [tail])
        null.append((np.abs(Emat @ (ys - ys.mean())) ** 2).max())
    return {"P_periodogram_mm": round(float(Pg[k]), 1), "power_ratio_to_median": r(pw[k] / np.median(pw), 2),
            "p_block_shuffle": round(float((1 + np.sum(np.array(null) >= pw[k])) / (1 + N_SURR)), 4),
            "phase_min_mm": round(float((-np.angle(Emat[k] @ y) / (2 * np.pi) * Pg[k] + Pg[k] / 2) % Pg[k]), 2)}


# ----------------------------------------------------------------------------------------------- z extents
def zprofile(prob, Z, valid, j0, j1, min_n=20, edges=None):
    sub = valid[:, j0:j1]
    zz = Z[:, j0:j1][sub].astype(np.float64)
    pp = prob[:, j0:j1][sub].astype(np.float64)
    if zz.size == 0:
        return None
    if edges is None:
        edges = np.arange(np.floor(zz.min() / DZ) * DZ, zz.max() + DZ, DZ)
    num, _ = np.histogram(zz, edges, weights=pp)
    den, _ = np.histogram(zz, edges)
    prof = np.where(den >= min_n, num / np.maximum(den, 1), np.nan)
    return {"cen": (edges[:-1] + edges[1:]) / 2, "prof": prof, "den": den, "edges": edges, "zz": zz}


def pitch(cen, prof, z5, z95):
    core = (cen >= z5) & (cen <= z95) & np.isfinite(prof)
    if core.sum() < 50:
        return None, None, None
    x, y = cen[core], prof[core]
    y = y - np.polyval(np.polyfit(x, y, 1), x)
    C = np.exp(-2j * np.pi * x[None, :] / PLINE[:, None]) @ y
    pw = np.abs(C) ** 2
    k = int(np.argmax(pw))
    P = float(PLINE[k])
    z0 = float((-np.angle(C[k]) / (2 * np.pi)) * P)
    return P, float(pw[k] / np.median(pw)), z0


def edge_search(s, cen, z_start, direction, level, P_line):
    """Walk from z_start (inside text) in `direction`; edge = first place where s stays < level for >= one
    line pitch. NaN runs > 2 mm end the walk ('missing data'); running off the profile = 'end of data'."""
    need = max(1, int(round(P_line / DZ)))
    i = int(np.clip(np.searchsorted(cen, z_start), 0, len(cen) - 1))
    below, nanrun, first_below, last_ok = 0, 0, None, i
    while 0 <= i < len(s):
        v = s[i]
        if not np.isfinite(v):
            nanrun += 1
            if nanrun * DZ > 2.0:
                return float(cen[last_ok] + direction * DZ / 2), "missing_data"
        else:
            nanrun = 0
            if v < level:
                if below == 0:
                    first_below = i
                below += 1
                if below >= need:
                    return float(cen[first_below] - direction * DZ / 2), "drop"
            else:
                below = 0
                last_ok = i
        i += direction
    return float(cen[last_ok] + direction * DZ / 2), "end_of_data"


def column_extent(prob, Z, valid, j0, j1, z5, z95, xlev):
    """Text extent in z for one column block (plan 4)."""
    zp = zprofile(prob, Z, valid, j0, j1)
    if zp is None:
        return {"status": "no valid cells"}
    cen, prof = zp["cen"], zp["prof"]
    P, strength, z0 = pitch(cen, prof, z5, z95)
    if P is None:
        return {"status": "too little text-band data"}
    w = max(3, int(round(P / DZ)))
    s = nan_boxcar(prof, w)
    core = (cen >= z5) & (cen <= z95) & np.isfinite(s)
    c = float(np.nanmedian(s[core]))
    b = float(np.nanpercentile(s[np.isfinite(s)], 5))
    out = {"x_mm": [round(mm(j0), 1), round(mm(j1), 1)], "width_mm": round(mm(j1 - j0), 1),
           "line_pitch_mm": round(P, 2), "pitch_strength": r(strength, 1), "core_level": r(c, 4), "bg_level": r(b, 4),
           "contrast_rel_x": r((c - b) / xlev, 3) if xlev > 0 else None}
    if xlev <= 0 or (c - b) < MIN_CONTRAST * xlev:
        out["status"] = "low contrast (no clear text)"
        return out
    E = {}
    for q in (0.2, 0.5, 0.8):
        lev = b + q * (c - b)
        E[q] = (edge_search(s, cen, z5, -1, lev, P), edge_search(s, cen, z95, +1, lev, P))
    zlo, zhi = E[0.5][0][0], E[0.5][1][0]
    # sub-strips (5 mm) for raggedness
    ns = int(round(SUB_MM / CELL_MM))
    sub_lo, sub_hi = [], []
    for a in range(j0, j1 - ns + 1, ns):
        zs = zprofile(prob, Z, valid, a, a + ns, min_n=10, edges=zp["edges"])
        if zs is None:
            continue
        ss = nan_boxcar(zs["prof"], w)
        cc = (cen >= z5) & (cen <= z95)
        if np.isfinite(ss[cc]).mean() < 0.5:
            continue
        lev = b + 0.5 * (c - b)
        el, fl = edge_search(ss, cen, z5, -1, lev, P)
        eh, fh = edge_search(ss, cen, z95, +1, lev, P)
        sub_lo.append(el)
        sub_hi.append(eh)
    zz = zp["zz"]
    zv_lo, zv_hi = float(np.percentile(zz, 0.5)), float(np.percentile(zz, 99.5))

    def rsd(v):
        if len(v) < 3:
            return None
        q1, q3 = np.percentile(v, [25, 75])
        return round(float((q3 - q1) / 1.349), 2)

    centres = [z0 + k * P for k in range(int(np.floor((zlo - z0) / P)) - 1, int(np.ceil((zhi - z0) / P)) + 2)]
    centres = [float(zc) for zc in centres if zlo <= zc <= zhi]
    out.update({
        "status": "ok",
        "z_lo_mm": round(zlo, 2), "z_lo_how": E[0.5][0][1], "z_hi_mm": round(zhi, 2), "z_hi_how": E[0.5][1][1],
        "edge_width_lo_mm": r(abs(E[0.2][0][0] - E[0.8][0][0]), 2), "edge_width_hi_mm": r(abs(E[0.2][1][0] - E[0.8][1][0]), 2),
        "text_extent_mm": round(zhi - zlo, 2), "lines_est": round((zhi - zlo) / P, 1),
        "line_centres_mm": [round(v, 2) for v in centres], "n_line_centres": len(centres),
        "first_line_centre_mm": r(min(centres), 2) if centres else None,
        "last_line_centre_mm": r(max(centres), 2) if centres else None,
        "n_substrips": len(sub_lo), "ragged_lo_mm": rsd(sub_lo), "ragged_hi_mm": rsd(sub_hi),
        "sub_edges_lo_mm": [round(v, 2) for v in sub_lo], "sub_edges_hi_mm": [round(v, 2) for v in sub_hi],
        "mesh_z_lo_mm": round(zv_lo, 2), "mesh_z_hi_mm": round(zv_hi, 2),
        "dist_to_mesh_end_lo_mm": round(zlo - zv_lo, 2), "dist_to_mesh_end_hi_mm": round(zv_hi - zhi, 2)})
    return out


# ----------------------------------------------------------------------------------------------- layout
def load_pooled(short):
    with np.load(OUT / short / "pooled.npz") as f:
        return {k: f[k] for k in f.files}


def text_band(nlab, Z, valid):
    ink = valid & (nlab.astype(np.float32) / (G * G) > 0.02)
    z5, z95 = np.percentile(Z[ink], [5, 95])
    return float(z5), float(z95), int(ink.sum())


def x_profile(prob, Z, valid, z5, z95):
    band = valid & (Z >= z5) & (Z <= z95)
    n = band.sum(0)
    m = np.where(n > 0, (np.where(band, prob, 0)).sum(0) / np.maximum(n, 1), np.nan)
    nref = np.percentile(n[n > 0], 90)
    m[n < 0.5 * nref] = np.nan
    return m, n


def layout(short, P_pool=None):
    d = OUT / short
    pooled = load_pooled(short)
    prob = pooled["mean"]
    gh, gw = prob.shape
    geo = load_grid(short, gh, gw)
    Z, valid = geo["Z"], geo["valid"]
    red = json.loads((d / "reduce.json").read_text())
    rng = np.random.default_rng(SEED)
    res = {"segment": short, "grid_hw": [gh, gw], "tifxyz_grid_hw": geo["shape_tifxyz"], "grid_note": geo["note"],
           "valid_frac": round(float(valid.mean()), 4)}
    res["registration_shift_diag"] = shift_diag(prob, pooled["nlab"], pooled["nev"])
    # z direction of canvas rows
    rowz = np.array([np.nanmedian(Z[i][valid[i]]) if valid[i].any() else np.nan for i in range(gh)])
    okr = np.isfinite(rowz)
    res["canvas_rows_vs_z_spearman"] = r(float(spearmanr(np.flatnonzero(okr), rowz[okr])[0]), 3)
    z5, z95, n_ink = text_band(pooled["nlab"], Z, valid)
    res["text_band_from_labels_z_mm"] = [round(z5, 2), round(z95, 2)]
    res["label_ink_cells"] = n_ink
    # --- x-profile and blocks
    m, nband = x_profile(prob, Z, valid, z5, z95)
    s3 = nan_boxcar(m, cells(SMOOTH_MM) | 1)
    x_mm = (np.arange(gw) + 0.5) * CELL_MM
    lo, hi = np.nanpercentile(s3, [5, 95])
    tb, t50, t10 = lo + T_BLOCK * (hi - lo), lo + T_HALF * (hi - lo), lo + T_CORE * (hi - lo)
    runs, islands = blocks_and_gaps(s3, tb)
    res["x_profile"] = {"lo": r(lo, 4), "hi": r(hi, 4), "t_block": r(tb, 4), "t50": r(t50, 4), "t10": r(t10, 4),
                        "valid_len_mm": round(mm(np.isfinite(s3).sum()), 1), "segment_len_mm": round(mm(gw), 1),
                        "nan_len_mm": round(mm((~np.isfinite(s3)).sum()), 1), "ink_islands_dissolved_mm": islands}
    # sensitivity: block/gap edges at 0.35 and 0.65
    sens = {}
    for q in (0.2, 0.4, 0.5):
        rr, _ = blocks_and_gaps(s3, lo + q * (hi - lo))
        sens[str(q)] = [[{HI: "block", LO: "gap", NAN: "nodata"}[t], round(mm(a), 1), round(mm(b), 1)] for t, a, b in rr]
    res["x_runs_sensitivity"] = sens
    blocks, gaps = [], []
    for i, (t, a, b) in enumerate(runs):
        left = runs[i - 1][0] if i > 0 else None
        right = runs[i + 1][0] if i + 1 < len(runs) else None
        if t == HI:
            blocks.append({"idx": i, "j": [a, b], "x_mm": [round(mm(a), 1), round(mm(b), 1)], "width_mm": round(mm(b - a), 1),
                           "touches_end_or_nodata": bool(left in (None, NAN) or right in (None, NAN))})
        elif t == LO:
            w = mm(b - a)
            if left == HI and right == HI:
                cls = "intercolumn_candidate" if GAP_MIN_MM <= w <= GAP_MAX_MM else ("wide" if w > GAP_MAX_MM else "narrow")
            else:
                cls = "edge"
            seg_s3 = s3[a:b]
            jm = a + int(np.nanargmin(seg_s3)) if np.isfinite(seg_s3).any() else (a + b) // 2
            ja_, jb_ = jm, jm
            while ja_ > 0 and np.isfinite(s3[ja_ - 1]) and s3[ja_ - 1] < t50:
                ja_ -= 1
            while jb_ < gw - 1 and np.isfinite(s3[jb_ + 1]) and s3[jb_ + 1] < t50:
                jb_ += 1
            gaps.append({"idx": i, "j": [a, b], "x_mm": [round(mm(a), 1), round(mm(b), 1)], "width_mm": round(w, 1),
                         "width_t50_mm": round(mm(jb_ + 1 - ja_), 1),
                         "centre_mm": round(mm((a + b) / 2), 2), "class": cls,
                         "min_level_rel": r((np.nanmin(seg_s3) - lo) / (hi - lo), 3),
                         "mean_level_rel": r((np.nanmean(seg_s3) - lo) / (hi - lo), 3)})
    res["blocks"], res["gaps"] = blocks, gaps
    icg = [g for g in gaps if g["class"] == "intercolumn_candidate"]
    fin = np.flatnonzero(np.isfinite(s3))
    x0, x1 = mm(fin.min()), mm(fin.max() + 1)
    L = res["x_profile"]["valid_len_mm"]
    lat = {"valid_len_mm": L, "long_segment": bool(L >= LONG_MM)}
    if L >= LONG_MM and len(icg) >= 2:
        lat.update(lattice_fit([g["centre_mm"] for g in icg], x0, x1, rng))
    elif len(icg) >= 2 and P_pool:
        sp = np.diff(sorted(g["centre_mm"] for g in icg))
        lat["spacings_mm"] = sp.round(2).tolist()
        lat["spacing_vs_P_pool"] = [{"spacing": round(float(v), 2), "k": int(max(1, round(v / P_pool))),
                                     "ratio": round(float(v / (P_pool * max(1, round(v / P_pool)))), 3),
                                     "consistent_15pct": bool(abs(v / (P_pool * max(1, round(v / P_pool))) - 1) <= 0.15
                                                              and round(v / P_pool) in (1, 2))} for v in sp]
    lat["periodogram"] = periodogram_x(s3, x_mm, rng) if L >= LONG_MM else None
    res["lattice"] = lat
    # lattice consistency per gap
    for g in icg:
        if lat.get("P_fit_mm"):
            k = round((g["centre_mm"] - lat["phase_mm"]) / lat["P_fit_mm"])
            g["lattice_residual_mm"] = round(g["centre_mm"] - (lat["phase_mm"] + k * lat["P_fit_mm"]), 2)
            g["lattice_consistent"] = bool(abs(g["lattice_residual_mm"]) <= RES_TOL * lat["P_fit_mm"])
        elif lat.get("spacing_vs_P_pool"):
            g["lattice_consistent"] = bool(all(v["consistent_15pct"] for v in lat["spacing_vs_P_pool"]))
        else:
            g["lattice_consistent"] = None
    # --- z extents per column block
    g1 = cells(COL_GUARD_MM)
    xlev = hi - lo
    cols = []
    for bk in blocks:
        a, b = bk["j"]
        if mm(b - a) < MIN_COL_MM:
            continue
        ce = column_extent(prob, Z, valid, a + g1, b - g1, z5, z95, xlev)
        ce["block_idx"] = bk["idx"]
        ce["touches_end_or_nodata"] = bk["touches_end_or_nodata"]
        cols.append(ce)
    res["columns"] = cols
    res["_internal"] = {"runs": runs, "t10": float(t10), "lo": float(lo), "hi": float(hi)}
    # --- candidates
    cand = candidates(short, res, prob, Z, valid, geo["P"], s3)
    res["candidates"] = cand["summary"]
    np.savez_compressed(d / "candidates_grid.npz", **cand["masks"])
    res["_internal"]["candidate_mask_keys"] = list(cand["masks"].keys())
    jdump(res, d / "layout.json")
    return res


def edt_cells(valid):
    v = np.pad(valid, 1, constant_values=False)
    return ndi.distance_transform_edt(v)[1:-1, 1:-1]


def stretch_ratio(P):
    """3D spacing between grid neighbours / nominal (20 px * 2.399 um), along x and along rows (per cell)."""
    ok = (P[..., 0] != -1)
    dx = np.full(P.shape[:2], np.nan, np.float32)
    dy = np.full(P.shape[:2], np.nan, np.float32)
    d1 = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=-1)
    m1 = ok[:, 1:] & ok[:, :-1]
    dx[:, :-1] = np.where(m1, d1, np.nan)
    d2 = np.linalg.norm(P[1:] - P[:-1], axis=-1)
    m2 = ok[1:] & ok[:-1]
    dy[:-1] = np.where(m2, d2, np.nan)
    return dx / G, dy / G


def tiles_inside(mask, i_rng, j_rng):
    """5 mm tiles wholly inside mask: (a) canvas-anchored 2084 px grid (Metal Detector stage-1 tiling),
    (b) region-anchored grid of 105-cell (5.04 mm) squares, best of 3 x 3 anchor offsets (0, 35, 70 cells)."""
    gh, gw = mask.shape
    na = 0
    for ta in range(int(i_rng[0] * G // TILE_PX), int(i_rng[1] * G // TILE_PX) + 1):
        r0, r1 = int(np.floor(ta * TILE_PX / G)), int(np.ceil((ta + 1) * TILE_PX / G))
        if r1 > gh:
            continue
        for tb in range(int(j_rng[0] * G // TILE_PX), int(j_rng[1] * G // TILE_PX) + 1):
            c0, c1 = int(np.floor(tb * TILE_PX / G)), int(np.ceil((tb + 1) * TILE_PX / G))
            if c1 > gw:
                continue
            if mask[r0:r1, c0:c1].all():
                na += 1
    T = 105
    i0, i1 = i_rng
    j0, j1 = j_rng
    sub = mask[i0:i1, j0:j1].astype(np.int32)
    ii = np.pad(sub.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best = 0
    for oi in (0, 35, 70):
        for oj in (0, 35, 70):
            n = 0
            for a in range(oi, sub.shape[0] - T + 1, T):
                for b in range(oj, sub.shape[1] - T + 1, T):
                    if ii[a + T, b + T] - ii[a, b + T] - ii[a + T, b] + ii[a, b] == T * T:
                        n += 1
            best = max(best, n)
    return na, best


def region_summary(name, kind, mask, prob, Z, valid, holes, sx, sy, lev, flags_extra):
    ii, jj = np.nonzero(mask)
    if ii.size == 0:
        return None
    i0, i1, j0, j1 = ii.min(), ii.max() + 1, jj.min(), jj.max() + 1
    zz = Z[mask]
    hb = holes[i0:i1, j0:j1]
    vb = valid[i0:i1, j0:j1]
    hole = float(hb.sum() / max(1, (hb | vb).sum()))
    na, nb_ = tiles_inside(mask, (i0, i1), (j0, j1))
    pm = float(prob[mask].mean())
    stx, sty = float(np.nanmedian(sx[mask])), float(np.nanmedian(sy[mask]))
    flags = list(flags_extra)
    if zz.max() >= TAPER_Z_MM:
        flags.append(f"reaches tapered roll end (z >= {TAPER_Z_MM} mm): {100 * float((zz >= TAPER_Z_MM).mean()):.0f}% of area")
    if hole > 0.05:
        flags.append(f"holes: {100 * hole:.0f}% interior mesh holes in bbox")
    if not (0.9 <= stx <= 1.1 and 0.9 <= sty <= 1.1):
        flags.append(f"mesh stretch x {stx:.2f} rows {sty:.2f} outside 0.9-1.1")
    rel = (pm - lev[0]) / (lev[1] - lev[0])
    if rel > 0.25:
        flags.append(f"render mean {rel:.2f} of text contrast (> 0.25)")
    return {"name": name, "kind": kind, "x_mm": [round(mm(j0), 2), round(mm(j1), 2)],
            "canvas_px_rows": [int(i0 * G), int(i1 * G)], "canvas_px_cols": [int(j0 * G), int(j1 * G)],
            "z_mm": [round(float(zz.min()), 2), round(float(zz.max()), 2)], "n_cells": int(ii.size),
            "area_mm2": round(ii.size * CELL_MM ** 2, 1), "tiles_5mm_canvas_anchored": na, "tiles_5mm_region_packed": nb_,
            "render_mean_prob": round(pm, 4), "render_level_rel": round(float(rel), 3),
            "hole_frac_bbox": r(hole, 3), "stretch_x": round(stx, 3), "stretch_rows": round(sty, 3),
            "unreliable_flags": flags, "reliable": not flags}


def candidates(short, res, prob, Z, valid, P3, s3):
    gh, gw = prob.shape
    lo, hi = res["_internal"]["lo"], res["_internal"]["hi"]
    t10 = res["_internal"]["t10"]
    runs = res["_internal"]["runs"]
    sx, sy = stretch_ratio(P3)
    edt = edt_cells(valid)
    far = edt >= cells(EDGE_GUARD_MM)
    holes = ndi.binary_fill_holes(valid) & ~valid
    colmap = {c["block_idx"]: c for c in res["columns"] if c.get("status") == "ok"}
    masks, summ = {}, []
    # per grid column text edges (for margins)
    e_lo = np.full(gw, np.nan)
    e_hi = np.full(gw, np.nan)
    owner = [None] * gw
    for bi, c in colmap.items():
        a, b = runs[bi][1], runs[bi][2]
        e_lo[a:b], e_hi[a:b] = c["z_lo_mm"], c["z_hi_mm"]
        for j in range(a, b):
            owner[j] = f"col@{mm(a):.0f}-{mm(b):.0f}mm"
    shrink = cells(CORE_SHRINK_MM)
    for g in res["gaps"]:
        if g["class"] != "intercolumn_candidate":
            continue
        a, b = g["j"]
        L, R = colmap.get(g["idx"] - 1), colmap.get(g["idx"] + 1)
        ext = [c for c in (L, R) if c]
        if not ext:
            g["core"] = "no neighbour column with a text extent"
            continue
        zlo = max(c["z_lo_mm"] for c in ext)
        zhi = min(c["z_hi_mm"] for c in ext)
        lo_edges = [c["z_lo_mm"] for c in ext]
        hi_edges = [c["z_hi_mm"] for c in ext]
        e_lo[a:b], e_hi[a:b] = min(lo_edges), max(hi_edges)
        for j in range(a, b):
            owner[j] = f"gap@{g['centre_mm']:.0f}mm"
        below = np.isfinite(s3[a:b]) & (s3[a:b] <= t10)
        rr = [q for q in rle(below.astype(int)) if q[0] == 1]
        if not rr:
            g["core"] = "no x with level <= t10"
            continue
        _, ca, cb = max(rr, key=lambda q: q[2] - q[1])
        ca, cb = a + ca + shrink, a + cb - shrink
        g["core_x_mm"] = [round(mm(ca), 2), round(mm(cb), 2)]
        g["core_width_mm"] = round(mm(cb - ca), 2)
        if mm(cb - ca) < CORE_MIN_MM:
            g["core"] = f"core {mm(cb - ca):.1f} mm < {CORE_MIN_MM} mm"
            continue
        m = np.zeros((gh, gw), bool)
        m[:, ca:cb] = True
        m &= valid & far & (Z >= zlo) & (Z <= zhi)
        flags = []
        if g.get("lattice_consistent") is False:
            flags.append("gap not lattice-consistent")
        if len(ext) < 2:
            flags.append("only one neighbour column has a text extent")
        elif max(abs(L["z_lo_mm"] - R["z_lo_mm"]), abs(L["z_hi_mm"] - R["z_hi_mm"])) > 5:
            flags.append("neighbour text extents differ by > 5 mm")
        key = f"ic_{g['centre_mm']:.0f}"
        sm = region_summary(key, "intercolumn", m, prob, Z, valid, holes, sx, sy, (lo, hi), flags)
        if sm:
            sm.update({"gap_x_mm": g["x_mm"], "gap_width_mm": g["width_mm"], "z_limits_from_neighbours_mm": [zlo, zhi]})
            summ.append(sm)
            masks[key] = m
    # margin bands per x-segment (column block or intercolumn-candidate gap), both ends
    segs = []
    j = 0
    while j < gw:
        if owner[j] is None:
            j += 1
            continue
        k = j
        while k < gw and owner[k] == owner[j]:
            k += 1
        segs.append((owner[j], j, k))
        j = k
    # the margin band is continuous under columns and intercolumns; x-segments only split it for reporting
    for name, a, b in segs:
        colsel = np.zeros(gw, bool)
        colsel[a:b] = True
        base = valid & far & colsel[None, :]
        parts = [("high_z", base & (Z >= (e_hi[None, :] + MARGIN_GUARD_MM)) & (Z < TAPER_Z_MM)),
                 ("high_z_taper", base & (Z >= (e_hi[None, :] + MARGIN_GUARD_MM)) & (Z >= TAPER_Z_MM)),
                 ("low_z", base & (Z <= (e_lo[None, :] - MARGIN_GUARD_MM)))]
        for end, m in parts:
            if not m.any():
                continue
            key = f"mg_{end}_{name}"
            sm = region_summary(key, f"margin_{end}", m, prob, Z, valid, holes, sx, sy, (lo, hi), [])
            if sm:
                sm["x_segment"] = name
                summ.append(sm)
                masks[key] = m
    return {"summary": summ, "masks": masks}


# ----------------------------------------------------------------------------------------------- orientation
def orient():
    from tools.margin.layout_inventory import orientation
    rng = np.random.default_rng(SEED)
    out = {}
    for short in SEG_ORDER:
        d = OUT / short
        if not (d / "layout.json").exists():
            out[short] = {"status": "not run"}
            continue
        lay = json.loads((d / "layout.json").read_text())
        pooled = load_pooled(short)
        gh, gw = pooled["mean"].shape
        geo = load_grid(short, gh, gw)
        Z, valid = geo["Z"], geo["valid"]
        runs = lay["_internal"]["runs"]
        text = np.zeros((gh, gw), bool)
        g1 = cells(COL_GUARD_MM)
        for c in lay["columns"]:
            if c.get("status") != "ok":
                continue
            _, a, b = runs[c["block_idx"]]
            text[:, a + g1:b - g1] |= (Z[:, a + g1:b - g1] >= c["z_lo_mm"]) & (Z[:, a + g1:b - g1] <= c["z_hi_mm"])
        a_ = {"L": pooled["nhi"].astype(np.float32) / (G * G), "Z": np.nan_to_num(Z, nan=-1.0), "valid": valid & text}
        o = orientation(a_, rng)
        o["text_cells"] = int((valid & text).sum())
        out[short] = o
        print(short, json.dumps(o)[:600], flush=True)
    ci = {s: o.get("skew_ci95") for s, o in out.items()}
    pos = sum(1 for v in ci.values() if v and v[0] > 0)
    neg = sum(1 for v in ci.values() if v and v[1] < 0)
    if pos >= 3 and neg == 0:
        verdict = "foot at HIGH z"
    elif neg >= 3 and pos == 0:
        verdict = "foot at LOW z"
    else:
        verdict = "UNDETERMINED"
    res = {"per_segment": out, "n_ci_above_0": pos, "n_ci_below_0": neg, "verdict": verdict,
           "rule": "layout_inventory.py docstring rule, unchanged: HIGH z if skew CI > 0 on >= 3 of 4 and none < 0; "
                   "LOW z if mirror; else UNDETERMINED"}
    jdump(res, OUT / "orientation.json")
    print("orientation verdict:", verdict, flush=True)
    return res


# ----------------------------------------------------------------------------------------------- pass 2 tripwire
def text_like(hi, mpx, label):
    """Amendment D recalibrated text-like flag on one region. hi = (p > 0.5) & region at full res, mpx = region
    (both cropped with the origin on the canvas 416 px = 1 mm tile grid)."""
    from tools.margin.deep_ld import blob_null, letter_S, shuffle_tiles
    from tools.margin.forensics import CELL
    from tools.margin.forensics_analyze import CMM, Profile, periodicity
    n_px = int(mpx.sum())
    fp = float(hi.sum() / n_px) if n_px else float("nan")
    nr, nc = hi.shape[0] // CELL, hi.shape[1] // CELL
    num = hi[:nr * CELL, :nc * CELL].reshape(nr, CELL, nc, CELL).sum(axis=(1, 3)).astype(np.float64)
    den = mpx[:nr * CELL, :nc * CELL].reshape(nr, CELL, nc, CELL).sum(axis=(1, 3)).astype(np.float64)
    per, per_flag, span_mm, ntl = None, False, 0.0, 0
    if den.sum() > 0:
        pr = Profile(num, den)
        tiles = pr.tiles()
        ntl = len(tiles)
        span_mm = len(pr.rows) * CMM
        if span_mm >= 10.0 and ntl >= 20:
            p = periodicity(pr, label)
            per = {k: p[k] for k in ("n_rows", "length_mm", "T1_ratio_pooled", "T1_peak_period_mm", "p_tile_T1",
                                     "tile_null_T1_median", "tile_null_T1_p95", "shuffle_cells_frac")}
            per_flag = bool(fp >= 0.001 and per["p_tile_T1"] <= 0.01)
    S, n_letter, area = letter_S(hi, UM)
    blob = {"n_components": int(len(area)), "letter_blobs_1_4mm2": n_letter, "S_max_in_1mm_row": S,
            "letter_blob_per_cm2": round(n_letter / (n_px * (UM / 1000) ** 2 / 100.0), 4) if n_px else None}
    blob_flag = False
    if S >= 3:
        tl = shuffle_tiles(den)
        if len(tl) >= 20:
            nullS = blob_null(hi, tl, UM)
            blob.update({"null_S_p99": float(np.percentile(nullS, 99)), "null_S_median": float(np.median(nullS)),
                         "n_shuffle_tiles": len(tl)})
            blob_flag = bool(S > blob["null_S_p99"])
        else:
            blob["null_note"] = f"only {len(tl)} whole 1 mm tiles: null not computable; S >= 3 reported"
    else:
        blob["null_note"] = "S < 3: flag cannot fire (pre-registered, as Amendment D)"
    return {"px": n_px, "area_mm2": round(n_px * (UM / 1000) ** 2, 1), "fp_frac_p05": round(fp, 5),
            "periodicity_testable": bool(per is not None), "row_span_mm": round(span_mm, 1), "n_1mm_tiles": ntl,
            "periodicity": per, "periodicity_flag": per_flag, "blobs": blob, "blob_flag": blob_flag,
            "text_like_flag": bool(per_flag or blob_flag)}


def trip(short, keep_raw=False, only=None):
    from tools.margin.forensics import CELL
    d = OUT / short
    lay = json.loads((d / "layout.json").read_text())
    red = json.loads((d / "reduce.json").read_text())
    with np.load(d / "candidates_grid.npz") as f:
        masks = {k: f[k] for k in f.files}
    raw = raw_path(short)
    if not raw.exists():
        raise SystemExit(f"raw render missing: {raw}; re-run reduce")
    lz = Lazy2D(raw)
    thr, scale, _ = value_scale(lz.dtype)
    H, W = red["canvas_hw"]
    SH = CELL * 4                   # 416 px = 1 mm shuffle tile
    out = {}
    names = [n for n in masks if (only is None or n in only)]
    stop = []
    t0 = time.time()
    for name in names:
        m = masks[name]
        ii, jj = np.nonzero(m)
        R0 = (ii.min() * G // SH) * SH
        R1 = min(H, -(-(ii.max() + 1) * G // SH) * SH)
        C0 = (jj.min() * G // SH) * SH
        C1 = min(W, -(-(jj.max() + 1) * G // SH) * SH)
        mpx = m[np.arange(R0, R1) // G][:, np.arange(C0, C1) // G]
        hi = np.zeros((R1 - R0, C1 - C0), bool)
        for r0 in range(R0, R1, STRIP):
            r1 = min(R1, r0 + STRIP)
            hi[r0 - R0:r1 - R0] = lz.rows(r0, r1, C0, C1) > thr
        hi &= mpx
        o = text_like(hi, mpx, name)
        out[name] = o
        per = o["periodicity"]
        print(f"  {name}: fp {o['fp_frac_p05']:.4f} p_tile {per and per['p_tile_T1']} "
              f"S {o['blobs']['S_max_in_1mm_row']} flag {o['text_like_flag']} ({time.time() - t0:.0f}s)", flush=True)
        del hi, mpx
        if o["text_like_flag"]:
            stop.append(name)
            break
    lz.close()
    res = {"segment": short, "regions": out, "n_regions_tested": len(out), "hard_stop": bool(stop), "stop_regions": stop,
           "expected_false_alarm_note": "per-region tests at p <= 0.01 (periodicity) and S > null p99 (blobs); "
                                        "no multiplicity correction (stop rule not weakened)"}
    jdump(res, d / "trip.json")
    if stop:
        jdump({"segment": short, "region": stop, "numbers": out[stop[0]]}, d / "HARD_STOP.json")
    if not keep_raw:
        raw.unlink()
        print(f"  deleted {raw.name}; {free_bytes(REPO) / 2**30:.1f} GB free", flush=True)
    if stop:
        print("HARD STOP: text-like structure inside a candidate blank region", stop, flush=True)
        sys.exit(3)
    return res


# ----------------------------------------------------------------------------------------------- canvas map
def canvas_map(short):
    """Official canvas -> HF canvas (where the audited preds live) via 3D nearest neighbour between tifxyz points."""
    from scipy.spatial import cKDTree
    d = OUT / short
    with np.load(d / "candidates_grid.npz") as f:
        masks = {k: f[k] for k in f.files}
    pooled = load_pooled(short)
    gh, gw = pooled["mean"].shape
    geo = load_grid(short, gh, gw)
    hf = REPO / "data" / f"1667_{short}" / "small"
    hm = json.loads((hf / "meta.json").read_text())
    Ph = np.stack([np.squeeze(tifffile.imread(hf / f"{c}.tif")).astype(np.float32) for c in "xyz"], -1)
    okh = (Ph[..., 0] != -1) & (Ph[..., 2] > 0)
    ih, jh = np.nonzero(okh)                       # full HF grid (deviation D3: was every 2nd point)
    tree = cKDTree(Ph[ih, jh])
    OUTLIER_PX = 2.0 / (UM / 1000)                 # matches > 2 mm off the region's median offset are rejected
    sxh, syh = float(hm["scale"][0]), float(hm["scale"][1])
    out = {"hf_grid_hw": list(okh.shape), "hf_scale_sx_sy": [sxh, syh], "regions": {}}
    hmasks = {}
    # global relation on text-band cells
    rng = np.random.default_rng(SEED)
    vi, vj = np.nonzero(geo["valid"])
    pick = rng.choice(len(vi), min(200000, len(vi)), replace=False)
    dist, idx = tree.query(geo["P"][vi[pick], vj[pick]])
    dy = ih[idx] / syh - vi[pick] * G
    dx = jh[idx] / sxh - vj[pick] * G
    out["global"] = {"nn_dist_um_median": round(float(np.median(dist) * UM), 1), "nn_dist_um_p95": round(float(np.percentile(dist, 95) * UM), 1),
                     "hf_minus_official_px_rows_median": round(float(np.median(dy)), 1),
                     "hf_minus_official_px_cols_median": round(float(np.median(dx)), 1),
                     "rows_iqr_px": round(float(np.subtract(*np.percentile(dy, [75, 25]))), 1),
                     "cols_iqr_px": round(float(np.subtract(*np.percentile(dx, [75, 25]))), 1)}
    for name, m in masks.items():
        ii, jj = np.nonzero(m)
        dist, idx = tree.query(geo["P"][ii, jj])
        oy = ih[idx] / syh - ii * G
        ox = jh[idx] / sxh - jj * G
        keep = (np.abs(oy - np.median(oy)) <= OUTLIER_PX) & (np.abs(ox - np.median(ox)) <= OUTLIER_PX)
        hm_ = np.zeros(okh.shape, bool)
        hm_[ih[idx[keep]], jh[idx[keep]]] = True
        hm_ = ndi.binary_closing(hm_, np.ones((3, 3), bool)) & okh
        hmasks[name] = hm_
        a, b = np.nonzero(hm_)
        out["regions"][name] = {"nn_dist_um_median": round(float(np.median(dist) * UM), 1),
                                "nn_dist_um_p95": round(float(np.percentile(dist, 95) * UM), 1),
                                "rejected_frac_offset_gt_2mm": round(float(1 - keep.mean()), 6),
                                "median_offset_px_rows_cols": [round(float(np.median(oy)), 1), round(float(np.median(ox)), 1)],
                                "hf_canvas_px_rows": [int(a.min() / syh), int((a.max() + 1) / syh)],
                                "hf_canvas_px_cols": [int(b.min() / sxh), int((b.max() + 1) / sxh)],
                                "hf_grid_cells": int(hm_.sum())}
    np.savez_compressed(d / "candidates_hf_grid.npz", **hmasks)
    jdump(out, d / "canvas_map.json")
    print(short, "canvas map", out["global"], flush=True)
    return out


def p_pool():
    """Inverse-variance pooled period from the long segments (w018, w023) whose lattice was detected."""
    vals = []
    for s in ("w018", "w023"):
        f = OUT / s / "layout.json"
        if f.exists():
            lat = json.loads(f.read_text())["lattice"]
            if lat.get("lattice_detected") and lat.get("P_fit_mm"):
                vals.append((s, lat["P_fit_mm"], lat.get("P_fit_se_mm") or 1.0))
    if not vals:
        return None
    w = np.array([1 / se ** 2 for _, _, se in vals])
    P = float(np.sum(w * np.array([p for _, p, _ in vals])) / w.sum())
    return {"P_pool_mm": round(P, 2), "se_mm": round(float(np.sqrt(1 / w.sum())), 2), "inputs": vals}


# ----------------------------------------------------------------------------------------------- synthetic checks
def synth_canvas(rng, desc_sign=+1, L_mm=300.0, H_mm=100.0):
    """Synthetic flattened segment: columns (period 64 mm, width 50 mm), 10 lines at 5.8 mm pitch (centres
    22.0-74.2 mm), straight high-z end, ragged low-z end (first 0-2 lines missing per 5 mm strip), one half-contrast
    column, a mesh hole, descender tails toward desc_sign * z. Returns cell-level arrays."""
    gh, gw = cells(H_mm), cells(L_mm)
    x = (np.arange(gw) + 0.5) * CELL_MM
    Z = ((np.arange(gh)[:, None] + 0.5) * CELL_MM + 2.0 + 0.002 * x[None, :]).astype(np.float32)
    valid = (Z >= 4.0) & (Z <= 90.0)
    valid[:, :cells(2)] = False
    valid[:, -cells(2):] = False
    prob = (0.03 + 0.02 * rng.random((gh, gw))).astype(np.float32)
    P, Wc, x0 = 64.0, 50.0, 12.0
    centres = 22.0 + 5.8 * np.arange(10)
    chunk = cells(0.5)
    for k in range(10):
        c0 = x0 + k * P
        if c0 >= L_mm:
            break
        amp = 0.35 if k == 2 else 0.7
        ja, jb = int(c0 / CELL_MM), min(gw, int((c0 + Wc) / CELL_MM))
        for s0 in range(ja, jb, cells(5.0)):
            s1 = min(jb, s0 + cells(5.0))
            k0 = rng.integers(0, 3)
            for li, zc in enumerate(centres):
                if li < k0:
                    continue
                for c in range(s0, s1, chunk):
                    if rng.random() < 0.7:
                        body = (Z[:, c:c + chunk] >= zc - 1.2) & (Z[:, c:c + chunk] <= zc + 1.2)
                        prob[:, c:c + chunk][body] = amp
                    if rng.random() < 0.2:
                        lo_, hi_ = (zc + 1.2, zc + 2.8) if desc_sign > 0 else (zc - 2.8, zc - 1.2)
                        tail = (Z[:, c:c + chunk] >= lo_) & (Z[:, c:c + chunk] <= hi_)
                        prob[:, c:c + chunk][tail] = amp
    hole = (Z >= 40) & (Z <= 45) & (x[None, :] >= 95) & (x[None, :] <= 100)
    valid &= ~hole
    nlab = np.zeros((gh, gw), np.uint16)
    lab_x = (x >= 80) & (x <= 120)
    nlab[:, lab_x] = np.where(prob[:, lab_x] > 0.5, G * G, 0)
    return prob, Z, valid, nlab


def synth_layout_check(rng):
    from tools.margin.layout_inventory import orientation
    prob, Z, valid, nlab = synth_canvas(rng, +1)
    z5, z95, _ = text_band(nlab, Z, valid)
    m, _ = x_profile(prob, Z, valid, z5, z95)
    s3 = nan_boxcar(m, cells(SMOOTH_MM) | 1)
    lo, hi = np.nanpercentile(s3, [5, 95])
    runs, islands = blocks_and_gaps(s3, lo + T_BLOCK * (hi - lo))
    gaps = [(mm(a), mm(b)) for i, (t, a, b) in enumerate(runs)
            if t == LO and 0 < i < len(runs) - 1 and runs[i - 1][0] == HI and runs[i + 1][0] == HI
            and GAP_MIN_MM <= mm(b - a) <= GAP_MAX_MM]
    fin = np.flatnonzero(np.isfinite(s3))
    lat = lattice_fit([(a + b) / 2 for a, b in gaps], mm(fin.min()), mm(fin.max() + 1), np.random.default_rng(SEED))
    cols = []
    for i, (t, a, b) in enumerate(runs):
        if t == HI and mm(b - a) >= MIN_COL_MM:
            ce = column_extent(prob, Z, valid, a + cells(1), b - cells(1), z5, z95, hi - lo)
            cols.append(ce)
    ok_cols = [c for c in cols if c.get("status") == "ok"]
    zl = [c["z_lo_mm"] for c in ok_cols]
    zh = [c["z_hi_mm"] for c in ok_cols]
    rl = [c["ragged_lo_mm"] for c in ok_cols if c["ragged_lo_mm"] is not None]
    rh = [c["ragged_hi_mm"] for c in ok_cols if c["ragged_hi_mm"] is not None]
    text = np.zeros_like(valid)
    for i, (t, a, b) in enumerate(runs):
        if t == HI and mm(b - a) >= MIN_COL_MM:
            text[:, a + cells(1):b - cells(1)] = True
    text &= (Z >= 19.1) & (Z <= 77.1)
    L = (prob > 0.5).astype(np.float32)
    o_pos = orientation({"L": L, "Z": Z, "valid": valid & text}, np.random.default_rng(SEED))
    prob2, Z2, valid2, _ = synth_canvas(np.random.default_rng(1), -1)
    o_neg = orientation({"L": (prob2 > 0.5).astype(np.float32), "Z": Z2, "valid": valid2 & text},
                        np.random.default_rng(SEED))
    res = {"gaps_mm": [[round(a, 1), round(b, 1)] for a, b in gaps], "expected_gap_centres": [69, 133, 197, 261],
           "P_fit_mm": lat["P_fit_mm"], "lattice_detected": lat["lattice_detected"], "p_R": lat["p_R_hardcore_null"],
           "z_lo_median": r(np.median(zl)), "z_hi_median": r(np.median(zh)), "expected_text_cells": [19.1, 77.1],
           "pitch_median": r(np.median([c["line_pitch_mm"] for c in ok_cols])),
           "ragged_lo_median": r(np.median(rl)), "ragged_hi_median": r(np.median(rh)),
           "skew_ci_plus_desc": o_pos.get("skew_ci95"), "skew_ci_minus_desc": o_neg.get("skew_ci95"),
           "n_cols_ok": len(ok_cols)}
    checks = {
        "4 intercolumn gaps near 69/133/197/261": len(gaps) == 4 and all(abs((a + b) / 2 - e) < 2 for (a, b), e in zip(gaps, [69, 133, 197, 261])),
        "P_fit within 1 mm of 64": lat["P_fit_mm"] is not None and abs(lat["P_fit_mm"] - 64) < 1,
        "lattice detected": lat["lattice_detected"],
        "straight high-z edge within 1 mm of 77.1": abs(np.median(zh) - 77.1) < 1.0,
        "pitch within 0.1 mm of 5.8": abs(np.median([c["line_pitch_mm"] for c in ok_cols]) - 5.8) < 0.1,
        "ragged low end > straight high end + 1 mm": np.median(rl) > np.median(rh) + 1.0,
        "+z descenders: skew CI > 0": o_pos.get("skew_ci95", [0, 0])[0] > 0,
        "-z descenders: skew CI < 0": o_neg.get("skew_ci95", [0, 0])[1] < 0,
    }
    return res, checks


def synth_trip_check(rng):
    """Blank region with small noise blobs vs the same region with planted letter rows (5.8 mm pitch)."""
    H, W = 416 * 12, 416 * 20                      # 12 x 20 mm, origin on the 1 mm tile grid
    mpx = np.ones((H, W), bool)
    blank = np.zeros((H, W), bool)
    for _ in range(40):                            # sub-letter specks (< 1 mm2)
        rr, cc = rng.integers(0, H - 100), rng.integers(0, W - 100)
        blank[rr:rr + 60, cc:cc + 60] = True
    a = text_like(blank & mpx, mpx, "synth_blank")
    text = blank.copy()
    bh, bw = int(1.2 / (UM / 1000)), int(1.3 / (UM / 1000))   # 1.2 x 1.3 mm = 1.56 mm2 letters
    for zc_mm in np.arange(1.5, 12, 5.8):
        r0 = int(zc_mm / (UM / 1000))
        for xc_mm in np.arange(0.5, 19.5, 2.4):
            c0 = int(xc_mm / (UM / 1000))
            text[r0:r0 + bh, c0:c0 + bw] = True
    b = text_like(text & mpx, mpx, "synth_rows")
    checks = {"blank region: no flag": not a["text_like_flag"], "planted letter rows: flag fires": b["text_like_flag"]}
    return {"blank": a, "planted_rows": b}, checks


def synth():
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    lay, c1 = synth_layout_check(rng)
    print("layout synth:", json.dumps(lay), flush=True)
    tr, c2 = synth_trip_check(np.random.default_rng(SEED + 1))
    print("trip synth:", json.dumps({k: {"S": v["blobs"]["S_max_in_1mm_row"], "p": (v["periodicity"] or {}).get("p_tile_T1"),
                                          "flag": v["text_like_flag"]} for k, v in tr.items()}), flush=True)
    checks = {**c1, **c2}
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k, flush=True)
    out = {"layout": lay, "trip": {k: {kk: vv for kk, vv in v.items()} for k, v in tr.items()},
           "checks": {k: bool(v) for k, v in checks.items()}, "all_pass": bool(all(checks.values())),
           "seconds": round(time.time() - t0, 1)}
    jdump(out, OUT / "synth.json")
    return out


# ----------------------------------------------------------------------------------------------- report
def straight_ragged(cols):
    out = {}
    for end in ("lo", "hi"):
        rag = [c[f"ragged_{end}_mm"] for c in cols if c.get(f"ragged_{end}_mm") is not None]
        edges = [c[f"z_{end}_mm"] for c in cols]
        dist = [c[f"dist_to_mesh_end_{end}_mm"] for c in cols]
        how = [c[f"z_{end}_how"] for c in cols]
        R, A, D = float(np.median(rag)), float(np.std(edges)), float(np.median(dist))
        cls = ("STRAIGHT" if (R <= 1.5 and A <= 2.0 and D >= 3.0) else
               "RAGGED" if (R >= 3.0 or D < 2.0) else "INTERMEDIATE")
        out["low_z" if end == "lo" else "high_z"] = {
            "median_raggedness_mm": round(R, 2), "n_cols_raggedness": len(rag), "across_column_sd_mm": round(A, 2),
            "edge_range_mm": [round(min(edges), 1), round(max(edges), 1)], "median_dist_to_mesh_end_mm": round(D, 2),
            "n_cols": len(edges), "edge_found_by": {h: how.count(h) for h in sorted(set(how))}, "class": cls}
    s = [k for k, v in out.items() if v["class"] == "STRAIGHT"]
    out["secondary_reading"] = (f"foot at the {s[0]} end" if len(s) == 1 else
                                "none (no end, or both ends, classed STRAIGHT)")
    return out


def git_head():
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:           # noqa: BLE001
        return None


def report():
    segs = {}
    all_cols = []
    for s in SEG_ORDER:
        d = OUT / s
        red = json.loads((d / "reduce.json").read_text())
        lay = json.loads((d / "layout.json").read_text())
        trp = json.loads((d / "trip.json").read_text()) if (d / "trip.json").exists() else {"regions": {}}
        cmap = json.loads((d / "canvas_map.json").read_text()) if (d / "canvas_map.json").exists() else {}
        lay.pop("_internal", None)
        cols = [c for c in lay["columns"] if c.get("status") == "ok"]
        for c in cols:
            c["segment"] = s
        all_cols += cols
        for c in lay["candidates"]:
            c["tripwire"] = trp["regions"].get(c["name"], "NOT TESTED (run stopped at an earlier flag)")
            c["hf_canvas"] = cmap.get("regions", {}).get(c["name"])
        segs[s] = {"registration": {k: red[k] for k in ("render_url", "render_bytes", "render_sha256", "dtype",
                                                       "canvas_hw", "render_shape_equals_label_canvas",
                                                       "registration_auroc", "registration_ok",
                                                       "recall_p05_labeled_ink", "fpr_p05_unlabeled_in_mask",
                                                       "official_label_px", "hf_inklabels_v2_px")},
                   "shift_diag": lay["registration_shift_diag"], "rows_vs_z_spearman": lay["canvas_rows_vs_z_spearman"],
                   "text_band_from_labels_z_mm": lay["text_band_from_labels_z_mm"], "x_profile": lay["x_profile"],
                   "x_runs_sensitivity": lay["x_runs_sensitivity"], "blocks": lay["blocks"], "gaps": lay["gaps"],
                   "lattice": lay["lattice"], "columns": lay["columns"], "candidates": lay["candidates"],
                   "tripwire": {k: trp.get(k) for k in ("n_regions_tested", "hard_stop", "stop_regions")},
                   "canvas_map_global": cmap.get("global")}
    orient_res = json.loads((OUT / "orientation.json").read_text())
    sr = straight_ragged(all_cols)
    pp = p_pool()
    cand = [dict(c, segment=s) for s in SEG_ORDER for c in segs[s]["candidates"]]

    def tot(sel):
        return {"n": len(sel), "area_mm2": round(sum(c["area_mm2"] for c in sel), 1),
                "tiles_canvas_anchored": sum(c["tiles_5mm_canvas_anchored"] for c in sel),
                "tiles_region_packed": sum(c["tiles_5mm_region_packed"] for c in sel)}

    def stopped(c):
        return isinstance(c["tripwire"], dict) and c["tripwire"].get("text_like_flag")

    kinds = {"intercolumn": [c for c in cand if c["kind"] == "intercolumn"],
             "margin_high_z (< 81 mm)": [c for c in cand if c["kind"] == "margin_high_z"],
             "margin_high_z_taper (>= 81 mm)": [c for c in cand if c["kind"] == "margin_high_z_taper"],
             "margin_low_z": [c for c in cand if c["kind"] == "margin_low_z"]}
    totals = {}
    for k, v in kinds.items():
        totals[k] = {"all": tot(v), "reliable_not_stopped": tot([c for c in v if c["reliable"] and not stopped(c)]),
                     "stopped": tot([c for c in v if stopped(c)]),
                     "untested": tot([c for c in v if not isinstance(c["tripwire"], dict)])}
    res = {"title": "Edition route: PHerc.1667 column lattice on the org's 2.4 um ink renders (numbers only)",
           "date": "2026-09-26", "plan": "results/edition_lattice_plan.json (commit 2bc58d8, pushed before any render)",
           "code": "tools/margin/lattice.py", "code_commit_at_report": git_head(),
           "status": "HARD STOP (text rule): row periodicity in the w031 intercolumn candidate ic_33. The authors decide.",
           "cols_1_3": json.loads(PLAN_JSON.read_text())["0_cols_1_3_check"],
           "segments": segs, "P_pool": pp,
           "edition_comparison": {"P_fit_w018_mm": segs["w018"]["lattice"].get("P_fit_mm"),
                                  "vs_64mm_pct": r(100 * (segs["w018"]["lattice"]["P_fit_mm"] / EDITION_P - 1), 1),
                                  "within_15pct_of_64": bool(abs(segs["w018"]["lattice"]["P_fit_mm"] / EDITION_P - 1) <= 0.15),
                                  "vs_75mm_pct": r(100 * (segs["w018"]["lattice"]["P_fit_mm"] / DIEGO_P - 1), 1),
                                  "implied_columns_in_1400mm": r(ROLL_MM / segs["w018"]["lattice"]["P_fit_mm"], 1)},
           "orientation": orient_res, "straight_vs_ragged": sr, "candidate_totals": totals,
           "synthetic_checks": json.loads((OUT / "synth.json").read_text())["checks"]}
    jdump(res, RES_JSON)
    print("wrote", RES_JSON, flush=True)
    return res


# ----------------------------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["synth", "geom", "reduce", "layout", "trip", "orient", "map", "report"])
    ap.add_argument("--seg", choices=SEG_ORDER)
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args()
    if a.cmd == "synth":
        synth()
    elif a.cmd == "geom":
        geom(a.seg)
    elif a.cmd == "reduce":
        reduce(a.seg)
    elif a.cmd == "layout":
        pp = p_pool()
        if pp:
            jdump(pp, OUT / "P_pool.json")
        layout(a.seg, pp["P_pool_mm"] if pp else None)
    elif a.cmd == "trip":
        trip(a.seg, keep_raw=a.keep_raw)
    elif a.cmd == "orient":
        orient()
    elif a.cmd == "map":
        canvas_map(a.seg)
    elif a.cmd == "report":
        report()


if __name__ == "__main__":
    main()
