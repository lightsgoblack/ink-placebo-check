"""Amendment E1 validation (before any v0-E pred is read): 100 seeded synthetic fills of each real guarded region mask
(w018 ic_201, w023 ic_132) per case, run through tools/margin/tripwire_e1.py exactly as the real run uses it, at the E1
fire rule. Synthetic maps only: no scroll data except the two region masks, and no images.

    .venv/bin/python -m tools.margin.e1_validation run --seg w018 --case Q1        # counted run (resumable)
    .venv/bin/python -m tools.margin.e1_validation run --seg w018 --case Q1 --dev  # development seeds (never counted)
    .venv/bin/python -m tools.margin.e1_validation table                           # fires / 100 and the gates

The generator parameters (GEN_SPEC) are fixed in writing and committed before the counted run.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from tools.margin import tripwire_e1 as E
from tools.margin.common import REPO, full_to_grid

SEGS = ("w018", "w023")
SEG_NO = {"w018": 18, "w023": 23}
CASES = ("Q1", "Q2", "Q3", "F1", "F2", "F3")
CASE_NO = {c: i + 1 for i, c in enumerate(CASES)}
N_FILLS = 100
FILL_SEED = 20260926            # counted fills: default_rng([20260926, case_no, seg_no, fill])
DEV_SEED = 20260927             # development fills (debugging only): default_rng([20260927, ...]); never counted
VOX = 2.399
PPM = 1000.0 / VOX              # px per mm
REG_JSON = REPO / "vault" / "results" / "lie_detector_v0e_regions.json"
OUT = REPO / "data" / "v0e" / "e1_validation"

GEN_SPEC = {
    "status": "Fixed in writing and committed before the counted validation run. Development runs use other seeds "
              "(default_rng([20260927, ...])) and only debug code; no generator parameter is changed after the "
              "counted run starts.",
    "regions": "The real guarded full-resolution masks of v0-E: w018 ic_201 (crop rows 2080-33696, cols 81952-85696) and "
               "w023 ic_132 (crop rows 2496-36192, cols 55328-59904), rebuilt from data/v0e/regions_<seg>.npz exactly "
               "as the v0-E harness does and checked against the committed crop_mask_sha256.",
    "pixel": "2.399 um per px (416.84 px per mm) on both axes of the flattened canvas.",
    "fills": "100 fills per case per region; fill f of case c on segment s uses numpy default_rng([20260926, c, s, f]) "
             "with c = 1..6 in the order Q1, Q2, Q3, F1, F2, F3 and s = 18 or 23. Every fill is a bool p > 0.5 map "
             "inside the region mask; nothing else is on it.",
    "letter_blob": "An upright filled ellipse standing in for one letter as a model's p > 0.5 blob: height h ~ U[1.5, "
                   "3.0] mm (letters 1.5-3 mm tall), ink area A ~ U[1.2, 3.5] mm2 (inside the 1-4 mm2 letter class, "
                   "with room for rasterization), width w = 4A / (pi h). Pixel (dy, dx) from the centre pixel is inked "
                   "iff (dy / ry)^2 + (dx / rx)^2 <= 1 with ry = h/2, rx = w/2 in px. The rasterized area is checked to "
                   "be 1-4 mm2.",
    "Q1": "Must stay quiet. One letter blob; its centre is drawn uniformly over region pixels (uniform in the region's "
          "bounding box, redrawn off the mask) until the whole blob lies inside the region.",
    "Q2": "Must stay quiet. Four letter blobs, placed one after another as in Q1; a draw that overlaps or touches "
          "(8-neighbourhood) an earlier blob is redrawn.",
    "Q3": "Must stay quiet. Structureless pixel noise: every region pixel is independently p > 0.5 with probability "
          "0.0278, the placebo FP area of the one pred processed so far (2.778%, about 2.8%). Drawn in 1024-row strips "
          "as rng.random(dtype=float32) < 0.0278.",
    "F1": "Must fire. Planted letter rows at 6 mm pitch: line centres at r_top + (phi + 6 k) mm, phi ~ U[0, 6) mm, "
          "r_top = the region's first row, k = 0, 1, ... while the line centre is inside the region's rows. Along "
          "each line, letter blobs (as above) are set left to right: the first starts g ~ U[0.3, 0.8] mm right of "
          "the region's left edge on the line-centre row, each next one after a gap g ~ U[0.3, 0.8] mm; each "
          "letter's centre row is the line centre + U[-0.2, 0.2] mm (baseline wobble). The line ends when the next "
          "letter would pass the region's right edge on that row. A letter not fully inside the region is left out "
          "(the line continues). Letters never touch: gaps >= 0.3 mm, and lines are 6 mm apart with letters <= 3 mm "
          "tall.",
    "F2": "Must fire. Pure periodic rows, no letters, at 4 mm pitch: row centres at r_top + (phi + 4 k) mm, phi ~ "
          "U[0, 4) mm. Each row is a solid band of thickness t ~ U[0.8, 1.2] mm over the longest run of columns "
          "where the region covers every row of the band, shortened by 0.1 mm at each end. A band that would leave "
          "the region's rows, or whose length is < 90% of the region's median row width, is left out, so every band "
          "is > 4 mm2 (not letter-sized).",
    "F3": "Must fire. As F2 at 6 mm pitch (phi ~ U[0, 6) mm).",
    "tripwire": "tools.margin.tripwire_e1.Region(mask).tripwire(fill, n_surr=1000, seed0=20260925, stop_early=True): "
                "the code the real run uses, at the E1 fire rule (see its INTERPRETATIONS, 'validation_stop_early').",
    "gates": {"quiet": "Q1, Q2, Q3: fires <= 2 of 100 on each region", "fire": "F1, F2, F3: fires >= 95 of 100 on "
              "each region", "rule": "All 12 (case, region) cells must pass. If any fails, E1 is not used and v0-E "
              "stays stopped (frozen). No iteration to make a case pass."},
}
QUIET, FIRE = ("Q1", "Q2", "Q3"), ("F1", "F2", "F3")
NOISE_P = 0.0278


# ------------------------------------------------------------------------------------------------ regions
def region_mask(seg: str) -> np.ndarray:
    from tools.margin.v0e import sha_mask
    R = json.loads(REG_JSON.read_text())["segments"][seg]
    with np.load(REPO / "data" / "v0e" / f"regions_{seg}.npz") as z:
        hm = z["hf_grid"].copy()
    H, W = R["hf"]["canvas_hw"]
    sy, sx = R["hf"]["grid_factor_sy_sx"]
    R0, R1 = R["full_res"]["crop_rows"]
    C0, C1 = R["full_res"]["crop_cols"]
    gy, gx = full_to_grid(H, sy, hm.shape[0]), full_to_grid(W, sx, hm.shape[1])
    m = hm[gy[R0:R1]][:, gx[C0:C1]]
    if sha_mask(m) != R["full_res"]["crop_mask_sha256"] or int(m.sum()) != R["full_res"]["px"]:
        raise SystemExit(f"{seg}: rebuilt region mask does not match the committed hash")
    if R0 % E.CELL or C0 % E.CELL:
        raise SystemExit(f"{seg}: crop origin not on the 104 px cell grid")
    return m


class Geo:
    """Row extents of a region mask (for the generators)."""

    def __init__(self, mask: np.ndarray):
        self.mask = mask
        self.H, self.W = mask.shape
        ra, ca = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
        self.bbox = (int(ra[0]), int(ra[-1]) + 1, int(ca[0]), int(ca[-1]) + 1)
        anyr = mask.any(1)
        self.left = np.where(anyr, mask.argmax(1), -1)
        self.right = np.where(anyr, self.W - mask[:, ::-1].argmax(1), -1)       # exclusive
        self.median_width = float(np.median(mask.sum(1)[anyr]))


# ------------------------------------------------------------------------------------------------ generators
def ellipse(rng):
    """One letter blob (bool array, odd shape, centred); its height/area are drawn from the spec."""
    h = rng.uniform(1.5, 3.0)
    A = rng.uniform(1.2, 3.5)
    w = 4.0 * A / (np.pi * h)
    ry, rx = h / 2 * PPM, w / 2 * PPM
    Ry, Rx = int(np.ceil(ry)), int(np.ceil(rx))
    yy, xx = np.mgrid[-Ry:Ry + 1, -Rx:Rx + 1]
    b = (yy / ry) ** 2 + (xx / rx) ** 2 <= 1.0
    a_mm2 = b.sum() / PPM ** 2
    if not 1.0 <= a_mm2 <= 4.0:
        raise AssertionError(f"letter blob area {a_mm2:.3f} mm2 outside 1-4")
    return b


def _fits(mask, hi, b, r, c, no_touch):
    """Blob b centred at (r, c) lies inside the mask (and, if no_touch, does not overlap/touch hi)."""
    Ry, Rx = b.shape[0] // 2, b.shape[1] // 2
    r0, c0 = r - Ry, c - Rx
    if r0 < 1 or c0 < 1 or r0 + b.shape[0] + 1 > mask.shape[0] or c0 + b.shape[1] + 1 > mask.shape[1]:
        return False
    if not mask[r0:r0 + b.shape[0], c0:c0 + b.shape[1]][b].all():
        return False
    if no_touch:
        from scipy import ndimage as ndi
        g = hi[r0 - 1:r0 + b.shape[0] + 1, c0 - 1:c0 + b.shape[1] + 1]
        if g.any() and (ndi.binary_dilation(np.pad(b, 1), E.ST8) & g).any():
            return False
    return True


def _stamp(hi, b, r, c):
    Ry, Rx = b.shape[0] // 2, b.shape[1] // 2
    hi[r - Ry:r - Ry + b.shape[0], c - Rx:c - Rx + b.shape[1]] |= b


def _random_blobs(geo, rng, k):
    hi = np.zeros_like(geo.mask)
    r0, r1, c0, c1 = geo.bbox
    for _ in range(k):
        b = ellipse(rng)
        for _t in range(100000):
            r, c = int(rng.integers(r0, r1)), int(rng.integers(c0, c1))
            if geo.mask[r, c] and _fits(geo.mask, hi, b, r, c, True):
                _stamp(hi, b, r, c)
                break
        else:
            raise RuntimeError("could not place a blob")
    return hi


def gen_Q1(geo, rng):
    return _random_blobs(geo, rng, 1)


def gen_Q2(geo, rng):
    return _random_blobs(geo, rng, 4)


def gen_Q3(geo, rng):
    hi = np.zeros_like(geo.mask)
    for r in range(0, geo.H, 1024):
        r1 = min(geo.H, r + 1024)
        hi[r:r1] = (rng.random((r1 - r, geo.W), dtype=np.float32) < NOISE_P) & geo.mask[r:r1]
    return hi


def gen_F1(geo, rng):
    hi = np.zeros_like(geo.mask)
    R0, R1 = geo.bbox[0], geo.bbox[1]
    phi = rng.uniform(0, 6.0)
    k = 0
    while True:
        rc = int(round(R0 + (phi + 6.0 * k) * PPM))
        k += 1
        if rc >= R1:
            break
        if geo.left[rc] < 0:
            continue
        x = geo.left[rc] + rng.uniform(0.3, 0.8) * PPM
        while True:
            b = ellipse(rng)
            wpx = b.shape[1]
            if x + wpx > geo.right[rc]:
                break
            r = int(round(rc + rng.uniform(-0.2, 0.2) * PPM))
            c = int(round(x)) + wpx // 2
            if _fits(geo.mask, hi, b, r, c, False):
                _stamp(hi, b, r, c)
            x += wpx + rng.uniform(0.3, 0.8) * PPM
    return hi


def gen_rows(geo, rng, pitch):
    hi = np.zeros_like(geo.mask)
    R0, R1 = geo.bbox[0], geo.bbox[1]
    phi = rng.uniform(0, pitch)
    k = 0
    while True:
        rc = R0 + (phi + pitch * k) * PPM
        k += 1
        t = rng.uniform(0.8, 1.2) * PPM
        ra, rb = int(round(rc - t / 2)), int(round(rc + t / 2))
        if ra >= R1:
            break
        if ra < R0 or rb > R1:
            continue
        full = geo.mask[ra:rb].all(0)
        d = np.diff(np.r_[0, full.astype(np.int8), 0])
        s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        if not s.size:
            continue
        j = int(np.argmax(e - s))
        a, b = int(s[j] + round(0.1 * PPM)), int(e[j] - round(0.1 * PPM))
        if b - a < 0.9 * geo.median_width:
            continue
        hi[ra:rb, a:b] = True
    return hi & geo.mask


def gen_F2(geo, rng):
    return gen_rows(geo, rng, 4.0)


def gen_F3(geo, rng):
    return gen_rows(geo, rng, 6.0)


GEN = {"Q1": gen_Q1, "Q2": gen_Q2, "Q3": gen_Q3, "F1": gen_F1, "F2": gen_F2, "F3": gen_F3}


# ------------------------------------------------------------------------------------------------ runner
def out_file(seg, case, dev):
    return OUT / ("dev" if dev else "counted") / f"{seg}_{case}.jsonl"


def run(seg: str, case: str, dev: bool = False, fills: int = N_FILLS):
    mask = region_mask(seg)
    geo = Geo(mask)
    reg = E.Region(mask, VOX)
    f = out_file(seg, case, dev)
    f.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if f.exists():
        done = {json.loads(x)["fill"] for x in f.read_text().splitlines() if x.strip()}
    ns = DEV_SEED if dev else FILL_SEED
    for i in range(fills):
        if i in done:
            continue
        t0 = time.time()
        rng = np.random.default_rng([ns, CASE_NO[case], SEG_NO[seg], i])
        hi = GEN[case](geo, rng)
        tg = time.time() - t0
        res = reg.tripwire(hi, n_surr=E.N_SURR, seed0=E.SEED0, stop_early=True)
        del hi
        rec = {"fill": i, "fire": res["fire"], "fire_a": res["a"]["fire"], "fire_b": res["b"]["fire"],
               "T1": res["a"]["T1"], "peak_mm": res["a"]["peak_period_mm"], "n_ge_a": res["a"]["n_null_ge"],
               "null_max_T1": res["a"]["null_max"], "S": res["b"]["S"], "n_ge_b": res["b"]["n_null_ge"],
               "null_max_S": res["b"]["null_max"], "n_computed": res["n_computed"],
               "n_components": res["n_components"], "letter_blobs": res["letter_blobs"],
               "fp_area": res["fp_area_frac"], "unplaced_total": res["placement"]["unplaced_total"],
               "implementation": res["placement"]["implementation"], "gen_s": round(tg, 1),
               "s": round(time.time() - t0, 1),
               "rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
        with open(f, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(seg, case, "dev" if dev else "counted", json.dumps(rec), flush=True)
    reg.release()


def table(dev: bool = False) -> dict:
    rows, ok_all, complete = [], True, True
    for case in CASES:
        for seg in SEGS:
            f = out_file(seg, case, dev)
            recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
            recs = {r["fill"]: r for r in recs}
            n = len(recs)
            fires = sum(r["fire"] for r in recs.values())
            gate = "<= 2" if case in QUIET else ">= 95"
            ok = (fires <= 2) if case in QUIET else (fires >= 95)
            ok = bool(ok and n == N_FILLS)
            complete &= n == N_FILLS
            ok_all &= ok
            rows.append({"case": case, "segment": seg, "fills": n, "fires": fires,
                         "fires_a": sum(r["fire_a"] for r in recs.values()),
                         "fires_b": sum(r["fire_b"] for r in recs.values()), "gate": gate,
                         "pass": ok,
                         "median_components": float(np.median([r["n_components"] for r in recs.values()])) if n else None,
                         "median_letter_blobs": float(np.median([r["letter_blobs"] for r in recs.values()])) if n else None,
                         "median_fp_area": float(np.median([r["fp_area"] for r in recs.values()])) if n else None,
                         "median_T1": float(np.median([r["T1"] for r in recs.values()])) if n else None,
                         "median_peak_mm": float(np.median([r["peak_mm"] for r in recs.values()])) if n else None,
                         "max_S": int(max(r["S"] for r in recs.values())) if n else None,
                         "unplaced_total": int(sum(r["unplaced_total"] for r in recs.values())),
                         "surrogates_computed_total": int(sum(r["n_computed"] for r in recs.values())),
                         "seconds_total": round(sum(r["s"] for r in recs.values()), 1)})
    return {"complete": bool(complete), "all_pass": bool(ok_all and complete), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "table"])
    ap.add_argument("--seg", choices=SEGS)
    ap.add_argument("--case", choices=CASES)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--fills", type=int, default=N_FILLS)
    a = ap.parse_args()
    t0 = time.time()
    if a.cmd == "run":
        segs = [a.seg] if a.seg else list(SEGS)
        cases = [a.case] if a.case else list(CASES)
        for s in segs:
            for c in cases:
                run(s, c, a.dev, a.fills)
    else:
        print(json.dumps(table(a.dev), indent=1))
    print(f"[{a.cmd}] wall {time.time() - t0:.0f} s, peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
