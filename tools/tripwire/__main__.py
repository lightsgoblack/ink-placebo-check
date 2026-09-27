"""CLI for the Amendment E1 text tripwire. Prints scalars only; never writes or shows an image.

In plain English: this is a smoke alarm for AI that invents rows of letters on blank papyrus. You give it a
model's prediction map and a mask of the region that is supposed to be blank, and it tells you -- in numbers,
never pictures -- whether the "ink" it finds is shaped like rows of writing (a structured, calibrated-unlikely
pattern) rather than a random smear or a single blob.

    python -m tools.tripwire check --pred PRED.tif --region REGION_MASK.tif --voxel-um 2.4 \
        [--threshold 0.5] [--surrogates 1000] [--seed 20260925] [--json out.json]

    python -m tools.tripwire calibrate --region REGION_MASK.tif --voxel-um 2.4 [--fills 20] [--json out.json]

Registration (lining up PRED and REGION_MASK pixel-for-pixel, and cropping PRED to the region's canvas) is the
caller's job: this tool assumes the two arrays already share a shape and a pixel grid. See README.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from tools.margin import tripwire_e1 as E
from tools.tripwire.io import load_array, to_bool_mask, to_prob

DEFAULT_THRESHOLD = 0.5
DEFAULT_SURROGATES = E.N_SURR       # 1000
DEFAULT_SEED = E.SEED0              # 20260925
DEFAULT_FILLS = 20


def _load_hi_and_region(pred_path, region_path, voxel_um, threshold):
    pred = to_prob(load_array(pred_path))
    region = to_bool_mask(load_array(region_path))
    if pred.shape != region.shape:
        raise SystemExit(f"shape mismatch: --pred is {pred.shape}, --region is {region.shape}. Registration "
                          f"(cropping/aligning PRED to REGION_MASK's pixel grid) is the caller's job -- fix that "
                          f"first, this tool does not resample or align images.")
    hi = pred > threshold
    outside = int((hi & ~region).sum())
    hi = hi & region
    reg = E.Region(region, vox=voxel_um)
    return reg, hi, outside


def cmd_check(a) -> dict:
    reg, hi, outside = _load_hi_and_region(a.pred, a.region, a.voxel_um, a.threshold)
    res = reg.tripwire(hi, n_surr=a.surrogates, seed0=a.seed, stop_early=not a.no_stop_early)
    out = {
        "voxel_um": a.voxel_um, "threshold": a.threshold, "region_px": reg.px,
        "pred_px_outside_region_masked": outside,
        "part_a": {"stat": "row_periodicity_T1", "T1": res["a"]["T1"], "peak_period_mm": res["a"]["peak_period_mm"],
                   "p_a": res["a"]["p"], "p_a_lower_bound": res["a"]["p_lower_bound"], "fire": res["a"]["fire"]},
        "part_b": {"stat": "letter_row_S", "S": res["b"]["S"], "p_b": res["b"]["p"],
                   "p_b_lower_bound": res["b"]["p_lower_bound"], "fire": res["b"]["fire"]},
        "fire": res["fire"],
        "n_components": res["n_components"], "letter_blobs": res["letter_blobs"],
        "fp_area_frac": res["fp_area_frac"], "n_surrogates_computed": res["n_computed"],
        "stopped_early": res["stopped_early"],
    }
    return out


def cmd_calibrate(a) -> dict:
    from tools.margin.e1_validation import GEN, CASES, QUIET, Geo

    region = to_bool_mask(load_array(a.region))
    geo = Geo(region)
    reg = E.Region(region, vox=a.voxel_um)
    rows = []
    for case in CASES:
        fires = 0
        t0 = time.time()
        for i in range(a.fills):
            rng = np.random.default_rng([a.seed, hash(case) & 0xFFFF, i])
            hi = GEN[case](geo, rng)
            res = reg.tripwire(hi, n_surr=a.surrogates, seed0=a.seed, stop_early=True)
            fires += res["fire"]
        gate = "<= 1 of 5 fills (quiet case)" if case in QUIET else "most fills (must-fire case)"
        rows.append({"case": case, "fills": a.fills, "fires": fires, "expected": gate,
                     "seconds": round(time.time() - t0, 1)})
        print(f"  {case}: {fires}/{a.fills} fires (expect {gate})", file=sys.stderr)
    return {"region_px": reg.px, "voxel_um": a.voxel_um, "fills_per_case": a.fills, "surrogates": a.surrogates,
            "note": "This reruns the six synthetic cases of results/lie_detector_v0e_e1_validation.json "
                    "(three that must stay quiet: Q1/Q2/Q3, three that must fire: F1/F2/F3) on YOUR region mask, "
                    "with fewer fills for speed. It is a sanity check on your geometry, not a re-validation of the "
                    "frozen rule -- see README.md.",
            "rows": rows}


def main():
    ap = argparse.ArgumentParser(prog="python -m tools.tripwire", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("check", help="run the E1 tripwire on one prediction map against one region mask")
    pc.add_argument("--pred", required=True, help="prediction map: .tif/.tiff, .npy or .png (probabilities or bool)")
    pc.add_argument("--region", required=True, help="region mask: .tif/.tiff, .npy or .png (nonzero = inside)")
    pc.add_argument("--voxel-um", type=float, required=True, help="pixel size in micrometers (both axes)")
    pc.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"ink threshold on PRED (default {DEFAULT_THRESHOLD})")
    pc.add_argument("--surrogates", type=int, default=DEFAULT_SURROGATES, help=f"null surrogates (default {DEFAULT_SURROGATES})")
    pc.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"surrogate seed base (default {DEFAULT_SEED})")
    pc.add_argument("--no-stop-early", action="store_true", help="compute all surrogates even after the fire "
                                                                  "decision is settled (slower, same result)")
    pc.add_argument("--json", help="write the result JSON to this path (also printed to stdout)")

    pb = sub.add_parser("calibrate", help="rerun the synthetic quiet/must-fire cases on your own region mask")
    pb.add_argument("--region", required=True, help="your region mask: .tif/.tiff, .npy or .png (nonzero = inside)")
    pb.add_argument("--voxel-um", type=float, default=E.VOX_UM, help=f"pixel size in micrometers (default {E.VOX_UM}, "
                                                                      f"the validated geometry -- see README.md)")
    pb.add_argument("--fills", type=int, default=DEFAULT_FILLS, help=f"synthetic fills per case (default {DEFAULT_FILLS})")
    pb.add_argument("--surrogates", type=int, default=DEFAULT_SURROGATES, help=f"null surrogates per fill (default {DEFAULT_SURROGATES})")
    pb.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"seed base (default {DEFAULT_SEED})")
    pb.add_argument("--json", help="write the result JSON to this path (also printed to stdout)")

    a = ap.parse_args()
    out = cmd_check(a) if a.cmd == "check" else cmd_calibrate(a)
    text = json.dumps(out, indent=1)
    if a.json:
        with open(a.json, "w") as fh:
            fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
