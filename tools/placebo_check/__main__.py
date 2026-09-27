"""python -m tools.placebo_check run --pred PRED.tif --segment w018|w023 [--json out.json]

The drop-in placebo check CLI: a lie-detector test for an ink-prediction TIFF on PHerc.1667 w018 or w023,
against the project's frozen v0-E edition-defined intercolumn placebo regions. Prints scalars only; never
writes, shows, or saves any image, crop, or rendering. See README.md for the method and "In plain English".

Order (mirrors the frozen v0-E / E1 pipeline, code unmodified, imported directly):
  1. registration: pixel AUROC of PRED against the published inklabels_v2 inside the evaluated-label window
     (tools.margin.v0e_e1.registration_pass). AUROC < 0.75 -> excluded before any placebo pixel is read.
  2. placebo false-positive rate on the guarded intercolumn region (tools.margin.v0e_e1.placebo_pass), reported
     against the same pred's in-mask unlabeled text-block FP rate (the frozen v0-E measures; a ratio near 1 means
     the model is no more "confident" in the blank patch than in ordinary unlabeled text-block surface).
  3. the E1 text tripwire on the placebo p > threshold map (tools.margin.tripwire_e1.Region.tripwire, the same
     call tools/tripwire's `check` subcommand makes): row periodicity + letter-row structure vs a component-
     shuffle null.

This tool assumes PRED already shares the labeled canvas's pixel grid (same shape, or an exact integer
downsample of it -- tools.margin.v0e.Pred upsamples that case automatically). It does not resample or align
an unrelated canvas; see README.md "Limits".
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tools.harness.fetch import sha256
from tools.margin import tripwire_e1 as E
from tools.margin import v0e as V
from tools.margin import v0e_e1 as E1
from tools.placebo_check import regions

SEGS = regions.SEGS


def run(a) -> dict:
    short = a.segment
    mask, R = regions.region_mask(short)
    H, W = R["hf"]["canvas_hw"]
    vox = R["full_res"]["px_um"]
    region_px = R["full_res"]["px"]
    px_mm2 = (vox / 1000.0) ** 2
    pred_path = Path(a.pred)
    if not pred_path.exists():
        raise SystemExit(f"--pred not found: {pred_path}")
    t0 = time.time()
    digest = sha256(pred_path)
    cm, EVp, POp = V.ev_cache(short)
    P = V.Pred(pred_path, H, W)
    out = {"pred": str(pred_path), "sha256": digest, "segment": short, "region": R["name"],
           "region_area_mm2": R["full_res"]["area_mm2"], "file_info": dict(P.lz.info),
           "upsample_note": P.note, "upsample_factor": P.f}
    if P.f == 1 and P.note:
        P.lz.close()
        out["excluded"] = P.note
        out["registration_ok"] = None
        out["runtime_s"] = round(time.time() - t0, 1)
        return out
    dt = np.dtype(P.lz.dtype)
    rg = E1.registration_pass(P, dt, cm, EVp, POp)
    rg.pop("_hist")
    out.update(rg)
    if not out["registration_ok"]:
        P.lz.close()
        out["excluded"] = f"registration AUROC {out['registration_auroc']:.3f} < 0.75: excluded before any placebo pixel is read"
        out["runtime_s"] = round(time.time() - t0, 1)
        return out
    hi, vmin, vmax = E1.placebo_pass(P, dt, R, mask)
    P.lz.close()
    fp_px = int(hi.sum())
    out["value_min_max_placebo"] = [vmin, vmax]
    out["placebo"] = {
        "region_px": region_px, "fp_px": fp_px, "fp_area_frac": fp_px / region_px,
        "reference_fp_p05_inmask_unlabeled": out["reference_fp_p05_inmask_unlabeled"],
        "placebo_over_reference": (fp_px / region_px) / out["reference_fp_p05_inmask_unlabeled"]
                                  if out["reference_fp_p05_inmask_unlabeled"] else None,
    }
    reg = E.Region(mask, vox=vox)
    tw = reg.tripwire(hi, n_surr=a.surrogates, seed0=a.seed, stop_early=not a.no_stop_early)
    reg.release()
    out["tripwire_e1"] = {
        "part_a": {"stat": "row_periodicity_T1", "T1": tw["a"]["T1"], "peak_period_mm": tw["a"]["peak_period_mm"],
                   "p_a": tw["a"]["p"], "fire": tw["a"]["fire"]},
        "part_b": {"stat": "letter_row_S", "S": tw["b"]["S"], "p_b": tw["b"]["p"], "fire": tw["b"]["fire"]},
        "fire": tw["fire"], "n_components": tw["n_components"], "letter_blobs": tw["letter_blobs"],
        "letter_blob_per_cm2": tw["letter_blobs"] / (region_px * px_mm2 / 100.0),
        "n_surrogates_computed": tw["n_computed"], "stopped_early": tw["stopped_early"],
    }
    out["fire"] = tw["fire"]
    out["runtime_s"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser(prog="python -m tools.placebo_check", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run", help="registration -> placebo FP rate -> E1 tripwire on one prediction TIFF")
    pr.add_argument("--pred", required=True, help="prediction TIFF: same canvas as the segment's labels, or an "
                                                    "exact integer downsample of it")
    pr.add_argument("--segment", required=True, choices=SEGS, help="PHerc.1667 segment (w018 or w023)")
    pr.add_argument("--surrogates", type=int, default=E.N_SURR, help=f"tripwire null surrogates (default {E.N_SURR})")
    pr.add_argument("--seed", type=int, default=E.SEED0, help=f"tripwire surrogate seed base (default {E.SEED0})")
    pr.add_argument("--no-stop-early", action="store_true", help="compute all surrogates even after the fire "
                                                                   "decision is settled (slower, same result)")
    pr.add_argument("--json", help="write the result JSON to this path (also printed to stdout)")
    a = ap.parse_args()
    out = run(a)
    text = json.dumps(out, indent=1)
    if a.json:
        Path(a.json).write_text(text + "\n")
    print(text)
    if out.get("fire"):
        print("\nTEXT-LIKE RESULT IN A PLACEBO REGION. Under this project's text rule, that is a hard stop: "
              "scalars only, save no images, and a human decides next. See tools/placebo_check/README.md.")


if __name__ == "__main__":
    main()
