"""Check 6b: replication on w029's band with the single model that fired on w031 (+ render only if the pred
does NOT fire). Pre-registered: if the pred fires (v0-M rule) -> text-rule trigger for a new region: record
scalars, do not pull the render, stop all w029 work.

    .venv/bin/python -m tools.margin.forensics_w029 run
"""
from __future__ import annotations

import json
import sys

import numpy as np

from tools.harness.fetch import list_remote
from tools.margin.forensics import OUT, do_static, pull_reduce
from tools.margin.forensics_analyze import Profile, fullres_T1, periodicity, load
from tools.margin.common import seg

MODEL = "preds/merged_confidence_vote_betti_ema_640_forward.tif"


def fired(short="w029"):
    st, info, mods = load(short)
    o, a = mods[MODEL.split("/")[-1]]
    t1 = fullres_T1(o, a, st)
    return bool(o["band_fp_frac"] >= 0.001 and ((t1.get("ratio") or 0) >= 10 or o["band_blobs"]["T2_flag"])), t1


def summarize():
    st, info, mods = load("w029")
    out = {"model": MODEL}
    k = MODEL.split("/")[-1]
    if k not in mods:
        return {"status": "not run"}
    o, a = mods[k]
    f, t1 = fired()
    bb = o["band_blobs"]
    out.update({"registration_auroc": o["registration_auroc"], "band_fp_frac": o["band_fp_frac"],
                "fpr_p05_text_unlabeled": o["fpr_p05_text_unlabeled"], "letter_blobs": bb["letter_blobs"],
                "max_letter_blobs_in_one_row": bb["max_letter_blobs_in_one_row"], "T2_flag": bb["T2_flag"],
                "T1_fullres": {"ratio": t1.get("ratio"), "peak_period_mm": t1.get("peak_period_mm")},
                "note": "stop rule: only scalars from the reduction pass are recorded; no null/phase analysis on w029",
                "fires_v0M_rule": f})
    rk = [m for m in mods if m.endswith("max_22_42.tif")]
    out["render"] = "not pulled (pred fired: new-region stop)" if f else ("reduced" if rk else "pending")
    out["status"] = "TEXT-RULE TRIGGER (new region): w029 work stopped" if f else "pred did not fire"
    return out


def run():
    do_static("w029")
    s = seg("w029")
    remote = list_remote(s)
    pull_reduce("w029", MODEL, "pred", remote)
    f, t1 = fired()
    print("w029 fires:", f, t1, flush=True)
    if f:
        print("TEXT-RULE TRIGGER on w029 band: render NOT pulled; w029 stopped.", flush=True)
        return
    pull_reduce("w029", s.max_render, "render", remote)


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        run()
    else:
        print(json.dumps(summarize(), indent=1, default=float))
