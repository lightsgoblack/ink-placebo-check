"""Canvas/registration diagnostic for the AUROC-0.697 v0-E exclusion (redteam_ink.md item 9).

The 1667_2um_pred.tif exclusion (AUROC 0.495, chance-level) was called a likely canvas/provenance problem on the
strength of its registration numbers alone (near-chance AUROC, near-zero recall at p > 0.5) -- no image was ever
made or viewed, and the authors' ruling (the settled-rules file 2026-09-26) closed that file with no further work: no
re-download, no additional diagnostic. This script does NOT touch 1667_2um_pred.tif.

It runs one extra, lightweight canvas check on the OTHER v0-E registration exclusion,
`ps512_scale1_dino_frozen_w018_..._forward_220326.tif` (AUROC 0.697), that was never run for it: does a simple
canvas mirror (row flip / column flip / both, tested inside the same evaluated-label window used for
registration) noticeably improve its AUROC against the published labels? A real canvas/provenance mismatch
(wrong flattening, mirrored axis) should show much higher AUROC under the correct mirror than under identity,
the way flipping would be expected to recover alignment for a genuinely swapped axis. A merely below-threshold
but correctly-registered model should not improve much under any mirror.

Scalars only. No image is made, viewed, or saved (PHerc.1667 is a read scroll: rule (a) applies). The pred TIFF
is re-downloaded through the harness (single file), streamed in row strips, and the raw file is deleted
afterwards. The pre-registered v0-E exclusion itself is UNCHANGED by this script -- it only adds a documented
reason for why the exclusion is read as "below-threshold model", not "misregistered file", alongside the
existing AUROC/recall numbers already on record.

    .venv/bin/python -m tools.margin.canvas_diag_0697
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from tools.harness.fetch import RAW, download, free_bytes, list_remote, sha256
from tools.margin import v0e as V
from tools.margin.common import REPO, seg

SHORT = "w018"
REL = "preds/ps512_scale1_dino_frozen_w018_20240304144031_2um_ckpt_023000_forward_220326.tif"
OUT_JSON = REPO / "vault" / "results" / "lie_detector_v0e_canvas_diag_0697.json"
MIN_MEM_GB, MIN_DISK_GB = 4.0, 8.0


def mem_avail_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def auroc_from_hist(h_pos: np.ndarray, h_neg: np.ndarray, thr: float, qf: int) -> dict:
    cn = np.cumsum(h_neg) - h_neg
    npos, nneg = int(h_pos.sum()), int(h_neg.sum())
    auc = float((h_pos * (cn + 0.5 * h_neg)).sum() / (npos * nneg)) if npos and nneg else float("nan")
    hb = np.arange(len(h_pos)) > thr * qf
    return {"auroc": auc, "recall_p05": float(h_pos[hb].sum() / npos) if npos else None,
            "reference_fp_p05": float(h_neg[hb].sum() / nneg) if nneg else None,
            "n_pos_px": npos, "n_neg_px": nneg}


def main():
    assert REL != "preds/1667_2um_pred.tif", "this script never touches the closed 1667_2um_pred.tif stop"
    cm, EVp, POp = V.ev_cache(SHORT)
    er0, er1 = cm["ev_bbox_rows"]
    ec0, ec1 = cm["ev_bbox_cols"]
    nrows, ncols = er1 - er0, ec1 - ec0
    H, W = cm["canvas_hw"]

    s = seg(SHORT)
    remote = list_remote(s)
    if REL not in remote:
        raise SystemExit(f"{REL} missing from the current bucket listing")
    bf = remote[REL]
    m, d = mem_avail_gb(), free_bytes(REPO) / 2**30
    if m < MIN_MEM_GB or d - bf.size / 2**30 < MIN_DISK_GB:
        raise SystemExit(f"resources short: MemAvailable {m:.1f} GB, disk {d:.1f} GB")
    raw = RAW / Path(REL).name
    t0 = time.time()
    download(bf, raw)
    download_s = round(time.time() - t0, 1)
    digest = sha256(raw)

    P = V.Pred(raw, H, W)
    thr, nb, qf = V.value_scale(P.lz.dtype)
    variants = ("identity", "row_flip", "col_flip", "both_flip")
    hist = {v: {"pos": np.zeros(nb, np.int64), "neg": np.zeros(nb, np.int64)} for v in variants}

    t1 = time.time()
    for r0 in range((er0 // V.SUB) * V.SUB, er1, V.SUB):
        o0, o1 = max(r0, er0), min(er1, r0 + V.SUB)
        av = P.rows(o0, o1, ec0, ec1)
        # identity-order label bits for this strip
        ev_i = np.unpackbits(EVp[o0 - er0:o1 - er0], axis=1, count=ncols).view(bool)
        pos_i = np.unpackbits(POp[o0 - er0:o1 - er0], axis=1, count=ncols).view(bool)
        # row-flipped label bits: bbox row i <-> nrows-1-i, so this strip's rows pair with the mirrored block,
        # read in reverse order
        a, b = o0 - er0, o1 - er0
        ev_rblk = np.unpackbits(EVp[nrows - b:nrows - a], axis=1, count=ncols).view(bool)[::-1]
        pos_rblk = np.unpackbits(POp[nrows - b:nrows - a], axis=1, count=ncols).view(bool)[::-1]
        pairs = {
            "identity": (pos_i, ev_i),
            "row_flip": (pos_rblk, ev_rblk),
            "col_flip": (pos_i[:, ::-1], ev_i[:, ::-1]),
            "both_flip": (pos_rblk[:, ::-1], ev_rblk[:, ::-1]),
        }
        for k0 in range(0, o1 - o0, 128):
            qa = av[k0:k0 + 128]
            q = qa if P.lz.dtype.kind != "f" else np.clip(np.rint(qa * qf), 0, nb - 1).astype(np.int64)
            for v in variants:
                pk, ek = pairs[v][0][k0:k0 + 128], pairs[v][1][k0:k0 + 128]
                hist[v]["pos"] += np.bincount(q[pk].ravel(), minlength=nb)[:nb]
                hist[v]["neg"] += np.bincount(q[ek & ~pk].ravel(), minlength=nb)[:nb]
        del av, ev_i, pos_i, ev_rblk, pos_rblk
    reduce_s = round(time.time() - t1, 1)

    raw.unlink(missing_ok=True)
    free_after = round(free_bytes(REPO) / 2**30, 1)

    out = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "redteam_ink.md item 9: canvas/registration diagnostic for the AUROC-0.697 v0-E exclusion, "
                   "numbers only, no images. Does NOT touch or re-download 1667_2um_pred.tif (the authors' ruling, "
                   "the settled-rules file 2026-09-26, stands unchanged).",
        "pred": REL, "model": "ps512_scale1_dino_frozen", "segment": SHORT, "sha256": digest,
        "bytes": bf.size, "download_s": download_s, "reduce_s": reduce_s,
        "dtype": str(P.lz.dtype), "upsample_note": P.note,
        "eval_window": {"bbox_rows_px": [er0, er1], "bbox_cols_px": [ec0, ec1], "canvas_hw": [H, W]},
        "method": "AUROC recomputed from the same evaluated-label window used for registration, under 4 canvas "
                  "hypotheses: identity (the registered result, on record), row_flip (label window mirrored "
                  "top-bottom), col_flip (mirrored left-right), both_flip (180-degree rotation). Pred pixel "
                  "values are read once, in normal order; only which label bits they are paired against changes.",
        "variants": {v: auroc_from_hist(hist[v]["pos"], hist[v]["neg"], thr, qf) for v in variants},
        "on_record_registration_auroc": 0.6970301393200028,
        "free_disk_gb_after_delete": free_after,
    }
    V.jdump(out, OUT_JSON)
    print(json.dumps({v: round(out["variants"][v]["auroc"], 4) for v in variants}, indent=1))
    print("wrote", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
