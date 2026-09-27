"""The two frozen v0-E guarded placebo regions (PHerc.1667 w018 ic_201, w023 ic_132), rebuilt from the committed
geometry file (results/lie_detector_v0e_regions.json) plus a small shipped coarse-grid mask per segment
(regions_data/regions_<seg>.npz, ~40 KB each -- a bool grid at the tifxyz cell resolution, not a full-resolution
image). No raw scan data is shipped or required to build the region masks.

In plain English: this file answers "which pixels count as the blank patch we trust?" using the project's frozen,
pre-registered definition. It does not look at any prediction map.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.margin.common import REPO, full_to_grid
from tools.margin.v0e import sha_mask

PKG_DIR = Path(__file__).resolve().parent
REGIONS_DIR = PKG_DIR / "regions_data"
REG_JSON = REPO / "vault" / "results" / "lie_detector_v0e_regions.json"
SEGS = ("w018", "w023")


def region_info(short: str) -> dict:
    """The segment's full geometry record from the committed regions file (source_candidate, guard, official,
    canvas_map, hf, full_res -- see results/lie_detector_v0e_regions.json for the field definitions)."""
    if short not in SEGS:
        raise SystemExit(f"unknown --segment {short!r}; must be one of {SEGS}")
    return json.loads(REG_JSON.read_text())["segments"][short]


def region_mask(short: str) -> tuple[np.ndarray, dict]:
    """The guarded placebo region as a full-resolution bool crop (True = inside the guarded region) and its
    geometry record. Rebuilt from the shipped coarse-grid NPZ and checked against the committed SHA-256 hash, so
    a tampered or stale shipped file is caught rather than silently trusted."""
    R = region_info(short)
    npz = REGIONS_DIR / f"regions_{short}.npz"
    if not npz.exists():
        raise SystemExit(f"missing shipped region file {npz}")
    with np.load(npz) as z:
        hm = z["hf_grid"].copy()
    H, W = R["hf"]["canvas_hw"]
    sy, sx = R["hf"]["grid_factor_sy_sx"]
    R0, R1 = R["full_res"]["crop_rows"]
    C0, C1 = R["full_res"]["crop_cols"]
    gy, gx = full_to_grid(H, sy, hm.shape[0]), full_to_grid(W, sx, hm.shape[1])
    m = hm[gy[R0:R1]][:, gx[C0:C1]]
    if sha_mask(m) != R["full_res"]["crop_mask_sha256"] or int(m.sum()) != R["full_res"]["px"]:
        raise SystemExit(f"{short}: rebuilt region mask does not match the committed hash in {REG_JSON}")
    return m, R
