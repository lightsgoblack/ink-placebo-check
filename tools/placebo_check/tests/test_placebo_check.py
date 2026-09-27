"""Synthetic-array tests for the placebo_check CLI: region-hash tamper detection, registration exclusion, a
quiet single-blob placebo (must not fire), and a periodic-band placebo (must fire). No scroll data, no images."""
from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

from tools.margin.common import full_to_grid
from tools.margin.v0e import sha_mask
from tools.placebo_check import __main__ as pc_main
from tools.placebo_check import regions as pc_regions

VOX = 2.399
PPM = 1000.0 / VOX


# ------------------------------------------------------------------------------------------------ regions.py
def test_region_mask_rebuilds_and_checks_hash(tmp_path, monkeypatch):
    GH, GW = 20, 30                      # small coarse tifxyz grid
    hm = np.zeros((GH, GW), bool)
    hm[5:12, 8:20] = True                # the "guarded region" on the coarse grid
    cell_px = 4                          # 4 full-res px per grid cell
    sy = sx = 1.0 / cell_px              # grid index = round(full_px * s)
    H, W = GH * cell_px, GW * cell_px
    gy, gx = full_to_grid(H, sy, GH), full_to_grid(W, sx, GW)
    R0, R1, C0, C1 = 4 * cell_px, 14 * cell_px, 6 * cell_px, 22 * cell_px     # aligned to the grid so the crop is exact
    m = hm[gy[R0:R1]][:, gx[C0:C1]]

    npz_dir = tmp_path / "regions_data"
    npz_dir.mkdir()
    np.savez_compressed(npz_dir / "regions_w018.npz", hf_grid=hm)
    reg_json = tmp_path / "regions.json"
    reg_json.write_text(json.dumps({"segments": {"w018": {
        "hf": {"canvas_hw": [H, W], "grid_factor_sy_sx": [sy, sx]},
        "full_res": {"crop_rows": [R0, R1], "crop_cols": [C0, C1], "crop_mask_sha256": sha_mask(m),
                     "px": int(m.sum())},
    }}}))
    monkeypatch.setattr(pc_regions, "REGIONS_DIR", npz_dir)
    monkeypatch.setattr(pc_regions, "REG_JSON", reg_json)

    got, R = pc_regions.region_mask("w018")
    assert np.array_equal(got, m)
    assert R["full_res"]["px"] == int(m.sum())

    with pytest.raises(SystemExit):
        pc_regions.region_info("not_a_segment")

    # a corrupted shipped file is caught, not silently trusted
    bad = hm.copy()
    bad[6, 10] = False                   # flip a cell inside the crop -> hash mismatch
    np.savez_compressed(npz_dir / "regions_w018.npz", hf_grid=bad)
    with pytest.raises(SystemExit):
        pc_regions.region_mask("w018")


# ------------------------------------------------------------------------------------------------ CLI run()
def _disk(shape, r0, c0, rad_px):
    a = np.zeros(shape, bool)
    R = int(np.ceil(rad_px))
    dy, dx = np.mgrid[-R:R + 1, -R:R + 1]
    d = dy ** 2 + dx ** 2 <= rad_px ** 2
    y0, y1, x0, x1 = max(0, r0 - R), min(shape[0], r0 + R + 1), max(0, c0 - R), min(shape[1], c0 + R + 1)
    a[y0:y1, x0:x1] = d[y0 - r0 + R:y1 - r0 + R, x0 - c0 + R:x1 - c0 + R]
    return a


class _A:
    def __init__(self, **kw):
        self.pred = None
        self.segment = "w018"
        self.surrogates = 1000
        self.seed = 20260925
        self.no_stop_early = False
        self.json = None
        self.__dict__.update(kw)


def _build_canvas(placebo_fill):
    """H x W uint8 canvas: an evaluated-label window (rows [0, 300), cols [0, 300)) with a clean positive
    rectangle, and a placebo region crop below it filled by `placebo_fill(mask_shape) -> bool array`."""
    h_mm, w_mm = 26.0, 4.0
    h_px, w_px = int(round(h_mm * PPM)), int(round(w_mm * PPM))
    mask = np.ones((h_px, w_px), bool)
    R0, C0 = 300, 0
    H, W = R0 + h_px, max(w_px, 300)
    pred = np.zeros((H, W), np.uint8)
    pred[50:150, 50:150] = 200                          # labeled ink, matched by the "good" pred
    fill = placebo_fill((h_px, w_px))
    pred[R0:R0 + h_px, C0:C0 + w_px] = np.where(fill, np.uint8(200), np.uint8(0))
    cm = {"ev_bbox_rows": [0, 300], "ev_bbox_cols": [0, 300]}
    ev = np.ones((300, 300), bool)
    pos = np.zeros((300, 300), bool)
    pos[50:150, 50:150] = True
    EVp = np.packbits(ev, axis=1)
    POp = np.packbits(pos, axis=1)
    R = {"name": "test_region", "hf": {"canvas_hw": [H, W]},
         "full_res": {"crop_rows": [R0, R0 + h_px], "crop_cols": [C0, C0 + w_px], "px": int(mask.sum()),
                     "px_um": VOX, "area_mm2": round(mask.sum() * (VOX / 1000.0) ** 2, 1)}}
    return pred, mask, R, cm, EVp, POp


def _run(tmp_path, monkeypatch, pred_arr, mask, R, cm, EVp, POp, pred_val_ok=True):
    pred_path = tmp_path / "pred.tif"
    tifffile.imwrite(pred_path, pred_arr)
    monkeypatch.setattr(pc_main.regions, "region_mask", lambda short: (mask, R))
    monkeypatch.setattr(pc_main.V, "ev_cache", lambda short: (cm, EVp, POp))
    return pc_main.run(_A(pred=str(pred_path)))


def test_registration_excludes_low_auroc(tmp_path, monkeypatch):
    def fill(shape):
        return np.zeros(shape, bool)
    pred, mask, R, cm, EVp, POp = _build_canvas(fill)
    pred[0:300, 0:300] = 100                      # positives and negatives at the same value -> AUROC ~ 0.5
    out = _run(tmp_path, monkeypatch, pred, mask, R, cm, EVp, POp)
    assert out["registration_ok"] is False
    assert "excluded" in out
    assert "placebo" not in out


def test_quiet_single_blob_does_not_fire(tmp_path, monkeypatch):
    def fill(shape):
        r0, c0 = shape[0] // 2, shape[1] // 2
        area_mm2 = 2.0
        rad_px = np.sqrt(area_mm2 / np.pi) * PPM
        return _disk(shape, r0, c0, rad_px)
    pred, mask, R, cm, EVp, POp = _build_canvas(fill)
    out = _run(tmp_path, monkeypatch, pred, mask, R, cm, EVp, POp)
    assert out["registration_ok"] is True
    assert out["placebo"]["fp_px"] > 0
    assert out["tripwire_e1"]["fire"] is False
    assert out["fire"] is False


def test_periodic_bands_fire(tmp_path, monkeypatch):
    def fill(shape):
        a = np.zeros(shape, bool)
        thick_px = int(round(1.0 * PPM))
        for k in range(4):
            r0 = int(round(k * 6.0 * PPM))
            if r0 + thick_px > shape[0]:
                break
            a[r0:r0 + thick_px, :] = True
        return a
    pred, mask, R, cm, EVp, POp = _build_canvas(fill)
    out = _run(tmp_path, monkeypatch, pred, mask, R, cm, EVp, POp)
    assert out["registration_ok"] is True
    assert out["tripwire_e1"]["fire"] is True
    assert out["tripwire_e1"]["part_a"]["fire"] is True
    assert out["fire"] is True
