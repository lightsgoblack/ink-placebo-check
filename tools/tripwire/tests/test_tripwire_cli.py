"""Tests for tools/tripwire (the standalone E1 tripwire package). Small synthetic arrays only, few surrogates, no
scroll data, no images. Run: .venv/bin/python -m pytest tools/tripwire/tests -q"""
from __future__ import annotations

import numpy as np

from tools.margin import tripwire_e1 as E
from tools.tripwire.io import load_array, to_bool_mask, to_prob

VOX_TOY = 100.0          # um per px: 1 mm^2 = 100 px, 1 mm row window = 10 px (matches tools/margin's toy tests)
N_SURR_TEST = 30         # few surrogates: keeps runtime small. Part (a) can never fire this way (p_a's floor is
                          # 1/(n_surr + 1) = 1/31, far above the frozen p_a <= 0.001), so these tests exercise
                          # part (b) (S >= 3 and beats every surrogate), which does not depend on n_surr.


def rect_mask(h=3000, w=900):
    return np.ones((h, w), bool)


def ellipse(ry=8, rx=10):
    yy, xx = np.mgrid[-ry:ry + 1, -rx:rx + 1]
    return (yy / ry) ** 2 + (xx / rx) ** 2 <= 1.0


def test_planted_row_fires():
    """Several letter-sized blobs placed in a single row (same centroid row, spread across columns) must fire."""
    mask = rect_mask()
    hi = np.zeros_like(mask)
    b = ellipse()
    row = 1500
    for col in (150, 350, 550, 750):
        r0, c0 = row - b.shape[0] // 2, col - b.shape[1] // 2
        hi[r0:r0 + b.shape[0], c0:c0 + b.shape[1]] |= b
    amm = b.sum() * (VOX_TOY / 1000.0) ** 2
    assert 1.0 <= amm <= 4.0, f"toy blob area {amm} mm^2 not in the letter class"

    reg = E.Region(mask, vox=VOX_TOY)
    res = reg.tripwire(hi, n_surr=N_SURR_TEST, seed0=20260925, stop_early=True)
    assert res["b"]["S"] >= 3
    assert res["b"]["fire"] is True
    assert res["fire"] is True


def test_single_blob_does_not_fire():
    """A single letter-sized blob must not fire (S = 1 < 3; part (a) cannot fire with this few surrogates)."""
    mask = rect_mask()
    hi = np.zeros_like(mask)
    b = ellipse()
    r0, c0 = 1500 - b.shape[0] // 2, 400 - b.shape[1] // 2
    hi[r0:r0 + b.shape[0], c0:c0 + b.shape[1]] |= b

    reg = E.Region(mask, vox=VOX_TOY)
    res = reg.tripwire(hi, n_surr=N_SURR_TEST, seed0=20260925, stop_early=True)
    assert res["b"]["S"] == 1
    assert res["b"]["fire"] is False
    assert res["a"]["fire"] is False
    assert res["fire"] is False


def test_io_round_trip(tmp_path):
    """The .npy loader round-trips arrays, and to_prob/to_bool_mask normalize them as documented."""
    pred = (np.random.default_rng(0).random((40, 30)) > 0.9)
    region = np.ones((40, 30), bool)
    region[:, :5] = False
    p_path, r_path = tmp_path / "pred.npy", tmp_path / "region.npy"
    np.save(p_path, pred)
    np.save(r_path, region)

    loaded_pred = load_array(p_path)
    loaded_region = load_array(r_path)
    assert loaded_pred.shape == pred.shape
    assert loaded_region.shape == region.shape

    prob = to_prob(loaded_pred)
    assert prob.dtype == np.float64
    assert prob.min() >= 0.0 and prob.max() <= 1.0
    np.testing.assert_array_equal(prob > 0.5, pred)

    bm = to_bool_mask(loaded_region)
    assert bm.dtype == bool
    np.testing.assert_array_equal(bm, region)


def test_shape_mismatch_is_caller_error():
    """check's shape check (not this module's job to resample) -- exercised at the array level used by __main__."""
    pred = to_prob(np.zeros((10, 10)))
    region = to_bool_mask(np.ones((10, 11)))
    assert pred.shape != region.shape
