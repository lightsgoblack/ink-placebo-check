"""Unit tests for the Amendment E1 text tripwire (tools/margin/tripwire_e1.py). Synthetic toy maps only, no scroll data,
no images. Run: .venv/bin/python -m pytest tools/margin/tests -q"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from tools.margin import tripwire_e1 as E
from tools.margin.deep_ld import letter_S
from tools.margin.forensics_analyze import CMM, Profile
from tools.margin.reduce_pred import label_components, periodogram_stat

VOX_TOY = 100.0          # um per px in the toys: 1 mm2 = 100 px, 1 mm window = 10 px (letter logic at toy scale)


# ------------------------------------------------------------------------------------------------ toy builders
def toy_mask(seed=0, H=2288, W=832):
    """Ragged vertical strip (22 x 8 cells) with a hole, so some rows have two runs."""
    rng = np.random.default_rng(seed)
    m = np.zeros((H, W), bool)
    for r0 in range(150, H - 200, 20):
        a = 100 + int(rng.integers(-15, 16))
        b = W - 120 + int(rng.integers(-15, 16))
        m[r0:r0 + 20, a:b] = True
    m[400:440, 300:340] = False
    return m


def blob(rng, hmin=3, hmax=40):
    """A random blob: union of a few ellipses (8-connected, possibly non-convex)."""
    h, w = int(rng.integers(hmin, hmax)), int(rng.integers(hmin, hmax))
    yy, xx = np.mgrid[:h, :w]
    b = ((yy - h / 2 + 0.5) / (h / 2)) ** 2 + ((xx - w / 2 + 0.5) / (w / 2)) ** 2 <= 1
    if rng.random() < 0.5 and h > 6 and w > 6:     # carve a notch for non-convex shapes
        b[h // 3:2 * h // 3, : w // 2] = False
    lab, n = ndi.label(b, E.ST8)
    if n > 1:
        b = lab == (1 + int(np.argmax(np.bincount(lab.ravel())[1:])))
    return b


def toy_map(mask, rng, n_blobs=20, n_single=10, sizes=(3, 40)):
    """Non-touching random blobs and singletons fully inside the mask."""
    hi = np.zeros_like(mask)
    shapes = [blob(rng, *sizes) for _ in range(n_blobs)] + [np.ones((1, 1), bool)] * n_single
    for b in shapes:
        for _ in range(500):
            r = int(rng.integers(0, mask.shape[0] - b.shape[0]))
            c = int(rng.integers(0, mask.shape[1] - b.shape[1]))
            win = (slice(r, r + b.shape[0]), slice(c, c + b.shape[1]))
            if not mask[win][b].all():
                continue
            gr = (slice(max(r - 1, 0), r + b.shape[0] + 1), slice(max(c - 1, 0), c + b.shape[1] + 1))
            if hi[gr].any():
                continue
            hi[win] |= b
            break
    return hi


def render(comps, ka, kb, shape):
    out = np.zeros(shape, np.int32)
    for c in range(comps.n):
        rr, cc = comps.pixels(c)
        out[rr + E.CELL * ka[c], cc + E.CELL * kb[c]] += 1
    return out


# ------------------------------------------------------------------------------------------------ tests
def test_shift_table_matches_brute_force():
    rng = np.random.default_rng(1)
    cell = 7
    for _ in range(40):
        A = rng.random((25, 30)) < 0.15
        B = rng.random((12, 9)) < 0.3
        Ar, A0, A1 = E._bitmap_runs(A, 0, 0)
        Br, B0, B1 = E._bitmap_runs(B, 0, 0)
        T = E.shift_table(Ar, A0, A1, Br, B0, B1, -3, 4, -2, 5, cell)
        for a in range(-3, 5):
            for b in range(-2, 6):
                big = np.zeros((80, 90), bool)
                big[30:55, 30:60] = A
                sh = np.zeros_like(big)
                r, c = 30 + cell * a, 30 + cell * b
                sh[r:r + 12, c:c + 9] = B
                assert T[a + 3, b + 2] == bool((big & sh).any())


def test_pool_and_row_pool():
    rng = np.random.default_rng(2)
    a = rng.random((1000, 333)) < 0.1
    from tools.margin.v0e import pool as v0e_pool
    assert np.array_equal(E.pool(a), v0e_pool(a))
    assert np.array_equal(E.row_pool(a), E.pool(a).sum(1))


def test_components_match_frozen_label_components_and_letter_S():
    mask = toy_mask(3)
    rng = np.random.default_rng(3)
    for seed in range(3):
        hi = toy_map(mask, rng, n_blobs=30, n_single=15)
        C = E.Components(hi, VOX_TOY)
        lc = label_components(hi, (VOX_TOY / 1000) ** 2)
        assert C.n == len(lc["area_px"])
        o = np.lexsort((lc["cent_c"], lc["cent_r"]))
        cent_c = np.array([C.pixels(c)[1].mean() for c in range(C.n)])
        mine = np.lexsort((cent_c, C.cent_r))
        assert np.array_equal(C.area[mine], lc["area_px"][o].astype(np.int64))
        assert np.array_equal(C.cent_r[mine], lc["cent_r"][o])
        S, n_let, _ = letter_S(hi, VOX_TOY)
        zero = np.zeros((1, C.n), np.int64)
        assert int(C.s_values(zero)[0]) == S and C.letter.size == n_let
        assert np.array_equal(C.rowcounts(zero)[0], E.pool(hi).sum(1))
        # placement order: area descending
        assert np.all(np.diff(C.area) <= 0)


def test_inside_tables_match_brute_force():
    mask = toy_mask(4)
    reg = E.Region(mask, VOX_TOY)
    hi = toy_map(mask, np.random.default_rng(4), n_blobs=12, n_single=4)
    C = E.Components(hi, VOX_TOY)
    sh = E.Shuffle(reg, C, "tables")
    for c in range(C.n):
        pr, pc = C.pixels(c)
        V = sh.V[sh.voff[c]:sh.voff[c] + sh.nA[c] * sh.nB[c]].reshape(sh.nA[c], sh.nB[c])
        for a in range(sh.nA[c]):
            for b in range(sh.nB[c]):
                rr = pr + E.CELL * (sh.kaMin[c] + a)
                cc = pc + E.CELL * (sh.kbMin[c] + b)
                assert V[a, b] == bool(mask[rr, cc].all())


def test_pair_tables_match_brute_force():
    mask = toy_mask(5)
    reg = E.Region(mask, VOX_TOY)
    hi = toy_map(mask, np.random.default_rng(5), n_blobs=8, n_single=3)
    C = E.Components(hi, VOX_TOY)
    sh = E.Shuffle(reg, C, "tables")
    tab = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(sh.pi, sh.pj))}
    for i in range(C.n):
        ir, ic = C.pixels(i)
        r0, c0 = C.r0[i] - 1, C.c0[i] - 1
        Di = np.zeros((C.r1[i] - C.r0[i] + 2, C.c1[i] - C.c0[i] + 2), bool)
        Di[ir - r0, ic - c0] = True
        Di = ndi.binary_dilation(Di, E.ST8)
        for j in range(i + 1, C.n):
            jr, jc = C.pixels(j)
            for da in range(sh.kaMin[j] - sh.kaMax[i], sh.kaMax[j] - sh.kaMin[i] + 1):
                for db in range(sh.kbMin[j] - sh.kbMax[i], sh.kbMax[j] - sh.kbMin[i] + 1):
                    r = jr + E.CELL * da - r0
                    c = jc + E.CELL * db - c0
                    ok = (r >= 0) & (r < Di.shape[0]) & (c >= 0) & (c < Di.shape[1])
                    truth = bool(Di[r[ok], c[ok]].any())
                    k = tab.get((i, j))
                    got = False
                    if k is not None:
                        a, b = da - sh.pa_lo[k], db - sh.pb_lo[k]
                        if 0 <= a < sh.pa_n[k] and 0 <= b < sh.pb_n[k]:
                            got = bool(sh.PT[sh.poff[k] + a * sh.pb_n[k] + b])
                    assert got == truth, (i, j, da, db)


@pytest.mark.parametrize("seed", [6, 7, 8])
def test_tables_and_bitmap_give_identical_surrogates(seed):
    mask = toy_mask(seed)
    reg = E.Region(mask, VOX_TOY)
    rng = np.random.default_rng(seed)
    hi = toy_map(mask, rng, n_blobs=40, n_single=25, sizes=(3, 60))
    C = E.Components(hi, VOX_TOY, rings=True)
    st = E.Shuffle(reg, C, "tables")
    sb = E.Shuffle(reg, C, "bitmap")
    seeds = [E.SEED0 + k for k in range(12)] + [[E.SEED0 + k, 3] for k in range(4)]
    KA, KB, PL, RD = st.place_tables(seeds)
    for s, sd in enumerate(seeds):
        ka, kb, pl, rd = sb.place_bitmap(sd)
        assert np.array_equal(ka, KA[s]) and np.array_equal(kb, KB[s]) and np.array_equal(pl, PL[s])
        assert rd == RD[s]
    assert (reg.owner() == E.I32MAX).all()          # the bitmap is reset after every surrogate


def test_identical_with_unplaceable_components(monkeypatch):
    """Few tries so that some components stay at their original positions; both implementations agree."""
    monkeypatch.setattr(E, "MAX_TRIES", 3)
    mask = toy_mask(9)
    reg = E.Region(mask, VOX_TOY)
    hi = toy_map(mask, np.random.default_rng(9), n_blobs=60, n_single=10, sizes=(20, 90))
    C = E.Components(hi, VOX_TOY, rings=True)
    st, sb = E.Shuffle(reg, C, "tables"), E.Shuffle(reg, C, "bitmap")
    seeds = [E.SEED0 + k for k in range(10)]
    KA, KB, PL, RD = st.place_tables(seeds)
    assert (~PL).sum() > 0
    for s, sd in enumerate(seeds):
        ka, kb, pl, rd = sb.place_bitmap(sd)
        assert np.array_equal(ka, KA[s]) and np.array_equal(kb, KB[s]) and np.array_equal(pl, PL[s])
        assert np.all(KA[s][~PL[s]] == 0) and np.all(KB[s][~PL[s]] == 0)


@pytest.mark.parametrize("impl", ["tables", "bitmap"])
def test_surrogates_valid_and_statistics_equal_rendered_map(impl):
    mask = toy_mask(10)
    reg = E.Region(mask, VOX_TOY)
    rng = np.random.default_rng(10)
    hi = toy_map(mask, rng, n_blobs=35, n_single=20, sizes=(3, 50))
    C = E.Components(hi, VOX_TOY, rings=True)
    sh = E.Shuffle(reg, C, impl)
    seeds = [E.SEED0 + k for k in range(8)]
    if impl == "tables":
        KA, KB, PL, _ = sh.place_tables(seeds)
    else:
        out = [sh.place_bitmap(s) for s in seeds]
        KA, KB, PL = (np.array([o[i] for o in out]) for i in range(3))
    rc, sv = C.rowcounts(KA), C.s_values(KA)
    for s in range(len(seeds)):
        assert PL[s].all()
        img = render(C, KA[s], KB[s], mask.shape)
        assert img.max() == 1                           # no overlap
        m = img > 0
        assert not (m & ~mask).any()                    # fully inside the region
        lab, n = ndi.label(m, E.ST8)                    # no touching: the same components come back
        assert n == C.n
        assert np.array_equal(np.sort(np.bincount(lab.ravel())[1:]), np.sort(C.area))
        pr = Profile(E.pool(m), reg.bcells)
        t1_frozen = periodogram_stat(pr.prof(), CMM * 1000)["ratio"]
        assert reg.t1_rows(rc[s]) == t1_frozen
        assert int(sv[s]) == letter_S(m, VOX_TOY)[0]
        assert reg.t1_rows(rc[s]) == reg.t1(E.pool(m))[0]


def test_observed_t1_is_the_frozen_statistic():
    mask = toy_mask(11)
    reg = E.Region(mask, VOX_TOY)
    hi = toy_map(mask, np.random.default_rng(11), n_blobs=20, n_single=5)
    obs = reg.observed(hi)
    pr = Profile(E.pool(hi), E.pool(mask))
    st = periodogram_stat(pr.prof(), CMM * 1000)
    assert obs["T1"] == st["ratio"] and obs["peak_period_mm"] == st["peak_period_mm"]


def tall_mask(seed=0, H=31200, W=520):
    """A 300-cell (75 mm) strip so that the T1 band (periods 2.5-8 mm) is populated, as on the real regions."""
    rng = np.random.default_rng(seed)
    m = np.zeros((H, W), bool)
    for r0 in range(300, H - 400, 40):
        m[r0:r0 + 40, 60 + int(rng.integers(-10, 11)):W - 60 + int(rng.integers(-10, 11))] = True
    return m


def periodic_rows_map(mask, pitch_cells=16, thick=200, phase=0):
    """Full-width bands every pitch_cells cells (16 cells = 4 mm on the frozen 0.25 mm rows)."""
    hi = np.zeros_like(mask)
    r0, r1, c0, c1 = E.Region(mask).bbox
    for r in range(r0 + 60 + phase, r1 - thick - 60, pitch_cells * E.CELL):
        cc = np.flatnonzero(mask[r:r + thick].all(0))
        hi[r:r + thick, cc[0] + 10:cc[-1] - 10] = True
    return hi & mask


def test_fire_rule_periodic_rows_fire_and_single_blob_quiet():
    mask = tall_mask(12)
    reg = E.Region(mask, VOX_TOY)
    res = reg.tripwire(periodic_rows_map(mask), n_surr=1000)
    assert res["a"]["fire"] and res["fire"] and res["a"]["n_null_ge"] == 0
    assert res["a"]["p"] == pytest.approx(1 / 1001)
    assert 3.5 < res["a"]["peak_period_mm"] < 4.5
    one = toy_map(mask, np.random.default_rng(12), n_blobs=1, n_single=0, sizes=(20, 60))
    r1 = reg.tripwire(one, n_surr=1000)
    assert r1["n_components"] == 1
    assert r1["a"]["p"] == (1 + r1["a"]["n_null_ge"]) / 1001


def test_part_b_can_fire_on_a_letter_row():
    """Toy scale (VOX_TOY): 5 letter-sized blobs (1-4 mm2) with centroids inside one 1 mm row window."""
    mask = toy_mask(13)
    reg = E.Region(mask, VOX_TOY)
    hi = np.zeros_like(mask)
    for k in range(5):
        c = 180 + 110 * k
        hi[1000:1015, c:c + 12] = True               # 180 px = 1.8 mm2 each
    res = reg.tripwire(hi, n_surr=1000)
    assert res["b"]["S"] == 5 and res["b"]["fire"] and res["b"]["null_max"] < 5


def test_stop_early_gives_the_same_decision():
    mask = toy_mask(14)
    reg = E.Region(mask, VOX_TOY)
    rng = np.random.default_rng(14)
    tall = tall_mask(14)
    rt = E.Region(tall, VOX_TOY)
    cases = [(reg, toy_map(mask, rng, n_blobs=4, n_single=0)), (rt, periodic_rows_map(tall, 24, 250)),
             (reg, toy_map(mask, rng, n_blobs=30, n_single=200, sizes=(2, 8)))]
    for reg, hi in cases:
        full = reg.tripwire(hi, n_surr=1000)
        fast = reg.tripwire(hi, n_surr=1000, stop_early=True)
        assert full["fire"] == fast["fire"] and full["a"]["fire"] == fast["a"]["fire"]
        assert full["b"]["fire"] == fast["b"]["fire"]
        if fast["stopped_early"]:
            assert not full["fire"] and fast["n_computed"] < 1000


def test_bitmap_path_tripwire_equals_tables_path():
    mask = toy_mask(15)
    reg = E.Region(mask, VOX_TOY)
    hi = toy_map(mask, np.random.default_rng(15), n_blobs=40, n_single=40, sizes=(2, 30))
    a = reg.tripwire(hi, n_surr=60, impl="tables", keep_null=True)
    b = reg.tripwire(hi, n_surr=60, impl="bitmap", keep_null=True)
    assert np.array_equal(a["_null"]["T1"], b["_null"]["T1"]) and np.array_equal(a["_null"]["S"], b["_null"]["S"])
    assert a["placement"]["implementation"] == "tables" and b["placement"]["implementation"] == "bitmap"


def test_many_components_use_the_bitmap_path():
    mask = toy_mask(16)
    reg = E.Region(mask, VOX_TOY)
    rng = np.random.default_rng(16)
    hi = (rng.random(mask.shape) < 0.01) & mask
    res = reg.tripwire(hi, n_surr=5)
    assert res["n_components"] > E.TABLES_MAX_N and res["placement"]["implementation"] == "bitmap"
    assert res["placement"]["unplaced_total"] == 0


def test_consensus_seeds_and_observed_mean():
    mask = toy_mask(17)
    reg = E.Region(mask, VOX_TOY)
    rng = np.random.default_rng(17)
    his = [toy_map(mask, rng, n_blobs=10, n_single=5) for _ in range(3)]
    c1 = reg.consensus(his, [1, 4, 6], n_surr=40, keep_null=True)
    c2 = reg.consensus(his, [1, 4, 6], n_surr=40, keep_null=True)
    assert np.array_equal(c1["_null"], c2["_null"])
    c3 = reg.consensus(his, [1, 4, 7], n_surr=40, keep_null=True)
    assert not np.array_equal(c1["_null"], c3["_null"])            # model id enters the seed
    mean_rows = sum(E.pool(h).sum(1) for h in his) / 3
    assert c1["T1"] == reg.t1_rows(mean_rows)
    # model m's surrogate k is the same placement as a single-map shuffle with seed [20260925 + k, m]
    C = E.Components(his[1], VOX_TOY)
    KA, _, _, _ = E.Shuffle(reg, C).place_tables([[E.SEED0 + k, 4] for k in range(40)])
    assert C.rowcounts(KA).shape == (40, reg.bcells.shape[0])


def test_empty_map_is_quiet():
    mask = toy_mask(18)
    reg = E.Region(mask, VOX_TOY)
    res = reg.tripwire(np.zeros_like(mask), n_surr=20)
    assert res["n_components"] == 0 and not res["fire"] and res["a"]["p"] == 1.0


def test_frozen_cross_check_and_dense_map_labelling():
    """The audit used in the real run: observed T1 via the frozen 2D Profile path and S via the frozen letter_S equal
    the tripwire's values; dense noise (one slab) and sparse maps (many slabs) label like label_components."""
    mask = toy_mask(19)
    reg = E.Region(mask, VOX_TOY)
    rng = np.random.default_rng(19)
    for hi in ((rng.random(mask.shape) < 0.05) & mask, toy_map(mask, rng, n_blobs=25, n_single=5)):
        res = reg.tripwire(hi, n_surr=3)
        ok = reg.frozen_check(hi, res)
        assert all(ok.values())
        C = E.Components(hi, VOX_TOY)
        lc = label_components(hi, (VOX_TOY / 1000) ** 2)
        assert np.array_equal(np.sort(C.area), np.sort(lc["area_px"].astype(np.int64)))
        assert np.allclose(np.sort(C.cent_r), np.sort(lc["cent_r"]), rtol=0, atol=0)


def test_region_row_extents():
    mask = toy_mask(20)
    reg = E.Region(mask, VOX_TOY)
    for r in range(0, mask.shape[0], 7):
        cols = np.flatnonzero(mask[r])
        if not cols.size:
            assert reg.rowL[r] == -1 and reg.rowR[r] == -1
            continue
        assert reg.rowL[r] == cols[0] and reg.rowR[r] == cols[-1] + 1
        assert reg.rowMulti[r] == bool((np.diff(cols) > 1).any())


@pytest.mark.parametrize("seed", [21, 22])
def test_bitmap_and_pixel_storage_modes_are_identical(seed):
    mask = toy_mask(seed)
    reg = E.Region(mask, VOX_TOY)
    hi = toy_map(mask, np.random.default_rng(seed), n_blobs=40, n_single=15, sizes=(2, 70))
    A, B = E.Components(hi, VOX_TOY, rings=False), E.Components(hi, VOX_TOY, rings=True)
    assert A.bms is not None and B.bms is None and A.n == B.n
    for f in ("area", "r0", "r1", "c0", "c1", "sum_r", "cent_r", "letter", "h_comp", "h_row", "h_cnt"):
        assert np.array_equal(getattr(A, f), getattr(B, f)), f
    assert A.n_multi == B.n_multi
    for c in range(A.n):
        for fa, fb in ((A.pixels(c), B.pixels(c)), (A.runs(c), B.runs(c)), (A.dilated_runs(c), B.dilated_runs(c)),
                       (A.row_extents(c), B.row_extents(c))):
            assert all(np.array_equal(x, y) for x, y in zip(fa, fb))
    seeds = [E.SEED0 + k for k in range(6)]
    ra = E.Shuffle(reg, A, "tables").place_tables(seeds)
    rb = E.Shuffle(reg, B, "tables").place_tables(seeds)
    assert all(np.array_equal(x, y) for x, y in zip(ra, rb))
