"""Text tripwire under Amendment E1: a component-shuffle null. Scalars only; this module never writes an image.

Frozen rule: prereg/lie_detector_v0e_e1.md (P5, SHA-256 a298a10c53c2d1eb...). Where the frozen text is silent,
the choices are fixed in INTERPRETATIONS below; the results JSON repeats them under "interpretations_fixed_before_preds".

    reg = Region(mask)                   # guarded placebo region: full-res bool crop, origin on the 104 px cell grid
    res = reg.tripwire(hi)               # hi: bool p > 0.5 map on the same crop, zero outside the region -> scalars
    con = reg.consensus([hi_a, hi_b], model_ids=[1, 4])     # part (a) on the mean of several maps

Text-like definition (the text rule's, unchanged):
  (a) row periodicity: the frozen T1 statistic on the 0.25 mm pooled row profile (FP px / region px per 104 px row),
      peak periodogram power over periods 2.5-8 mm / median power over 0.5-8 mm (forensics_analyze.Profile +
      reduce_pred.periodogram_stat, as in the frozen periodicity());
  (b) letter rows: S = the most letter-sized (1-4 mm2, 8-connected) component centroids in any 1 mm row window
      (deep_ld.letter_S).
Null: 1,000 surrogates; surrogate k uses numpy default_rng(20260925 + k) (consensus: default_rng([20260925 + k, m])).
Each surrogate moves the map's own connected components (translation only, on the 104 px cell lattice, pixel-exact
footprints) to random positions fully inside the region, where they neither overlap nor touch.
Fire: (a) p_a = (1 + #{T1_k >= T1}) / 1001 <= 0.001, i.e. T1 beats every surrogate; or (b) S >= 3 and S > max_k S_k.

Two implementations place the components; both follow the same procedure and the same random draws, so they give
identical surrogates (tools/margin/tests/test_tripwire_e1.py checks this):
  * "tables" (<= TABLES_MAX_N components): exact inside-region and pairwise overlap/touch tables over lattice shifts,
    built once from run-length footprints; all surrogates are placed together with table look-ups.
  * "bitmap" (more components): one surrogate at a time on a full-resolution owner array (int32).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from tools.margin.deep_ld import letter_S, row_window_max
from tools.margin.forensics_analyze import CMM, Profile
from tools.margin.reduce_pred import periodogram_stat

CELL = 104                  # px; one 0.25 mm pooled row (0.2495 mm at 2.399 um) and the translation lattice step
VOX_UM = 2.399
N_SURR = 1000
SEED0 = 20260925
MAX_TRIES = 1000
P_FIRE = 0.001
S_MIN = 3
LETTER_MM2 = (1.0, 4.0)
TABLES_MAX_N = 128          # "tables" implementation up to this many components, "bitmap" above (identical results)
BIG_PX = 256                # bitmap implementation: components this large test "inside the region" by table look-up
CHUNK = 250                 # surrogates placed together by the tables implementation
CHUNK_EARLY = 50            # the same when stop_early is set
ST8 = np.ones((3, 3), bool)
I32MAX = np.iinfo(np.int32).max
PREREG = "prereg/lie_detector_v0e_e1.md"
PREREG_SHA256 = "a298a10c53c2d1ebca9e1a247c403538652851b868f242bb2a8039093d017273"

INTERPRETATIONS = {
    "registration_first": "Registration is computed in a first pass that reads only the window of the evaluated label "
                          "mask (val_v2 | sup_v2 | sup_v1) and uses only its pixels: pixel AUROC of inklabels_v2 > 0 "
                          "vs == 0, as frozen in P2. The placebo crop is read in a second pass, and only if AUROC >= "
                          "0.75. An excluded pred keeps its registration numbers (AUROC, recall at p > 0.5, in-mask "
                          "unlabeled FP, pixel counts); no placebo statistic is computed or stored for it.",
    "components": "The region's own connected components of (p > 0.5) inside the guarded region at full resolution, "
                  "8-connected (the connectivity of the frozen label_components).",
    "lattice": "Components are translated by whole 104 px cells (one 0.25 mm pooled row, 0.2495 mm) in both axes. "
               "Footprints are pixel-exact. Row counts per 0.25 mm row and centroids therefore move exactly with "
               "each component.",
    "proposal": "A try draws a lattice translation uniformly among those that keep the component's bounding box "
                "inside the region's bounding box. It succeeds if every pixel lands inside the guarded region mask "
                "and no pixel overlaps or touches (8-neighbourhood) a pixel of another placed component.",
    "non_overlapping": "'Non-overlapping' is read as neither overlapping nor touching, so every placed component stays "
                       "a separate connected component and keeps its size and shape exactly (E1: 'the shuffle keeps "
                       "sizes and shapes').",
    "order_and_rounds": "Components are placed in order of area (largest first; ties in raster order of the first "
                        "pixel), in rounds. In each round every component not yet placed makes one try. A try is "
                        "rejected if it leaves the region, overlaps or touches a component placed in an earlier "
                        "round, or overlaps or touches a same-round try of a larger (earlier) component that lies "
                        "inside the region. At most 1,000 rounds, so at most 1,000 tries per component. A component "
                        "still not placed keeps its original position; these are counted and reported.",
    "random_draws": "Surrogate k of a pred uses numpy default_rng(20260925 + k); in the consensus, model m uses "
                    "default_rng([20260925 + k, m]), m = the model's index in the pre-registered pred list of its "
                    "segment (regions file order). In each round the generator draws rng.integers(0, n_rows) and then "
                    "rng.integers(0, n_cols) for the unplaced components in placement order.",
    "surrogate_statistics": "A surrogate's statistics come from its component list: the 0.25 mm row counts are the sum "
                            "of each component's row counts shifted by its row translation, and S uses the letter-"
                            "sized components' centroids shifted by the same translation. Because placed components "
                            "never overlap or touch, these equal the statistics of the rendered surrogate map (checked "
                            "in the unit tests). A component that keeps its original position counts there.",
    "part_a": "T1 = the frozen statistic (forensics_analyze.Profile rows: 0.25 mm pooled rows whose region px >= 10% of "
              "the fullest row, the full span between the first and last such row; reduce_pred.periodogram_stat: "
              "linear detrend, max power over periods 2.5-8 mm / median power over 0.5-8 mm), recomputed on every "
              "surrogate. The observed map and every surrogate go through the same Profile.prof and periodogram_stat "
              "calls, fed with 0.25 mm row counts (Profile.prof sums a row over its cells, so row counts give the "
              "identical profile). p_a = (1 + #{T1_k >= T1}) / 1001; part (a) fires iff p_a <= 0.001.",
    "part_b": "S = the frozen letter_S statistic (letter-sized = 1-4 mm2 at 2.399 um px; centroid rows; 1 mm window = "
              "416.84 px; deep_ld.row_window_max), computed from the component list with the arithmetic of the frozen "
              "label_components (area in px, centroid row = sum of rows / area). Unit tests check it equals letter_S "
              "on maps and rendered surrogates; the real run also checks every pred's observed S and T1 against the "
              "frozen letter_S and the frozen 2D Profile path. p_b = (1 + #{S_k >= S}) / 1001 is reported. Part (b) "
              "fires iff S >= 3 and S > max_k S_k.",
    "consensus": "Per segment, after all its preds: part (a) on the mean of the p > 0.5 maps of the preds that pass "
                 "registration (AUROC >= 0.75), family-row preds included (as in P2's consensus). The mean map's row "
                 "profile is the mean of the models' 0.25 mm row counts. Null: each model's components are shuffled "
                 "independently per surrogate (seeds above). Same fire rule.",
    "two_implementations": "Two exact implementations of this one procedure (pair tables for <= 128 components, a "
                           "full-resolution bitmap above). Same draws, same accept/reject decisions, identical "
                           "surrogates (unit-tested); the choice only changes speed.",
    "validation_stop_early": "The validation counts fires with this same code and seeds. For speed it stops computing "
                             "surrogates for a fill once neither part can fire (a surrogate has reached the observed "
                             "T1, and S < 3 or a surrogate has reached S). That cannot change the decision: firing "
                             "needs the observed value to beat every one of the 1,000 surrogates, and every surrogate "
                             "is computed the same way whether or not later ones are. The real run computes all "
                             "1,000 surrogates, so its p values are exact.",
    "stop_semantics": "Any E1 fire (pred or consensus) is a HARD STOP under text rule (a): computing stops, only "
                      "scalars are kept, and the saved full-resolution p > 0.5 crops are deleted. No image is made at "
                      "any point.",
    "saved_maps": "Each registered pred's placebo p > 0.5 crop is kept as a bit-packed array (data/v0e/, never in git, "
                  "never an image) for the consensus null, and deleted when the run ends or stops.",
    "planted_copy": "The planted-blob sanity copy is synthetic, not a model's output; the tripwire is not run on it.",
    "excluded_1667": "1667_2um_pred.tif is excluded under E1's registration rule (AUROC 0.495); its registration numbers "
                     "come from its earlier record, and it is not downloaded again.",
    "registration_pause": "If 3 or more of the 18 remaining preds fail registration, the run pauses and reports before "
                          "anything else (lead's instruction; P2 treats registration failure as a possible harness "
                          "bug).",
}


# ------------------------------------------------------------------------------------------------ small helpers
def pool(a: np.ndarray, c: int = CELL) -> np.ndarray:
    """Sum over c x c cells anchored at (0, 0); a trailing partial cell is kept (the same code as v0e.pool)."""
    rs = np.add.reduceat(a, np.arange(0, a.shape[0], c), axis=0, dtype=np.int64)
    return np.add.reduceat(rs, np.arange(0, a.shape[1], c), axis=1).astype(np.float64)


def row_pool(a: np.ndarray, c: int = CELL) -> np.ndarray:
    """Pixel count per 104 px row band (= pool(a).sum(1), exact integers as float64)."""
    return np.add.reduceat(a.sum(1, dtype=np.int64), np.arange(0, a.shape[0], c)).astype(np.float64)


def _csr_take(start: np.ndarray, size: np.ndarray, sel: np.ndarray):
    """Payload indices of the segments `sel` of a CSR layout, concatenated in order, and their sizes."""
    sz = size[sel]
    tot = int(sz.sum())
    if tot == 0:
        return np.zeros(0, np.int64), sz
    first = np.cumsum(sz) - sz
    return np.repeat(start[sel] - first, sz) + np.arange(tot, dtype=np.int64), sz


def _seg_starts(sz: np.ndarray) -> np.ndarray:
    return np.cumsum(sz) - sz


def _bitmap_runs(bm: np.ndarray, r_off: int, c_off: int):
    """Row runs of a bool array (rows ascending): row, first col, end col (half-open), offset to crop coords."""
    p = np.zeros((bm.shape[0], bm.shape[1] + 2), np.int8)
    p[:, 1:-1] = bm
    d = np.diff(p, axis=1)
    rs, cs = np.nonzero(d == 1)
    _, ce = np.nonzero(d == -1)
    return ((rs + r_off).astype(np.int64), (cs + c_off).astype(np.int64), (ce + c_off).astype(np.int64))


def shift_table(Ar, A0, A1, Br, B0, B1, a_lo: int, a_hi: int, b_lo: int, b_hi: int, cell: int = CELL):
    """Exact overlap table of two run-length pixel sets under lattice shifts.

    A: runs (rows Ar ascending, columns [A0, A1)); B: runs (Br, [B0, B1)). Returns bool T with shape
    (a_hi - a_lo + 1, b_hi - b_lo + 1): T[a - a_lo, b - b_lo] is True iff A and B shifted by (cell * a, cell * b)
    share at least one pixel."""
    na, nb = a_hi - a_lo + 1, b_hi - b_lo + 1
    if na <= 0 or nb <= 0 or not len(Ar) or not len(Br):
        return np.zeros((max(na, 0), max(nb, 0)), bool)
    rmin, rmax = int(Ar[0]), int(Ar[-1])
    ptr = np.searchsorted(Ar, np.arange(rmin, rmax + 2))
    sh = cell * np.arange(a_lo, a_hi + 1, dtype=np.int64)
    tr = np.asarray(Br, np.int64)[None, :] + sh[:, None]
    ki, bi = np.nonzero((tr >= rmin) & (tr <= rmax))
    T = np.zeros((na, nb), bool)
    if not ki.size:
        return T
    t = tr[ki, bi] - rmin
    s = ptr[t]
    cnt = ptr[t + 1] - s
    keep = cnt > 0
    ki, bi, s, cnt = ki[keep], bi[keep], s[keep], cnt[keep]
    if not ki.size:
        return T
    rep = np.repeat(np.arange(ki.size), cnt)
    ai = np.repeat(s - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum()))
    ki, bi = ki[rep], bi[rep]
    lo = (np.asarray(A0, np.int64)[ai] - np.asarray(B1, np.int64)[bi]) // cell + 1
    hi = -((np.asarray(B0, np.int64)[bi] - np.asarray(A1, np.int64)[ai]) // cell) - 1
    lo = np.maximum(lo, b_lo) - b_lo
    hi = np.minimum(hi, b_hi) - b_lo
    g = lo <= hi
    D = np.zeros((na, nb + 1), np.int32)
    np.add.at(D, (ki[g], lo[g]), 1)
    np.add.at(D, (ki[g], hi[g] + 1), -1)
    return np.cumsum(D, axis=1)[:, :nb] > 0


def _dedup_sorted(key: np.ndarray) -> np.ndarray:
    """Sorted unique values (np.unique is slow on this numpy build)."""
    key = np.sort(key)
    if key.size:
        key = key[np.r_[True, key[1:] != key[:-1]]]
    return key


# ------------------------------------------------------------------------------------------------ components
class Components:
    """8-connected components of a p > 0.5 map (crop coordinates), in placement order: area descending, ties in
    raster order of the first pixel.

    Labelling runs on horizontal slabs split at empty rows (a component cannot cross an empty row); this gives the
    same components in the same raster order as labelling the whole map. Two storage modes with identical content:
    few components (<= TABLES_MAX_N, no rings requested) keep one bitmap per component; otherwise CSR pixel lists
    (start, area; raster order within a component) plus, for the bitmap implementation, rings."""

    def __init__(self, hi: np.ndarray, vox: float = VOX_UM, rings: bool | None = None):
        hi = np.asarray(hi, bool)
        self.shape = hi.shape
        H, W = hi.shape
        self.vox = float(vox)
        self.n_cell_rows = -(-H // CELL)
        rows = np.flatnonzero(hi.any(1))
        e64, e32 = np.zeros(0, np.int64), np.zeros(0, np.int32)
        self.has_rings = False
        self.bms = None
        if not rows.size:
            self.n = 0
            self.area = self.r0 = self.r1 = self.c0 = self.c1 = self.letter = e64
            self.start = np.zeros(1, np.int64)
            self.rr = self.cc = e32
            self.cent_r = self.sum_r = np.zeros(0)
            self.h_comp = self.h_row = e64
            self.h_cnt = np.zeros(0)
            self.n_multi = 0
            return
        brk = np.flatnonzero(np.diff(rows) > 1)
        slabs, n = [], 0
        for a, b in zip(np.r_[rows[0], rows[brk + 1]], np.r_[rows[brk] + 1, rows[-1] + 1]):
            sub = hi[a:b]
            ca = np.flatnonzero(sub.any(0))
            c0, c1 = int(ca[0]), int(ca[-1]) + 1
            lab, k = ndi.label(sub[:, c0:c1], structure=ST8)
            slabs.append((int(a), c0, lab, n))
            n += k
        self.n = int(n)
        if rings is None:
            rings = self.n > TABLES_MAX_N
        if self.n <= TABLES_MAX_N and not rings:
            self._from_bitmaps(slabs)
        else:
            self._from_pixels(slabs, hi, rings)

    def _finish(self, area_l, sum_l, r0, r1, c0, c1, hk, hr, hc):
        """Common tail: placement order and per-component summaries (label-order inputs)."""
        n = self.n
        prio = np.lexsort((np.arange(n), -area_l))
        newid = np.empty(n, np.int64)
        newid[prio] = np.arange(n)
        self._prio, self._newid = prio, newid
        self.area = area_l[prio].astype(np.int64)
        self.n_multi = int((self.area > 1).sum())          # multi-pixel components first, single pixels last
        self.start = np.r_[0, np.cumsum(self.area)].astype(np.int64)
        self.r0, self.r1, self.c0, self.c1 = r0[prio], r1[prio], c0[prio], c1[prio]
        self.sum_r = sum_l[prio]                            # exact integers in float64
        self.cent_r = self.sum_r / self.area
        amm = self.area * (self.vox / 1000.0) ** 2
        self.letter = np.flatnonzero((amm >= LETTER_MM2[0]) & (amm <= LETTER_MM2[1]))
        hk = newid[hk]
        o = np.lexsort((hr, hk))
        self.h_comp, self.h_row, self.h_cnt = hk[o], hr[o], hc[o].astype(np.float64)

    def _from_bitmaps(self, slabs):
        bms, area, sumr, r0, r1, c0, c1, hk, hr, hc = [], [], [], [], [], [], [], [], [], []
        for a, cc0, lab, off in slabs:
            for j, sl in enumerate(ndi.find_objects(lab)):
                bm = lab[sl] == j + 1
                ra, ca = a + sl[0].start, cc0 + sl[1].start
                rs = bm.sum(1)
                rows = ra + np.arange(bm.shape[0])
                bms.append(bm)
                area.append(int(rs.sum()))
                sumr.append(float((rs * rows).sum()))
                r0.append(ra); r1.append(ra + bm.shape[0]); c0.append(ca); c1.append(ca + bm.shape[1])
                cr = rows // CELL
                u = np.unique(cr)
                hk.append(np.full(u.size, off + j)); hr.append(u); hc.append(np.bincount(cr - u[0], rs)[u - u[0]])
        i64 = lambda x: np.asarray(x, np.int64)
        self._finish(i64(area), np.asarray(sumr, np.float64), i64(r0), i64(r1), i64(c0), i64(c1),
                     np.concatenate(hk).astype(np.int64), np.concatenate(hr).astype(np.int64), np.concatenate(hc))
        self.bms = [bms[int(i)] for i in self._prio]
        self.rr = self.cc = None

    def _from_pixels(self, slabs, hi, rings):
        rl, cl, ll = [], [], []
        for a, c0, lab, off in slabs:
            flat = np.flatnonzero(lab)
            ll.append(lab.ravel()[flat].astype(np.int64) - 1 + off)
            w = lab.shape[1]
            rl.append((flat // w + a).astype(np.int32))
            cl.append((flat % w + c0).astype(np.int32))
            del flat
        del slabs
        rr, cc, lv = np.concatenate(rl), np.concatenate(cl), np.concatenate(ll)
        del rl, cl, ll
        n = self.n
        area_l = np.bincount(lv, minlength=n)
        st_l = np.r_[0, np.cumsum(area_l)][:-1]
        o = np.argsort(lv, kind="stable")                   # label order, raster order inside a label
        rr, cc, lv = rr[o], cc[o], lv[o]
        del o
        sum_l = np.bincount(lv, weights=rr, minlength=n)
        r0 = np.minimum.reduceat(rr, st_l).astype(np.int64)
        r1 = np.maximum.reduceat(rr, st_l).astype(np.int64) + 1
        c0 = np.minimum.reduceat(cc, st_l).astype(np.int64)
        c1 = np.maximum.reduceat(cc, st_l).astype(np.int64) + 1
        nrc = self.n_cell_rows
        k = lv * nrc + rr // CELL
        key = _dedup_sorted(k)
        cnt = np.bincount(np.searchsorted(key, k), minlength=key.size)
        del k
        self._finish(area_l, sum_l, r0, r1, c0, c1, key // nrc, key % nrc, cnt)
        # regroup the pixels in placement order (blocks of whole labels, raster order kept inside each)
        idx, _ = _csr_take(st_l.astype(np.int64), area_l.astype(np.int64), self._prio)
        self.rr, self.cc = rr[idx], cc[idx]
        cid = np.repeat(np.arange(n, dtype=np.int64), self.area)
        del idx, rr, cc, lv
        if rings:
            self._rings(hi, cid)

    def _rings(self, hi, cid):
        """Ring = pixels that touch a component but are not in it (8-neighbourhood); with the component it forms the
        3 x 3 dilated footprint. Single pixels: their 8 neighbours (implicit). Larger components: CSR (rstart, rsize,
        rflat), flat indices in the 1-px padded crop."""
        H, W = self.shape
        Wp = W + 2
        multi = self.area[cid] > 1
        r, c, cm = self.rr[multi].astype(np.int64), self.cc[multi].astype(np.int64), cid[multi]
        big = np.int64((H + 2) * Wp)
        keys = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                qr, qc = r + dr, c + dc
                bg = np.ones(qr.size, bool)
                inb = (qr >= 0) & (qr < H) & (qc >= 0) & (qc < W)
                bg[inb] = ~hi[qr[inb], qc[inb]]
                keys.append(cm[bg] * big + (qr[bg] + 1) * Wp + (qc[bg] + 1))
        key = _dedup_sorted(np.concatenate(keys)) if keys else np.zeros(0, np.int64)
        del keys
        comp = key // big
        self.rflat = key % big
        self.rsize = np.bincount(comp, minlength=self.n).astype(np.int64)
        self.rstart = np.r_[0, np.cumsum(self.rsize)][:-1].astype(np.int64)
        self.has_rings = True

    def pixels(self, c: int):
        """Rows and columns of component c's pixels (raster order), in either storage mode."""
        if self.bms is not None:
            r, q = np.nonzero(self.bms[c])
            return r + int(self.r0[c]), q + int(self.c0[c])
        s, e = self.start[c], self.start[c + 1]
        return self.rr[s:e], self.cc[s:e]

    def pflat(self) -> np.ndarray:
        """Flat index of every pixel in the 1-px padded crop (pixel-list mode)."""
        if self.rr is None:
            raise ValueError("pixel lists not built (bitmap storage mode)")
        return (self.rr.astype(np.int64) + 1) * (self.shape[1] + 2) + (self.cc.astype(np.int64) + 1)

    def _bitmap(self, c: int, pad: int):
        r0, c0 = int(self.r0[c]), int(self.c0[c])
        if self.bms is not None:
            bm = np.pad(self.bms[c], pad) if pad else self.bms[c]
            return bm, r0 - pad, c0 - pad
        s, e = self.start[c], self.start[c + 1]
        bm = np.zeros((int(self.r1[c]) - r0 + 2 * pad, int(self.c1[c]) - c0 + 2 * pad), bool)
        bm[self.rr[s:e] - r0 + pad, self.cc[s:e] - c0 + pad] = True
        return bm, r0 - pad, c0 - pad

    def runs(self, c: int):
        """Row runs of component c (crop coordinates)."""
        return _bitmap_runs(*self._bitmap(c, 0))

    def dilated_runs(self, c: int):
        """Row runs of component c dilated by the 3 x 3 square (crop coordinates)."""
        bm, r0, c0 = self._bitmap(c, 1)
        return _bitmap_runs(ndi.binary_dilation(bm, ST8), r0, c0)

    def row_extents(self, c: int):
        """Rows of component c with the first and last column of its pixels in each row."""
        if self.bms is not None:
            bm = self.bms[c]
            w = bm.shape[1]
            return (int(self.r0[c]) + np.arange(bm.shape[0], dtype=np.int64),
                    int(self.c0[c]) + bm.argmax(1).astype(np.int64),
                    int(self.c0[c]) + w - 1 - bm[:, ::-1].argmax(1).astype(np.int64))
        s, e = self.start[c], self.start[c + 1]
        rr, cc = self.rr[s:e], self.cc[s:e]
        b = np.flatnonzero(np.r_[True, rr[1:] != rr[:-1]])
        return rr[b].astype(np.int64), cc[b].astype(np.int64), np.maximum.reduceat(cc, b).astype(np.int64)

    def rowcounts(self, KA: np.ndarray) -> np.ndarray:
        """0.25 mm row counts for row translations KA (cells), shape (S, N) -> (S, n_cell_rows)."""
        KA = np.atleast_2d(KA)
        S, nrc = KA.shape[0], self.n_cell_rows
        if not self.n:
            return np.zeros((S, nrc))
        rows = self.h_row[None, :] + KA[:, self.h_comp]
        idx = (np.arange(S, dtype=np.int64)[:, None] * nrc + rows).ravel()
        w = np.broadcast_to(self.h_cnt, rows.shape).ravel()
        return np.bincount(idx, weights=w, minlength=S * nrc).reshape(S, nrc)

    def s_values(self, KA: np.ndarray) -> np.ndarray:
        """S (most letter-sized centroids in any 1 mm row window) for row translations KA, shape (S, N) -> (S,)."""
        KA = np.atleast_2d(KA)
        if not self.letter.size:
            return np.zeros(KA.shape[0], np.int64)
        win = 1000.0 / self.vox
        L = self.letter
        # the centroid of the moved component, computed as the frozen label_components computes it on a map
        cr = (self.sum_r[L][None, :] + (self.area[L] * CELL)[None, :] * KA[:, L]) / self.area[L][None, :]
        return np.array([row_window_max(x, win) for x in cr], np.int64)


# ------------------------------------------------------------------------------------------------ region
class Region:
    """A guarded placebo region (full-res bool crop). The crop origin must lie on the 104 px cell grid of the canvas
    (true for the v0-E crops, which are aligned to 416 px) so that the pooled rows are the frozen 0.25 mm rows."""

    def __init__(self, mask: np.ndarray, vox: float = VOX_UM):
        self.mask = np.ascontiguousarray(mask, bool)
        self.vox = float(vox)
        self.H, self.W = self.mask.shape
        anyr = self.mask.any(1)
        ra, ca = np.flatnonzero(anyr), np.flatnonzero(self.mask.any(0))
        if not ra.size:
            raise ValueError("empty region mask")
        self.bbox = (int(ra[0]), int(ra[-1]) + 1, int(ca[0]), int(ca[-1]) + 1)
        self.px = int(self.mask.sum())
        self.bcells = pool(self.mask)
        self._pr = Profile(self.bcells, self.bcells)     # its rows come from the region (den); num is always passed
        r0, r1, c0, c1 = self.bbox
        self.nm = _bitmap_runs(~self.mask[r0:r1, c0:c1], r0, c0)      # outside-region runs inside the bbox
        # per crop row: first and last+1 region column (-1 if none), and whether the row has more than one run
        self.rowL = np.where(anyr, self.mask.argmax(1), -1).astype(np.int64)
        self.rowR = np.where(anyr, self.W - self.mask[:, ::-1].argmax(1), -1).astype(np.int64)
        nruns = np.zeros(self.H, np.int64)
        np.add.at(nruns, _bitmap_runs(self.mask, 0, 0)[0], 1)
        self.rowMulti = nruns > 1
        self._maskp = None
        self._own = None

    # ---- statistics (frozen code paths)
    def t1(self, num: np.ndarray):
        """Frozen T1 on the 0.25 mm row profile: num is the pooled FP map (2D) or row counts as a column."""
        st = periodogram_stat(self._pr.prof(num), CMM * 1000)
        if st.get("ratio") is None:
            raise ValueError("row profile too short for T1")
        return float(st["ratio"]), float(st["peak_period_mm"]), int(st["n"]), float(st["length_mm"])

    def t1_rows(self, rc: np.ndarray) -> float:
        return self.t1(np.asarray(rc, float)[:, None])[0]

    def maskp(self) -> np.ndarray:
        if self._maskp is None:
            self._maskp = np.pad(self.mask, 1).ravel()
        return self._maskp

    def owner(self) -> np.ndarray:
        if self._own is None:
            self._own = np.full((self.H + 2) * (self.W + 2), I32MAX, np.int32)
        return self._own

    def release(self):
        """Free the large bitmap-path arrays."""
        self._maskp = None
        self._own = None

    def frozen_check(self, hi: np.ndarray, res: dict) -> dict:
        """Audit for real preds: the observed T1 through the frozen 2D path (Profile(pool(map), pool(region))) and the
        observed S through the frozen letter_S must equal the tripwire's values. Raises if not."""
        self.release()
        pr = Profile(pool(hi), self.bcells)
        st = periodogram_stat(pr.prof(), CMM * 1000)
        S, n_letter, _ = letter_S(hi, self.vox)
        ok = {"T1_equal": bool(st["ratio"] == res["a"]["T1"]), "S_equal": bool(int(S) == res["b"]["S"]),
              "letter_blobs_equal": bool(int(n_letter) == res["letter_blobs"])}
        if not all(ok.values()):
            raise AssertionError(f"frozen cross-check failed: {ok}")
        return ok

    # ---- the tripwire
    def observed(self, hi: np.ndarray, rings: bool | None = None) -> dict:
        hi = np.asarray(hi, bool)
        if hi.shape != self.mask.shape:
            raise ValueError(f"map shape {hi.shape} != region shape {self.mask.shape}")
        if (hi & ~self.mask).any():
            raise ValueError("p > 0.5 map has pixels outside the region")
        rc = row_pool(hi)
        T1, peak, n_rows, length = self.t1(rc[:, None])
        comps = Components(hi, self.vox, rings)
        zero = np.zeros((1, comps.n), np.int64)
        if not np.array_equal(comps.rowcounts(zero)[0], rc):
            raise AssertionError("component row counts disagree with the map")
        amm = comps.area * (self.vox / 1000.0) ** 2
        return {"T1": T1, "peak_period_mm": peak, "n_rows": n_rows, "length_mm": length,
                "S": int(comps.s_values(zero)[0]), "n_letter": int(comps.letter.size), "n_components": comps.n,
                "fp_px": int(comps.area.sum()), "largest_component_mm2": float(amm.max()) if comps.n else 0.0,
                "_comps": comps}

    def tripwire(self, hi: np.ndarray, n_surr: int = N_SURR, seed0: int = SEED0, stop_early: bool = False,
                 impl: str | None = None, keep_null: bool = False) -> dict:
        obs = self.observed(hi, True if impl == "bitmap" else None)
        comps = obs.pop("_comps")
        sh = Shuffle(self, comps, impl)
        seeds = [seed0 + k for k in range(n_surr)]
        T1s, Ss, unpl, rounds = [], [], [], []
        n_ge_a = n_ge_b = 0
        stopped = False
        for KA, placed, rnd in sh.surrogates(seeds, stop_early):
            rc = comps.rowcounts(KA)
            t = [self.t1_rows(x) for x in rc]
            s = comps.s_values(KA)
            T1s += t
            Ss += s.tolist()
            unpl += (comps.n - placed.sum(1)).tolist()
            rounds += rnd.tolist()
            n_ge_a += sum(x >= obs["T1"] for x in t)
            n_ge_b += int((s >= obs["S"]).sum())
            if stop_early and len(T1s) < n_surr and n_ge_a > 0 and (obs["S"] < S_MIN or n_ge_b > 0):
                stopped = True
                break
        m = len(T1s)
        T1a, Sa = np.array(T1s), np.array(Ss, np.int64)
        p_a = (1 + n_ge_a) / (1 + n_surr)
        p_b = (1 + n_ge_b) / (1 + n_surr)
        fire_a = bool(m == n_surr and p_a <= P_FIRE)
        fire_b = bool(m == n_surr and obs["S"] >= S_MIN and obs["S"] > int(Sa.max()))
        up = np.array(unpl, np.int64)
        res = {
            "n_components": obs["n_components"], "fp_px": obs["fp_px"], "fp_area_frac": obs["fp_px"] / self.px,
            "letter_blobs": obs["n_letter"], "largest_component_mm2": obs["largest_component_mm2"],
            "a": {"T1": obs["T1"], "peak_period_mm": obs["peak_period_mm"], "n_rows": obs["n_rows"],
                  "length_mm": obs["length_mm"], "n_null_ge": int(n_ge_a),
                  "p": p_a if m == n_surr else None, "p_lower_bound": p_a,
                  "null_max": float(T1a.max()), "null_median": float(np.median(T1a)),
                  "null_p95": float(np.percentile(T1a, 95)), "fire": fire_a},
            "b": {"S": obs["S"], "n_null_ge": int(n_ge_b), "p": p_b if m == n_surr else None, "p_lower_bound": p_b,
                  "null_max": int(Sa.max()), "null_median": float(np.median(Sa)), "fire": fire_b},
            "fire": bool(fire_a or fire_b),
            "n_surrogates": n_surr, "n_computed": m, "stopped_early": stopped,
            "placement": {"implementation": sh.impl, "unplaced_total": int(up.sum()),
                          "surrogates_with_unplaced": int((up > 0).sum()), "max_unplaced": int(up.max()),
                          "max_rounds": int(max(rounds)) if rounds else 0},
        }
        if keep_null:
            res["_null"] = {"T1": T1a, "S": Sa}
        return res

    def consensus(self, his: list, model_ids: list, n_surr: int = N_SURR, seed0: int = SEED0,
                  impl: str | None = None, keep_null: bool = False) -> dict:
        """Part (a) on the mean of several p > 0.5 maps; each model's components shuffled independently."""
        M = len(his)
        if M == 0:
            raise ValueError("no maps")
        nrc = self._pr.den.shape[0]
        acc = np.zeros((n_surr, nrc))
        obs_rc = np.zeros(nrc)
        per = []
        for m, h in enumerate(his):
            h = np.asarray(h, bool)
            if h.shape != self.mask.shape or (h & ~self.mask).any():
                raise ValueError("map not on the region")
            comps = Components(h, self.vox, True if impl == "bitmap" else None)
            obs_rc += comps.rowcounts(np.zeros((1, comps.n), np.int64))[0]
            sh = Shuffle(self, comps, impl)
            seeds = [[seed0 + k, int(model_ids[m])] for k in range(n_surr)]
            k0, up = 0, []
            for KA, placed, rnd in sh.surrogates(seeds, False):
                acc[k0:k0 + len(KA)] += comps.rowcounts(KA)
                up += (comps.n - placed.sum(1)).tolist()
                k0 += len(KA)
            up = np.array(up, np.int64)
            per.append({"model_id": int(model_ids[m]), "n_components": comps.n, "implementation": sh.impl,
                        "unplaced_total": int(up.sum()), "surrogates_with_unplaced": int((up > 0).sum())})
            del comps, sh
        T1, peak, n_rows, length = self.t1(obs_rc[:, None] / M)
        T1n = np.array([self.t1_rows(x / M) for x in acc])
        n_ge = int((T1n >= T1).sum())
        p = (1 + n_ge) / (1 + n_surr)
        res = {"n_models": M, "model_ids": [int(x) for x in model_ids], "T1": T1, "peak_period_mm": peak,
               "n_rows": n_rows, "length_mm": length, "n_null_ge": n_ge, "p": p, "null_max": float(T1n.max()),
               "null_median": float(np.median(T1n)), "null_p95": float(np.percentile(T1n, 95)),
               "fire": bool(p <= P_FIRE), "n_surrogates": n_surr, "models": per}
        if keep_null:
            res["_null"] = T1n
        return res


# ------------------------------------------------------------------------------------------------ the shuffle
class Shuffle:
    """Component-shuffle surrogates of one map on one region (see INTERPRETATIONS)."""

    def __init__(self, region: Region, comps: Components, impl: str | None = None):
        self.R, self.C = region, comps
        n = comps.n
        self.impl = impl or ("tables" if n <= TABLES_MAX_N else "bitmap")
        if self.impl not in ("tables", "bitmap"):
            raise ValueError(self.impl)
        MR0, MR1, MC0, MC1 = region.bbox
        self.kaMin = -((comps.r0 - MR0) // CELL)
        self.kaMax = (MR1 - comps.r1) // CELL
        self.kbMin = -((comps.c0 - MC0) // CELL)
        self.kbMax = (MC1 - comps.c1) // CELL
        self.nA = self.kaMax - self.kaMin + 1
        self.nB = self.kbMax - self.kbMin + 1
        if n and (self.nA.min() < 1 or self.nB.min() < 1):
            raise AssertionError("a component lies outside the region bounding box")
        if not n:
            return
        if self.impl == "tables":
            self._vtables(np.arange(n))
            self._pairtables()
        else:
            if not comps.has_rings:
                raise ValueError("bitmap implementation needs Components(..., rings=True)")
            self.isbig = comps.area >= BIG_PX
            self._vtables(np.flatnonzero(self.isbig))
            self.Pf = comps.pflat()
            self.pS = self.Pf[comps.start[comps.n_multi:-1]]          # the one pixel of each single-pixel component
            Wp = region.W + 2
            self.off9 = np.array([dr * Wp + dc for dr in (-1, 0, 1) for dc in (-1, 0, 1)], np.int64)

    # ---- exact tables
    def _inside_table(self, c: int) -> np.ndarray:
        """bool table over (ka - kaMin, kb - kbMin): True iff component c lies fully inside the region. Rows where the
        region row is a single run use its two edges; lattice rows that meet a region row with several runs are
        computed with the general run-overlap method."""
        R = self.R
        ka = np.arange(self.kaMin[c], self.kaMax[c] + 1)
        kbs = np.arange(self.kbMin[c], self.kbMax[c] + 1)
        rows, cmin, cmax = self.C.row_extents(c)
        tr = rows[None, :] + CELL * ka[:, None]
        Lt, Rt = R.rowL[tr], R.rowR[tr]
        empty = (Lt < 0).any(1)
        lo = (-((cmin[None, :] - Lt) // CELL)).max(1)          # smallest kb with cmin + CELL kb >= L
        hi = ((Rt - 1 - cmax[None, :]) // CELL).min(1)           # largest kb with cmax + CELL kb <= R - 1
        V = (kbs[None, :] >= lo[:, None]) & (kbs[None, :] <= hi[:, None]) & ~empty[:, None]
        general = R.rowMulti[tr].any(1) & ~empty
        if general.any():
            runs = self.C.runs(c)
            for i in np.flatnonzero(general):
                V[i] = ~shift_table(*R.nm, *runs, int(ka[i]), int(ka[i]), int(kbs[0]), int(kbs[-1]))[0]
        return V

    def _vtables(self, sel: np.ndarray):
        n = self.C.n
        self.voff = np.zeros(n, np.int64)
        parts, off = [], 0
        for c in sel:
            c = int(c)
            V = self._inside_table(c)
            if not V[-self.kaMin[c], -self.kbMin[c]]:
                raise AssertionError("a component is not inside the region at its own position")
            self.voff[c] = off
            parts.append(V.ravel())
            off += V.size
        self.V = np.concatenate(parts) if parts else np.zeros(0, bool)

    def _pairtables(self):
        """Pair tables for i < j (placement order): conflict iff i dilated by 3 x 3 and j shifted by (da, db) cells
        relative to i share a pixel (overlap or touch)."""
        C = self.C
        n = C.n
        P = [C.runs(c) for c in range(n)]
        D = [C.dilated_runs(c) for c in range(n)]
        pi, pj, alo, an, blo, bn, off, parts = [], [], [], [], [], [], [], []
        o = 0
        for i in range(n):
            for j in range(i + 1, n):
                # shifts (in cells) at which j's rows/cols can meet i's dilated rows [r0_i - 1, r1_i] / cols,
                # intersected with the shifts both components can actually take
                a_lo = max(-((C.r1[j] - C.r0[i]) // CELL), self.kaMin[j] - self.kaMax[i])
                a_hi = min((C.r1[i] - C.r0[j]) // CELL, self.kaMax[j] - self.kaMin[i])
                b_lo = max(-((C.c1[j] - C.c0[i]) // CELL), self.kbMin[j] - self.kbMax[i])
                b_hi = min((C.c1[i] - C.c0[j]) // CELL, self.kbMax[j] - self.kbMin[i])
                if a_lo > a_hi or b_lo > b_hi:
                    continue
                T = shift_table(*D[i], *P[j], int(a_lo), int(a_hi), int(b_lo), int(b_hi))
                if not T.any():
                    continue
                if a_lo <= 0 <= a_hi and b_lo <= 0 <= b_hi and T[-a_lo, -b_lo]:
                    raise AssertionError("two components touch at their own positions")
                pi.append(i); pj.append(j); alo.append(a_lo); an.append(T.shape[0]); blo.append(b_lo)
                bn.append(T.shape[1]); off.append(o); parts.append(T.ravel()); o += T.size
        self.pi, self.pj = np.array(pi, np.int64), np.array(pj, np.int64)
        self.pa_lo, self.pa_n = np.array(alo, np.int64), np.array(an, np.int64)
        self.pb_lo, self.pb_n = np.array(blo, np.int64), np.array(bn, np.int64)
        self.poff = np.array(off, np.int64)
        self.PT = np.concatenate(parts) if parts else np.zeros(0, bool)
        if self.pi.size:
            self.oj = np.argsort(self.pj, kind="stable")
            self.uj, self.sj = np.unique(self.pj[self.oj], return_index=True)
            self.oi = np.argsort(self.pi, kind="stable")
            self.ui, self.si = np.unique(self.pi[self.oi], return_index=True)

    # ---- surrogates
    def surrogates(self, seeds: list, stop_early: bool = False):
        """Yield (KA (s, N) row translations in cells, placed (s, N), rounds (s,)) for consecutive chunks of seeds."""
        n = self.C.n
        if not n:
            for k in range(0, len(seeds), CHUNK):
                s = len(seeds[k:k + CHUNK])
                yield np.zeros((s, 0), np.int64), np.zeros((s, 0), bool), np.zeros(s, np.int64)
            return
        if self.impl == "tables":
            ch = CHUNK_EARLY if stop_early else CHUNK
            for k in range(0, len(seeds), ch):
                KA, KB, placed, rounds = self.place_tables(seeds[k:k + ch])
                yield KA, placed, rounds
        else:
            for sd in seeds:
                ka, kb, placed, rounds = self.place_bitmap(sd)
                yield ka[None, :], placed[None, :], np.array([rounds])

    def place_tables(self, seeds: list):
        n, S = self.C.n, len(seeds)
        nA, nB, kaMin, kbMin = self.nA, self.nB, self.kaMin, self.kbMin
        KA = np.zeros((S, n), np.int64)
        KB = np.zeros((S, n), np.int64)
        placed = np.zeros((S, n), bool)
        rounds = np.zeros(S, np.int64)
        rngs = [np.random.default_rng(sd) for sd in seeds]
        npair = self.pi.size
        for _ in range(MAX_TRIES):
            act = np.flatnonzero(~placed.all(1))
            if not act.size:
                break
            pl = placed[act]
            cand = ~pl
            cA, cB = KA[act].copy(), KB[act].copy()
            for ii, k in enumerate(act):
                U = np.flatnonzero(cand[ii])
                cA[ii, U] = kaMin[U] + rngs[k].integers(0, nA[U])
                cB[ii, U] = kbMin[U] + rngs[k].integers(0, nB[U])
            rounds[act] += 1
            v1 = np.zeros_like(cand)
            ai, ci = np.nonzero(cand)
            v1[ai, ci] = self.V[self.voff[ci] + (cA[ai, ci] - kaMin[ci]) * nB[ci] + (cB[ai, ci] - kbMin[ci])]
            rej = np.zeros_like(cand)
            if npair:
                pi, pj = self.pi, self.pj
                a = cA[:, pj] - cA[:, pi] - self.pa_lo
                b = cB[:, pj] - cB[:, pi] - self.pb_lo
                ins = (a >= 0) & (a < self.pa_n) & (b >= 0) & (b < self.pb_n)
                conf = np.zeros(ins.shape, bool)
                conf[ins] = self.PT[(self.poff + a * self.pb_n + b)[ins]]
                rj = conf & cand[:, pj] & (pl[:, pi] | v1[:, pi])
                ri = conf & cand[:, pi] & pl[:, pj]
                rej[:, self.uj] |= np.logical_or.reduceat(rj[:, self.oj], self.sj, axis=1)
                rej[:, self.ui] |= np.logical_or.reduceat(ri[:, self.oi], self.si, axis=1)
            acc = v1 & ~rej
            KA[act] = np.where(acc, cA, KA[act])
            KB[act] = np.where(acc, cB, KB[act])
            placed[act] |= acc
        return KA, KB, placed, rounds

    def place_bitmap(self, seed):
        """One surrogate on the full-resolution owner array: -1 = placed pixel, rank = a claim by a same-round try
        inside the region (the smallest rank wins), I32MAX = free. Multi-pixel components use CSR pixel lists and
        rings; single-pixel components (the last block in placement order) use their pixel and its 8 neighbours."""
        R, C = self.R, self.C
        n, nm = C.n, C.n_multi
        rng = np.random.default_rng(seed)
        own, mp = R.owner(), R.maskp()
        Wp = R.W + 2
        ka = np.zeros(n, np.int64)
        kb = np.zeros(n, np.int64)
        placed = np.zeros(n, bool)
        rounds = 0
        for _ in range(MAX_TRIES):
            U = np.flatnonzero(~placed)
            if not U.size:
                break
            rounds += 1
            a = rng.integers(0, self.nA[U])
            b = rng.integers(0, self.nB[U])
            cka, ckb = self.kaMin[U] + a, self.kbMin[U] + b
            shift = CELL * (cka * Wp + ckb)
            k = int(np.searchsorted(U, nm))                  # U[:k] multi-pixel, U[k:] single pixels
            # (1) inside the region
            v1 = np.empty(U.size, bool)
            if k:
                Um = U[:k]
                big = self.isbig[Um]
                vm = np.empty(k, bool)
                if big.any():
                    ub = Um[big]
                    vm[big] = self.V[self.voff[ub] + a[:k][big] * self.nB[ub] + b[:k][big]]
                sm = ~big
                if sm.any():
                    idx, sz = _csr_take(C.start, C.area, Um[sm])
                    vm[sm] = np.logical_and.reduceat(mp[self.Pf[idx] + np.repeat(shift[:k][sm], sz)],
                                                     _seg_starts(sz))
                    del idx
                v1[:k] = vm
            tS = self.pS[U[k:] - nm] + shift[k:]
            v1[k:] = mp[tS]
            # (2) claims by every try inside the region
            Vm, shm = U[:k][v1[:k]], shift[:k][v1[:k]]
            idx, szm = _csr_take(C.start, C.area, Vm)
            tgP = self.Pf[idx] + np.repeat(shm, szm)
            del idx
            np.minimum.at(own, tgP, np.repeat(Vm.astype(np.int32), szm))
            vs = v1[k:]
            Vs, tS = U[k:][vs], tS[vs]
            np.minimum.at(own, tS, Vs.astype(np.int32))
            # (3) checks: nothing placed and no smaller-rank claim on the 3 x 3 dilated footprint
            accm = np.zeros(Vm.size, bool)
            if Vm.size:
                mn = np.minimum.reduceat(own[tgP], _seg_starts(szm))
                idx, sz = _csr_take(C.rstart, C.rsize, Vm)
                mn = np.minimum(mn, np.minimum.reduceat(own[C.rflat[idx] + np.repeat(shm, sz)], _seg_starts(sz)))
                del idx
                accm = mn >= Vm
            m = np.full(Vs.size, I32MAX, np.int32)
            for o in self.off9:
                np.minimum(m, own[tS + o], out=m)
            accs = m >= Vs
            # (4) clear all claims, then mark the accepted pixels
            for t in (tgP, tS):
                vals = own[t]
                own[t[vals >= 0]] = I32MAX
            own[tgP[np.repeat(accm, szm)]] = -1
            own[tS[accs]] = -1
            done = np.r_[Vm[accm], Vs[accs]]
            dka = np.r_[cka[:k][v1[:k]][accm], cka[k:][vs][accs]]
            dkb = np.r_[ckb[:k][v1[:k]][accm], ckb[k:][vs][accs]]
            placed[done] = True
            ka[done], kb[done] = dka, dkb
            del tgP, tS
        P = np.flatnonzero(placed)
        if P.size:
            idx, sz = _csr_take(C.start, C.area, P)
            own[self.Pf[idx] + np.repeat(CELL * (ka[P] * Wp + kb[P]), sz)] = I32MAX
        return ka, kb, placed, rounds
