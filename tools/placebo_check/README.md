# tools.placebo_check -- a lie-detector test for ink AI on PHerc.1667

**In plain English: a lie-detector test for ink AI: we point it at papyrus scholars know is blank and check
whether it sees letters.** Two small patches of PHerc.1667 (segments w018 and w023) sit between the columns of
text that the published edition already transcribed. Because the edition maps out where the columns are, we
know these patches are blank -- not because a model or a label file says so, but because a human papyrologist's
published reading of the scroll leaves them empty. We take an AI model's "here is where I think there's ink"
map, crop it to those blank patches, and ask two questions: (1) does the model even line up with the parts of
the scroll that *are* labeled (a sanity check, called **registration**), and (2) inside the blank patch, does
the model draw anything shaped like rows of handwriting rather than a random smear (**the tripwire**)? Every
answer is a number; the tool never makes, shows, or saves a picture.

## What this package is

A drop-in CLI over the frozen v0-E harness (`results/lie_detector_v0e.md`, verdict **PASS**), for running
the same check on a new prediction file:

```
python -m tools.placebo_check run --pred PRED.tif --segment w018   # or w023
python -m tools.placebo_check run --pred PRED.tif --segment w023 --json out.json
```

It imports `tools/margin/v0e.py`, `tools/margin/v0e_e1.py` and `tools/margin/tripwire_e1.py` unmodified --
those modules are the frozen, tested engine; this package only wraps them behind a CLI and ships the small
region-definition files (see below) so the check can be re-run without redoing the original geometry work.

`PRED.tif` must already share the labeled segment's pixel canvas (same shape, or an exact integer downsample of
it, which is upsampled automatically). Getting a prediction onto that canvas is the caller's job; this tool does
not resample or align an unrelated grid.

## Method

1. **Registration first.** Pixel AUROC of `PRED > threshold` against the published `inklabels_v2`, inside the
   window covered by the edition's own evaluated-region masks (`val_v2 | sup_v2 | sup_v1`). **AUROC < 0.75
   excludes the file before any placebo pixel is read** -- if a model doesn't line up with text scholars have
   already read, its placebo reading isn't trustworthy either.
2. **Placebo false-positive rate**, on the guarded intercolumn region (below), reported against the same file's
   false-positive rate on ordinary in-mask *unlabeled* text-block pixels (the frozen v0-E measures). A ratio near
   1 means the model is no more "confident" in the blank gap than it is on unremarkable unlabeled surface inside
   the text block; a ratio well above 1 means it isn't.
3. **The E1 text tripwire**, on the placebo `p > threshold` map: row periodicity (part a) and letter-row
   structure (part b), each compared to 1,000 surrogates built by moving the map's own connected blobs to random
   non-overlapping positions inside the region ("component-shuffle"). Fire = the observed statistic beats every
   surrogate. Full statistical definition: `tools/tripwire/README.md` and the docstring of
   `tools/margin/tripwire_e1.py`.

### Where "blank" comes from

The two regions are **not** defined by any model's own output. They are the two gaps between text columns that
`results/edition_lattice.md` located from the published edition's column layout (22 columns on w018's ~1.4
m wrap) and geometric measurement of the org's own flattened mesh -- then shrunk by a 1.5 mm guard against
line-end spill, kept 2 mm from any mesh edge, and capped below the taper zone (z < 81 mm). **The two segments
reach "blank" differently:** on w018, a periodic column lattice was fit and detected (period 72.45 +/- 0.25 mm,
but from only 3 observed gaps); on w023 that lattice fit did NOT reach detection, so its region (ic_132) instead
comes from a directly observed blank gap in the render's column-level profile, not from lattice extrapolation.
Both candidate regions independently passed their own local text-structure check on the org's own render before
either was adopted (`edition_lattice.md`, "Candidate blank regions"). Region geometry:

| Segment | Region | Guarded core (official canvas) | Area (HF full-res canvas) |
|---|---|---|---|
| w018 | ic_201 | x 196.91-204.25 mm | 545.8 mm2 |
| w023 | ic_132 | x 133.00-142.98 mm | 770.6 mm2 |

1,316.4 mm2 total. **Limits:** this is two intercolumns on two segments of one scroll (PHerc.1667); the margins
row (the z < 81 mm band above the text, conditional on the org confirming the column foot is at high z) was
**not run**; and two of the 19 files checked in the v0-E run were excluded by the registration gate before any
placebo pixel was read (one unidentifiable-provenance file at AUROC 0.495, one named model at AUROC 0.697).

## Region files shipped with this package

`regions_data/regions_w018.npz` and `regions_data/regions_w023.npz` (~40 KB each) hold a coarse tifxyz-grid bool
mask of the guarded region -- not a full-resolution image, not raw scan data. Combined with the geometry record
in `results/lie_detector_v0e_regions.json` (committed, plain JSON), `tools/placebo_check/regions.py`
rebuilds the exact full-resolution region crop and checks it against the geometry file's committed SHA-256
hash, so a stale or tampered shipped file is caught rather than silently trusted.

## The frozen rules and their public hashes

Two pre-registration documents, hashed and posted (`#robots`, 2026-09-26) **before** any result they govern
existed:

| Label | File | SHA-256 |
|---|---|---|
| P2 | `prereg/lie_detector_v0e.md` | `a3dcaea682f49e561dca9bcfeac6eb23a9b34e6f87efdfe2a4f611a24fc54010` |
| P5 | `prereg/lie_detector_v0e_e1.md` | `a298a10c53c2d1ebca9e1a247c403538652851b868f242bb2a8039093d017273` |

P2 defines the placebo regions, the measures, the tripwire's original form, and the PASS/KILL rule. P5
("Amendment E1") replaces only the tripwire's null with the calibrated component-shuffle version used here,
after the original flag turned out to fire on a single isolated letter-sized blob (documented in
`results/lie_detector_v0e.md`) -- the text-like *definition* never changed, only its statistics.
`prereg/verify_prereg.sh` in the public release tree verifies both files byte-for-byte against these hashes.

## v0-E results (18 models, PASS)

| Segment | Models registered / listed | Registration fail | Tripwire fires | Placebo FP range |
|---|---|---|---|---|
| w018 | 9 / 11 | 2 (both excluded pre-placebo) | 0 of 9 + 1 consensus | 1.33% - 8.12% |
| w023 | 8 / 8 | 0 | 0 of 8 + 1 consensus | 2.97% - 8.30% |

**No tripwire fire** on any of the 16 registered models or the 2 per-segment consensus checks. Verdict **PASS**:
34 of 64 model pairs show placebo false-positive ratio >= 1.5 with a 95% CI excluding 1, and the pattern of
*where* models put extra ink in the placebo is consistent between the two segments (Spearman rho = 0.857, n =
8; post-hoc 95% bootstrap CI [0.289, 1.000], exact permutation p = 0.011 two-tailed -- see
`results/lie_detector_v0e.md` "Post-hoc"). PASS is a statement about the harness -- these models reliably
distinguish something between the two segments' blank patches in a way that replicates -- not a claim that any
model found text; rule (a) already says none did, because nothing fired. **The 34/64 pairs are not 64
independent tests** -- in plain English, they're drawn from only 17 distinct models (9 + 8), so many pairs share
a model with other pairs; read the fraction as how often ratio-consistency held across that pool of 17 models,
not as 64 separately-powered trials. Full per-model table: `results/lie_detector_v0e.md`.

One file, `1667_2um_pred.tif`, stopped the run's first pass on 2026-09-26 under the pre-E1 tripwire; it fails
registration (AUROC 0.495) and was closed as a registration/alignment artifact, not a text-like result --
covered in detail in `results/lie_detector_v0e.md`.

## E1 validation evidence

Before any of the 18 remaining predictions was read, the E1 tripwire's calibration was checked with 100 seeded
synthetic fills of each real guarded region mask, split into 3 cases that must stay quiet (one letter-sized
blob; four letter-sized blobs; structureless pixel noise at the observed placebo FP rate) and 3 that must fire
(planted letter rows at 6 mm pitch; pure periodic bands at 4 mm and 6 mm pitch, no letters). Gate: quiet cases
fire in <= 2 of 100 seeds, must-fire cases in >= 95 of 100, and all 12 (case, region) cells must pass or E1 is
not used. **Result: 12/12 cells pass** -- every quiet case fired **0/100** times on both regions; every
must-fire case fired effectively **100%** on both regions. Full numbers:
`results/lie_detector_v0e_e1_validation.json`. Run `python -m tools.tripwire calibrate --region
YOUR_MASK.tif` for a smaller, unvalidated version of the same battery on your own region.

## How this differs from InkSurf and ink-disagree

Two independently published tools test adjacent ground on PHerc.1667 false positives -- credit to both authors
for public, pre-registered work in the same space:

- **[InkSurf](https://github.com/BioMarco/Inksurf)** (BioMarco) freezes a protocol before pixel reveal and
  reports false-positive **area** per negative cm2 on unannotated surface. It explicitly declines to review
  micro-patches or connected components.
- **[ink-disagree](https://github.com/ibarapascal/ink-disagree)** (ibarapascal) pre-registers several protocol
  versions and distinguishes labeled-blank from not-yet-annotated surface, reporting a base-model ink rate of
  0.1% on unannotated vs 23% on annotated surface -- a useful false-positive baseline.

Neither anchors "blank" to the published edition's column layout, and neither tests whether the false-positive
ink is *shaped* like writing. This package differs on three points:

1. **Blank is defined by the published edition's column geometry**, not by any model's own labels or by
   "unannotated" status -- a scribe's layout, not an algorithm's opinion, says these gaps have no text.
2. **A calibrated text-structure tripwire**, not just an area rate: row periodicity and letter-row layout,
   compared to a component-shuffle null built from the model's own blobs (a single big non-periodic splotch will
   not fire this, by design -- see "Limits" below). Pair the two kinds of check: an area-based tool tells you
   *how much* false-positive ink there is; this one tells you whether what's there is *shaped like text*.
3. **Pre-registered in two stages, both hashed and posted before their results existed** (P2 for the original
   protocol, P5/Amendment E1 for the recalibrated null after a documented false-positive-rate problem in the
   first version) -- the amendment history is public, not just the final rule.

## Limits

- **Two segments of one scroll (PHerc.1667), 1,316 mm2 over two intercolumns.** This does not test any other
  scroll, any margin region, or any placebo/control region beyond the two studied here.
- **The margins row (z < 81 mm above the text) was not run** -- it is conditional on the org confirming the
  column foot is at the high-z end of the mesh, which remains unconfirmed.
- **Two of 19 files were excluded by the registration gate** and never entered the placebo measures.
- **A single large, non-periodic false-positive blob will not fire the tripwire** -- the component-shuffle null
  keeps clumpiness by design (see `tools/tripwire/README.md` "Limits"), so this is not a general false-positive
  detector; pair it with an area-based check (InkSurf, ink-disagree, or the placebo FP-area rate this CLI already
  prints alongside the tripwire result) for that.
- **A single isolated mark -- a paragraphos, coronis, or other one-off annotation stroke -- may not fire this
  tripwire even if genuine**, because the tripwire is built to catch row-like structure (periodic rows or >= 3
  letter-sized blobs in a line), not any single mark. This tool does not claim to catch every kind of mark that
  could sit in a "blank" gap, only writing-like rows.
- **Registration first** means a file that fails the AUROC gate never gets a placebo reading -- if that gate is
  wrong for your file (wrong canvas, flipped axis), you will see an exclusion, not a placebo result.
- **This tool does not itself decide the scroll is blank** -- it tests structure inside a region this project
  asserts is blank on the published edition's authority. If you disagree with that geometry, the tripwire result
  does not settle it.

## Tests

`python -m pytest tools/placebo_check/tests -q` -- synthetic arrays only (a tiny in-memory region + planted
letter rows / a single blob), no scroll data, no images, no network. A planted-rows fill must FIRE; a single
blob must not; the region-hash check must reject a corrupted shipped file.
