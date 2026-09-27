# tools.tripwire -- the Amendment E1 text tripwire

**In plain English: a smoke alarm for AI that invents rows of letters on blank papyrus.** You give it a model's
"here's where I think there's ink" map and a mask of a region that is supposed to be blank (a margin, a placebo
patch, an unlabeled control area). It tells you -- as numbers only, never as pictures -- whether the shape of
that "ink" looks like structured rows of writing rather than a random smear or a single blob, by comparing it
to a calibrated null model built from the map's own blobs. It never shows or saves an image.

This is a thin, documented wrapper around `tools/margin/tripwire_e1.py`, the frozen statistical engine used by
the project's v0-E placebo run. This package imports that module unchanged; it does not alter it.

## The frozen rule

For a prediction map `hi = (p > threshold)` cropped to a guarded blank region:

- **Part (a), row periodicity.** T1 = peak periodogram power (over 2.5-8 mm row periods) divided by median
  power (over 0.5-8 mm), on the region's 0.25 mm pooled row profile. Compare T1 to 1,000 surrogates built by
  taking the map's own connected components and moving each one (translation only, on a 0.25 mm lattice,
  pixel-exact shape) to a random position fully inside the region where it neither overlaps nor touches another
  moved component ("component-shuffle"). `p_a = (1 + #surrogates with T1 >= observed) / 1001`. Fires if
  `p_a <= 0.001`, i.e. the observed T1 beats every one of the 1,000 surrogates.
- **Part (b), letter rows.** S = the most letter-sized (1-4 mm^2, 8-connected) component centroids found in any
  1 mm row window. Fires if `S >= 3` and S beats every surrogate's S.
- **Fire = (a) or (b).** In plain English: fire means "this looks like rows of letters and not chance."

The full frozen text, including the choices for everything the pre-registration document left open, is in the
docstring and `INTERPRETATIONS` dict of `tools/margin/tripwire_e1.py`. That module's own math is not repeated
here; this README only documents how to run it as a standalone tool.

## Calibration evidence

The rule above was validated (before any real prediction was read) with 100 seeded synthetic fills of each of
two real guarded region masks, split into 3 "must stay quiet" cases (Q1: one letter blob, Q2: four letter blobs,
Q3: structureless pixel noise at the observed placebo false-positive rate) and 3 "must fire" cases (F1: planted
letter rows at 6 mm pitch, F2/F3: pure periodic rows at 4/6 mm pitch, no letters). Gate: quiet cases fire <= 2
of 100, must-fire cases fire >= 95 of 100, all 12 (case, region) cells must pass or the rule is not used.

Result (`results/lie_detector_v0e_e1_validation.json`): **12/12 cells pass** -- every quiet case fired
**0/100** times on both regions, every planted-row case fired the required threshold (effectively **100%** on
both regions). Full numbers, including median T1/S per cell, are in that file.

Run `python -m tools.tripwire calibrate --region YOUR_MASK.tif` to see this same battery, with fewer fills, on
your own region's geometry -- a sanity check on your data, not a re-run of the frozen validation above (which
used the project's own two segments and is not re-derived here).

## How this differs from InkSurf / ink-disagree

Both [InkSurf](https://github.com/BioMarco/Inksurf) (BioMarco) and
[ink-disagree](https://github.com/ibarapascal/ink-disagree) (ibarapascal) are false-positive **area** auditors,
and both are credited here as independent, pre-registered work on adjacent ground: they measure how much of a
"should be blank" surface a model paints as ink (e.g. false-positive area per negative cm^2, or % ink rate on
unannotated vs. annotated surface). Neither asks whether that ink is *shaped* like text.

This tool measures text-like **structure** with a calibrated null: it asks whether the ink's row-periodicity
and letter-sized-blob layout is more regular than a component-shuffle of the model's own blobs would produce by
chance. A model that paints a single large blob, or scattered noise at a realistic false-positive rate, does not
fire this tripwire even though it may fire an area-based one. Pair the two: a "blank-by-papyrology" region
definition (as InkSurf and ink-disagree use) tells you *where* to look; this tool tells you whether what's there
looks like writing.

## Limits

- **Assumes the region really is blank** (by papyrology/edition, not just "the model didn't label it"). This
  tool does not establish blankness; it only tests structure inside a region you assert is blank.
- **Needs registration first.** `check` requires `--pred` and `--region` to already share a pixel grid and
  shape; it does not resample, align, or crop. Getting your prediction map and region mask onto the same grid
  is your job -- document how you did it.
- **The component-shuffle null keeps clumpiness.** Surrogates move the map's own blobs as rigid shapes; they do
  not break up or smooth a single very large or oddly-shaped blob. A model that paints one big, non-periodic
  splotch will not fire this tripwire (by design -- that's not text-like structure), so this tool is not a
  general false-positive detector; pair it with an area-based check for that.
- **`calibrate`'s synthetic geometry (letter size, row pitch) is defined in millimeters** and converted to
  pixels using the voxel size you pass (default: the validated 2.399 um/px geometry). If your mask is at a very
  different resolution or a very different shape than the validated regions, treat the calibration numbers as
  approximate, not as a repeat of the exact validation above.
- **`calibrate` needs a region several millimeters across.** The must-fire cases plant rows of letters or bands
  at 4-6 mm pitch; a mask much smaller than that has no room to plant them (or can even stall while it tries),
  and a tiny mask can leave no room for a second letter blob in the quiet cases either. Point it at a real
  guarded region, not a small test crop.
- **Part (a) needs enough rows** to build a periodogram (a very short or narrow region may not have one); the
  tool raises rather than silently returning a meaningless number.
- **A single isolated mark** (a paragraphos, coronis, or other one-off annotation) is not, by itself, row-like
  structure, so it may not fire this tripwire even if genuine. The tripwire targets rows of writing, not any
  single mark in a blank region.

## Tests

`python -m pytest tools/tripwire/tests -q` -- small synthetic arrays only, few surrogates, no scroll data, no
images. A planted-rows region must FIRE; a single blob must not.
