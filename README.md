# ink-placebo-check

A pre-registered audit of AI ink-detection models on PHerc.1667, plus a drop-in tool to run it on your own
predictions. Its pass/fail rules were frozen and hashed publicly before any result existed.

**In plain English: a lie-detector test for ink AI: we point it at papyrus scholars know is blank and check
whether it sees letters.** Two patches of PHerc.1667 sit between text columns that the published edition already
transcribed -- so we know they're blank not because a model or a label file says so, but because a human
papyrologist's reading of the scroll leaves them empty. We ran 18 public ink-prediction files through those
patches and asked: does anything look like rows of handwriting? Nothing did. Full numbers below.

## Result: PASS, no text-like finding

18 predictions on segments w018 and w023 (2 excluded by a registration sanity gate before their placebo patch
was ever read); **0 of 16 registered predictions and 0 of 2 per-segment consensus checks** triggered the
text-structure tripwire. The harness itself passes its own pre-registered validity test (model pairs replicate
across segments; Spearman rho = 0.857). Full report: [results/lie_detector_v0e.md](results/lie_detector_v0e.md).
Method, hashes, and how this differs from InkSurf and ink-disagree:
[tools/placebo_check/README.md](tools/placebo_check/README.md).

## Run it on your own prediction

```bash
git clone https://github.com/lightsgoblack/ink-placebo-check && cd ink-placebo-check
python -m pip install -e .
python -m tools.placebo_check run --pred YOUR_PRED.tif --segment w018   # or w023
```

`YOUR_PRED.tif` must share the labeled segment's pixel canvas (or be an exact integer downsample of it). Prints
scalars only -- registration AUROC, placebo false-positive rate, and the E1 tripwire's fire/no-fire decision.
Never writes, shows, or saves an image. Full CLI docs: [tools/placebo_check/README.md](tools/placebo_check/README.md).
The text-tripwire engine on its own, usable on any region you assert is blank: [tools/tripwire/README.md](tools/tripwire/README.md).

Running the check above on your own file needs no scan data at all -- the two guarded regions ship as small
(~40 KB each) coarse-grid masks. Reproducing the full 19-file, 2-segment v0-E audit from scratch instead needs
each public prediction TIFF (roughly 0.6-1.3 GiB apiece) plus the labeled segment volumes from the scrollprize
Hugging Face bucket, which are not shipped in this repository.

## Verify the pre-registration

```bash
sh prereg/verify_prereg.sh
```

Both frozen protocol texts (P2, the original placebo protocol; P5/Amendment E1, the recalibrated tripwire null)
must match the SHA-256 hashes posted in the Vesuvius Challenge Discord (#robots) on Sep 26, 2026, before any
score or verdict existed: [#robots hash log](https://discord.com/channels/1079907749569237093/1553425799162372096/1553425799162372096).

## What's here

| Path | What it is |
|---|---|
| `tools/placebo_check/` | The drop-in CLI: registration -> placebo FP rate -> E1 tripwire, on one prediction file |
| `tools/tripwire/` | The text-structure tripwire engine on its own (`check`, `calibrate`) |
| `tools/margin/` | The full research harness these tools wrap (v0-E, Amendment E1, the edition-lattice geometry) |
| `tools/harness/` | Single-file pulls from the scrollprize Hugging Face bucket (never syncs whole folders) |
| `results/` | Result reports (md) and all numbers (json), including the E1 calibration validation |
| `prereg/` | The frozen protocol texts (P2, P5), their hashes and the verify script |

## Rules we follow

- No scan data and no images of PHerc.1667 in this repo.
- "Blank" is defined by the published edition's column layout, never by a model's own output.
- Every tripwire fire is reported in full, including the one that led to the recalibrated null (Amendment E1) --
  the amendment history is public, not just the final rule.

## Citation and license

Code: MIT. Derived label files: CC BY-NC 4.0 (Vesuvius Challenge data terms). To cite this work,
use [CITATION.cff](CITATION.cff).

Analysis built with Claude; I directed it and checked the results.
