# Lie Detector v0-E: HARD STOP (text rule) on the first pred, w018 intercolumn placebo (numbers only)
Run 2026-09-26 by builder (with reviewer discipline). Frozen rule: `prereg/lie_detector_v0e.md` (the frozen-criteria history v0-E block at 7538c31, SHA-256 `a3dcaea682f49e56...`, verified). Placebo regions, harness checks and the analysis plan were committed and pushed before any pred was downloaded (48d2cc4, 7086eb2: `lie_detector_v0e_regions.json`). Code: `tools/margin/v0e.py`. Scalars: `lie_detector_v0e.json`. No images were made or viewed, and no glyph was read. $0.

**CURRENT STATUS (2026-09-27): PASS under Amendment E1 on the 18 remaining preds, no tripwire fire; see "E1 resume" below. The section that follows is the historical record of the first-pred stop (closed as an alignment artifact).** Historical status at the time: The recalibrated tripwire (Amendment D flag) fired inside the w018 placebo region for the first pred processed. Per the frozen rule ("Any model firing inside a placebo region = HARD STOP, scalars only, The authors decide") nothing further was computed: no other pred was downloaded, and no pair ratio, Spearman or verdict exists. The authors decide.

## Plain-language summary
- **What stopped.** `1667_2um_pred.tif` (w018 only, provenance unknown) is the first pred in the pre-registered order. In the w018 intercolumn placebo, its p > 0.5 map has a strong row rhythm: T1 91.4 against a 1 mm tile-shuffle null median of 28.5 (p95 51.5). That gives p = 0.005, the smallest value 200 surrogates allow, with the peak at 4.16 mm.
- **Same pred fails registration.** Its AUROC against the labels is 0.495 (chance; the threshold is 0.75), and it marks only 3.4% of labeled ink at p > 0.5. Its values are unrelated to this segment's labels: either the file is not on this canvas, or it is not a usable ink map here. Either way, what sits at the placebo coordinates in this file may not be the model's reading of the intercolumn. It would have been excluded from every ratio. The plan still ran the tripwire on it, because the stop rule covers every model (fixed before any pred was read).
- **No letter rows.** Placebo FP area is 2.78%, the same as this pred's FP on in-mask unlabeled text-block pixels (2.90%). It has 77 components, 2 of them letter-sized (1-4 mm2), and the largest is 9.1 mm2. At most 1 letter blob sits in any 1 mm row, so the blob-row part cannot fire. Only the periodicity part fired. The 4.2 mm peak is inside the 2.5-8 mm text band but not at the 5.5-6.1 mm line pitch.
- **Known property of the frozen flag (recorded before any pred).** On synthetic 7.4 x 60 mm strips, the periodicity part fires on non-periodic structure. A single isolated letter-sized blob gives p <= 0.01 in 20/20 strips, and so do 4 blobs (20/20). Structureless pixel noise gives 0/50, and sub-millimetre blobs give 0/20. The 1 mm tile shuffle destroys any FP structure coherent over more than 1 mm, so a firing means that kind of structure is present, not specifically rows.
- **Readings the scalars can't separate (The authors decide).**
  - **Misregistration or not an ink map here.** The file's values are unrelated to this canvas's labels (AUROC 0.495). If it is on another flattening or mirrored, the rhythm comes from wherever these pixels really are on the surface, possibly text columns.
  - **Clumpy non-text FP.** The flag responds to any structure coherent over more than 1 mm; here the largest component is 9.1 mm2.
  - **Model output with a row rhythm in blank papyrus.** This is the hallucination signature the test looks for, or unrecognized writing. The scalars can't rule it out.

## Scalar evidence
| Item | Value |
|---|---|
| Segment / region | w018 / ic_201 (guarded core 7.34 mm, 545.8 mm2 on the HF canvas) |
| Pred | `preds/1667_2um_pred.tif`; sha256 `63bb1b2acfa3f47b...`; uint8, same shape as the labels; Software tag `inference_ome_zarr.py`; not identifiable as the ink_canonical_2um family (no keyword in name or metadata) |
| Registration AUROC / recall at p > 0.5 | 0.495 / 3.4% (FAILS >= 0.75) |
| Placebo FP area at p > 0.5 | 2.778% (2,634,821 px) |
| Reference FP, in-mask unlabeled | 2.902% |
| Components / letter blobs (per cm2) / largest | 77 / 2 (0.366) / 9.1 mm2 |
| Blob rows: S (max letter blobs in a 1 mm row) | 1 (< 3: cannot fire) |
| Row periodicity: T1 / peak period | 91.4 / 4.16 mm over a 74.8 mm profile (300 rows of 0.25 mm) |
| 1 mm tile-shuffle null: median / p95; p | 28.5 / 51.5; p = 0.0050 (flag at <= 0.01) |

## What was and was not done
- **Done before any pred (committed and pushed).** Placebo regions after guards: w018 ic_201 545.8 mm2 (core 7.34 mm) and w023 ic_132 770.6 mm2 (core 9.98 mm), 1,316.4 mm2 in total. Neither was dropped. Both were mapped through the saved canvas map, rebuilt bit for bit (non-constant offset). Mask hashes, tile blocks and the analysis plan are in the regions file.
- **Harness checks before any pred.** All synthetic checks pass: letter rows fire, the null is calibrated on pixel noise, planted blobs are recovered, a true 2x pair passes, self-pairs are exact and tile sums agree. A plumbing run with the label file as a stand-in pred gives AUROC 1.000, and 0 labeled-ink px in either placebo.
- **Preds.** 1 of 19 processed (download, reduce, raw deleted). The run stopped on the first. None of the other 10 w018 preds or 8 w023 preds was downloaded.
- **Not computed (stop).** Planted-blob sanity on a real pred, self-pair on real preds, pair ratios, bootstrap CIs, Spearman, the consensus tripwire and the verdict. The margins row is not run in any case: the column-foot answer is not in.
- **Kept.** The per-pred scalars and 5 mm tile counts are kept. This pred's 0.25 mm pooled FP map was not kept (stop semantics). No image exists.

## Options for the authors (no action taken)
1. **Treat it as a registration artifact.** This pred fails the frozen registration gate and would never enter a ratio. Resume v0-E on the other 18 preds under the same stop rule. Note that given the flag property above, another pred whose placebo FP holds even one letter-sized blob may stop the run again.
2. **Diagnose the canvas of `1667_2um_pred.tif` first**, for example flips or another flattening against the labels. This needs a re-download and more computation on the stopped file.
3. **Keep v0-E stopped.** The reviewer reviews the tripwire's specificity (the frozen text can't change; any change would be a new, separately approved amendment).

## Deviations
- **None during the run.** The run followed the plan committed before any pred. The interpretations fixed before any pred (listed in the regions file) are:
  - bootstrap units are all 5 mm tiles that touch the region;
  - the guard is 32 cells = 1.535 mm;
  - the edge and z guards are applied on both meshes;
  - PASS requires that no KILL clause holds;
  - the tripwire also runs on preds that fail registration, and on the consensus;
  - there is a label bit cache;
  - after a firing, the blob-row null is skipped and the pooled map is not kept.
- **The stop came from the "tripwire also on registration failures" item.** It was committed in 7086eb2, before this pred was downloaded.

## Resources
- **Downloads.** One pred, 253 MB in 3 s; reduced in 33 s; raw deleted.
- **Disk.** 19-20 GB free throughout (floor 6 GB). `data/v0e/` holds a 1.3 GB label bit cache and the scalars.
- **Memory.** Peak 2.2 GB for the reduce process; the other jobs on the box used about 10 GB.
- **Cost.** CPU only, $0.

---

# E1 resume (2026-09-27): the 18 remaining preds -- PASS, no stop

Amendment E1 (`prereg/lie_detector_v0e_e1.md`, P5) fixed the tripwire's statistics after the counted validation
above passed 12/12 cells (`results/lie_detector_v0e_e1_validation.json`, all_pass). This run processes the 18
preds the original stop above left untouched, in the pre-registered order: w018's 10 remaining preds, then w023's 8.
`1667_2um_pred.tif` is not re-downloaded; its old numbers are reused and it stays excluded by the registration rule
(closed by the authors as an alignment artifact, not a text-like result). Code: `tools/margin/v0e_e1.py` +
`tools/margin/tripwire_e1.py`. Scalars only, everywhere: `results/lie_detector_v0e.json` key `e1.run`,
`data/v0e/e1/analysis.json` (gitignored, full detail). No image was made or viewed. $0.

## In plain English
We asked, for each of 18 AI guesses at where ink is on scroll segments w018 and w023: (1) does this guess line up
with where the known, published readings say ink actually is (**registration**, a sanity check, not the text test),
and (2) inside a patch of papyrus that has **no** published reading (the **placebo** -- a blank stand-in), does the
guess draw anything that looks like rows of handwriting (**the tripwire**)? A tripwire firing on a placebo is a hard
stop under rule (a): it would mean a model appears to "see" writing where none has been read, which needs a human
to look at, not more automated computing. **Nothing fired.** Two of the eighteen guesses didn't even pass the
sanity check (their pixels don't line up with the known readings) and were set aside before their placebo patch was
ever looked at, per the rule; the rest (16) were checked, plus two extra "vote of all models together" checks, and
every one came back clean. Separately, the same 16 guesses' patterns of *where* they put extra ink in the placebo
line up well between the two segments (a consistency check, `Spearman rho`), and several pairs of guesses show
significantly more agreement with each other than chance would (the `pair ratio` / `CI` tests) -- both of which
this pre-registered test needed to call the harness itself trustworthy ("**PASS**"). PASS is a statement about
whether the measuring tool works, not a claim that any text was found -- rule (a) already tells us none was, since
nothing fired.

## Verdict: PASS
`>= 1 model pair with placebo FP ratio >= 1.5 and 95% CI excluding 1, AND Spearman >= 0.5` -- both held (34/64
pairs pass at ratio >= 1.5 with CI excluding 1; Spearman rho = 0.857, n = 8). All sanity checks true (registration
all-used >= 0.75, registration-fail <= 50% per segment, planted blob recovered at ratio 1.51 (CI [1.26, 1.82],
gate >= 1.3), all self-pairs exact 1.00, frozen cross-check all true). Margins row: **not run**, unchanged
(conditional on the org confirming the column foot at high z; still unconfirmed).

## Registration (pass 1, before any placebo pixel is read)
| Segment | Listed | Registration fail | Which |
|---|---|---|---|
| w018 | 11 | 2 | `1667_2um_pred.tif` (AUROC 0.495, closed stop, earlier record, not re-downloaded); `ps512_scale1_dino_frozen_...` (AUROC 0.697) |
| w023 | 8 | 0 | none |

2 of 18 registration failures, both in w018 -- well under the "3+ of 18 -> pause" gate, so the run continued
without pausing.

## Per-pred placebo + E1 tripwire (registered preds only; scalars, no images)
w018 (guarded region ic_201; 9 registered of 10 remaining):
| Model | AUROC | Placebo FP area | T1 (periodicity) | p_a | S (letter rows) | p_b | Fire |
|---|---|---|---|---|---|---|---|
| ps256_bcesmooth0.1_dicesmooth0.1 | 0.970 | 5.924% | 9.1 | 0.931 | 0 | 1.000 | No |
| ps256_mean | 0.961 | 1.331% | 15.2 | 0.509 | 0 | 1.000 | No |
| ps48_640_640_smooth_0.1 | 0.967 | 6.633% | 29.6 | 0.730 | 1 | 1.000 | No |
| ps512_betti_ema_0.9995_meanteacher | 0.984 | 3.734% | 20.4 | 0.561 | 0 | 1.000 | No |
| ps512_ema_0.9995 | 0.982 | 5.541% | 16.6 | 0.997 | 1 | 1.000 | No |
| ps512_max_betti | 0.979 | 5.329% | 22.6 | 0.709 | 1 | 1.000 | No |
| ps512_max_boundary | 0.954 | 8.123% | 14.4 | 0.970 | 1 | 1.000 | No |
| ps512_max | 0.970 | 5.720% | 11.4 | 0.979 | 1 | 1.000 | No |
| ps512_no_zstride | 0.933 | 2.324% | 29.2 | 0.483 | 1 | 1.000 | No |

w023 (guarded region ic_132; 8 registered of 8):
| Model | AUROC | Placebo FP area | T1 (periodicity) | p_a | S (letter rows) | p_b | Fire |
|---|---|---|---|---|---|---|---|
| ps256_bcesmooth0.1_dicesmooth0.1 | 0.856 | 7.462% | 12.3 | 0.819 | 1 | 1.000 | No |
| ps256_mean | 0.847 | 3.185% | 21.7 | 0.349 | 1 | 1.000 | No |
| ps48_640_640_smooth_0.1 | 0.926 | 7.003% | 15.6 | 0.990 | 1 | 1.000 | No |
| ps512_betti_ema_0.9995_meanteacher | 0.910 | 6.885% | 18.2 | 0.897 | 1 | 1.000 | No |
| ps512_max_betti | 0.916 | 4.774% | 30.4 | 0.368 | 1 | 1.000 | No |
| ps512_max_boundary | 0.888 | 8.297% | 35.4 | 0.326 | 2 | 0.307 | No |
| ps512_max | 0.912 | 8.240% | 15.9 | 0.943 | 1 | 1.000 | No |
| ps512_no_zstride | 0.882 | 2.967% | 9.6 | 0.990 | 0 | 1.000 | No |

Fire rule: (a) p_a <= 0.001, or (b) S >= 3 and S beats every one of 1,000 surrogates. No pred came remotely close
(smallest p_a = 0.326; largest S = 2, gate needs >= 3). No family-keyword preds, no merged-model duplicates.

## Consensus tripwire (part (a) on the mean p > 0.5 map of the registered preds)
| Segment | Models | T1 | Null max | p | Fire |
|---|---|---|---|---|---|
| w018 | 9 | 14.8 | 111.5 | 0.970 | No |
| w023 | 8 | 21.8 | 80.8 | 0.788 | No |

## Planted-blob sanity (P2; not a stop criterion)
Synthetic copy of `ps256_bcesmooth0.1_dicesmooth0.1` (w018): recovery ratio 1.51 (95% CI [1.26, 1.82]), expected
1.51 -- CI excludes 1 and clears the >= 1.3 gate. Self-pairs exact 1.00 everywhere. This is a planted, synthetic
blob to check the ratio machinery works, not a model result; the tripwire is never run on it.

## Deviations
- **Resources only, no correctness issue.** Peak RSS during the w023 reduce (including its 8-model consensus)
  reached 3.47 GB, above the <= 3 GB target; the harness's own MemAvailable (>= 4 GB) and disk (>= 8 GB) gates were
  satisfied before every download throughout, and nothing crashed, swapped, or was killed. w018's reduce peaked at
  2.89 GB, inside budget. No code change was made in response; logged here as instructed.
- **No other deviations.** Code review of `tools/margin/v0e_e1.py` against the Do-list before running: pre-registered
  order (w018 then w023, regions-file order) confirmed; `1667_2um_pred.tif` skipped and its earlier record reused
  confirmed; registration-before-placebo and the AUROC < 0.75 exclusion confirmed; per-pred E1 tripwire with frozen
  cross-check confirmed; per-segment independent-per-model-shuffle consensus with seeds `[20260925+k, m]` confirmed
  (`tools/margin/tripwire_e1.py`, unit-tested, 23/23 tests pass); frozen P2 analysis (pair ratios, bootstrap CIs,
  Spearman, sanity, verdict) reused unchanged via `tools/margin/v0e.py`; margins row left not-run. No bugs found,
  nothing fixed.

## Resources (E1 run)
- **Reduce w018.** Wall 5,806 s (~97 min); peak RSS 2.89 GB; 9 registered preds downloaded (0.58-1.22 GiB each,
  6-12 s each) and deleted after reduce; per-pred tripwire 61-602 s (1,000 surrogates each); consensus (9 models)
  2,681 s.
- **Reduce w023.** Wall 13,206 s (~220 min); peak RSS 3.47 GB (see deviations); 8 registered preds (0.67-1.26 GiB
  each); per-pred tripwire 185-3,214 s (the slowest pred had 3,278 components, bitmap implementation); consensus
  (8 models) 6,349 s.
- **Analyze.** Wall 0 s (pure aggregation of saved per-pred JSON), peak RSS 0.11 GB.
- **Disk.** Free disk stayed >= 12 GB throughout (floor checked before every download); each raw `.tif` deleted
  right after reduce.
- **Cost.** CPU only (1-2 BLAS threads, `nice -n 10`), $0.

## Overnight queue
`internal/OVERNIGHT_QUEUE.md` job 2 updated to DONE with this verdict.
