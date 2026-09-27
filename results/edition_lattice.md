# Edition route: PHerc.1667 column lattice on the org's own 2.4 um ink renders (numbers only)
Run 2026-09-26 by builder (with reviewer discipline), under the settled-rules file 2026-09-26 "Edition route GO (numbers only)". Plan: `edition_lattice_plan.json`, committed and pushed (2bc58d8) before any render byte was downloaded. Code: `tools/margin/lattice.py`. Raw numbers: `edition_lattice.json`. No images were made or viewed, and no glyph was read. Reduced arrays live only in `data/edition_lattice/` (gitignored). $0.

**STATUS: HARD STOP (text rule).** The tripwire fired inside the w031 intercolumn candidate. The authors decide (details in "Hard stop" below).

## Plain-language summary
- **What was done.** The org's own 2.4 um ink render (model `20260417190342-new_canon_autoresearch_recipe`) was pulled for w018, w023, w029 and w031, one file at a time. Each was reduced to numbers and deleted. We then looked for the regular column/intercolumn pattern along the lines, the text height per column, and which z end is the column foot.
- **Registration is good.** All four renders sit on the official label canvas (same shape). Pixel AUROC against the republished `inklabels_v2` inside the annotation masks is 0.88 to 0.93 (threshold 0.75). The best label/render alignment is within one 48 um cell of zero. The official labels match our HF `inklabels_v2` pixel counts to within 0.5%.
- **The column lattice is clear on w018 and only there.**
  - w018 has 3 blank intercolumns, 10 to 14 mm wide (13 to 16 mm at the half level). They sit on a lattice with period **72.45 +- 0.25 mm**, residual 0.2 mm, null p = 0.018. The periodogram agrees (72.4 mm).
  - That period is 13% above the 1.4 m / 22 = 64 mm estimate (inside the pre-set +-15% band) and 3% below Diego-dcv's 75 mm. Read literally, it gives 19.3 columns over 1.4 m instead of 22.
  - **w023: lattice NOT detected.** The ink profile is patchy. There is one 115 mm stretch with no blank intercolumn, and the two clear gaps are 78.6 mm apart (p = 0.72).
  - w029 and w031 are single wraps (~70 mm), with one intercolumn each (14.0 and 10.3 mm wide). There is nothing to check a spacing against.
- **Text height per column.** Lines have a 5.5 to 6.1 mm pitch. The render shows 9 to 13 line centres per column, the same range as the edition's 10 to 13 preserved lines.
  - **At low z** the text runs right down to where every mesh stops (z 7 to 21 mm), so that end is censored by the mesh.
  - **At high z** the text stops between **67 and 86 mm**. The stopping height varies by column (SD 6.1 mm across columns, ~4 mm within a column), and 3 to 20 mm of papyrus remain beyond it.
  - On w023, w029, w031 and one w018 column, the render puts text 10 to 18 mm above the labeled edge (66.5 to 68.4 mm). This fits the forensics' H2 ("labels stop short").
- **Orientation: UNDETERMINED** under the pre-set descender-skew rule.
  - 3 of 4 skews point to the foot at high z, but only w023's CI excludes zero. w018 just misses (CI -0.002 to 0.245), and w029 leans the other way with a wide CI.
  - The straight-vs-ragged check classes the high-z end RAGGED and the low-z end INTERMEDIATE, so it gives no secondary reading either.
- **Candidate blank regions** (geometry only, NOT audited):
  - **Intercolumns:** 7 regions, 4,336 mm2, 68 canvas-anchored 5 mm tiles (110 region-packed). Only **2 are unflagged and passed the tripwire**: w018 ic_201 (772 mm2, 13/28 tiles) and w023 ic_132 (1,010 mm2, 15/29 tiles), **1,782 mm2 and 28/57 tiles** together.
  - **Margin bands:** high z below 81 mm is 1,969 mm2 (28/36 tiles, mostly w018). The tapered zone at 81 mm and above is 2,508 mm2 (unreliable, 2/5 tiles). Low z is 76 mm2 (0 tiles).
  - The high-z bands are a lower margin **only if the foot is at high z**, which is still open.
- **Hard stop.** The w031 intercolumn candidate ic_33 shows strong row periodicity at 6.2 mm (about the line pitch): T1 198 vs shuffle-null median 27, p = 0.005, the minimum possible. The letter-blob test did not fire (at most 2 blobs in a 1 mm row).
  - It could be marginalia. It could also be line ends or column lean spilling into a narrow (5.1 mm) blank core.
  - Scalars only; nothing more was computed on that region. Three tiny w031 margin slivers (63 mm2) were left untested.
- **Independence caveat.** The render model is the `ink_canonical_2um` ResNet-152 production checkpoint. Its training data are not published, and whether it saw 1667 labels (these four segments included) is **unknown**. `hecate` is a fine-tune of the same model.

## 1. Registration
| Seg | Canvas (rows x cols) | AUROC | Recall p>0.5 | Unlabeled FPR | Best shift (cells) | r at 0 | Official / HF label px | Render MB |
|---|---|---|---|---|---|---|---|---|
| w018 | 42380 x 98100 | 0.927 | 0.73 | 0.044 | 0, +1 | 0.77 | 56.83 M / 56.80 M | 340 |
| w023 | 41860 x 102360 | 0.884 | 0.64 | 0.067 | 0, 0 | 0.66 | 22.67 M / 22.64 M | 386 |
| w029 | 38000 x 31320 | 0.902 | 0.78 | 0.136 | -1, 0 | 0.68 | 7.02 M / 7.01 M | 121 |
| w031 | 37480 x 32180 | 0.892 | 0.65 | 0.055 | +1, +1 | 0.70 | 13.75 M / 13.68 M | 89 |

- Canvas rows increase with z on all four (Spearman 1.0).
- The official and HF canvases are different flattenings of the same surface. The 3D nearest-neighbour distance is a median 19 to 21 um. The pixel offset is not constant: median +55 to +1463 rows, IQR 41 to 348 px.
- **Regions must therefore be moved between canvases through the 3D map** (saved in `data/edition_lattice/<seg>/candidates_hf_grid.npz`), never by a fixed shift.

## 2. Column lattice (x = along the lines, canvas mm)
| Seg | Valid length (mm) | Column blocks (x mm, width) | Intercolumn gaps: x mm, width at 0.3 / 0.5 level | Lattice |
|---|---|---|---|---|
| w018 | 229 | 2.4-49.2 (46.7, edge), 62.4-122.9 (60.5), 132.8-193.8 (61.0), 207.6-231.5 (23.9, edge) | 49.2-62.4: 13.2 / 15.1; 122.9-132.8: 9.9 / 13.3; 193.8-207.6: 13.8 / 15.7 | **P = 72.45 +- 0.25 mm**, residual RMS 0.2 mm, R 0.9998, p 0.018 (2000 hard-core draws); periodogram 72.4 mm, block-shuffle p 0.025. DETECTED |
| w023 | 240 | 2.9-117.6 (114.7), 146.0-189.0 (43.0), 194.5-203.4 (8.9), 217.4-242.6 (25.2, edge) | 117.6-146.0: 28.4 / 42.1; 189.0-194.5: 5.5 / 39.3 (off-lattice); 203.4-217.4: 14.1 / 16.6 | R 0.79, p 0.72; periodogram p 0.91. NOT DETECTED (the 2 consistent gaps are 78.6 mm apart) |
| w029 | 70 | 12.5-24.5 (12.0, weak), 38.4-74.8 (36.3, edge) | 24.5-38.4: 14.0 / 16.4 | one gap, no spacing check |
| w031 | 68 | 13.9-27.4 (13.6), 37.7-72.3 (34.5, edge) | 27.4-37.7: 10.3 / 12.3 | one gap, no spacing check |

- **Threshold sensitivity** (pre-registered, levels 0.2 / 0.4 / 0.5): the w018 gaps stay put at every level. The w023 gaps move a lot (a gap appears at 30.7-37.4 mm only at 0.5).
- **w023 level profile.** Levels (0 = gap floor, 1 = 95th pct) run 0.3 to 1.1 across 0-115 mm. They dip to 0.31 at x = 32 mm and to 0.41 at 68 mm, but nowhere reach the ~0 of a blank intercolumn. Clear blank gaps appear only at 131-143 and 207-215 mm.
- **Edition comparison.** 72.45 mm is 13.2% above 64 mm (inside the pre-set +-15%) and 3.4% below 75 mm, which implies 19.3 columns over 1.4 m.

## 3. Text extent in z per column (render, half level of the one-pitch-smoothed profile)
| Seg | Column x (mm) | Pitch (mm) | Low edge (mm) | High edge (mm) | High edge width 20-80% (mm) | Lines est. / line centres | Raggedness low / high (mm) | Papyrus beyond: low / high (mm) |
|---|---|---|---|---|---|---|---|---|
| w018 | 3.5-48.2 | 5.77 | 7.0 (mesh end) | 67.3 | 17.6 | 10.5 / 10 | 1.33 / 3.63 | 2.2 / 20.1 |
| w018 | 63.4-121.9 | 5.60 | 7.0 (mesh end) | 78.1 | 16.2 | 12.7 / 13 | 1.26 / 6.89 | 2.2 / 9.3 |
| w018 | 133.8-192.8 | 5.88 | 8.0 (mesh end) | 68.3 | 14.5 | 10.3 / 10 | 1.63 / 1.56 | 3.1 / 19.1 |
| w018 | 208.6-230.5 | 6.02 | 8.5 (mesh end) | 69.6 | 19.2 | 10.1 / 10 | 0.82 / 5.95 | 3.8 / 17.6 |
| w023 | 3.9-116.6 | 6.06 | 9.1 (mesh end) | 79.6 | 4.4 | 11.6 / 12 | 1.72 / 6.06 | 3.9 / 8.6 |
| w023 | 147.0-188.0 | 5.80 | 7.4 (mesh end) | 76.7 | 18.4 | 11.9 / 12 | 0.76 / 3.45 | 2.4 / 11.4 |
| w023 | 218.5-241.6 | 6.14 | 8.3 (mesh end) | 77.8 | 18.0 | 11.3 / 11 | 2.84 / 2.02 | 3.1 / 10.2 |
| w029 | 13.5-23.5 (weak) | 6.07 | 13.8 (mesh end) | 69.8 | 21.9 | 9.2 / 9 | n/a | 1.9 / 18.7 |
| w029 | 39.4-73.7 | 5.63 | 9.7 (mesh end) | 83.3 | 4.3 | 13.1 / 13 | 2.80 / 3.95 | 0.4 / 4.2 |
| w031 | 14.9-26.4 | 6.12 | 21.2 (mesh end) | 85.8 (mesh end) | 3.3 | 10.6 / 10 | n/a | 2.1 / 3.1 |
| w031 | 38.7-71.3 | 5.64 | 14.3 (mesh end) | 80.6 | 6.9 | 11.8 / 12 | 2.17 / 4.58 | -0.9 / 7.0 |

**Straight scribal edge vs ragged break** (pre-set rule, all 11 columns pooled):

| End | Median raggedness | SD across columns | Edge range | Median papyrus beyond | How edges were found | Class |
|---|---|---|---|---|---|---|
| low z | 1.63 mm | 4.17 mm | 7.0-21.2 mm | 2.2 mm | 11/11 at the mesh end | INTERMEDIATE |
| high z | 3.95 mm | 6.07 mm | 67.3-85.8 mm | 10.2 mm | 10 drop, 1 mesh end | RAGGED |

**Secondary reading: none.** Neither end is STRAIGHT.
- In plain terms, the low-z "edge" is where the official meshes stop, so it says nothing about the writing.
- At high z the text ends at different heights in different columns. On w018 and w023 the density fades over ~15-19 mm (about 3 lines) rather than stopping at a crisp last line.

## 4. Orientation (pre-set rule from tools/margin/layout_inventory.py, unchanged)
| Seg | Strips | Pitch (mm) | Skew | 95% CI | Strips positive | tail_asym [CI] |
|---|---|---|---|---|---|---|
| w018 | 22 | 5.49 | +0.141 | [-0.002, 0.245] | 77% | +0.031 [-0.007, 0.070] |
| w023 | 20 | 6.17 | +0.180 | **[0.027, 0.232]** | 65% | +0.075 [0.031, 0.109] |
| w029 | 5 | 5.54 | -0.135 | [-0.241, 0.104] | 20% | -0.074 [-0.143, -0.001] |
| w031 | 6 | 5.64 | +0.073 | [-0.206, 0.250] | 67% | +0.052 [-0.065, 0.120] |

**Verdict: UNDETERMINED.**
- The rule needs a CI above 0 on at least 3 of 4 segments and none below 0. Here 1 CI is above 0 and none is below.
- Descriptively, the lean is toward the foot at high z (3 of 4 point estimates). But w029's secondary statistic (tail_asym) points the other way.

## 5. Candidate blank regions (geometry only; NOT audited)
Only regions of 25 mm2 or more are listed; every region is in the JSON. "Tiles" = 5 mm tiles wholly inside the region: canvas-anchored 2084 px grid / region-packed. "Level" = render mean relative to the text contrast (0 = gap floor). "Trip" = tripwire result: periodicity p, and max letter blobs in a 1 mm row (S).

| Seg | Region | x (mm) | z (mm) | Area (mm2) | Tiles | Level | Flags | Trip |
|---|---|---|---|---|---|---|---|---|
| w018 | ic_56 | 51.4-61.1 | 7.0-67.3 | 675 | 13 / 13 | 0.00 | neighbour extents differ > 5 mm | p 0.020, S 1: pass |
| w018 | ic_128 | 126.6-130.7 | 8.0-68.3 | 295 | 0 / 0 | 0.04 | neighbour extents differ > 5 mm | p 0.39, S 0: pass |
| w018 | **ic_201** | 195.4-205.8 | 8.5-68.3 | **772** | **13 / 28** | 0.01 | none | p 0.59, S 0: pass |
| w018 | high z, col 2-49 | 3.0-49.2 | 69.3-81.0 | 579 | 10 / 14 | 0.10 | none | p 0.040, S 1: pass |
| w018 | high z, col 133-194 | 132.8-193.8 | 70.3-81.0 | 692 | 12 / 14 | 0.00 | none | p 0.43, S 0: pass |
| w018 | high z, gap 201 | 193.8-207.6 | 71.6-81.0 | 137 | 2 / 2 | -0.12 | none | p 0.72, S 0: pass |
| w018 | high z, col 208-232 | 207.6-228.3 | 71.6-81.0 | 202 | 3 / 4 | 0.12 | none | p 0.52, S 0: pass |
| w018 | high z, col 62-123 | 62.4-122.9 | 80.1-81.0 | 57 | 0 / 0 | -0.06 | none | not testable, S 0 |
| w018 | taper (>= 81 mm), 7 regions | 2.8-227.7 | 81.0-86.5 | 1,172 | 0 / 0 | -0.79 to 0.06 | tapered roll end | not testable, S 0 |
| w023 | **ic_132** | 131.5-144.5 | 9.1-76.7 | **1,010** | **15 / 29** | 0.03 | none | p 0.035, S 1: pass |
| w023 | ic_210 | 206.9-215.3 | 8.3-77.8 | 703 | 15 / 16 | 0.09 | only one neighbour column has an extent | p 0.075, S 1: pass |
| w023 | high z, col 146-189 | 146.0-189.0 | 78.7-81.0 | 103 | 0 / 0 | -0.01 | none | not testable, S 0 |
| w023 | low z, col 3-118 | 10.5-117.6 | 5.9-7.1 | 63 | 0 / 0 | -0.23 | none | not testable, S 0 |
| w023 | taper (>= 81 mm), 6 regions | 10.9-234.4 | 81.0-87.3 | 1,195 | 2 / 5 | -0.58 to 0.11 | tapered roll end | not testable, S <= 1 |
| w029 | ic_31 | 28.3-36.7 | 14.5-69.8 | 536 | 12 / 12 | 0.04 | neighbour extents differ > 5 mm | p 0.030, S 1: pass |
| w029 | high z, col 12-24 | 12.5-24.5 | 71.8-81.0 | 124 | 1 / 2 | 0.18 | none (weak column) | p 0.47, S 1: pass |
| w029 | taper, col 12-24 | 12.5-24.5 | 81.0-87.4 | 63 | 0 / 0 | 0.25 | tapered roll end | not testable, S 1 |
| w031 | **ic_33** | 28.4-33.5 | 21.3-80.6 | 345 | 0 / 12 | -0.04 | neighbour extents differ > 5 mm | **p 0.005, S 2: FLAG -> HARD STOP** |
| w031 | taper, col 38-72 | 57.0-72.3 | 82.6-88.0 | 63 | 0 / 0 | 0.26 | tapered end; level > 0.25 | NOT TESTED (after the stop) |

**Totals.**

| Kind | Regions | Area (mm2) | Tiles (canvas / packed) | Unflagged and passed |
|---|---|---|---|---|
| Intercolumn | 7 | 4,336 | 68 / 110 | 2 regions, 1,782 mm2, 28 / 57 tiles |
| High-z band < 81 mm | 12 | 1,969 | 28 / 36 | 12 regions, 1,969 mm2, 28 / 36 tiles (a margin only if the foot is at high z) |
| High-z taper >= 81 mm | 17 | 2,508 | 2 / 5 | none (all flagged tapered) |
| Low-z band | 7 | 76 | 0 / 0 | 5 regions, 74 mm2, 0 tiles (plus 1 untested 0.1 mm2 w031 sliver) |

**What makes candidates unreliable:**
- **Tapered roll end.** Everything at z >= 81 mm is flagged.
- **Uneven neighbour columns.** Three intercolumns have neighbour columns whose high-z text edges differ by more than 5 mm, so their top is set by the lower neighbour.
- **Orientation.** The high-z bands sit on the side where the text edge is ragged and fading. That side may be the break, not a margin.
- **w023 lattice.** The lattice is not established there, so its intercolumns are not confirmed as lattice nodes.
- **Holes and stretch.** No region has interior mesh holes > 5% or mesh stretch outside 0.9 to 1.1.
- **HF canvas.** Every region has an HF-canvas bbox via the 3D map (JSON: `hf_canvas`). At most 0.02% of cells were rejected as 3D self-coincidences.

## Hard stop (text rule) for the authors
| Where | Numbers | Action |
|---|---|---|
| w031 intercolumn candidate ic_33 (x 28.4-33.5 mm, z 21.3-80.6 mm, 345 mm2; blank core only 5.1 mm wide; render level at the gap floor) | Row periodicity T1 198 vs 1 mm tile-shuffle null median 27 / p95 46, **p = 0.005** (the 200-surrogate minimum), peak period 6.21 mm; region p > 0.5 fraction 6.0%; letter blobs 6 (1.7 per cm2), max 2 in a 1 mm row (blob flag cannot fire) | Scalars only. The raw render was already due for deletion and was deleted. No further computation on this region. The 3 remaining w031 regions were not tested. No images anywhere |

- **Readings we can't separate here (The authors decide):**
  - Marginalia or other writing in the intercolumn.
  - Line ends of the left column or line starts of the right column reaching into the 5.1 mm core (column lean, "Maas's law", would do this).
  - A model response that follows the line rhythm next to text.
- **Context.** The neighbouring w031 columns' text runs 80.6 and 85.8 mm high. The other six intercolumn candidates on w018, w023 and w029 all passed, with p from 0.020 to 0.59 and S <= 1.

## Independence caveat
- **Render model.** `20260417190342-new_canon_autoresearch_recipe` is the `ink_canonical_2um` checkpoint (HF card: ResNet-152 3D encoder + 3D decoder, recipe `new_canon_autoresearch_recipe`, epoch 13). The card does not state its training data.
- **What villa docs/37 says.** The autoresearch recipe was found while "training only on PHerc. 139 data", and its Dice was "computed on pseudo-labels" of PHerc. 1667. So 1667 pseudo-labels were at least used to *select* the recipe.
- **What stays open.** Whether this checkpoint was *trained* on 1667 labels, these four segments included, is **UNKNOWN**. `hecate` is a fine-tune of this same model, so it is not independent of the render.
- **Consequence.** The lattice (x) is a coarse, robust feature and should carry over. The per-column text extent, and so the margin bands, inherit whatever this model does near the labeled edge.

## Columns 1-3 (text rule)
- By unrolled length from the outer end (sum of official mesh lengths, merged mesh 1,399.8 mm), w031 starts 597 mm in, w029 735 mm, w023 872 mm and w018 1,134 mm.
- Cols 1-3 (at most ~225 mm) fall in w037-w041. **No target segment is in cols 1-3**, so none was stopped for that reason.

## Deviations
- **D1** (before any render, recorded in the plan). The block threshold was changed from the 0.5 to the 0.3 level after the synthetic test, because a half-contrast column vanished at 0.5.
- **D2** (after the stop). The plan's stop clause says to pull no further render; all four renders had already been pulled and deleted. Three steps followed:
  - the pre-set orientation rule, run on text-region cells of the already pooled arrays;
  - the geometry-only canvas map;
  - this write-up.

  None of them reads a raw render or the flagged region. the authors may prefer the orientation to be held back until he rules. It is published-text only (rule a).
- **D3.** The canvas map was fixed after its first run.
  - What changed: it now uses the full HF grid (not every 2nd point) and rejects matches more than 2 mm off the region's median offset.
  - Why: the first version let a handful of 3D self-coincident HF points stretch bboxes across whole segments.
  - Effect: reporting only; no decision uses it.
- **D4.** `orientation()` was reused unchanged. Its output fields `labeled_top_edge_z_mm` / `labeled_bottom_edge_z_mm` are render-derived here (naming only).
- **D5.** P_pool comes from w018 alone (the w023 lattice was not detected). w029 and w031 had one intercolumn each, so the pre-set spacing check could not run.
- **D6.** `report()` / `straight_ragged()` were added to the code after the runs (reporting only). All segment computations ran on the pushed plan code (2bc58d8).

## Resources
- **Downloads:** 4 renders (936 MB) plus official tifxyz and label shards (316 MB), one render at a time, each deleted after its tripwire pass.
- **Disk:** 23.6 GB free at the end (floor 6 GB); `data/edition_lattice/` holds 375 MB.
- **Memory:** peak per process under ~3 GB (2080-row strips). Other jobs on the box used 6-9 GB.
- **Cost:** CPU only, about 1.5 h, $0.
- **Synthetic checks (before real data):** 10/10 pass.
  - Lattice P within 1 mm of 64.
  - Straight vs ragged edges separated.
  - +z / -z descender skew signs recovered.
  - A blank region does not flag; planted letter rows do.
