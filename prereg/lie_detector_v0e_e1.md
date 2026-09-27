### Amendment E1 to Lie Detector v0-E (skeptic, 2026-09-26, PRE-RESULT for the 18 preds not yet processed): a text-specific tripwire
**Status: APPROVED by Colin 2026-09-26 ("Criteria approved"), frozen at this commit. TESTING.** Basis: results/lie_detector_v0e.md.
- v0-E stopped on its first pred (1667_2um_pred.tif, w018). That file fails registration (AUROC 0.495; its pixels may not be on this canvas). Its rhythm is 4.2 mm, not the 5.5-6.4 mm line pitch, and it has no letter rows.
- Before any pred, the builder documented that the frozen flag fires on a single letter-sized blob (20/20 synthetic strips). The 1 mm tile-shuffle null destroys any clump larger than 1 mm, so clumpiness alone reads as "significant".
- E1 keeps the text rule's definition word for word and fixes only the statistics.
- Revised before approval: fire only when the pattern beats all 1,000 random rearrangements. At the old per-map threshold, 20 maps would give about a 1 in 3 chance of at least one stop from chance alone; E1 caps that at 4%.

| Item | Specification |
|------|---------------|
| Registration first | A pred with registration AUROC < 0.75 on the segment is EXCLUDED before any placebo pixel is read: no tripwire, no ratios. Its registration numbers are reported. Its placebo coordinates do not correspond to the placebo papyrus. |
| Tripwire (replaces the Amendment D flag, v0-E only) | Same text-like definition as the text rule: (a) row periodicity, the frozen T1 statistic (peak in 2.5-8 mm), OR (b) >= 3 letter-sized (1-4 mm2) components aligned in one 1 mm row. **Null changed to a COMPONENT-SHUFFLE**: 1,000 surrogates (seeds 20260925+k). Each surrogate places the region's own connected p > 0.5 components at random non-overlapping positions fully inside the guarded placebo. The components are translated only, never rotated. A component that cannot be placed after 1,000 tries keeps its position, and the count is reported. The shuffle keeps sizes and shapes (clumpiness) and destroys the arrangement. **Fire = (a) or (b) beats all 1,000 surrogates (p <= 0.001).** For (b), that means S >= 3 and S > the null maximum. |
| False-stop budget | At most 20 maps (10 w018 + 8 w023 preds + 2 consensus) x 2 parts x 0.001 gives at most a 4% chance of any stop from chance alone across the whole run. At the D threshold (p <= 0.01) that chance is about 1 in 3. |
| Validation before any pred is read (all must pass, committed) | Run on 100 seeded synthetic fills of each real guarded region mask (w018 ic_201, w023 ic_132), at the E1 fire rule. **Must stay quiet:** a single isolated letter-sized blob fires in <= 2%; 4 randomly placed letter-sized blobs <= 2%; structureless pixel noise <= 2%. **Must fire:** planted letter rows at 6 mm pitch >= 95%; planted pure periodic rows (no letters) at 4 and 6 mm >= 95%. If any case fails, E1 is not used and v0-E stays stopped. |
| Consensus tripwire | Part (a) on the mean of the registered preds' p > 0.5 maps. Null: each model's components are shuffled independently (the same 1,000 seeds). Same fire rule. |
| Action on firing | Unchanged: HARD STOP, scalars only, Colin decides. |
| The stopped file | 1667_2um_pred.tif is excluded under the registration rule, retroactively and by rule (it would never have entered a ratio); its stop is reported in full. |
| Everything else | v0-E exactly as frozen (regions, measures, PASS/KILL, sanity). |
