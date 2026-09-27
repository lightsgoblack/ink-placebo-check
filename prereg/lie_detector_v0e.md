### Lie Detector v0-E (skeptic, 2026-09-26, PRE-TEST): edition-defined placebo arm (intercolumns)
**Status: APPROVED by Colin 2026-09-26 ("Criteria approved"), frozen at this commit. TESTING.** Basis: results/edition_lattice.md (commit 1b84e89). The edition route defines blank by scribal layout, not by any audited model: w018 column period 72.45 +/- 0.25 mm (3 gaps). The w031 candidate ic_33 hard-stopped: core only 5.1 mm wide, row rhythm at line pitch (6.2 mm), no letter rows. That is most consistent with line-end spill into a too-narrow gap, so this draft guards against spill.

| Item | Specification |
|------|---------------|
| Placebo regions (fixed before any pred is read) | The two edition-lattice intercolumns marked reliable that passed the tripwire on the org render: w018 x 195.4-205.8 mm (z 8.5-68.3) and w023 x 131.5-144.5 mm (z 9.1-76.7), 1,782 mm2 in total. Each is shrunk by a 1.5 mm guard on both x sides (against line-end spill) and kept >= 2 mm from any mesh edge and below z 81 mm (taper). A region whose guarded core is < 5 mm wide is dropped. Mapped through the saved official-to-HF canvas map, not a fixed shift. |
| Why blank | Scribal layout (edition: 22 columns; fitted period), not the audited models. **Independence caveat:** the gaps were located with the org's ink_canonical_2um render. That model, and any pred identifiable as trained on the same data, is reported in a separate row. |
| Models | Published non-reverse preds on w018 (11) and w023 (8), as in v0-M / D. merged_* count as one model. |
| Measures | Per model: placebo FP area at p > 0.5; letter-blob density (1-4 mm2 components) per cm2; the same model's FP on in-mask unlabeled text-block pixels (reference only, not a PASS input). |
| Tripwire (text rule b) | The recalibrated flag from Amendment D (1 mm tile-shuffle, p <= 0.01 at 2.5-8 mm; blob rows vs the shuffled 99th pct). Any model firing inside a placebo region = HARD STOP, scalars only, Colin decides. That is the hallucination signature we are looking for, or unrecognized writing. |
| PASS | >= 1 model pair with placebo FP ratio >= 1.5 and 95% CI excluding 1 (paired bootstrap over 5 mm tile blocks, B = 2000, seed 20260925), AND a stable ranking across w018 vs w023 (Spearman >= 0.5 on shared models). |
| KILL | All pair CIs include 1; OR every model's placebo FP < 0.1%; OR Spearman < 0.5. INCONCLUSIVE = KILL (no retry). |
| Sanity (failure = harness bug) | Registration AUROC >= 0.75 per pred on labeled text; planted 2 mm2 blobs in one pred's placebo region recovered (ratio >= 1.3); self-pair ratio = 1.00. |
| Margins (conditional) | High-z bands (z < 81 mm, above each column's text end + 2 mm) join as a separate row ONLY if the org confirms the column foot is at high z. |
| Cost | $0. One pred at a time, placebo and reference crops only, raw deleted. No images. |
