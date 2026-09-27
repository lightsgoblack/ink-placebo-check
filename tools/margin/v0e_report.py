"""Write results/lie_detector_v0e.md from lie_detector_v0e.json + lie_detector_v0e_regions.json (numbers only).

    .venv/bin/python -m tools.margin.v0e report
"""
from __future__ import annotations

import json

import numpy as np

from tools.margin.v0e import REG_JSON, RES_JSON, RES_MD, SEGS

FROZEN_PASS = (">= 1 model pair with placebo FP ratio >= 1.5 and 95% CI excluding 1 (paired bootstrap over 5 mm "
               "tile blocks, B = 2000, seed 20260925), AND a stable ranking across w018 vs w023 (Spearman >= 0.5 on "
               "shared models).")
FROZEN_KILL = ("All pair CIs include 1; OR every model's placebo FP < 0.1%; OR Spearman < 0.5. INCONCLUSIVE = KILL "
               "(no retry).")


def f(x, n=2, pct=False):
    if x is None:
        return "n/a"
    if isinstance(x, float) and not np.isfinite(x):
        return "inf" if x > 0 else "n/a"
    return f"{100 * x:.{n}f}%" if pct else f"{x:.{n}f}"


def short_model(m):
    return (m.replace("_0.9995", "").replace("bcesmooth0.1_dicesmooth0.1", "bcesmooth")
            .replace("_640_640_smooth_0.1", "_640").replace("_meanteacher", "_mt"))


def report(extra_deviations: list | None = None):
    res = json.loads(RES_JSON.read_text())
    if res.get("verdict") == "HARD STOP":
        return stop_report()
    reg = json.loads(REG_JSON.read_text())
    L = []
    w = L.append
    v = res["verdict"]
    sp = res["spearman"]
    ctx = res["verdict_context"]
    segs = res["segments"]
    main = {s: [m for m in segs[s]["models"] if m["role"] == "main"] for s in SEGS}
    allm = [m for s in SEGS for m in main[s]]
    fps = [m["placebo_fp_area"] for m in allm]
    allp = [p for s in SEGS for p in segs[s]["pairs"]]
    best = max(allp, key=lambda p: (p["ci95"][0], p["ratio"])) if allp else None
    n_flag = sum(m["text_like_flag"] for s in SEGS for m in segs[s]["models"])
    planted = next((segs[s]["planted"] for s in SEGS if "planted" in segs[s]), None)

    w("# Lie Detector v0-E: edition-defined intercolumn placebo, PHerc.1667 w018 + w023 (numbers only)")
    w(f"Run {res['created_utc'][:10]} by builder (with reviewer discipline). Frozen rule: `prereg/lie_detector_v0e.md` "
      f"(the frozen-criteria history v0-E block at 7538c31, SHA-256 `{res['frozen_rule']['sha256'][:16]}...`, verified before the run). "
      "Placebo regions and the analysis plan were committed and pushed before the first pred was downloaded "
      "(`lie_detector_v0e_regions.json`). Code: `tools/margin/v0e.py`. Raw numbers: `lie_detector_v0e.json`. "
      "No images were made or viewed, and no glyph was read. $0.")
    w("")
    w(f"**VERDICT: {v}.** {res['verdict_rule']}.")
    w("")
    w("## Plain-language summary")
    w("- **What was tested.** Do the published ink models differ in how much \"ink\" they report on papyrus that the "
      "scribe left blank? The blank areas are two intercolumns (gaps between text columns) fixed by the scroll's "
      "layout, not by any model. If models differ there, and the ordering repeats on a second segment, the audit "
      "has something to say. If not, the test is killed.")
    tot = reg["total_placebo_area_mm2_hf"]
    rw = [f"{s} {reg['segments'][s]['official']['core_width_mm']:.2f} mm wide, {reg['segments'][s]['full_res']['area_mm2']:.0f} mm2"
          for s in SEGS]
    w(f"- **Placebo regions after the guards.** {tot:,.0f} mm2 in total ({'; '.join(rw)}), down from 1,782 mm2 "
      "before the 1.5 mm spill guard on each side. Neither region was dropped (both cores >= 5 mm).")
    w(f"- **Sanity.** Registration: every used pred has AUROC >= 0.75 "
      f"({f(min(m['registration_auroc'] for m in allm), 3)} to {f(max(m['registration_auroc'] for m in allm), 3)}). "
      + (f"Planted 2 mm2 blobs recovered: ratio {f(planted['ratio'])} (CI {f(planted['ci95'][0])}-{f(planted['ci95'][1])}), "
         if planted else "Planted blobs: not run. ")
      + f"self-pair ratios exactly 1.00: {'yes' if res['sanity']['self_pair_exact_all'] else 'NO'}.")
    w(f"- **Placebo false-positive area.** Across the {len(allm)} used model-segment rows it ranges from "
      f"{f(min(fps), 3, True)} to {f(max(fps), 3, True)} at p > 0.5.")
    w(f"- **Tripwire (text rule).** {'It fired in ' + str(n_flag) + ' model(s): HARD STOP.' if n_flag else 'It did not fire for any model in either placebo region; the model-mean consensus did not fire either.'}")
    if best:
        w(f"- **Model differences.** {res['counts']['pairs_ci_excl_1']} of {res['counts']['pairs']} pairs have a 95% CI "
          f"excluding 1, and {res['counts']['pairs_pass']} meet ratio >= 1.5 with the CI excluding 1. The strongest pair "
          f"is {short_model(best['num'])} / {short_model(best['den'])} = {f(best['ratio'])} (CI {f(best['ci95'][0])}-{f(best['ci95'][1])}).")
    w(f"- **Ranking stability.** Spearman between the w018 and w023 placebo FP rankings of the {sp['n']} shared models: "
      f"rho = {f(sp['rho'], 3)}.")
    w(f"- **Verdict.** {v}. " + ("; ".join(ctx["kill_clauses_true"]) if ctx["kill_clauses_true"] else "") )
    w("")
    w("## Verdict against the frozen rule")
    w(f"- PASS (frozen): \"{FROZEN_PASS}\"")
    w(f"- KILL (frozen): \"{FROZEN_KILL}\"")
    w("")
    w("| Clause | Value | Holds |")
    w("|---|---|---|")
    w(f"| >= 1 pair with ratio >= 1.5 and CI excluding 1 | {res['counts']['pairs_pass']} of {res['counts']['pairs']} pairs | {'yes' if ctx['any_pass_pair'] else 'no'} |")
    w(f"| Spearman >= 0.5 on shared models | rho = {f(sp['rho'], 3)} (n = {sp['n']}) | {'yes' if (sp['rho'] is not None and np.isfinite(sp['rho']) and sp['rho'] >= 0.5) else 'no'} |")
    w(f"| KILL: all pair CIs include 1 | {res['counts']['pairs_ci_excl_1']} pairs exclude 1 | {'yes' if ctx['all_ci_include_1'] else 'no'} |")
    w(f"| KILL: every model's placebo FP < 0.1% | max {f(max(fps), 3, True)} | {'yes' if ctx['all_fp_below_0.1pct'] else 'no'} |")
    w(f"| KILL: Spearman < 0.5 | rho = {f(sp['rho'], 3)} | {'yes' if not (sp['rho'] is not None and np.isfinite(sp['rho']) and sp['rho'] >= 0.5) else 'no'} |")
    w("")
    w("## Sanity (failure = harness bug)")
    w("| Check | Result | Pass |")
    w("|---|---|---|")
    regs = [(s, m["model"], m["registration_auroc"]) for s in SEGS for m in segs[s]["models"]]
    w(f"| Registration AUROC >= 0.75 per pred (labeled text, val_v2 / sup_v2 / sup_v1) | {f(min(r[2] for r in regs), 3)} to {f(max(r[2] for r in regs), 3)} over {len(regs)} preds; excluded: {sum(len(segs[s]['excluded']) for s in SEGS)} | {'yes' if res['sanity']['registration_used_all_ge_0.75'] else 'no'} |")
    if planted:
        w(f"| Planted 2 mm2 blobs recovered (ratio >= 1.3) | {planted['model']} ({'w018' if 'planted' in segs['w018'] else 'w023'}): {planted['n_disks']} disks, {planted['planted_px']:,} px; ratio {f(planted['ratio'])} (expected {f(planted['expected_ratio'])}), CI {f(planted['ci95'][0])}-{f(planted['ci95'][1])}; letter blobs {planted['letter_blobs_before']} -> {planted['letter_blobs_after']} | {'yes' if planted['recovered'] else 'no'} |")
    w(f"| Self-pair ratio = 1.00 | every used model vs itself: ratio 1.00, CI [1, 1] | {'yes' if res['sanity']['self_pair_exact_all'] else 'no'} |")
    sy = res.get("synthetic_checks")
    if sy:
        w(f"| Synthetic harness checks (before any pred) | {', '.join(k for k in sy['checks'])} | {'all pass' if sy['all_harness_checks_pass'] else 'FAIL'} |")
    w("")
    w("## Placebo regions (fixed before any pred was read)")
    w("| Seg | Region | Official x (mm), after guard | Core width | z (mm) | HF area | Tile blocks (5 mm) | 1 mm shuffle tiles | Map offset rows / cols p5-p95 (px) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for s in SEGS:
        r = reg["segments"][s]
        cm = r["canvas_map"]
        w(f"| {s} | {r['name']} | {r['official']['x_mm'][0]}-{r['official']['x_mm'][1]} | {r['official']['core_width_mm']:.2f} mm | "
          f"{r['official']['z_mm'][0]}-{r['official']['z_mm'][1]} | {r['full_res']['area_mm2']} mm2 | {r['full_res']['n_tile_blocks']} | "
          f"{r['full_res']['n_1mm_shuffle_tiles']} | {cm['offset_px_rows_p5_p95'][0]}..{cm['offset_px_rows_p5_p95'][1]} / "
          f"{cm['offset_px_cols_p5_p95'][0]}..{cm['offset_px_cols_p5_p95'][1]} |")
    w("")
    w("- Guards: 1.5 mm on each x side (32 cells = 1.535 mm); >= 2 mm from the official and the HF mesh edge; z < 81 mm "
      "on both meshes. Only the x guard removed anything.")
    w("- The official-to-HF canvas map was rebuilt from the saved inputs and reproduced the saved HF masks bit for bit. "
      "The guarded region is that map restricted to the guarded cells (non-constant offset, never a fixed shift).")
    w("- Mask hashes (recipe in the regions file): " + "; ".join(
        f"{s} official `{reg['segments'][s]['official']['mask_sha256'][:12]}`, HF grid `{reg['segments'][s]['hf']['mask_sha256'][:12]}`, "
        f"full-res crop `{reg['segments'][s]['full_res']['crop_mask_sha256'][:12]}`" for s in SEGS) + ".")
    w("")
    for s in SEGS:
        sg = segs[s]
        w(f"## Per-model table: {s} ({sg['region']}, {sg['region_area_mm2']} mm2, {sg['n_tile_blocks']} tile blocks)")
        w("| Model | Role | AUROC | Placebo FP | Ref FP (in-mask unlabeled) | Placebo / ref | Letter blobs (per cm2) | S (null p99) | T1 (peak mm) | p tile | Flag |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for m in sorted(sg["models"], key=lambda m: -m["placebo_fp_area"]):
            w(f"| {short_model(m['model'])} | {m['role']} | {f(m['registration_auroc'], 3)} | {f(m['placebo_fp_area'], 3, True)} | "
              f"{f(m['reference_fp'], 2, True)} | {f(m['placebo_over_reference'], 3)} | {m['letter_blobs']} ({f(m['letter_blob_per_cm2'], 3)}) | "
              f"{m['S_1mm_row']} ({f(m['S_null_p99'], 1) if m['S_null_p99'] is not None else '-'}) | {f(m['T1'], 1)} ({f(m['T1_peak_mm'], 1)}) | "
              f"{f(m['p_tile_T1'], 3)} | {'FIRE' if m['text_like_flag'] else 'no'} |")
        c = sg.get("consensus")
        if c:
            w(f"- Consensus (mean of {c['n_models']} used preds): T1 {f(c['T1_ratio_pooled'], 1)} at {f(c['T1_peak_period_mm'], 1)} mm, "
              f"tile-shuffle p {f(c['p_tile_T1'], 3)} (null median {f(c['tile_null_T1_median'], 1)}): {'FIRE' if c['flag'] else 'no flag'}.")
        if sg["excluded"]:
            w("- Excluded: " + "; ".join(f"{e['pred']} ({e['why']})" for e in sg["excluded"]) + ".")
        w("")
    w("## Pair ratios (paired bootstrap over 5 mm tile blocks, B = 2000, seed 20260925)")
    w("Top pairs by CI lower bound per segment; every pair is in the JSON.")
    w("")
    w("| Seg | Pair (higher / lower) | Ratio | 95% CI | CI excludes 1 | PASS pair |")
    w("|---|---|---|---|---|---|")
    for s in SEGS:
        for p in sorted(segs[s]["pairs"], key=lambda p: -p["ci95"][0])[:6]:
            w(f"| {s} | {short_model(p['num'])} / {short_model(p['den'])} | {f(p['ratio'])} | {f(p['ci95'][0])}-{f(p['ci95'][1])} | "
              f"{'yes' if p['ci_excludes_1'] else 'no'} | {'yes' if p['pass_pair'] else 'no'} |")
    w("")
    w(f"**Spearman** (placebo FP, shared models, w018 vs w023): rho = {f(sp['rho'], 3)}, n = {sp['n']}.")
    w("")
    w("| Model | w018 placebo FP | w023 placebo FP |")
    w("|---|---|---|")
    for k, a, b in zip(sp["models"], sp["w018"], sp["w023"]):
        w(f"| {short_model(k)} | {f(a, 3, True)} | {f(b, 3, True)} |")
    w("")
    w("## Independence caveat")
    fam = [x for s in SEGS for x in segs[s]["family_rows"]]
    w("- The intercolumns were located with the org's `ink_canonical_2um` render (lattice run). Rule: a pred whose "
      "name or TIFF metadata identifies it as that family is reported in a separate row and left out of the verdict "
      "inputs. " + (f"Identified: {', '.join(fam)}." if fam else "No pred was identifiable as that family from its "
      "name or metadata (keywords: canonical, new_canon, autoresearch, hecate, resnet152)."))
    w("- `1667_2um_pred.tif` (w018 only) has unknown provenance (same '1667_2um' tag as the reverse pred, which scout "
      "rated as likely a different model). It is not identifiable as the family and stays in the main set; it is not "
      "among the Spearman models (it has no w023 twin).")
    w("")
    w("## Deviations and interpretations")
    w("| # | What | Why | Effect |")
    w("|---|---|---|---|")
    devs = [
        ("I1 (pre-registered, before any pred)", "Bootstrap units = every HF-canvas 5 mm tile holding >= 1 region px (Amendment D plan used tiles >= 50% in the band)",
         "The strips are 7-10 mm wide; a 50% rule would drop region px from the measured FP area", "Point estimate = whole-region FP ratio"),
        ("I2 (pre-registered)", "x guard = 32 cells = 1.535 mm", "Smallest whole-cell guard >= 1.5 mm on the 20 px grid", "0.035 mm extra per side"),
        ("I3 (pre-registered)", "Edge and z guards applied on both the official and the HF mesh", "'Any mesh edge'; the preds live on the HF canvas", "None removed any cell"),
        ("I4 (pre-registered)", "PASS needs no KILL clause to hold", "The frozen KILL line lists 'every model's placebo FP < 0.1%' regardless of ratios", "Only matters if both held"),
        ("I5 (pre-registered)", "Tripwire also run on preds failing registration, plus the model-mean consensus", "Stricter stop, as the Amendment D harness", "No firing"),
        ("I6 (pre-registered)", "Registration histograms read a bit cache of val_v2 | sup_v2 | sup_v1 and inklabels_v2 built from the label files", "Speed; same pixels as deep_ld", "None on values"),
    ]
    for d in devs + [tuple(x) for x in (extra_deviations or [])]:
        w("| " + " | ".join(d) + " |")
    w("")
    w("- **Margins row: NOT run.** It is conditional on the org confirming that the column foot is at high z, and that answer is not in.")
    w("")
    rs = res.get("resources") or {}
    if rs:
        w("## Resources")
        for k, val in rs.items():
            w(f"- **{k}.** {val}")
        w("")
    RES_MD.write_text("\n".join(L) + "\n")
    print("wrote", RES_MD)


def stop_report():
    """HARD STOP write-up: scalars already computed, nothing further. No images."""
    res = json.loads(RES_JSON.read_text())
    reg = json.loads(REG_JSON.read_text())
    hs = res["hard_stop"]
    seg_, pred = hs["segment"], hs["pred"]
    rec = next(m for m in res["segments"][seg_]["models"] if m["pred"] == pred)
    per, bl = hs["periodicity"], hs["blobs"]
    sy = reg["synthetic_harness_checks_before_any_pred"]
    ch = sy["tripwire_characterization"]
    cm = reg["segments"][seg_]
    n_done = sum(res["segments"][s]["n_reduced"] for s in res["segments"])
    L = []
    w = L.append
    w("# Lie Detector v0-E: HARD STOP (text rule) on the first pred, w018 intercolumn placebo (numbers only)")
    w(f"Run {res['created_utc'][:10]} by builder (with reviewer discipline). Frozen rule: `prereg/lie_detector_v0e.md` "
      f"(the frozen-criteria history v0-E block at 7538c31, SHA-256 `{res['frozen_rule']['sha256'][:16]}...`, verified). Placebo regions, "
      "harness checks and the analysis plan were committed and pushed before any pred was downloaded (48d2cc4, 7086eb2: "
      "`lie_detector_v0e_regions.json`). Code: `tools/margin/v0e.py`. Scalars: `lie_detector_v0e.json`. "
      "No images were made or viewed, and no glyph was read. $0.")
    w("")
    w("**STATUS: HARD STOP. No verdict.** The recalibrated tripwire (Amendment D flag) fired inside the w018 placebo "
      "region for the first pred processed. Per the frozen rule (\"Any model firing inside a placebo region = HARD "
      "STOP, scalars only, The authors decide\") nothing further was computed: no other pred was downloaded, and no pair "
      "ratio, Spearman or verdict exists. The authors decide.")
    w("")
    w("## Plain-language summary")
    w(f"- **What stopped.** `{pred.split('/')[-1]}` (w018 only, provenance unknown) is the first pred in the pre-registered "
      f"order. In the w018 intercolumn placebo, its p > 0.5 map has a strong row rhythm: T1 {per['T1_ratio_pooled']:.1f} against a "
      f"1 mm tile-shuffle null median of {per['tile_null_T1_median']:.1f} (p95 {per['tile_null_T1_p95']:.1f}). That gives "
      f"p = {per['p_tile_T1']:.3f}, the smallest value 200 surrogates allow, with the peak at {per['T1_peak_period_mm']:.2f} mm.")
    w(f"- **Same pred fails registration.** Its AUROC against the labels is {rec['registration_auroc']:.3f} (chance; the "
      f"threshold is 0.75), and it marks only {100 * rec['recall_p05']:.1f}% of labeled ink at p > 0.5. Its values are "
      "unrelated to this segment's labels: either the file is not on this canvas, or it is not a usable ink map here. "
      "Either way, what sits at the placebo coordinates in this file may not be the model's reading of the "
      "intercolumn. It would have been excluded from every ratio. The plan still ran the tripwire on it, "
      "because the stop rule covers every model (fixed before any pred was read).")
    w(f"- **No letter rows.** Placebo FP area is {100 * rec['placebo_fp_area']:.2f}%, the same as this pred's FP on "
      f"in-mask unlabeled text-block pixels ({100 * rec['reference_fp']:.2f}%). It has {bl['n_components']} components, "
      f"{bl['letter_blobs']} of them letter-sized (1-4 mm2), and the largest is {bl['largest_component_mm2']:.1f} mm2. At "
      f"most {bl['S_max_letter_blobs_1mm_row']} letter blob sits in any 1 mm row, so the blob-row part cannot fire. "
      f"Only the periodicity part fired. The {per['T1_peak_period_mm']:.1f} mm peak is inside the 2.5-8 mm text band but "
      "not at the 5.5-6.1 mm line pitch.")
    w("- **Known property of the frozen flag (recorded before any pred).** On synthetic 7.4 x 60 mm strips, the periodicity "
      f"part fires on non-periodic structure. A single isolated letter-sized blob gives p <= 0.01 in "
      f"{int(ch['periodicity_rate_1_letter_blob']['frac_p_le_0.01'] * 20)}/20 strips, and so do 4 blobs "
      f"({int(ch['periodicity_rate_4_letter_blobs']['frac_p_le_0.01'] * 20)}/20). Structureless pixel noise gives 0/50, "
      f"and sub-millimetre blobs give {int(ch['periodicity_rate_40_subtile_blobs_0.05-0.2mm2']['frac_p_le_0.01'] * 20)}/20. "
      "The 1 mm tile shuffle destroys any FP structure coherent over more than 1 mm, so a firing means that kind of "
      "structure is present, not specifically rows.")
    w("- **Readings the scalars can't separate (The authors decide).**")
    w("  - **Misregistration or not an ink map here.** The file's values are unrelated to this canvas's labels (AUROC "
      "0.495). If it is on another flattening or mirrored, the rhythm comes from wherever these pixels really are on "
      "the surface, possibly text columns.")
    w(f"  - **Clumpy non-text FP.** The flag responds to any structure coherent over more than 1 mm; here the largest "
      f"component is {bl['largest_component_mm2']:.1f} mm2.")
    w("  - **Model output with a row rhythm in blank papyrus.** This is the hallucination signature the test looks "
      "for, or unrecognized writing. The scalars can't rule it out.")
    w("")
    w("## Scalar evidence")
    w("| Item | Value |")
    w("|---|---|")
    w(f"| Segment / region | {seg_} / {hs['region']} (guarded core {cm['official']['core_width_mm']:.2f} mm, {cm['full_res']['area_mm2']} mm2 on the HF canvas) |")
    w(f"| Pred | `{pred}`; sha256 `{rec['sha256'][:16]}...`; uint8, same shape as the labels; Software tag `inference_ome_zarr.py`; not identifiable as the ink_canonical_2um family (no keyword in name or metadata) |")
    w(f"| Registration AUROC / recall at p > 0.5 | {rec['registration_auroc']:.3f} / {100 * rec['recall_p05']:.1f}% (FAILS >= 0.75) |")
    w(f"| Placebo FP area at p > 0.5 | {100 * rec['placebo_fp_area']:.3f}% ({rec['placebo_fp_px']:,} px) |")
    w(f"| Reference FP, in-mask unlabeled | {100 * rec['reference_fp']:.3f}% |")
    w(f"| Components / letter blobs (per cm2) / largest | {bl['n_components']} / {bl['letter_blobs']} ({rec['letter_blob_per_cm2']:.3f}) / {bl['largest_component_mm2']:.1f} mm2 |")
    w(f"| Blob rows: S (max letter blobs in a 1 mm row) | {bl['S_max_letter_blobs_1mm_row']} (< 3: cannot fire) |")
    w(f"| Row periodicity: T1 / peak period | {per['T1_ratio_pooled']:.1f} / {per['T1_peak_period_mm']:.2f} mm over a {per['length_mm']:.1f} mm profile ({per['n_rows']} rows of 0.25 mm) |")
    w(f"| 1 mm tile-shuffle null: median / p95; p | {per['tile_null_T1_median']:.1f} / {per['tile_null_T1_p95']:.1f}; p = {per['p_tile_T1']:.4f} (flag at <= 0.01) |")
    w("")
    w("## What was and was not done")
    w(f"- **Done before any pred (committed and pushed).** Placebo regions after guards: w018 ic_201 "
      f"{reg['segments']['w018']['full_res']['area_mm2']} mm2 (core {reg['segments']['w018']['official']['core_width_mm']:.2f} mm) "
      f"and w023 ic_132 {reg['segments']['w023']['full_res']['area_mm2']} mm2 (core {reg['segments']['w023']['official']['core_width_mm']:.2f} mm), "
      f"{reg['total_placebo_area_mm2_hf']:,} mm2 in total. Neither was dropped. Both were mapped through the saved canvas "
      "map, rebuilt bit for bit (non-constant offset). Mask hashes, tile blocks and the analysis plan are in the regions "
      "file.")
    w("- **Harness checks before any pred.** All synthetic checks pass: letter rows fire, the null is calibrated on "
      "pixel noise, planted blobs are recovered, a true 2x pair passes, self-pairs are exact and tile sums agree. A "
      "plumbing run with the label file as a stand-in pred gives AUROC 1.000, and 0 labeled-ink px in either placebo.")
    w(f"- **Preds.** {n_done} of 19 processed (download, reduce, raw deleted). The run stopped on the first. None of the "
      "other 10 w018 preds or 8 w023 preds was downloaded.")
    w("- **Not computed (stop).** Planted-blob sanity on a real pred, self-pair on real preds, pair ratios, bootstrap "
      "CIs, Spearman, the consensus tripwire and the verdict. The margins row is not run in any case: the column-foot "
      "answer is not in.")
    w("- **Kept.** The per-pred scalars and 5 mm tile counts are kept. This pred's 0.25 mm pooled FP map was not kept "
      "(stop semantics). No image exists.")
    w("")
    w("## Options for the authors (no action taken)")
    w("1. **Treat it as a registration artifact.** This pred fails the frozen registration gate and would never enter "
      "a ratio. Resume v0-E on the other 18 preds under the same stop rule. Note that given the flag property above, "
      "another pred whose placebo FP holds even one letter-sized blob may stop the run again.")
    w("2. **Diagnose the canvas of `1667_2um_pred.tif` first**, for example flips or another flattening against the "
      "labels. This needs a re-download and more computation on the stopped file.")
    w("3. **Keep v0-E stopped.** The reviewer reviews the tripwire's specificity (the frozen text can't change; any "
      "change would be a new, separately approved amendment).")
    w("")
    w("## Deviations")
    w("- **None during the run.** The run followed the plan committed before any pred. The interpretations fixed "
      "before any pred (listed in the regions file) are:")
    w("  - bootstrap units are all 5 mm tiles that touch the region;")
    w("  - the guard is 32 cells = 1.535 mm;")
    w("  - the edge and z guards are applied on both meshes;")
    w("  - PASS requires that no KILL clause holds;")
    w("  - the tripwire also runs on preds that fail registration, and on the consensus;")
    w("  - there is a label bit cache;")
    w("  - after a firing, the blob-row null is skipped and the pooled map is not kept.")
    w("- **The stop came from the \"tripwire also on registration failures\" item.** It was committed in 7086eb2, "
      "before this pred was downloaded.")
    w("")
    w("## Resources")
    w(f"- **Downloads.** One pred, 253 MB in {rec['download_s']:.0f} s; reduced in {rec['runtime_s']:.0f} s; raw deleted.")
    w("- **Disk.** 19-20 GB free throughout (floor 6 GB). `data/v0e/` holds a 1.3 GB label bit cache and the scalars.")
    w("- **Memory.** Peak 2.2 GB for the reduce process; the other jobs on the box used about 10 GB.")
    w("- **Cost.** CPU only, $0.")
    RES_MD.write_text("\n".join(L) + "\n")
    print("wrote", RES_MD)
