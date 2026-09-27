"""Lie Detector v0-E resumed under Amendment E1 (prereg/lie_detector_v0e_e1.md, P5; base rule P2 =
prereg/lie_detector_v0e.md). Scalars only; never writes an image.

    .venv/bin/python -m tools.margin.v0e_e1 reduce --seg w018     # one pred at a time, resumable
    .venv/bin/python -m tools.margin.v0e_e1 reduce --seg w023
    .venv/bin/python -m tools.margin.v0e_e1 analyze               # ratios, bootstrap CIs, Spearman, sanity, verdict

Per pred, in the pre-registered order (regions file order, w018 then w023; 1667_2um_pred.tif is not downloaded):
  1. registration pass: reads ONLY the evaluated-label window (val_v2 | sup_v2 | sup_v1 bbox); pixel AUROC as in P2.
     AUROC < 0.75 -> excluded, registration numbers kept, no placebo pixel read. 3+ failures among the 18 -> pause.
  2. placebo pass (registered preds only): the guarded region crop -> p > 0.5 map, frozen measures (FP area, 5 mm tile
     FP counts, letter blobs, reference FP from pass 1), then the E1 tripwire (1,000 component-shuffle surrogates,
     tools/margin/tripwire_e1.py) with the frozen cross-check. Any fire -> HARD STOP (exit 3): scalars only, the
     saved p > 0.5 crops are deleted.
  3. the planted-blob sanity copy (P2) on the first used pred with FP > 0; synthetic, never tripwired.
After a segment's preds: the E1 consensus tripwire (part (a) on the mean map of the registered preds).
Per-pred results are written as they finish, so a restart resumes at the next pred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from tools.harness.fetch import RAW, download, free_bytes, list_remote, sha256
from tools.margin import tripwire_e1 as E
from tools.margin import v0e as V
from tools.margin.common import REPO, seg
from tools.margin.deep_ld import letter_S
from tools.margin.e1_validation import region_mask

OUT = V.OUT / "e1"
STOP_FILE = V.OUT / "HARD_STOP_E1.json"
PAUSE_FILE = V.OUT / "PAUSE_E1_registration.json"
STOPPED_1667 = "preds/1667_2um_pred.tif"
MIN_MEM_GB, MIN_DISK_GB = 4.0, 8.0


def mem_avail_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def wait_resources(need_bytes: int):
    """MemAvailable >= 4 GB and free disk >= 8 GB (+ the file) before a download; bounded poll (590 s)."""
    t0 = time.time()
    while True:
        m, d = mem_avail_gb(), free_bytes(REPO) / 2**30
        if m >= MIN_MEM_GB and d - need_bytes / 2**30 >= MIN_DISK_GB:
            return round(m, 1), round(d, 1)
        if time.time() - t0 > 590:
            raise SystemExit(f"resources not available after 590 s: MemAvailable {m:.1f} GB, disk {d:.1f} GB")
        time.sleep(20)


def check_prereg():
    got = hashlib.sha256((REPO / E.PREREG).read_bytes()).hexdigest()
    if got != E.PREREG_SHA256:
        raise SystemExit(f"E1 frozen text hash mismatch: {got}")
    got2 = hashlib.sha256(V.PREREG.read_bytes()).hexdigest()
    if got2 != V.PREREG_SHA:
        raise SystemExit(f"P2 frozen text hash mismatch: {got2}")


def model_index(R: dict, rel: str) -> int:
    return [m["rel"] for m in R["models"]].index(rel)


def rec_path(short, rel):
    return OUT / short / f"{Path(rel).name}.json"


def hi_path(short, rel):
    return OUT / short / f"{Path(rel).name}.hi.npy"


def save_hi(short, rel, hi):
    np.save(hi_path(short, rel), np.packbits(hi, axis=1))


def load_hi(short, rel, shape):
    return np.unpackbits(np.load(hi_path(short, rel)), axis=1, count=shape[1]).view(bool)


def delete_saved_maps() -> int:
    n = 0
    for f in OUT.glob("*/*.hi.npy"):
        f.unlink()
        n += 1
    return n


def registration_failures() -> list:
    out = []
    for s in V.SEGS:
        for f in sorted((OUT / s).glob("*.json")) if (OUT / s).exists() else []:
            r = json.loads(f.read_text())
            if r.get("pred") and r["pred"] != STOPPED_1667 and r.get("registration_ok") is False:
                out.append(r["pred"])
    return out


# ------------------------------------------------------------------------------------------------ the two passes
def registration_pass(P: V.Pred, dt, cm: dict, EVp, POp) -> dict:
    """Pass 1: only the evaluated-label window is read (as in v0e.reduce_one's histogram part)."""
    thr, nb, qf = V.value_scale(dt)
    er0, er1 = cm["ev_bbox_rows"]
    ec0, ec1 = cm["ev_bbox_cols"]
    nbits = ec1 - ec0
    h_pos = np.zeros(nb, np.int64)
    h_neg = np.zeros(nb, np.int64)
    vmin, vmax = np.inf, -np.inf
    for r0 in range((er0 // V.SUB) * V.SUB, er1, V.SUB):
        o0, o1 = max(r0, er0), min(er1, r0 + V.SUB)
        av = P.rows(o0, o1, ec0, ec1)
        vmin, vmax = min(vmin, float(av.min())), max(vmax, float(av.max()))
        ev = np.unpackbits(EVp[o0 - er0:o1 - er0], axis=1, count=nbits).view(bool)
        pos = np.unpackbits(POp[o0 - er0:o1 - er0], axis=1, count=nbits).view(bool)
        for k0 in range(0, o1 - o0, 128):
            qa = av[k0:k0 + 128]
            q = qa if dt.kind != "f" else np.clip(np.rint(qa * qf), 0, nb - 1).astype(np.int64)
            pk, ek = pos[k0:k0 + 128], ev[k0:k0 + 128]
            h_pos += np.bincount(q[pk].ravel(), minlength=nb)[:nb]
            h_neg += np.bincount(q[ek & ~pk].ravel(), minlength=nb)[:nb]
        del av, ev, pos
    cn = np.cumsum(h_neg) - h_neg
    npos, nneg = int(h_pos.sum()), int(h_neg.sum())
    auc = float((h_pos * (cn + 0.5 * h_neg)).sum() / (npos * nneg)) if npos and nneg else float("nan")
    hb = np.arange(nb) > thr * qf
    return {"dtype": str(dt), "p05_threshold_raw": thr, "value_min_ev_window": vmin, "value_max_ev_window": vmax,
            "range_ok": bool((dt == np.uint8) or dt == np.uint16 or (dt.kind == "f" and vmin >= -1e-6 and vmax <= 1 + 1e-6)),
            "registration_auroc": auc, "registration_ok": bool(auc >= 0.75),
            "recall_p05_labeled_ink": float(h_pos[hb].sum() / npos) if npos else None,
            "reference_fp_p05_inmask_unlabeled": float(h_neg[hb].sum() / nneg) if nneg else None,
            "n_pos_px": npos, "n_neg_px": nneg, "_hist": {"h_pos": h_pos, "h_neg": h_neg}}


def placebo_pass(P: V.Pred, dt, R: dict, mask: np.ndarray) -> np.ndarray:
    """Pass 2: the guarded region crop only -> bool p > 0.5 map inside the region."""
    thr, _, _ = V.value_scale(dt)
    R0, R1 = R["full_res"]["crop_rows"]
    C0, C1 = R["full_res"]["crop_cols"]
    hi = np.zeros((R1 - R0, C1 - C0), bool)
    vmin, vmax = np.inf, -np.inf
    for r0 in range(R0, R1, V.SUB):
        r1 = min(R1, r0 + V.SUB)
        a = P.rows(r0, r1, C0, C1)
        vmin, vmax = min(vmin, float(a.min())), max(vmax, float(a.max()))
        hi[r0 - R0:r1 - R0] = (a > thr) & mask[r0 - R0:r1 - R0]
        del a
    return hi, vmin, vmax


def planted_done() -> bool:
    return any("planted" in json.loads(f.read_text()) for s in V.SEGS if (OUT / s).exists()
               for f in (OUT / s).glob("*.json") if f.name != "consensus.json")


def hard_stop(record: dict):
    n = delete_saved_maps()
    record["saved_p05_crops_deleted"] = n
    record["utc"] = time.strftime("%FT%TZ", time.gmtime())
    V.jdump(record, STOP_FILE)
    print("HARD STOP (E1 tripwire fired inside a placebo region). Scalars only; saved crops deleted:", n, flush=True)
    raise SystemExit(3)


def strip_tripwire(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


# ------------------------------------------------------------------------------------------------ reduce
def do_reduce(short: str):
    check_prereg()
    for f in (STOP_FILE, PAUSE_FILE):
        if f.exists():
            raise SystemExit(f"{f.name} recorded; nothing more is computed")
    reg = json.loads(V.REG_JSON.read_text())
    R = reg["segments"][short]
    if R["official"]["dropped_core_lt_5mm"]:
        print(short, "region dropped (core < 5 mm)")
        return
    vox = R["full_res"]["px_um"]
    region_px = R["full_res"]["px"]
    px_mm2 = (vox / 1000.0) ** 2
    R0, R1 = R["full_res"]["crop_rows"]
    C0, C1 = R["full_res"]["crop_cols"]
    TY, TX = R["full_res"]["tile_grid_TY_TX"]
    H, W = R["hf"]["canvas_hw"]
    mask = region_mask(short)                     # checked against the committed crop hash
    cm, EVp, POp = V.ev_cache(short)
    s = seg(short)
    remote = list_remote(s)
    od = OUT / short
    od.mkdir(parents=True, exist_ok=True)
    print(f"{short}: {len(R['models'])} preds (pre-registered order)", flush=True)
    for idx, mdl in enumerate(R["models"]):
        rel = mdl["rel"]
        name = Path(rel).name
        rp = rec_path(short, rel)
        if rp.exists():
            print("  done", name, flush=True)
            continue
        if rel == STOPPED_1667:
            old = json.loads((V.OUT / short / f"{name}.json").read_text())
            V.jdump({"pred": rel, "model": mdl["model"], "model_index": idx, "excluded": "E1 registration rule "
                     "(AUROC < 0.75); stop closed by the authors as an alignment artifact; not downloaded again",
                     "registration_ok": False, "registration_auroc": old["registration_auroc"],
                     "recall_p05_labeled_ink": old["recall_p05_labeled_ink"],
                     "reference_fp_p05_inmask_unlabeled": old["reference_fp_p05_inmask_unlabeled"],
                     "n_pos_px": old["n_pos_px"], "n_neg_px": old["n_neg_px"], "sha256": old.get("sha256"),
                     "source": f"data/v0e/{short}/{name}.json (earlier record)"}, rp)
            print("  excluded (earlier record)", name, flush=True)
            continue
        if rel not in remote:
            V.jdump({"pred": rel, "model": mdl["model"], "model_index": idx,
                     "excluded": "missing from the bucket at download time"}, rp)
            print("  MISSING", name, flush=True)
            continue
        mem, disk = wait_resources(remote[rel].size)
        raw = RAW / name
        t0 = time.time()
        print(f"  pull {name} ({remote[rel].size / 2**30:.2f} GiB), MemAvailable {mem} GB, disk {disk} GB", flush=True)
        for wait in (2, 4, 8, 16, None):
            try:
                if not (raw.exists() and raw.stat().st_size == remote[rel].size):
                    download(remote[rel], raw)
                break
            except SystemExit:
                raise
            except Exception as e:          # noqa: BLE001  (network)
                raw.unlink(missing_ok=True)
                if wait is None:
                    raise
                print(f"   download error ({type(e).__name__}: {e}); retry in {wait} s", flush=True)
                time.sleep(wait)
        td = time.time() - t0
        digest = sha256(raw)
        md, md_text = V.tiff_meta(raw)
        P = V.Pred(raw, H, W)
        out = {"pred": rel, "model": mdl["model"], "model_index": idx, "file_info": dict(P.lz.info),
               "upsample_note": P.note, "upsample_factor": P.f, "sha256": digest, "bytes": int(remote[rel].size),
               "tiff_metadata": md, "family_keyword_hits": V.family_hits(rel, md_text), "download_s": round(td, 1)}
        if P.f == 1 and P.note:
            P.lz.close()
            raw.unlink()
            out["excluded"] = P.note
            out["registration_ok"] = None
            V.jdump(out, rp)
            print("   EXCLUDED", P.note, flush=True)
            continue
        dt = np.dtype(P.lz.dtype)
        rg = registration_pass(P, dt, cm, EVp, POp)
        hist = rg.pop("_hist")
        out.update(rg)
        arrays = dict(hist)
        if not out["registration_ok"]:
            P.lz.close()
            raw.unlink()
            out["excluded"] = f"registration AUROC {out['registration_auroc']:.3f} < 0.75 (E1: before any placebo pixel)"
            out["runtime_s"] = round(time.time() - t0, 1)
            out["free_disk_gb_after_delete"] = round(free_bytes(REPO) / 2**30, 1)
            np.savez_compressed(od / f"{name}.npz", **arrays)
            V.jdump(out, rp)
            print(f"   AUROC {out['registration_auroc']:.3f} -> EXCLUDED before the placebo pass", flush=True)
            fails = registration_failures()
            if len(fails) >= 3:
                V.jdump({"utc": time.strftime("%FT%TZ", time.gmtime()), "failed_registration": fails,
                         "rule": "3+ of the 18 remaining preds fail registration: pause and report"}, PAUSE_FILE)
                print("PAUSE: 3+ preds failed registration", flush=True)
                raise SystemExit(4)
            continue
        hi, pmin, pmax = placebo_pass(P, dt, R, mask)
        P.lz.close()
        raw.unlink()
        out["free_disk_gb_after_delete"] = round(free_bytes(REPO) / 2**30, 1)
        out["value_min_max_placebo"] = [pmin, pmax]
        fp_px = int(hi.sum())
        tfp = V.tile_sum(hi, R0, C0, TY, TX)
        arrays["tile_fp"] = tfp
        # E1 tripwire (all 1,000 surrogates) + frozen cross-check
        t1 = time.time()
        rgn = E.Region(mask, vox)
        tw = rgn.tripwire(hi, n_surr=E.N_SURR, seed0=E.SEED0, stop_early=False)
        fc = rgn.frozen_check(hi, tw)
        rgn.release()
        del rgn
        tw = strip_tripwire(tw)
        tw["frozen_cross_check"] = fc
        tw["seconds"] = round(time.time() - t1, 1)
        out["placebo"] = {"region_px": region_px, "fp_px": fp_px, "fp_area_frac": fp_px / region_px,
                          "letter_blobs": tw["letter_blobs"],
                          "letter_blob_per_cm2": tw["letter_blobs"] / (region_px * px_mm2 / 100.0),
                          "hi_map_sha256": V.sha_mask(hi), "tripwire_e1": tw}
        if tw["fire"]:
            del hi
            out["runtime_s"] = round(time.time() - t0, 1)
            np.savez_compressed(od / f"{name}.npz", **arrays)
            V.jdump(out, rp)
            hard_stop({"segment": short, "region": R["name"], "pred": rel, "fire_a": tw["a"]["fire"],
                       "fire_b": tw["b"]["fire"], "tripwire_e1": tw, "fp_area_frac": fp_px / region_px})
        save_hi(short, rel, hi)
        # planted sanity (P2): first used pred in processing order with FP > 0; synthetic copy, never tripwired
        if not planted_done() and not out["family_keyword_hits"] and fp_px > 0:
            hp, info = V.plant(hi, mask, fp_px, region_px, vox)
            arrays["tile_fp_planted"] = V.tile_sum(hp, R0, C0, TY, TX)
            info["letter_blobs_after"] = int(letter_S(hp, vox)[1])
            info["letter_blobs_before"] = tw["letter_blobs"]
            out["planted"] = info
            del hp
        del hi
        out["runtime_s"] = round(time.time() - t0, 1)
        out["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
        np.savez_compressed(od / f"{name}.npz", **arrays)
        V.jdump(out, rp)
        a, b = tw["a"], tw["b"]
        print(f"   {out['dtype']} x{out['upsample_factor']} AUROC {out['registration_auroc']:.3f} "
              f"FP {fp_px / region_px:.4%} refFP {out['reference_fp_p05_inmask_unlabeled']:.3%} comps "
              f"{tw['n_components']} ({tw['placement']['implementation']}) letters {tw['letter_blobs']} | "
              f"T1 {a['T1']:.1f} null max {a['null_max']:.1f} p {a['p']:.4f} | S {b['S']} null max {b['null_max']} "
              f"| fire {tw['fire']} | fam {out['family_keyword_hits']} planted {'planted' in out} "
              f"(dl {td:.0f} s, tripwire {tw['seconds']} s, total {out['runtime_s']} s)", flush=True)
    consensus(short, R, mask, vox)


class LazyMaps:
    """The saved p > 0.5 crops, unpacked one at a time (keeps RSS low in the consensus)."""

    def __init__(self, short, rels, shape):
        self.short, self.rels, self.shape = short, rels, shape

    def __len__(self):
        return len(self.rels)

    def __iter__(self):
        for rel in self.rels:
            yield load_hi(self.short, rel, self.shape)


def consensus(short, R, mask, vox):
    od = OUT / short
    cf = od / "consensus.json"
    if cf.exists():
        print("  consensus done", flush=True)
        return json.loads(cf.read_text())
    recs = [json.loads(rec_path(short, m["rel"]).read_text()) for m in R["models"]]
    used = [r for r in recs if r.get("registration_ok") and "excluded" not in r]
    if not used:
        V.jdump({"n_models": 0, "note": "no registered pred"}, cf)
        return None
    rels = [r["pred"] for r in used]
    ids = [model_index(R, rel) for rel in rels]
    t0 = time.time()
    rgn = E.Region(mask, vox)
    c = rgn.consensus(LazyMaps(short, rels, mask.shape), ids, n_surr=E.N_SURR, seed0=E.SEED0)
    rgn.release()
    c = strip_tripwire(c)
    c["preds"] = rels
    c["seconds"] = round(time.time() - t0, 1)
    print(f"  consensus: {c['n_models']} maps, T1 {c['T1']:.1f} null max {c['null_max']:.1f} p {c['p']:.4f} "
          f"fire {c['fire']} ({c['seconds']} s)", flush=True)
    V.jdump(c, cf)
    if c["fire"]:
        hard_stop({"segment": short, "region": R["name"], "pred": "consensus", "consensus": c})
    return c


# ------------------------------------------------------------------------------------------------ analysis
def seg_analysis(short, R):
    od = OUT / short
    recs = [json.loads(rec_path(short, m["rel"]).read_text()) for m in R["models"]]
    excluded, family, dup, main = [], [], [], []
    merged_rep = None
    for r in recs:
        if "excluded" in r:
            excluded.append({"pred": r["pred"], "why": r["excluded"], "registration_auroc": r.get("registration_auroc")})
        elif r["family_keyword_hits"]:
            family.append(r)
        elif r["model"].startswith("merged_"):
            if merged_rep is None:
                merged_rep = r
                main.append(r)
            else:
                dup.append({"pred": r["pred"], "same_p05_map_as": merged_rep["pred"],
                            "identical": r["placebo"]["hi_map_sha256"] == merged_rep["placebo"]["hi_map_sha256"]})
        else:
            main.append(r)
    units = np.array([[u[0], u[1]] for u in R["full_res"]["tile_blocks"]])

    def unit_fp(r, key="tile_fp"):
        with np.load(od / f"{Path(r['pred']).name}.npz") as z:
            return z[key][units[:, 0], units[:, 1]].astype(float)

    fp = np.stack([unit_fp(r) for r in main]) if main else np.zeros((0, len(units)))
    names = [r["model"] for r in main]
    Wb = V.boot_weights(len(units))
    pairs = V.pair_table(fp, names, Wb) if len(main) > 1 else []
    selfp = V.self_pairs(fp, names, Wb) if main else {}
    rows = []
    for r in recs:
        base = {"model": r.get("model"), "pred": r["pred"], "registration_auroc": r.get("registration_auroc"),
                "recall_p05": r.get("recall_p05_labeled_ink"), "reference_fp": r.get("reference_fp_p05_inmask_unlabeled")}
        if "excluded" in r:
            rows.append({**base, "role": "excluded: " + r["excluded"]})
            continue
        p = r["placebo"]
        tw = p["tripwire_e1"]
        role = ("family (separate row)" if r["family_keyword_hits"] else
                "duplicate merged_*" if any(d["pred"] == r["pred"] for d in dup) else "main")
        rows.append({**base, "role": role, "upsample_factor": r["upsample_factor"],
                     "placebo_fp_area": p["fp_area_frac"], "placebo_fp_px": p["fp_px"],
                     "placebo_over_reference": (p["fp_area_frac"] / r["reference_fp_p05_inmask_unlabeled"]
                                                if r["reference_fp_p05_inmask_unlabeled"] else None),
                     "letter_blobs": p["letter_blobs"], "letter_blob_per_cm2": p["letter_blob_per_cm2"],
                     "n_components": tw["n_components"], "implementation": tw["placement"]["implementation"],
                     "unplaced_total": tw["placement"]["unplaced_total"],
                     "T1": tw["a"]["T1"], "T1_peak_mm": tw["a"]["peak_period_mm"], "T1_null_max": tw["a"]["null_max"],
                     "p_a": tw["a"]["p"], "S": tw["b"]["S"], "S_null_max": tw["b"]["null_max"], "p_b": tw["b"]["p"],
                     "fire": tw["fire"], "frozen_cross_check_ok": all(tw["frozen_cross_check"].values()),
                     "family_keyword_hits": r["family_keyword_hits"], "sha256": r["sha256"],
                     "download_s": r.get("download_s"), "runtime_s": r.get("runtime_s")})
    n_reg_fail = sum(1 for r in recs if r.get("registration_ok") is False)
    out = {"region": R["name"], "region_px": R["full_res"]["px"], "region_area_mm2": R["full_res"]["area_mm2"],
           "n_tile_blocks": len(units), "n_listed": len(R["models"]), "n_main": len(main),
           "n_registration_fail": n_reg_fail, "registration_fail_frac": n_reg_fail / len(recs),
           "excluded": excluded, "family_rows": [r["pred"] for r in family], "merged_duplicates": dup,
           "models": rows, "pairs": pairs, "self_pairs": selfp,
           "consensus": json.loads((od / "consensus.json").read_text()) if (od / "consensus.json").exists() else None}
    for r in recs:
        if "planted" in r:
            pt = V.pair_table(np.stack([unit_fp(r, "tile_fp_planted"), unit_fp(r)]), ["planted", "original"], Wb)[0]
            out["planted"] = {**r["planted"], "model": r["model"], "ratio": pt["ratio"], "ci95": pt["ci95"],
                              "recovered": bool(pt["ratio"] >= 1.3), "ci_excludes_1": pt["ci_excludes_1"]}
    return out


def analyze():
    if STOP_FILE.exists() or PAUSE_FILE.exists():
        raise SystemExit("stop/pause recorded: no analysis")
    reg = json.loads(V.REG_JSON.read_text())
    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code": "tools/margin/v0e_e1.py",
           "git_head": V.git_head(), "segments": {}}
    for s in V.SEGS:
        res["segments"][s] = seg_analysis(s, reg["segments"][s])
    m = {s: {r["model"]: r["placebo_fp_area"] for r in res["segments"][s]["models"] if r["role"] == "main"}
         for s in V.SEGS}
    common = sorted(set(m["w018"]) & set(m["w023"]))
    a, b = [m["w018"][k] for k in common], [m["w023"][k] for k in common]
    rho = float(spearmanr(a, b)[0]) if len(common) >= 3 and len(set(a)) > 1 and len(set(b)) > 1 else float("nan")
    res["spearman"] = {"models": common, "rho": rho, "n": len(common), "w018": a, "w023": b}
    allp = [p for s in V.SEGS for p in res["segments"][s]["pairs"]]
    allm = [r for s in V.SEGS for r in res["segments"][s]["models"] if r["role"] == "main"]
    planted = next((res["segments"][s]["planted"] for s in V.SEGS if "planted" in res["segments"][s]), None)
    sanity = {"registration_used_all_ge_0.75": all(r["registration_auroc"] >= 0.75 for r in allm),
              "registration_fail_le_half_each_segment": all(res["segments"][s]["registration_fail_frac"] <= 0.5
                                                            for s in V.SEGS),
              "planted_recovered": bool(planted and planted["recovered"]),
              "self_pair_exact_all": all(v["exact_1"] for s in V.SEGS for v in res["segments"][s]["self_pairs"].values()),
              "frozen_cross_check_all": all(r.get("frozen_cross_check_ok", True) for s in V.SEGS
                                            for r in res["segments"][s]["models"])}
    res["sanity"] = sanity
    any_pass = any(p["pass_pair"] for p in allp)
    all_incl = all(not p["ci_excludes_1"] for p in allp)
    all_clean = all(r["placebo_fp_area"] < 0.001 for r in allm)
    kill = []
    if all_incl:
        kill.append("All pair CIs include 1")
    if all_clean:
        kill.append("every model's placebo FP < 0.1%")
    if not (rho >= 0.5):
        kill.append(f"Spearman < 0.5 (rho = {rho:.3f})" if np.isfinite(rho) else "Spearman undefined (counted as < 0.5)")
    res["counts"] = {"pairs": len(allp), "pairs_ci_excl_1": sum(p["ci_excludes_1"] for p in allp),
                     "pairs_ratio_ge_1.5": sum(p["ratio"] >= 1.5 for p in allp),
                     "pairs_pass": sum(p["pass_pair"] for p in allp)}
    if not (sanity["planted_recovered"] and sanity["self_pair_exact_all"] and
            sanity["registration_fail_le_half_each_segment"] and sanity["frozen_cross_check_all"]):
        v, why = "NO VERDICT", "sanity failure = harness bug"
    elif kill:
        v, why = "KILL", "KILL: " + "; ".join(kill)
    elif any_pass and rho >= 0.5:
        v, why = "PASS", ">= 1 model pair with placebo FP ratio >= 1.5 and 95% CI excluding 1, AND Spearman >= 0.5"
    else:
        v, why = "KILL", "INCONCLUSIVE = KILL (no retry)"
    res["verdict"], res["verdict_rule"] = v, why
    res["verdict_context"] = {"any_pass_pair": any_pass, "all_ci_include_1": all_incl,
                              "all_fp_below_0.1pct": all_clean, "spearman": rho, "kill_clauses_true": kill}
    res["margins_row"] = "NOT RUN: conditional on the org confirming the column foot at high z (not confirmed)."
    V.jdump(res, OUT / "analysis.json")
    print(json.dumps({k: res[k] for k in ("verdict", "verdict_rule", "sanity", "counts", "spearman")}, indent=1,
                     default=V.LAT._js))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["reduce", "analyze"])
    ap.add_argument("--seg", choices=V.SEGS)
    a = ap.parse_args()
    t0 = time.time()
    try:
        if a.cmd == "reduce":
            do_reduce(a.seg)
        else:
            analyze()
    finally:
        print(f"[{a.cmd}] wall {time.time() - t0:.0f} s, peak RSS "
              f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
