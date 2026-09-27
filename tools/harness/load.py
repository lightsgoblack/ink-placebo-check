"""Load the harness segment as aligned numpy arrays.

    from tools.harness.load import load_segment
    d = load_segment()               # w028, both views; load_segment("w029_20251212185248662_2um") etc.
    d["crop"]["labels"], d["crop"]["preds"]["ps48_...tif"], d["overview"]["validation_mask"], d["meta"]

Views:
  crop      full resolution, validation-mask bbox + 256 px margin. Offset (y0, x0) in d["crop"]["offset_yx"].
  overview  whole segment, 4x mean pooled (ints rounded back to dtype; so a 0/255 mask becomes a
            coverage fraction scaled to 0..255, threshold > 127 for a majority mask).

Run `python -m tools.harness.load [segment]` to print the alignment report (shapes, dtypes, value ranges).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .fetch import DEFAULT_SEGMENT, DS, Seg, mean_pool, read_2d


def _npz(path: Path) -> np.ndarray:
    with np.load(path) as z:
        return z["data"]


def load_segment(segment: str = DEFAULT_SEGMENT, views=("crop", "overview"), label_version: str = "v2") -> dict:
    """Return {"crop": {...}, "overview": {...}, "meta": {...}} of aligned arrays.

    Each view holds: labels, validation_mask, supervision_mask, max_render,
    preds (dict keyed by exact pred filename). label_version picks v2 (default) or v1
    for both labels and supervision mask.
    """
    s = Seg(segment)
    bbox = json.loads(s.bbox_json.read_text())
    c = bbox["crop"]
    v = "_v2" if label_version == "v2" else ""
    small = {"labels": f"{s.seg_id}_inklabels{v}.tif",
             "validation_mask": f"{s.seg_id}_validation_mask_v2.tif",
             "supervision_mask": f"{s.seg_id}_supervision_mask{v}.tif"}
    full = {k: read_2d(s.small / f) for k, f in small.items()}
    MAX_RENDER = s.max_render

    pred_names = sorted(p.stem for p in s.crop.glob("*.tif.npz") if p.stem != MAX_RENDER)
    out = {
        "meta": {
            "segment_meta": json.loads((s.small / "meta.json").read_text()),
            "bbox": bbox,
            "files": {**small, "max_render": MAX_RENDER},
            "manifest": json.loads(s.manifest.read_text()),
            "tifxyz_dir": str(s.small),
        }
    }
    for view in views:
        d = {}
        if view == "crop":
            for k, a in full.items():
                d[k] = a[c["y0"]:c["y1"], c["x0"]:c["x1"]]
            src = s.crop
            d["offset_yx"] = (c["y0"], c["x0"])
            d["downsample"] = 1
        elif view == "overview":
            for k, a in full.items():
                d[k] = mean_pool(a, DS)
            src = s.over
            d["offset_yx"] = (0, 0)
            d["downsample"] = DS
        else:
            raise ValueError(view)
        d["max_render"] = _npz(src / f"{MAX_RENDER}.npz")
        d["preds"] = {n: _npz(src / f"{n}.npz") for n in pred_names}
        out[view] = d
    return out


def _arrays(view: dict):
    for k in ("labels", "validation_mask", "supervision_mask", "max_render"):
        yield k, view[k]
    for k, a in view["preds"].items():
        yield k, a


def report(d: dict) -> bool:
    ok = True
    for view in ("crop", "overview"):
        if view not in d:
            continue
        v = d[view]
        shapes = {k: a.shape for k, a in _arrays(v)}
        same = len(set(shapes.values())) == 1
        ok &= same
        print(f"\n== {view}: offset {v['offset_yx']}, downsample {v['downsample']}, "
              f"{len(v['preds'])} preds, all shapes equal: {same} {set(shapes.values())}")
        for k, a in _arrays(v):
            print(f"  {str(a.dtype):8s} {str(a.shape):16s} min {a.min():>10.4g} max {a.max():>10.4g}  {k}")
        vm = v["validation_mask"] > (127 if view == "overview" else 0)
        lab_in = int(np.count_nonzero(v["labels"][vm]))
        print(f"  validation px: {int(vm.sum()):,}; label-nonzero px inside validation: {lab_in:,}")
        ok &= lab_in > 0
    return ok


if __name__ == "__main__":
    import sys
    good = report(load_segment(*sys.argv[1:2]))
    print("\nALIGNMENT OK" if good else "\nALIGNMENT PROBLEM")
    raise SystemExit(0 if good else 1)
