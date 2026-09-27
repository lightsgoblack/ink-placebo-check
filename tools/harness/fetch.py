"""Idempotent pull + crop + delete pipeline for one ink segment from the scrollprize HF bucket.

Disk-frugal by design: large 2D TIFFs (preds, max render) are downloaded ONE AT A TIME to
data/raw/, reduced to (a) a full-resolution crop of the validation-mask bbox plus a margin and
(b) a 4x mean-pooled full-segment overview, then the raw file is deleted before the next pull.
Small files (labels, masks, meta.json, optionally tifxyz) are kept whole.

Only single-file downloads are used (huggingface_hub.download_bucket_files with explicit paths).
The bucket is never synced; .zarr folders are never touched.

Usage:
    .venv/bin/python -m tools.harness.fetch                                   # w028, everything
    .venv/bin/python -m tools.harness.fetch --segment w029_20251212185248662_2um --masks-only
    .venv/bin/python -m tools.harness.fetch --segment w029_20251212185248662_2um --preds-only \
        --skip-reverse
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

BUCKET = "scrollprize/datasets"
SCROLL = "1667"
DEFAULT_SEGMENT = "w028_20251208130119156_2um"

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"

MARGIN = 256          # px added around the validation-mask bbox
DS = 4                # overview downsample factor (mean pool)
MIN_FREE = 4 * 2**30  # stop if a download would leave less than this free


@dataclass(frozen=True)
class Seg:
    """Paths and file names for one ink segment."""
    segment: str                    # e.g. w029_20251212185248662_2um

    @property
    def seg_id(self) -> str:        # e.g. w029_20251212185248662
        return re.sub(r"_2um$", "", self.segment)

    @property
    def short(self) -> str:         # e.g. w029
        return self.segment.split("_")[0]

    @property
    def prefix(self) -> str:
        return f"ink/{SCROLL}/{self.segment}/"

    @property
    def data(self) -> Path:
        return REPO / "data" / f"{SCROLL}_{self.short}"

    small = property(lambda s: s.data / "small")
    crop = property(lambda s: s.data / "crop")
    over = property(lambda s: s.data / "overview")
    manifest = property(lambda s: s.data / "manifest.json")
    bbox_json = property(lambda s: s.data / "bbox.json")
    listing_json = property(lambda s: s.data / "listing.json")

    @property
    def max_render(self) -> str:
        return f"{self.segment}_max_22_42.tif"

    def mask_files(self, remote: dict) -> list[str]:
        """meta.json + every labels / supervision / validation TIFF version present at the root."""
        pat = re.compile(rf"^{re.escape(self.seg_id)}_(inklabels|supervision_mask|validation_mask)[^/]*\.tif$")
        return ["meta.json"] + sorted(k for k in remote if pat.match(k))

    def val_file(self) -> Path | None:
        c = sorted(self.small.glob(f"{self.seg_id}_validation_mask*.tif"))
        return c[-1] if c else None     # highest version (v2 sorts after bare)


def is_reverse(rel: str) -> bool:
    return "reverse" in Path(rel).name


# ---------- helpers ----------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def load_manifest(s: Seg) -> dict:
    if s.manifest.exists():
        return json.loads(s.manifest.read_text())
    return {"bucket": BUCKET, "prefix": s.prefix, "files": {}}


def save_manifest(s: Seg, m: dict) -> None:
    s.manifest.write_text(json.dumps(m, indent=2, sort_keys=True))


def list_remote(s: Seg) -> dict:
    """Map relative path -> BucketFile for the segment root and preds/ (non-recursive, no zarr walk)."""
    from huggingface_hub import list_bucket_tree
    out = {}
    for p in (s.prefix, s.prefix + "preds/"):
        for f in list_bucket_tree(BUCKET, prefix=p, recursive=False, token=False):
            if type(f).__name__ == "BucketFile":
                out[f.path[len(s.prefix):]] = f
    s.data.mkdir(parents=True, exist_ok=True)
    s.listing_json.write_text(json.dumps({k: v.size for k, v in sorted(out.items())}, indent=2))
    return out


def download(bf, dest: Path) -> None:
    """Single-file download of one BucketFile to dest, verified against the listed size."""
    from huggingface_hub import download_bucket_files
    dest.parent.mkdir(parents=True, exist_ok=True)
    need = bf.size + MIN_FREE
    have = free_bytes(dest.parent)
    if have < need:
        raise SystemExit(
            f"STOP: {have / 2**30:.2f} GiB free; pulling {bf.path} ({bf.size / 2**30:.2f} GiB) "
            f"would leave < {MIN_FREE / 2**30:.0f} GiB. Nothing downloaded."
        )
    download_bucket_files(BUCKET, [(bf, str(dest))], raise_on_missing_files=True, token=False)
    got = dest.stat().st_size
    if got != bf.size:
        raise SystemExit(f"Size mismatch for {bf.path}: got {got}, listing says {bf.size}")


def read_2d(path: Path) -> np.ndarray:
    a = tifffile.imread(path)
    a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"{path.name}: expected a 2D image, got shape {a.shape}")
    return a


def mean_pool(a: np.ndarray, f: int = DS) -> np.ndarray:
    """f x f mean pool, trailing partial blocks dropped. Integer dtypes are rounded back to dtype."""
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    out = np.empty((h // f, w // f), dtype=a.dtype)
    step = 1024 * f  # row strips keep the float64 temporaries small
    for r in range(0, h, step):
        blk = a[r:min(r + step, h), :w]
        m = blk.reshape(blk.shape[0] // f, f, w // f, f).mean(axis=(1, 3), dtype=np.float64)
        if np.issubdtype(a.dtype, np.integer):
            info = np.iinfo(a.dtype)
            m = np.clip(np.rint(m), info.min, info.max)
        elif a.dtype == np.bool_:
            m = m >= 0.5
        out[r // f:(r + blk.shape[0]) // f] = m
    return out


def validation_bbox(s: Seg) -> dict:
    """Bbox (y0, y1, x0, x1; half-open) of nonzero validation mask, plus margin clipped to image."""
    vf = s.val_file()
    if vf is None:
        raise SystemExit(f"{s.segment}: no validation mask in the bucket; nothing to crop to.")
    vm = read_2d(vf)
    ys, xs = np.nonzero(vm)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    H, W = vm.shape
    c = dict(
        y0=max(0, y0 - MARGIN), y1=min(H, y1 + MARGIN),
        x0=max(0, x0 - MARGIN), x1=min(W, x1 + MARGIN),
    )
    return {
        "validation_file": vf.name,
        "segment_shape": [H, W],
        "mask_bbox": {"y0": y0, "y1": y1, "x0": x0, "x1": x1},
        "mask_bbox_hw": [y1 - y0, x1 - x0],
        "mask_bbox_frac_of_segment": (y1 - y0) * (x1 - x0) / (H * W),
        "mask_nonzero_px": int(ys.size),
        "mask_nonzero_frac_of_segment": ys.size / (H * W),
        "margin": MARGIN,
        "crop": c,
        "crop_hw": [c["y1"] - c["y0"], c["x1"] - c["x0"]],
        "downsample": DS,
    }


def reduce_and_store(s: Seg, src: Path, rel: str, bbox: dict, overview: bool = True) -> dict:
    """Write crop (+ optional overview) npz with a sidecar json for one 2D TIFF."""
    a = read_2d(src)
    exp = tuple(bbox["segment_shape"])
    c = bbox["crop"]
    name = Path(rel).name
    crop = a[c["y0"]:c["y1"], c["x0"]:c["x1"]]
    s.crop.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(s.crop / f"{name}.npz", data=crop)
    side = {
        "source": s.prefix + rel,
        "source_shape": list(a.shape),
        "source_dtype": str(a.dtype),
        "shape_matches_labels": tuple(a.shape) == exp,
        "crop_offset_yx": [c["y0"], c["x0"]],
        "crop_shape": list(crop.shape),
        "min": float(a.min()), "max": float(a.max()),
    }
    (s.crop / f"{name}.json").write_text(json.dumps(side, indent=2))
    if overview:
        over = mean_pool(a, DS)
        s.over.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(s.over / f"{name}.npz", data=over)
        side = {**side, "downsample_factor": DS, "overview_shape": list(over.shape),
                "downsample_method": "mean pool, trailing partial blocks dropped, ints rounded back to dtype"}
        (s.over / f"{name}.json").write_text(json.dumps(side, indent=2))
    return side


# ---------- pipeline ----------

def fetch_small(s: Seg, remote: dict, man: dict, files: list[str]) -> None:
    for rel in files:
        dest = s.small / rel
        if rel in man["files"] and dest.exists() and dest.stat().st_size == man["files"][rel]["bytes"]:
            continue
        if rel not in remote:
            print(f"  missing in bucket: {rel}")
            continue
        print(f"  pull {rel} ({remote[rel].size:,} B)")
        download(remote[rel], dest)
        man["files"][rel] = {"bytes": dest.stat().st_size, "sha256": sha256(dest), "kept": "whole"}
        save_manifest(s, man)


def fetch_big(s: Seg, remote: dict, man: dict, bbox: dict, rels: list[str], overview: bool) -> None:
    for rel in rels:
        name = Path(rel).name
        done = (s.crop / f"{name}.npz").exists() and (not overview or (s.over / f"{name}.npz").exists())
        if rel in man["files"] and done:
            print(f"  skip {rel} (already reduced)")
            continue
        raw = RAW / name
        print(f"  pull {rel} ({remote[rel].size / 2**20:.0f} MiB), free {free_bytes(RAW.parent) / 2**30:.2f} GiB",
              flush=True)
        if not (raw.exists() and raw.stat().st_size == remote[rel].size):
            download(remote[rel], raw)
        digest = sha256(raw)
        side = reduce_and_store(s, raw, rel, bbox, overview=overview)
        raw.unlink()
        man["files"][rel] = {
            "bytes": remote[rel].size, "sha256": digest,
            "kept": "crop" + (" + 4x overview" if overview else "") + " (raw deleted)",
            "source_shape": side["source_shape"], "source_dtype": side["source_dtype"],
        }
        save_manifest(s, man)
        print(f"    -> {side['source_dtype']} {side['source_shape']}, raw deleted", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segment", default=DEFAULT_SEGMENT, help="segment folder under ink/1667/")
    ap.add_argument("--masks-only", action="store_true", help="only meta.json + labels/masks (all versions)")
    ap.add_argument("--small-only", action="store_true", help="masks + tifxyz x/y/z, no big files")
    ap.add_argument("--preds-only", action="store_true", help="big files: preds only (no max render)")
    ap.add_argument("--skip-reverse", action="store_true", help="never download the reverse pred")
    ap.add_argument("--no-overview", action="store_true", help="crop only, no 4x overview")
    args = ap.parse_args(argv)

    s = Seg(args.segment)
    for d in (s.small, RAW):
        d.mkdir(parents=True, exist_ok=True)
    man = load_manifest(s)
    remote = list_remote(s)
    print(f"{len(remote)} files listed under {s.prefix} (+ preds/)")

    small = s.mask_files(remote)
    if not args.masks_only:
        small += ["x.tif", "y.tif", "z.tif"]
    fetch_small(s, remote, man, small)
    if args.masks_only or args.small_only:
        if s.val_file() is not None:
            s.bbox_json.write_text(json.dumps(validation_bbox(s), indent=2))
        return
    bbox = validation_bbox(s)
    s.bbox_json.write_text(json.dumps(bbox, indent=2))
    print("validation bbox:", json.dumps(bbox))
    big = sorted(k for k in remote if k.startswith("preds/"))
    if args.skip_reverse:
        big = [k for k in big if not is_reverse(k)]
    if not args.preds_only:
        big = [s.max_render] + big
    fetch_big(s, remote, man, bbox, big, overview=not args.no_overview)
    total = sum(v["bytes"] for v in man["files"].values())
    print(f"done: {len(man['files'])} files, {total / 1e9:.2f} GB transferred in total")


if __name__ == "__main__":
    sys.exit(main())
