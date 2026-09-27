# harness

Shared data harness for the kill tests. Pulls ONE ink segment
(`ink/1667/w028_20251208130119156_2um/`) from the scrollprize HF bucket
(https://huggingface.co/buckets/scrollprize/datasets) and serves aligned numpy arrays.

## Run

```bash
.venv/bin/python -m tools.harness.fetch   # idempotent: pull, crop, downsample, delete raw
.venv/bin/python -m tools.harness.load    # alignment report: shapes, dtypes, value ranges
```

```python
from tools.harness.load import load_segment
d = load_segment()          # d["crop"], d["overview"], d["meta"]
```

## What it does

| Step | Detail |
|------|--------|
| Download | `huggingface_hub.download_bucket_files` (hub >= 2.0), one explicit file path per call, anonymous. Never syncs a folder; never touches the `.zarr` folders. Size checked against `list_bucket_tree`. |
| Small files | labels v1 + v2, supervision mask v1 + v2, validation_mask_v2, x/y/z.tif (tifxyz), meta.json. Kept whole in `data/1667_w028/small/`. |
| Big files | max render + 12 preds. One at a time into `data/raw/`, then (a) full-res crop of the validation-mask bbox + 256 px margin and (b) 4x mean-pooled whole segment, both `np.savez_compressed` (key `data`, dtype preserved) with a sidecar json (crop offset, downsample factor, source shape/dtype). Raw deleted before the next pull. |
| Disk guard | Refuses any download that would leave < 4 GiB free. |
| Manifest | `data/1667_w028/manifest.json`: file, bytes, sha256 (computed before deletion). |

`crop` view: full resolution, offset `(y0, x0)` in segment pixels. `overview` view: 4x mean pool,
trailing partial blocks dropped, integers rounded back to dtype (a 0/255 mask becomes coverage
0..255; threshold > 127 for a majority mask). tifxyz stays at its native 1/20 grid in `small/`.

## Rules

- `data/` is gitignored. Data and derivatives are CC BY-NC 4.0 (Vesuvius Challenge).
- PHerc.1667 is a read scroll: do not render or publish images of preds, labels or renders
  from this harness.

License: MIT (repo root).
