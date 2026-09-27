"""Shared helpers: segment table, strip-wise TIFF access (never loads a full-res image whole),
tifxyz-grid <-> full-res mapping, tile grid."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import tifffile
import zarr

from tools.harness.fetch import REPO, Seg

SEGMENTS = {
    "w023": "w023_20240304161941_2um",
    "w018": "w018_20240304144031_2um",
    "w029": "w029_20251212185248662_2um",
    "w031": "w031_2025122323_2um",
}
TILE_PX = 2084            # 5 mm at 2.399 um (Amendment C: "5 mm tiles (2084 px, full res)")
STRIP = 2084              # row strip height for streaming reads (= one tile row)
BAND_JSON = REPO / "data" / "margin_bands.json"
BAND_JSON_VAULT = REPO / "vault" / "results" / "margin_bands.json"


def seg(short: str) -> Seg:
    return Seg(SEGMENTS[short])


def voxel_um(meta: dict) -> float:
    m = re.search(r"_(\d+(?:\.\d+)?)um_", meta["volume"])
    return float(m.group(1))


def meta(short: str) -> dict:
    return json.loads((seg(short).small / "meta.json").read_text())


class Lazy2D:
    """Row-range reads from a (possibly huge) 2D TIFF via tifffile's zarr store."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.tf = tifffile.TiffFile(self.path)
        self.store = self.tf.aszarr()
        self.z = zarr.open(self.store, mode="r")
        if isinstance(self.z, zarr.Group):          # multi-level: take level 0
            self.z = self.z["0"]
        self.shape = tuple(self.z.shape[-2:])
        self.dtype = self.z.dtype
        p = self.tf.pages[0]
        self.info = {"shape": list(p.shape), "dtype": str(p.dtype), "compression": p.compression.name,
                     "tiled": bool(p.is_tiled), "tile": list(p.tile) if p.is_tiled else None,
                     "rowsperstrip": None if p.is_tiled else int(p.rowsperstrip), "pages": len(self.tf.pages)}

    def rows(self, r0: int, r1: int, c0: int = 0, c1: int | None = None) -> np.ndarray:
        c1 = self.shape[1] if c1 is None else c1
        a = self.z[..., r0:r1, c0:c1]
        return np.squeeze(np.asarray(a)).reshape(r1 - r0, c1 - c0)

    def close(self):
        try:
            self.store.close()
        finally:
            self.tf.close()


def grid_factors(m: dict, full_shape, grid_shape) -> tuple[float, float]:
    """(sy, sx): grid index = round(full_px * s). meta scale is [sx, sy] (checked against shapes)."""
    sx, sy = float(m["scale"][0]), float(m["scale"][1])
    # sanity: shapes agree to within 1 grid cell
    assert abs(full_shape[0] * sy - grid_shape[0]) < 2 and abs(full_shape[1] * sx - grid_shape[1]) < 2, \
        (full_shape, grid_shape, sx, sy)
    return sy, sx


def full_to_grid(n_full: int, s: float, n_grid: int) -> np.ndarray:
    """Grid index for every full-res index (nearest grid point), clipped."""
    return np.clip(np.rint(np.arange(n_full) * s).astype(np.int64), 0, n_grid - 1)


def grid_boundaries(n_full: int, s: float, n_grid: int) -> np.ndarray:
    """Start full-res index of each grid cell (for reduceat), from full_to_grid."""
    g = full_to_grid(n_full, s, n_grid)
    starts = np.searchsorted(g, np.arange(n_grid), side="left")
    return starts


def pool_count_to_grid(lz: Lazy2D, sy: float, sx: float, grid_shape, pred=lambda a: a > 0) -> np.ndarray:
    """Exact count of full-res pixels with pred(value) per grid cell (nearest-grid-point cells)."""
    H, W = lz.shape
    GH, GW = grid_shape
    gy = full_to_grid(H, sy, GH)
    cstart = grid_boundaries(W, sx, GW)
    valid_c = np.r_[np.diff(cstart) > 0, cstart[-1] < W]
    out = np.zeros(grid_shape, np.int64)
    for r0 in range(0, H, STRIP):
        r1 = min(H, r0 + STRIP)
        a = pred(lz.rows(r0, r1))
        u, st = np.unique(gy[r0:r1], return_index=True)
        rowsum = np.add.reduceat(a, st, axis=0, dtype=np.int32)       # (n grid rows, W), contiguous
        del a
        colsum = np.add.reduceat(rowsum, np.minimum(cstart, W - 1), axis=1)
        colsum[:, ~valid_c] = 0
        out[u] += colsum
    return out


def tile_grid(shape, t: int = TILE_PX) -> tuple[int, int]:
    """Number of full tiles (anchored at (0, 0); trailing partial tiles dropped, as stage 1)."""
    return shape[0] // t, shape[1] // t
