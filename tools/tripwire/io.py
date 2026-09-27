"""Array loading for the tripwire CLI: .tif/.tiff, .npy, .png. No image is ever written from here.

In plain English: this just reads a picture or array file into numbers for the tripwire to look at.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SUFFIXES = (".tif", ".tiff", ".npy", ".png")


def load_array(path: str | Path) -> np.ndarray:
    """Load a 2D array from .tif/.tiff, .npy or .png. Multi-channel images are collapsed with max() over channels
    (any channel lit counts), since these inputs are prediction/mask maps, not photographs."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".tif", ".tiff"):
        import tifffile
        a = np.squeeze(tifffile.imread(p))
    elif suf == ".npy":
        a = np.load(p)
    elif suf == ".png":
        from PIL import Image
        a = np.array(Image.open(p))
    else:
        raise ValueError(f"unsupported file type {suf!r} (want one of {SUFFIXES})")
    a = np.asarray(a)
    if a.ndim == 3:
        a = a.max(axis=-1)
    if a.ndim != 2:
        raise ValueError(f"{p}: expected a 2D array after squeezing/channel-collapse, got shape {a.shape}")
    return a


def to_prob(a: np.ndarray) -> np.ndarray:
    """Normalize a loaded prediction array to float probabilities in [0, 1]."""
    a = np.asarray(a)
    if a.dtype == bool:
        return a.astype(np.float64)
    a = a.astype(np.float64)
    if a.max() > 1.0:
        # 8/16-bit image encodings: scale by the dtype's natural max, not the observed max, so a mostly-zero
        # prediction map does not get rescaled into a false ink signal.
        a = a / 255.0 if a.max() <= 255.0 else a / 65535.0
    return np.clip(a, 0.0, 1.0)


def to_bool_mask(a: np.ndarray) -> np.ndarray:
    """Normalize a loaded region-mask array to bool (nonzero = inside the region)."""
    return np.asarray(a) != 0
