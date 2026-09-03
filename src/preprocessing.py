"""Validation and layout conversion for the public demo."""

from pathlib import Path

import numpy as np


def load_input(path: str | Path) -> np.ndarray:
    """Load a finite numeric 2D array and return NCHW float32 data."""
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError("input must be a numeric NumPy array")
    if array.ndim != 2:
        raise ValueError("input must be a 2D array")
    if min(array.shape) < 3:
        raise ValueError("input dimensions must be at least 3")
    if array.size > 1_000_000:
        raise ValueError("input is too large")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(converted).all():
        raise ValueError("input must contain only finite values")
    return converted[None, None, :, :]
