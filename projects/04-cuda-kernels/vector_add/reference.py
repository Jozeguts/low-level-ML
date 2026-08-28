from __future__ import annotations

import numpy as np


def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Trusted CPU reference for the CUDA implementation."""
    if a.shape != b.shape:
        raise ValueError("inputs must have identical shapes")
    if a.dtype != np.float32 or b.dtype != np.float32:
        raise TypeError("inputs must use float32")
    return a + b
