from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Storage:
    """Owns the raw NumPy buffer used by a tensor."""

    data: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise TypeError("Storage requires a NumPy array")
        if self.data.ndim != 1:
            raise ValueError("Storage must be one-dimensional")

    @property
    def size(self) -> int:
        return self.data.size


@dataclass(frozen=True)
class TensorLayout:
    """Describes how a tensor maps indices onto storage."""

    shape: tuple[int, ...]
    strides: tuple[int, ...]
    storage_offset: int = 0

    def __post_init__(self) -> None:
        if len(self.shape) != len(self.strides):
            raise ValueError("shape and strides must have equal rank")
        if any(d < 0 for d in self.shape):
            raise ValueError("shape dimensions must be non-negative")
        if self.storage_offset < 0:
            raise ValueError("storage_offset must be non-negative")

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) if self.shape else 1

    def is_contiguous(self) -> bool:
        expected = 1
        for size, stride in zip(reversed(self.shape), reversed(self.strides)):
            if size != 1 and stride != expected:
                return False
            expected *= size
        return True

    def offset(self, index: Iterable[int]) -> int:
        index = tuple(index)
        if len(index) != self.ndim:
            raise IndexError("index rank does not match tensor rank")
        result = self.storage_offset
        for i, size, stride in zip(index, self.shape, self.strides):
            if not 0 <= i < size:
                raise IndexError("tensor index out of bounds")
            result += i * stride
        return result
