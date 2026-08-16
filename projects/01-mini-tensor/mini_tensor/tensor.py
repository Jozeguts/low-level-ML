from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np


ArrayLike = np.ndarray | Sequence[float] | float | int


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduce a broadcast gradient back to the original operand shape."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


@dataclass(eq=False)
class Tensor:
    data: np.ndarray
    requires_grad: bool = False
    grad: Optional[np.ndarray] = None
    _parents: tuple["Tensor", ...] = ()
    _backward: Optional[Callable[[np.ndarray], None]] = None

    def __init__(self, data: ArrayLike, requires_grad: bool = False):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._parents = ()
        self._backward = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def strides(self) -> tuple[int, ...]:
        itemsize = self.data.itemsize
        return tuple(s // itemsize for s in self.data.strides)

    def __repr__(self) -> str:
        return f"Tensor(data={self.data!r}, requires_grad={self.requires_grad})"

    def _binary(self, other: ArrayLike, op: Callable, grad_self: Callable, grad_other: Callable) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(op(self.data, other.data), self.requires_grad or other.requires_grad)
        out._parents = (self, other)

        def backward(g: np.ndarray) -> None:
            if self.requires_grad:
                local = _unbroadcast(grad_self(g, self.data, other.data), self.shape)
                self.grad = local if self.grad is None else self.grad + local
            if other.requires_grad:
                local = _unbroadcast(grad_other(g, self.data, other.data), other.shape)
                other.grad = local if other.grad is None else other.grad + local

        out._backward = backward
        return out

    def __add__(self, other: ArrayLike) -> "Tensor":
        return self._binary(other, np.add, lambda g, a, b: g, lambda g, a, b: g)

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self + other

    def __mul__(self, other: ArrayLike) -> "Tensor":
        return self._binary(other, np.multiply, lambda g, a, b: g * b, lambda g, a, b: g * a)

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self * other

    def __matmul__(self, other: "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad)
        out._parents = (self, other)

        def backward(g: np.ndarray) -> None:
            if self.requires_grad:
                local = g @ np.swapaxes(other.data, -1, -2)
                self.grad = local if self.grad is None else self.grad + local
            if other.requires_grad:
                local = np.swapaxes(self.data, -1, -2) @ g
                other.grad = local if other.grad is None else other.grad + local

        out._backward = backward
        return out

    def sum(self) -> "Tensor":
        out = Tensor(self.data.sum(), self.requires_grad)
        out._parents = (self,)

        def backward(g: np.ndarray) -> None:
            if self.requires_grad:
                local = np.ones_like(self.data) * g
                self.grad = local if self.grad is None else self.grad + local

        out._backward = backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0), self.requires_grad)
        out._parents = (self,)

        def backward(g: np.ndarray) -> None:
            if self.requires_grad:
                local = g * (self.data > 0)
                self.grad = local if self.grad is None else self.grad + local

        out._backward = backward
        return out

    def backward(self, grad: Optional[ArrayLike] = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("backward() requires requires_grad=True")
        seed = np.ones_like(self.data) if grad is None else np.asarray(grad, dtype=np.float64)
        self.grad = seed

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._parents:
                build(parent)
            topo.append(node)

        build(self)
        for node in reversed(topo):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)

    def zero_grad(self) -> None:
        self.grad = None

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(self.data.reshape(*shape), self.requires_grad)
        out._parents = (self,)

        def backward(g: np.ndarray) -> None:
            if self.requires_grad:
                local = g.reshape(self.shape)
                self.grad = local if self.grad is None else self.grad + local

        out._backward = backward
        return out

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False)
