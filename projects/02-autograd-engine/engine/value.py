from __future__ import annotations

import math
from typing import Callable, Iterable


class Value:
    """Scalar value participating in a reverse-mode computation graph."""

    def __init__(
        self,
        data: float,
        _children: Iterable["Value"] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._op = _op
        self.label = label
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: float | "Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    def __radd__(self, other: float | "Value") -> "Value":
        return self + other

    def __neg__(self) -> "Value":
        return self * -1.0

    def __sub__(self, other: float | "Value") -> "Value":
        return self + (-other if isinstance(other, Value) else -other)

    def __rsub__(self, other: float | "Value") -> "Value":
        return other + (-self)

    def __mul__(self, other: float | "Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    def __rmul__(self, other: float | "Value") -> "Value":
        return self * other

    def __pow__(self, exponent: float) -> "Value":
        if not isinstance(exponent, (int, float)):
            raise TypeError("only scalar exponents are supported")
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def backward() -> None:
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = backward
        return out

    def exp(self) -> "Value":
        out_data = math.exp(self.data)
        out = Value(out_data, (self,), "exp")

        def backward() -> None:
            self.grad += out_data * out.grad

        out._backward = backward
        return out

    def log(self) -> "Value":
        if self.data <= 0:
            raise ValueError("log requires a positive input")
        out = Value(math.log(self.data), (self,), "log")

        def backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = backward
        return out

    def relu(self) -> "Value":
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def backward() -> None:
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = backward
        return out

    def zero_grad(self) -> None:
        self.grad = 0.0

    def backward(self) -> None:
        """Run reverse-mode differentiation from this scalar output."""
        topo: list[Value] = []
        visited: set[int] = set()

        def build(node: Value) -> None:
            identity = id(node)
            if identity in visited:
                return
            visited.add(identity)
            for parent in node._prev:
                build(parent)
            topo.append(node)

        build(self)

        for node in topo:
            node.grad = 0.0
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    def graph_nodes(self) -> list["Value"]:
        """Return nodes in deterministic topological order."""
        topo: list[Value] = []
        visited: set[int] = set()

        def build(node: Value) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._prev:
                build(parent)
            topo.append(node)

        build(self)
        return topo
