"""Python entry point for the native lowlevel_ops extension."""

from importlib import import_module


_C = import_module("lowlevel_ops._C")

vector_add = _C.vector_add

__all__ = ["vector_add"]
