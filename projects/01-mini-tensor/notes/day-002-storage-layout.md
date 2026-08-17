# Day 002: Storage, Strides, Views, and Layout

## Objective

Make tensor memory layout explicit.

The implementation now separates two ideas that are often hidden by a high-level tensor API:

- Storage: the underlying one-dimensional allocation.
- Layout: shape, strides, and storage offset describing how a tensor interprets that allocation.

## Core model

For an index `(i0, i1, ..., in)`, the storage position is:

`offset = storage_offset + sum(i_k * stride_k)`

Strides are measured in elements rather than bytes. This keeps the model close to the indexing rules used by tensor libraries while avoiding architecture-specific byte calculations at the Python layer.

For a contiguous `(2, 3)` tensor, the strides are `(3, 1)`.

After transposition, the shape becomes `(3, 2)` and the strides become `(1, 3)`. The data order changes from the user's perspective, but the underlying allocation remains shared.

## Why views matter

A view changes interpretation without allocating another numerical buffer. Slices and transposes therefore expose the difference between logical tensor structure and physical storage.

This matters for performance. A non-contiguous tensor often produces less convenient memory access patterns for a numerical kernel. Production frameworks frequently need to decide whether to operate directly on a strided layout or materialize a contiguous copy.

## Implementation increment

Project 01 now exposes:

- `Tensor.storage`
- `Tensor.layout`
- `Tensor.storage_offset`
- `Tensor.strides`
- `Tensor.is_contiguous`
- indexed tensor views
- transpose views
- view-aware backward propagation

The layout model is implemented in `mini_tensor/storage.py`.

## Verification

Tests verify:

1. contiguous stride calculation
2. multidimensional offset calculation
3. transpose stride changes
4. storage sharing
5. slice storage offsets
6. gradient propagation through slices
7. gradient propagation through transpose
8. storage buffer validation

## Engineering observations

Shape alone does not describe how tensor data is arranged in memory. Two tensors with the same shape may have different strides and therefore different access patterns.

A view also does not require ownership of a new data buffer. It needs metadata describing how to interpret an existing allocation.

This distinction becomes important in later CUDA work, where memory coalescing depends directly on how threads traverse physical memory.

## Next Project 01 increment

The next development increment will extend indexing and reshape/view semantics, then add stronger aliasing and contiguity tests before moving to the operator system.
