# Project 03: PyTorch C++ Operators

## Purpose

Cross the boundary from Python-level model code into native framework code. Build custom C++ operators, expose them to PyTorch, test their correctness, inspect dispatch behavior, and measure the cost of crossing language and framework boundaries.

## Why this matters

Production ML systems spend substantial time outside Python. Native operators provide control over memory access, CPU parallelism, data types, and device-specific execution. This project connects high-level PyTorch APIs to the lower-level operator layer used by real workloads.

## Workflow

1. Start with a Python reference implementation.
2. Define the operator contract and tensor shape rules.
3. Implement the operator in C++.
4. Register it with PyTorch.
5. Add CPU tests and error handling.
6. Add autograd support where appropriate.
7. Compare outputs with the reference implementation.
8. Benchmark Python, native C++, and PyTorch equivalents.
9. Inspect dispatch and memory behavior.
10. Extend the operator toward CUDA.

## Build phases

### Phase 1

Create a C++ extension for vector addition and reduction.

### Phase 2

Implement matrix multiplication and normalization with explicit shape validation.

### Phase 3

Add custom backward implementations and test gradient correctness.

### Phase 4

Add CUDA implementations and device dispatch.

### Phase 5

Package the extension and document ABI, compiler, CUDA, and PyTorch version requirements.

## Deliverables

- C++ source
- PyTorch operator registration
- Python wrapper
- Unit tests
- Gradient checks
- CPU benchmarks
- CUDA implementation
- Profiling results
- Build instructions
- Architecture notes

## Questions this project should answer

- What happens after a Python operator call?
- How does PyTorch select an implementation for a device and dtype?
- What work belongs in C++ versus Python?
- How does a custom operator participate in autograd?
- Where do extension overheads appear?

## Success criteria

The custom operator must match the reference implementation, pass gradient tests, run on the intended devices, and include measured performance rather than qualitative claims.
