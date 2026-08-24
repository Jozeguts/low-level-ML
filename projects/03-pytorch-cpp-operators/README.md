# Project 03: PyTorch C++ Operators

## Purpose

Cross the boundary from Python-level model code into native framework code. Build custom C++ operators, expose them to PyTorch, test correctness, inspect dispatch behavior, and measure the cost of crossing language and framework boundaries.

## Why this matters

Production ML systems spend substantial time outside Python. Native operators provide control over memory access, CPU parallelism, data types, and device-specific execution. This project connects high-level PyTorch APIs to the lower-level operator layer used by real workloads.

## Day 3 objective

Establish a reproducible native-extension foundation and implement the first CPU operator: vector addition.

Today's work deliberately stops before CUDA. The goal is to understand the Python-to-C++ boundary before introducing GPU execution.

## Workflow

1. Define a Python reference implementation.
2. Define the operator contract and supported tensor layouts.
3. Implement the operation in C++.
4. Register the operator with PyTorch's operator system.
5. Expose the operation to Python.
6. Add input validation and useful errors.
7. Compare outputs with PyTorch.
8. Add benchmarks.
9. Inspect tensor metadata and dispatch behavior.
10. Document compiler, Python, and PyTorch requirements.

## First operator: vector addition

For tensors `a` and `b` with the same supported shape, compute:

`out[i] = a[i] + b[i]`

The first implementation targets contiguous CPU tensors. This restriction is intentional. It keeps the first native kernel small enough to inspect while making memory access explicit.

## Architecture

```text
Python user code
      |
      v
Python wrapper
      |
      v
PyTorch operator registration
      |
      v
C++ implementation
      |
      v
CPU tensor memory
```

Later stages will replace the simple CPU implementation with device-aware dispatch and CUDA kernels.

## Correctness requirements

Test:

- matching shapes
- empty tensors
- floating-point values
- negative values
- large tensors
- invalid shapes
- unsupported layouts

Compare the custom result with `torch.add`.

## Measurement plan

Measure separately:

- Python reference time
- native operator time
- PyTorch native operator time
- extension loading time where relevant

Do not interpret one small benchmark as a general framework comparison. Record tensor sizes, dtype, hardware, PyTorch version, compiler, and number of repetitions.

## Investigation questions

- How does an operator registered in C++ become callable from Python?
- What metadata does the native implementation receive?
- Where does shape validation occur?
- How does dispatch distinguish CPU from other devices?
- What overhead exists before the arithmetic loop begins?
- How does contiguous layout simplify the kernel?

## Build phases

### Phase 1, CPU vector addition

Create the extension, register the operator, test it, and benchmark it.

### Phase 2, CPU reductions

Implement a reduction and investigate accumulation order and numerical behavior.

### Phase 3, matrix multiplication

Implement a baseline matrix multiplication and compare it with optimized PyTorch kernels.

### Phase 4, autograd

Add a custom backward path and verify gradients numerically.

### Phase 5, dispatch

Introduce explicit device and dtype dispatch.

### Phase 6, CUDA

Move the operator to GPU execution after the CPU execution path is understood.

### Phase 7, profiling

Use profiling to separate framework overhead, launch overhead, memory access, and arithmetic work.

## Deliverables

- C++ source
- PyTorch operator registration
- Python wrapper
- unit tests
- gradient checks
- CPU benchmarks
- CUDA implementation
- profiling results
- build instructions
- architecture notes
- limitations

## Success criteria

The custom operator must match the reference implementation, pass its tests, and include reproducible performance measurements. Later stages must explain the path from Python API to native execution rather than treating the extension mechanism as a black box.
