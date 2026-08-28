# Project 04: CUDA Kernel Laboratory

## Purpose

Move from framework-level native CPU operators into direct GPU programming.

The project builds a progression of CUDA kernels from simple elementwise operations toward reductions, matrix multiplication, normalization, and fused operations. Each kernel is accompanied by a CPU or PyTorch reference, correctness tests, benchmarks, and performance analysis.

## Why this matters

GPU execution is fundamental to modern ML systems. Understanding CUDA exposes the mechanisms hidden behind high-level tensor operations:

- grids and blocks
- threads and warps
- global, shared, constant, and register memory
- memory coalescing
- synchronization
- occupancy
- launch configuration
- arithmetic intensity
- kernel fusion

These concepts prepare the later GPU optimization and LLM inference projects.

## Workflow

1. Define a mathematical operation.
2. Build a trusted PyTorch or CPU reference.
3. Define tensor shapes, dtypes, and memory-layout assumptions.
4. Implement the simplest correct CUDA kernel.
5. Validate numerical correctness across multiple shapes.
6. Benchmark against the reference.
7. Inspect memory access and launch configuration.
8. Optimize only after establishing a baseline.
9. Record performance and explain the cause of changes.
10. Carry reusable lessons into the next kernel.

## Kernel progression

### Stage 1: Vector addition

Map one output element to one CUDA thread.

Study:

- thread indexing
- block dimensions
- grid dimensions
- bounds checks
- host-to-device and device-to-host transfers

### Stage 2: Elementwise multiply and fused multiply-add

Compare separate kernels with fused execution.

### Stage 3: Reduction

Implement a sum reduction and study:

- shared memory
- synchronization
- warp-level execution
- reduction trees

### Stage 4: Matrix multiplication

Build a naive matmul, then introduce tiling and shared memory.

### Stage 5: Softmax

Study multi-stage reductions, numerical stability, and memory traffic.

### Stage 6: Normalization

Implement LayerNorm or RMSNorm and analyze bandwidth versus computation.

### Stage 7: Fusion

Combine compatible operations to reduce intermediate memory traffic and kernel launches.

## Day 4 scope

Build the first standalone CUDA kernel project around vector addition.

Deliverables:

- CUDA source
- C++ host launcher
- Python reference
- correctness tests
- build instructions
- benchmark harness
- architecture notes
- Day 4 technical report

## Success criteria

The vector-add kernel must produce correct results for multiple sizes and configurations. The benchmark must separate kernel execution time from data-transfer overhead and document the baseline.

## Constraints

Correctness comes before optimization. No performance claim is accepted without a reproducible benchmark and a clear description of what was measured.
