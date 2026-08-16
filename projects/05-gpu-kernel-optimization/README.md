# Project 05: GPU Kernel Optimization

## Purpose

Turn correct CUDA kernels into efficient kernels by identifying bottlenecks through profiling and changing memory access, parallel decomposition, tiling, synchronization, and fusion strategies.

## Why this matters

Writing a GPU kernel and optimizing a GPU kernel are different engineering tasks. This project develops the ability to explain performance from hardware behavior rather than from source-code appearance.

## Workflow

1. Freeze a correct baseline.
2. Benchmark multiple input sizes.
3. Profile the baseline.
4. Identify the dominant bottleneck.
5. Form one optimization hypothesis.
6. Implement one change.
7. Re-run correctness tests.
8. Re-profile.
9. Record speedup and resource changes.
10. Keep or reject the optimization based on evidence.

## Optimization topics

- Coalesced loads and stores
- Shared-memory tiling
- Register blocking
- Warp-level primitives
- Reduction trees
- Occupancy and register pressure
- Bank conflicts
- Kernel fusion
- Persistent execution patterns
- Mixed precision
- Launch overhead
- Shape-dependent performance

## Case studies

- Naive versus tiled matrix multiplication
- Naive versus optimized reduction
- Separate versus fused normalization operations
- Naive versus optimized softmax
- Attention kernel bottlenecks

## Deliverables

- Baseline kernels
- Optimized variants
- Benchmark matrix
- Profiler captures or summaries
- Hardware specification
- Optimization diary
- Performance tables
- Regression tests

## Success criteria

Every speedup must be accompanied by a bottleneck explanation and correctness evidence. The final report should distinguish compute-bound, memory-bound, latency-bound, and launch-bound behavior.
