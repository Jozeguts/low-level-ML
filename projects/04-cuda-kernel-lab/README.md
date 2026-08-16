# Project 04: CUDA Kernel Lab

## Purpose

Build GPU kernels from first principles and learn the CUDA execution model through small, measurable kernels.

## Why this matters

Modern ML performance depends heavily on GPU memory movement, parallel execution, synchronization, and kernel launch behavior. Framework abstractions hide these details. This project exposes them.

## Workflow

1. Establish a CPU reference implementation.
2. Write the simplest correct CUDA kernel.
3. Validate numerical correctness.
4. Measure kernel latency and throughput.
5. Inspect grid, block, warp, and memory behavior.
6. Change one optimization variable at a time.
7. Compare against PyTorch and vendor libraries.
8. Record hardware, compiler, input shape, dtype, and measurement method.

## Kernel sequence

1. Vector addition
2. Elementwise activation
3. Reduction
4. Softmax
5. Layer normalization
6. RMS normalization
7. Tiled matrix multiplication
8. Transpose
9. Attention components
10. Fused operations

## Concepts

- Threads and blocks
- Warps
- SIMT execution
- Global memory
- Shared memory
- Registers
- Synchronization
- Atomics
- Memory coalescing
- Occupancy
- Launch configuration
- Numerical precision
- Kernel fusion

## Deliverables

Every kernel gets source code, a CPU reference, correctness tests, benchmark scripts, profiling notes, and a README explaining the execution model.

## Success criteria

Correctness comes first. Performance claims require reproducible measurements. Each optimization must explain which hardware bottleneck it targets.

## Questions this project should answer

- How does a GPU execute a kernel?
- Why do warps matter?
- What makes a memory access coalesced?
- When does shared memory help?
- Why does occupancy matter, and when does it stop being the limiting factor?
- Why are simple-looking operations expensive when launch overhead dominates?
