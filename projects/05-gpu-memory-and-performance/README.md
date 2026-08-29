# Project 05: GPU Memory and Performance

## Purpose

Study how GPU memory behavior affects ML kernel performance.

Day 05 starts with a controlled vector-add benchmark. The project compares launch configurations and records kernel-only timing using CUDA events.

## Questions

- How does block size affect execution time?
- What happens when the input does not divide evenly across blocks?
- Why is vector addition primarily a memory-throughput workload?
- How do global-memory access patterns affect effective bandwidth?
- How do kernel launch parameters influence performance?

## Workflow

1. Establish a correctness reference.
2. Run the same operation across several block sizes.
3. Warm up the GPU.
4. Time only the kernel with CUDA events.
5. Repeat measurements.
6. Calculate effective bandwidth.
7. Record the configuration and result.
8. Compare configurations without changing other variables.
9. Explain performance differences using the GPU execution model.

## Day 05 deliverables

- CUDA vector-add benchmark
- correctness harness
- block-size sweep
- CSV-style benchmark output
- Python analysis script
- reproducible build configuration
- performance notes

## Important distinction

Kernel-only timing excludes host-to-device and device-to-host transfers. This makes the result useful for studying the kernel itself. End-to-end latency requires a separate measurement.
