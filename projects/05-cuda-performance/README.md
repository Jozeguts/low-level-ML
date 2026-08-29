# Project 05: CUDA Performance Engineering

## Purpose

Study GPU performance systematically rather than treating kernel speed as a single number.

Day 5 starts with vector addition and measures the effect of CUDA block size while keeping the workload fixed.

## Questions

- How does block size affect a simple memory-bound kernel?
- What is kernel-only latency?
- What effective bandwidth does the kernel achieve?
- When does changing launch configuration stop producing useful gains?
- How should benchmark results be reported without confusing hardware peak bandwidth with measured throughput?

## Day 5 experiment

Fixed workload:

- 16,777,216 float32 elements
- 10 warm-up launches
- 100 measured launches
- vector addition

Variable:

- 64 threads per block
- 128 threads per block
- 256 threads per block
- 512 threads per block

The benchmark uses CUDA events around the measured kernel launches.

## Build

```bash
cmake -S . -B build
cmake --build build -j
./build/vector_add_benchmark
```

A CUDA-capable NVIDIA system is required for execution.

## Measurement model

Vector addition reads two arrays and writes one array. For N float32 elements:

`bytes = 3 * N * 4`

`effective_bandwidth = bytes / kernel_time`

The result is an effective bandwidth measurement for this workload. It is not the GPU's theoretical peak memory bandwidth.

## Experimental discipline

Keep the data size, dtype, operation, warm-up count, and iteration count fixed while changing block size. Record the GPU model, CUDA version, driver version, and benchmark output when collecting real results.

Do not report fabricated numbers. The repository contains the executable benchmark so results can be collected on an NVIDIA GPU.

## Next work

Extend the benchmark across input sizes and add launch-overhead analysis. Then inspect memory coalescing and use profiling tools to connect measured performance with hardware behavior.
