# Vector Add Benchmark

Day 05 adds kernel-only CUDA timing to the baseline vector-add implementation.

## Build

From `projects/04-cuda-kernels/vector_add`:

```bash
cmake -S . -B build
cmake --build build -j
./build/vector_add_benchmark
```

The benchmark uses CUDA events rather than host wall-clock timing for the measured kernel interval.

## Interpretation

Vector addition performs two global-memory reads and one global-memory write per element. It performs one floating-point addition. This makes it a useful first example of a bandwidth-dominated kernel.

The reported effective bandwidth is:

```text
3 * N * sizeof(float)
---------------------
   kernel_time
```

The value is useful for comparing configurations on the same machine. It should not be treated as the GPU's hardware peak bandwidth.
