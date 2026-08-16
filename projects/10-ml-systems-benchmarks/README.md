# Project 10: ML Systems Benchmark Suite

## Purpose

Create a reusable benchmark system for comparing tensor operations, kernels, inference runtimes, batching strategies, and serving architectures under controlled conditions.

## Why this matters

Performance claims without controlled measurement are weak. This project turns benchmarking into an engineering discipline and provides a common measurement layer for the earlier projects.

## Workflow

1. Define the question before writing the benchmark.
2. Select representative workloads.
3. Warm up the runtime where appropriate.
4. Separate setup time from steady-state execution.
5. Repeat measurements and report distributions.
6. Record hardware and software versions.
7. Measure memory alongside latency.
8. Compare throughput and tail latency.
9. Store machine-readable benchmark results.
10. Analyze regressions across implementations.

## Measurement areas

- Operation latency
- Kernel latency
- Throughput
- Tokens per second
- Requests per second
- Queue time
- Peak memory
- GPU memory
- CPU utilization
- GPU utilization
- Batch efficiency
- Scaling with sequence length

## Experimental discipline

Every benchmark records:

- Hardware
- Operating system
- Driver and CUDA versions when relevant
- Framework versions
- Compiler versions when relevant
- Input shapes
- Dtypes
- Batch sizes
- Warm-up policy
- Number of repetitions
- Measurement method

## Deliverables

- Benchmark runner
- Common result schema
- Workload definitions
- Statistical summaries
- CSV or JSON result output
- Regression detection
- Visualization scripts
- Cross-project reports

## Success criteria

The suite must produce repeatable measurements and make it easy to compare implementations without changing hidden experimental variables.

## Questions this project should answer

- What does a fair ML systems benchmark look like?
- How should latency distributions be reported?
- When is throughput more useful than latency?
- How do batch size and sequence length affect performance?
- How do we detect a performance regression automatically?
