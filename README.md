# Low-Level ML Engineering

A long-form engineering portfolio for understanding machine learning systems from mathematical abstractions down to hardware execution and production serving.

This repository is organized as a sequence of substantial projects. Each project has its own implementation, tests, experiments, benchmarks, workflow documentation, and engineering conclusions.

## Learning progression

```text
Tensor representation
        |
        v
Automatic differentiation
        |
        v
Native PyTorch operators
        |
        v
CUDA kernels
        |
        v
GPU optimization
        |
        v
LLM inference
        |
        v
Paged KV cache and scheduling
        |
        v
TensorFlow runtime concepts
        |
        v
Production model serving
        |
        v
Cross-system benchmarking
```

## Projects

### 01. Mini Tensor Library

Build a NumPy-backed tensor runtime covering storage, shape, strides, views, broadcasting, operators, matrix multiplication, and a first reverse-mode autograd implementation.

Importance: establishes the core abstraction shared by almost every ML framework.

### 02. Autograd Engine

Build a standalone differentiation engine, then extend it from scalars to tensors and use it to train a small neural network.

Importance: explains computational graphs, chain-rule propagation, gradient accumulation, and graph memory behavior.

### 03. PyTorch C++ Operators

Implement native PyTorch operators, register them, test them, connect them to autograd, and extend toward device-specific implementations.

Importance: connects Python model code to native framework execution.

### 04. CUDA Kernel Lab

Implement fundamental GPU kernels from simple elementwise operations through reductions, softmax, normalization, matrix multiplication, and attention components.

Importance: develops practical understanding of GPU execution and memory behavior.

### 05. GPU Kernel Optimization

Profile kernels, identify bottlenecks, and optimize memory access, tiling, synchronization, register use, occupancy, and fusion.

Importance: develops evidence-based GPU performance engineering.

### 06. LLM Inference Engine

Build a decoder-only transformer inference runtime with KV caching, batching, sampling, memory accounting, and prefill/decode benchmarks.

Importance: connects transformer mathematics to real inference constraints.

### 07. Mini vLLM

Build a simplified serving engine with paged KV cache management, block allocation, continuous batching, scheduling, and streaming.

Importance: explains how inference engines improve accelerator utilization under concurrent workloads.

### 08. TensorFlow Runtime Internals

Investigate eager execution, graph tracing, device placement, compilation, XLA concepts, SavedModel, and serving.

Importance: broadens framework-level understanding beyond one ecosystem.

### 09. Model Serving Platform

Build an inference service with an API, request queue, batching, concurrency control, health checks, metrics, load testing, and graceful shutdown.

Importance: turns an inference runtime into an operational service.

### 10. ML Systems Benchmark Suite

Create a common benchmark framework for latency, throughput, memory, batching, scaling, and regression analysis across all earlier projects.

Importance: establishes disciplined measurement for ML systems engineering.

## Project workflow

Every project follows the same engineering loop:

1. Define the systems question.
2. Study the minimum theory needed to answer it.
3. Build a simple reference implementation.
4. Add correctness tests.
5. Compare against a trusted implementation where possible.
6. Profile or benchmark the baseline.
7. Form an optimization or architecture hypothesis.
8. Implement the next increment.
9. Measure again.
10. Document the result and limitations.
11. Commit the working increment.
12. Use the result to define the next project milestone.

## Documentation standard

Every project README should explain:

- What is being built.
- Why it matters.
- The architecture.
- The implementation workflow.
- The theory behind the implementation.
- The experiments to run.
- The measurements to collect.
- The expected failure modes.
- The success criteria.
- The connection to real ML frameworks or production systems.

## Engineering principles

- Build before optimizing.
- Measure before making performance claims.
- Prefer small reproducible experiments.
- Keep reference implementations simple.
- Verify numerical correctness before profiling.
- Record hardware and software environments.
- Separate latency, throughput, memory, and utilization measurements.
- Explain why an optimization works, not only whether it works.
- Increase complexity only after the lower layer is understood.

## Repository goal

The end state is a connected body of working ML systems software. The projects should make it possible to trace a path from a tensor operation, through framework dispatch and native kernels, into GPU execution, through LLM inference and scheduling, and finally into a production serving system.
