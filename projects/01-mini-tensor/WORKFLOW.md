# Project 01 Workflow: Mini Tensor Library

## Mission

Build a small tensor runtime that exposes the foundations hidden by modern frameworks. The implementation is intentionally small, but the investigation is broad.

## Why the project matters

A tensor framework sits at the center of ML systems. Tensor storage, metadata, broadcasting, views, operator execution, and automatic differentiation influence memory use and performance throughout training and inference.

## Learning path

### Stage 1, representation

Implement storage, dtype, shape, strides, storage offset, indexing, and contiguous-layout checks.

Questions:

- What does a tensor object need to describe its data?
- How do strides map multidimensional indices to storage?
- Why do views avoid copies?
- When does a view become non-contiguous?

### Stage 2, operator system

Implement arithmetic, reductions, reshape, transpose, broadcasting, matrix multiplication, and activation functions.

Document each operator's input contract, output shape, forward computation, and backward rule.

### Stage 3, autograd

Build a graph of operations. Store parents and local backward functions. Perform topological traversal and gradient accumulation.

### Stage 4, numerical verification

Use finite differences to verify gradients. Compare outputs and gradients against NumPy and PyTorch.

### Stage 5, model training

Train a small multilayer perceptron using the library. This forces the tensor and autograd layers to work together in a realistic workload.

### Stage 6, performance

Benchmark contiguous and strided operations, broadcasting, matrix multiplication, and autograd overhead. Record input shapes and hardware for every result.

### Stage 7, framework mapping

Map each component to the concepts found in PyTorch, including tensor metadata, dispatch, autograd, and native kernels.

## Project structure

```text
01-mini-tensor/
├── README.md
├── WORKFLOW.md
├── pyproject.toml
├── mini_tensor/
│   └── tensor.py
├── tests/
├── benchmarks/
├── examples/
└── notes/
```

## Required evidence

Every milestone must include working code, tests, and a short engineering explanation. Performance work must include measurements. Gradient work must include numerical verification.

## Final deliverable

A small tensor framework plus an engineering report explaining how storage, metadata, operator execution, and autograd form the foundation for larger frameworks.

## Transition to Project 02

Project 01 establishes tensor mechanics. Project 02 isolates and expands automatic differentiation so the graph engine itself becomes a major system under investigation.
