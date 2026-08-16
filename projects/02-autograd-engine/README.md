# Project 02: Autograd Engine

## Purpose

Build a standalone reverse-mode automatic differentiation engine and use it to understand how computation graphs, local derivatives, gradient accumulation, topological execution, and backward propagation work.

## Why this matters

Autograd sits between mathematical models and tensor kernels. Understanding it explains how PyTorch turns a forward program into a backward program, why gradients accumulate, how graph lifetime affects memory, and where framework overhead enters training workloads.

## Workflow

1. Define a scalar value object.
2. Store parents and the local backward rule for every operation.
3. Build a directed acyclic computation graph during the forward pass.
4. Topologically order the graph for reverse execution.
5. Seed the output gradient.
6. Propagate gradients through every operation.
7. Accumulate gradients at shared ancestors.
8. Verify every derivative with finite differences.
9. Extend the engine from scalars to tensors.
10. Compare the result with PyTorch autograd.

## Core concepts

- Computational graphs
- Reverse-mode differentiation
- Chain rule
- Jacobian-vector products
- Local derivative rules
- Leaf and non-leaf values
- Gradient accumulation
- Topological sorting
- Graph retention and freeing
- Numerical gradient checking
- Higher-order differentiation

## Build phases

### Phase 1, scalar engine

Implement `Value`, addition, multiplication, power, exponential, logarithm, ReLU, and backward propagation.

### Phase 2, graph correctness

Add graph visualization, shared subgraphs, gradient accumulation tests, deterministic topological ordering, and finite-difference checks.

### Phase 3, tensor engine

Introduce shapes, broadcasting, matrix multiplication, reductions, views, and tensor-aware backward rules.

### Phase 4, training workload

Train a small multilayer perceptron using only the custom engine. Compare convergence and runtime against PyTorch.

### Phase 5, framework analysis

Trace the equivalent PyTorch operations and document where a production framework adds dispatch, device management, memory management, kernel selection, and graph optimization.

## Deliverables

- `engine/` implementation
- Unit tests for every derivative rule
- Numerical gradient checker
- Graph visualization
- MLP training example
- PyTorch comparison
- Runtime and memory measurements
- Engineering report

## Success criteria

A derivative is accepted only when analytical and numerical gradients agree within a documented tolerance. The final project must train a small model and explain every major step from forward operation to gradient update.

## Questions this project should answer

- Why does reverse mode fit neural-network training?
- Why do shared graph nodes need gradient accumulation?
- Why does graph topology matter during backward execution?
- Where does autograd memory usage come from?
- What changes when tensors replace scalars?
- What work belongs to the graph engine versus the tensor kernel?

## Final outcome

A compact autodiff engine that makes the mechanics of modern automatic differentiation inspectable rather than abstract.
