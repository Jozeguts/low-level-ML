# Project 01: Mini Tensor Library

## Mission

Build a small tensor and automatic differentiation runtime from first principles.

The project uses NumPy as a low-level numerical backend, while deliberately rebuilding the abstractions that modern ML frameworks place around numerical kernels.

The purpose is to understand what happens between a high-level ML expression and the numerical work required to execute it.

## Why this project matters

A tensor library provides the foundation for training and inference systems.

The important ideas are broader than tensor arithmetic:

- Memory representation determines how data is accessed.
- Shape and stride metadata determines how views and indexing work.
- Broadcasting determines how operations align different shapes.
- Operator implementations determine numerical behavior.
- Computational graphs determine differentiation order.
- Gradient accumulation determines correct optimization behavior.
- Profiling exposes framework overhead.

These concepts lead directly into later projects involving PyTorch internals, ATen, C++ operators, CUDA kernels, GPU memory, LLM inference, and serving.

## What you will build

By the end of the project, the repository should contain:

1. A tensor data model.
2. Shape, stride, and storage metadata.
3. Views and slicing.
4. A collection of numerical operators.
5. Broadcasting support.
6. Matrix multiplication.
7. A reverse-mode autograd engine.
8. Numerical gradient checking.
9. A small neural-network training workload.
10. NumPy and PyTorch comparison experiments.
11. Performance benchmarks.
12. Profiling and optimization experiments.
13. A technical report explaining the design.

## Architecture

```text
User code
   |
   v
Tensor API
   |
   +---- Tensor metadata
   |       shape
   |       strides
   |       dtype
   |       storage offset
   |
   +---- Operators
   |       arithmetic
   |       reductions
   |       matmul
   |       activations
   |
   +---- Autograd graph
   |       parents
   |       local backward rules
   |       topological traversal
   |       gradient accumulation
   |
   v
NumPy numerical backend
   |
   v
CPU memory and compute
```

This architecture is intentionally simple. Each layer exists so its responsibilities and limitations are visible.

## Development workflow

Each feature follows the same engineering loop:

```text
Concept
  -> hypothesis
  -> minimal implementation
  -> unit tests
  -> numerical verification
  -> experiment
  -> benchmark
  -> framework comparison
  -> documentation
```

Do not treat documentation as a replacement for implementation. Do not treat a successful example as proof of correctness.

## Stage 1: Tensor representation

Implement:

- dtype
- shape
- ndim
- strides
- storage offset
- contiguous checks
- indexing
- slicing
- reshape
- transpose
- views
- storage sharing

Questions to answer:

- How does a multidimensional index map to memory?
- What is the difference between shape and strides?
- Why does transpose often avoid copying data?
- Why do some reshapes require contiguous storage?
- How does a view affect aliasing?

Evidence:

- unit tests
- memory-sharing tests
- stride experiments
- documented examples

## Stage 2: Operator system

Implement and test:

- addition
- subtraction
- multiplication
- division
- negation
- power
- sum
- mean
- max
- exponential
- logarithm
- square root
- ReLU
- matrix multiplication

For every operator document its input contract, output shape, numerical operation, edge cases, and backward rule.

## Stage 3: Broadcasting

Implement broadcasting explicitly.

Test:

- scalar with tensor
- vector with matrix
- singleton dimensions
- leading dimensions
- compatible shapes
- incompatible shapes
- gradients through broadcast operations

Compare behavior against NumPy and PyTorch.

## Stage 4: Autograd

Build a directed acyclic computational graph.

Each differentiable operation records its parents and the information required to compute local derivatives.

Implement:

- graph construction
- parent tracking
- topological sorting
- reverse traversal
- local backward functions
- gradient accumulation
- zeroing gradients
- detach

Test shared subgraphs carefully. If one tensor contributes to multiple downstream paths, its gradients must accumulate correctly.

## Stage 5: Numerical gradient checking

Implement finite-difference verification.

For representative scalar functions, compare analytical gradients with numerical approximations.

Cover:

- scalar inputs
- vectors
- matrices
- broadcasting
- matrix multiplication
- nonlinear operations
- chained expressions
- shared graph nodes

Record tolerances and failures explicitly.

## Stage 6: Neural-network workload

Use the library for a complete training experiment.

Implement:

- Linear layer
- ReLU
- MSE loss
- parameter registration
- SGD
- gradient clearing
- training loop

Train a small multilayer perceptron on a deterministic synthetic dataset.

Record the seed, model architecture, learning rate, number of steps, initial loss, final loss, and learning curve.

## Stage 7: NumPy and PyTorch comparison

Run equivalent workloads in NumPy and PyTorch.

Compare:

- forward outputs
- gradients
- parameter updates
- convergence
- operation semantics
- execution time

Then map the concepts in this project to PyTorch concepts such as Tensor metadata, Storage, ATen operators, dispatch, and autograd nodes.

The goal is conceptual understanding, not source-code duplication.

## Stage 8: Performance investigation

Benchmark:

- tensor creation
- elementwise operations
- broadcasting
- contiguous operations
- strided operations
- matrix multiplication
- graph construction
- backward execution
- complete training iterations

Record tensor shape, dtype, software versions, hardware, iteration count, and timing methodology.

Separate Python framework overhead from numerical computation where practical.

## Stage 9: Profiling and optimization

Profile before optimizing.

Investigate:

- temporary allocations
- Python dispatch
- graph traversal
- memory layout
- redundant computation
- contiguous versus strided access

Every optimization must include a correctness test and before/after measurements.

## Project structure

```text
01-mini-tensor/
├── README.md
├── WORKFLOW.md
├── pyproject.toml
├── mini_tensor/
│   ├── __init__.py
│   ├── tensor.py
│   ├── ops.py
│   ├── autograd.py
│   └── nn.py
├── tests/
│   ├── test_tensor.py
│   ├── test_ops.py
│   ├── test_autograd.py
│   └── test_gradcheck.py
├── benchmarks/
├── examples/
├── notes/
└── results/
```

## Daily project rule

Only one project receives active development per day.

A Project 01 day must stay focused on Project 01. The day's work should produce one coherent increment rather than scattered changes across future projects.

A daily increment should include code or an experiment plus appropriate tests and documentation.

## Definition of done

Project 01 is complete when:

- tensor representation is tested
- operators are tested
- broadcasting is tested
- views and strides are investigated
- autograd is verified numerically
- a small neural network trains successfully
- results match reference implementations within defined tolerances
- performance has been measured
- at least one optimization has been evaluated
- limitations are documented
- the final engineering report explains what the implementation teaches about real tensor frameworks

## Final engineering questions

1. What minimum abstraction is required to represent a tensor?
2. How do strides affect memory access?
3. Why are views important?
4. How does broadcasting alter gradient propagation?
5. How does reverse-mode autograd traverse a graph?
6. Where does a Python implementation become a performance bottleneck?
7. Which responsibilities belong in a tensor object versus an operator system?
8. Which parts of this design correspond to PyTorch internals?
9. What would need to change to support efficient GPU execution?
10. Which limitations prevent this implementation from becoming a production ML framework?
