# Project 01 Workflow: Mini Tensor Library

## Mission

Build a small tensor runtime that exposes the foundations hidden by modern ML frameworks. The implementation stays compact, while the investigation covers representation, execution, differentiation, correctness, and performance.

## Why the project matters

Tensor systems sit underneath training and inference. Storage layout, metadata, broadcasting, views, operator execution, and automatic differentiation directly affect correctness, memory traffic, and performance.

This project provides the foundation for later work on PyTorch internals, ATen, C++ extensions, CUDA kernels, GPU memory behavior, LLM inference, and serving systems.

## Development rule

One project receives active development per day.

On a Project 01 day, all substantive work stays inside Project 01. Do not begin Project 02 during the same development day.

Each day must produce a meaningful engineering increment. Examples include an implementation milestone, test expansion, numerical experiment, benchmark, profiling session, optimization, or framework comparison.

Every daily increment records:

- Objective
- Concepts studied
- Implementation
- Tests
- Experiments
- Measurements
- Findings
- Limitations
- Next step

## Stage 1: Tensor representation

Study how numerical data maps to memory.

Implement and investigate:

- dtype
- shape
- ndim
- strides
- storage offset
- contiguous layout
- indexing
- slicing
- reshape
- transpose
- views
- aliasing

Questions:

- What does a tensor object describe?
- How does an index become a storage offset?
- Why do views avoid data copies?
- Which transformations preserve a contiguous layout?
- What happens when a requested view is incompatible with the current strides?

Deliverable: tests demonstrating contiguous tensors, transposed tensors, sliced tensors, views, and storage sharing.

## Stage 2: Operator execution

Implement operators independently from autograd first.

Required operations:

- add
- subtract
- multiply
- divide
- negation
- power
- sum
- mean
- max
- exp
- log
- sqrt
- ReLU
- matrix multiplication

For every operator document:

- Input contract
- Output shape
- Forward computation
- Memory behavior
- Backward rule
- Edge cases

Introduce a small dispatch layer so operations follow a consistent execution path.

## Stage 3: Broadcasting

Implement broadcasting rules explicitly.

Cover:

- scalar expansion
- singleton dimensions
- leading dimensions
- compatible shapes
- incompatible shapes
- gradient reduction across broadcast dimensions

Compare behavior against NumPy and PyTorch.

## Stage 4: Autograd engine

Represent differentiable computation as a directed acyclic graph.

Each operation should retain the information required by its local backward function.

Implement:

- parent tracking
- graph construction
- topological ordering
- reverse traversal
- local derivatives
- gradient accumulation
- gradient clearing
- non-differentiable detach

Test shared subgraphs so a tensor used more than once receives the correct accumulated gradient.

## Stage 5: Numerical gradient verification

Build finite-difference gradient checking.

For scalar functions, compare analytical derivatives against numerical approximations.

Test:

- scalar functions
- vectors
- matrices
- broadcasting
- matrix multiplication
- nonlinear functions
- chained expressions
- shared graph nodes

Use explicit absolute and relative error thresholds. Record failures instead of silently loosening tolerances.

## Stage 6: Neural network workload

Use the library for a complete small training workload.

Implement:

- Linear layer
- ReLU
- MSE loss
- parameter registration
- SGD
- zeroing gradients
- training loop

Train a multilayer perceptron on a deterministic synthetic dataset.

Required evidence:

- reproducible seed
- initial loss
- final loss
- learning curve
- learned outputs

## Stage 7: Framework comparison

Implement the same workload using PyTorch.

Compare:

- forward outputs
- gradients
- parameter updates
- convergence
- operation behavior
- execution time

Explain differences rather than assuming identical implementations.

Map the project concepts to PyTorch concepts such as Tensor metadata, Storage, operators, ATen, dispatch, and autograd nodes.

## Stage 8: Performance investigation

Benchmark representative workloads.

Include:

- tensor construction
- elementwise operations
- broadcasting
- contiguous operations
- strided operations
- matrix multiplication
- graph construction
- backward execution
- complete training iterations

Record:

- tensor shapes
- dtype
- iteration count
- hardware
- Python version
- NumPy version
- PyTorch version where relevant

Separate framework overhead from numerical kernel time where practical.

## Stage 9: Profiling and optimization

Profile before changing implementation details.

Investigate:

- temporary allocations
- Python dispatch overhead
- graph traversal overhead
- memory layout
- redundant work
- contiguous versus strided access

Apply optimizations only when measurements justify them.

Every optimization requires a before/after benchmark and a correctness test.

## Stage 10: Engineering report

The final project documentation should explain:

1. Tensor representation
2. Memory layout
3. Strides and views
4. Operator dispatch
5. Broadcasting
6. Computational graphs
7. Reverse-mode differentiation
8. Gradient accumulation
9. Numerical verification
10. Training workflow
11. Performance characteristics
12. Comparison with NumPy
13. Comparison with PyTorch
14. Design tradeoffs
15. Known limitations
16. What changes would be required for a production tensor runtime

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

## Definition of done

Project 01 is complete only when:

- The tensor implementation passes the complete test suite.
- Numerical gradient checks pass across representative operations.
- A small neural network trains successfully.
- NumPy comparisons validate numerical behavior.
- PyTorch comparisons validate forward and backward behavior.
- Performance measurements identify the major overheads.
- At least one optimization is measured before and after.
- The final README explains both the implementation and the underlying systems concepts.

Code running once is not sufficient evidence of completion.

## Daily development cadence

Each Project 01 day follows this sequence:

1. Select one narrow systems objective.
2. Read the relevant implementation concepts.
3. Form an engineering hypothesis.
4. Implement the smallest useful experiment.
5. Add or update tests.
6. Run correctness checks.
7. Measure performance when relevant.
8. Document the result.
9. Commit the complete increment.
10. Define the next day's objective.

The next project starts only after its scheduled project day begins.
