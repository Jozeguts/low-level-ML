# Project 02 Workflow

## Day 2 scope

Build the first complete scalar reverse-mode automatic differentiation engine.

This day stays entirely within Project 02.

## Architecture

```text
Forward computation
        |
        v
   Value nodes
        |
        +--> data
        +--> parents
        +--> operation
        +--> local backward rule
        +--> gradient
        |
        v
Topological ordering
        |
        v
Reverse traversal
        |
        v
Accumulated gradients
```

## Implementation workflow

### 1. Represent values

Every `Value` stores a scalar numerical value and its accumulated derivative.

### 2. Record graph structure

An operation creates a new node and records the operands as parents.

### 3. Attach local derivative rules

Each operator defines how an incoming gradient is transformed before it reaches each parent.

### 4. Construct a topological ordering

The backward pass requires children to run before their parents. A depth-first traversal collects each node once, producing a deterministic dependency order.

### 5. Seed the output

For a scalar output `L`, set `dL/dL = 1`.

### 6. Execute reverse mode

Visit nodes in reverse topological order and execute each node's local backward rule.

### 7. Accumulate gradients

If a value appears on multiple paths, every path contributes to its gradient. The engine therefore adds contributions rather than overwriting them.

## Operators implemented today

- addition
- subtraction
- multiplication
- negation
- scalar power
- exponential
- logarithm
- ReLU

## Verification strategy

Every derivative rule receives an analytic test.

Representative functions also use central finite differences:

`f'(x) ≈ (f(x + ε) - f(x - ε)) / (2ε)`

The numerical check validates the complete chain of local derivative rules instead of testing only isolated formulas.

## Important experiment

For:

`y = x² + x`

both branches depend on `x`.

The derivative is:

`dy/dx = 2x + 1`

At `x = 3`, the expected gradient is `7`.

This experiment verifies gradient accumulation at shared ancestors.

## Engineering lessons

Reverse-mode autodiff is a graph execution problem as well as a calculus problem.

The derivative of an operator is local, but the final gradient depends on the entire dependency graph.

Topological ordering guarantees that all downstream gradient contributions reach a node before its backward rule finishes propagating them.

## Current limitations

- scalar values only
- no tensor shapes
- no broadcasting
- no higher-order gradients
- no graph retention policy
- no explicit saved-tensor lifecycle
- no memory profiling

These limitations are deliberate. They define the boundary for the next development stages.

## Next increment

Add graph inspection and visualization, stronger graph correctness tests, and a reusable numerical gradient checker before extending the engine to tensors.
