# Project 06: LLM Inference Engine

## Purpose

Build a small decoder-only transformer inference runtime and understand the complete path from token IDs to generated tokens.

## Why this matters

LLM inference is a systems problem. Model weights, attention state, memory bandwidth, batching, scheduling, numerical precision, and kernel execution all affect latency and throughput.

## Workflow

1. Load a small pretrained-compatible model or a controlled local model.
2. Implement token input and output handling.
3. Implement embedding, normalization, attention, MLP, residual paths, and logits.
4. Implement autoregressive generation.
5. Add KV caching.
6. Measure prefill and decode separately.
7. Add batching.
8. Add sampling strategies.
9. Add memory accounting.
10. Compare with a reference framework.

## Architecture

```text
request
  -> tokenizer
  -> input preparation
  -> embedding
  -> transformer blocks
      -> normalization
      -> attention + KV cache
      -> residual
      -> MLP
      -> residual
  -> logits
  -> sampler
  -> next token
```

## Core topics

- Prefill versus decode
- KV cache
- Attention complexity
- Memory bandwidth
- Batch size
- Sequence length
- Sampling
- Temperature and top-k/top-p
- Quantization concepts
- CPU versus GPU execution
- Latency versus throughput

## Deliverables

- Inference runtime
- Model loader
- KV-cache implementation
- Sampling module
- Batch runner
- Memory accounting
- Latency benchmarks
- Tokens-per-second benchmarks
- Reference comparison

## Success criteria

The runtime must produce numerically validated outputs for a controlled model and report separate prefill latency, decode latency, memory use, and throughput.

## Questions this project should answer

- Why does decoding become memory-bandwidth sensitive?
- Why is KV caching important?
- Why do prefill and decode behave differently?
- What limits batch size?
- Which optimizations belong in kernels and which belong in scheduling?
