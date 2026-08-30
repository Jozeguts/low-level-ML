# Project 06: LLM Inference Engine

## Scope

Build a controlled decoder-only Transformer inference runtime and study the systems behavior behind autoregressive language-model inference.

The implementation is structured like a small inference runtime: configuration, weights, tokenizer interface, execution state, KV cache, sampling, batching, memory accounting, profiling, and reference validation are separate components.

The runtime uses a deterministic tiny Transformer checkpoint generated locally. This keeps tests reproducible and avoids depending on a remote model artifact.

## Architecture

```text
Request
  |
  v
Token IDs
  |
  v
Request state
  |
  +---- prefill ----> Transformer ----> logits
  |                       |               |
  |                       +--> KV cache   v
  |                                   sampler
  |                                       |
  +---- decode <------------------------ next token
```

Each block performs:

```text
x = x + Attention(RMSNorm(x), KV-cache)
x = x + MLP(RMSNorm(x))
```

## Prefill versus decode

Prefill processes the prompt in parallel and populates the cache. Decode consumes one new token per step and reuses the keys and values from previous steps.

The runtime measures both phases independently because they expose different bottlenecks. Prefill offers substantial parallel work. Decode performs relatively little new arithmetic per request while repeatedly reading model weights and the growing KV cache.

## KV cache

For each layer and request, keys and values are stored as:

```text
K: [kv_heads, sequence, head_dim]
V: [kv_heads, sequence, head_dim]
```

The cache tracks capacity, current length, dtype, and byte usage. Overflow is rejected explicitly.

## Sampling

Supported policies:

- greedy
- temperature
- top-k
- top-p
- seeded stochastic sampling

Sampling is isolated from model execution so generation policy does not affect the numerical core.

## Batching

The batch runner executes independent request states together. Request state remains isolated, which provides a baseline for later continuous batching and scheduler work in Project 07.

## Memory accounting

The runtime reports parameter memory, KV-cache memory, activation estimates, and total estimated memory.

For a standard cache, the dominant storage scales approximately with:

```text
layers × sequence × KV_heads × head_dim × 2 × bytes_per_element
```

The factor of two represents K and V.

## Benchmarking

The benchmark reports prompt length, generated tokens, batch size, prefill latency, decode latency, end-to-end latency, tokens per second, and KV-cache bytes.

No benchmark numbers are hard-coded. Hardware measurements must be produced on the machine running the benchmark.

## Validation

The test suite covers deterministic weights, shape checks, KV-cache correctness, greedy generation, seeded sampling, attention equivalence, cache accounting, and prefill/decode equivalence.

## Real-world extension path

The design prepares for paged KV caching, continuous batching, prefix caching, quantized weights, fused CUDA kernels, tensor parallelism, request cancellation, admission control, token streaming, and an HTTP serving layer.

## Run

```bash
python -m pip install numpy pytest
pytest -q
python examples/generate.py
python benchmarks/benchmark.py
```

## Research basis

The project studies the practical consequences of autoregressive decoding, memory bandwidth, cache reuse, batching, and sampling. It also provides the numerical boundary needed before Project 07 investigates PagedAttention-style memory management and continuous batching.
