# Project 06: LLM Inference Engine

## Objective

Build a realistic, inspectable reference inference engine that exposes the systems problems behind production LLM serving before replacing individual components with optimized C++ and CUDA implementations.

The project models the complete request path:

```text
client
  ↓
request validation
  ↓
admission control
  ↓
prefill
  ↓
KV-cache residency
  ↓
continuous decode
  ↓
sampling
  ↓
completion
  ↓
metrics
```

## Why this project exists

A Transformer forward pass is only one component of an inference service. Production systems also need to manage variable-length requests, KV-cache memory, dynamic batching, request cancellation, latency, throughput, and failures.

This project therefore treats inference as a systems problem.

## Architecture

```text
                    ┌────────────────────┐
                    │  Client / HTTP API  │
                    └──────────┬─────────┘
                               │
                               v
                    ┌────────────────────┐
                    │ Admission Control  │
                    └──────────┬─────────┘
                               │
                               v
                    ┌────────────────────┐
                    │ Request Scheduler  │
                    └──────┬─────┬───────┘
                           │     │
                     prefill     decode
                           │     │
                           v     v
                    ┌────────────────────┐
                    │  Model Executor    │
                    │ NumPy Transformer  │
                    └─────────┬──────────┘
                              │
                    ┌─────────┴─────────┐
                    v                   v
             ┌──────────────┐    ┌──────────────┐
             │ KV Manager   │    │   Sampler    │
             │ contiguous + │    │ greedy/top-k │
             │ paged model  │    │ /top-p/temp  │
             └──────────────┘    └──────────────┘
                              │
                              v
                    ┌────────────────────┐
                    │ Metrics / Results  │
                    └────────────────────┘
```

## Numerical model

`runtime.py` implements a deterministic tiny decoder-only Transformer.

Each block follows:

```text
x = x + Attention(RMSNorm(x), KV-cache)
x = x + MLP(RMSNorm(x))
```

The default demo uses grouped-query attention with fewer K/V heads than query heads. This makes the cache behavior closer to modern LLM configurations while keeping the model small enough for CPU tests.

The checkpoint is generated locally from a fixed seed. No external model download is required.

## Prefill and decode

Prefill processes the prompt and constructs the initial KV state.

Decode processes one new token per active request and reuses the existing K/V state.

The separation is important because prefill is prompt-heavy while decode is repeatedly executed and is sensitive to cache access, scheduling, and per-step overhead.

## KV-cache systems

`runtime.py` provides a contiguous numerical KV cache for the model.

`paged_kv.py` provides a separate reference implementation of a physical block allocator.

The paged allocator supports:

- fixed-size physical blocks
- variable-length sequences
- logical-to-physical block tables
- block allocation
- block growth
- block release
- random physical placement
- K/V writes
- K/V reads
- capacity errors
- memory accounting
- allocator snapshots

This makes KV fragmentation explicit. It is a reference implementation of the memory abstraction, not a reproduction of vLLM's optimized GPU kernels.

## Continuous batching

`scheduler.py` models request lifecycle independently of numerical execution.

```text
QUEUED → PREFILL → DECODING → FINISHED
             │          │
             └──────────┴────→ CANCELLED / FAILED
```

The scheduler supports:

- FIFO-style ordering
- priority ordering
- maximum active requests
- token budgets
- decode reservation
- prefill selection
- cancellation
- EOS completion
- maximum generation limits
- request snapshots

`engine.py` provides the executable end-to-end model loop over requests.

## Engine API

Example:

```python
import numpy as np
from engine import GenerationRequest, build_demo_engine

engine = build_demo_engine()
request = GenerationRequest(
    prompt=np.array([4, 8, 15, 16, 23, 42]),
    max_new_tokens=16,
    strategy="greedy",
    request_id="demo-1",
)

engine.admit(request)
engine.run_until_complete()
print(engine.completed["demo-1"].generated)
```

The engine records TTFT and request-level lifecycle state.

## Sampling

`sampling.py` keeps token selection separate from model execution.

Supported strategies:

- greedy
- temperature sampling
- top-k
- top-p / nucleus sampling
- deterministic seeded sampling

The sampler validates parameters and uses a numerically stable softmax path.

## Workload generation

`workload.py` creates deterministic benchmark cases:

- short prompts
- mixed prompt lengths
- long prompts
- controlled concurrency

The same workload is reproducible from the same seed, making comparisons between runtime changes easier to interpret.

## Benchmarking

`benchmark_engine.py` reports:

- prompt tokens
- generated tokens
- wall-clock runtime
- generated tokens per second
- TTFT p50
- TTFT p95
- mean TTFT

Run:

```bash
python benchmark_engine.py
```

For machine-readable output:

```bash
python benchmark_engine.py --json
```

These are CPU-reference measurements. They are not GPU performance claims.

## Research basis

The implementation is informed by:

- Hugging Face cache strategies
- Hugging Face continuous batching
- Hugging Face continuous batching architecture
- vLLM architecture and serving documentation
- PagedAttention
- FlashAttention
- CUDA memory and execution guidance

The detailed research report is in `research.md`.

## Production mapping

| Reference component | Production analogue |
|---|---|
| NumPy Transformer | PyTorch / C++ / CUDA executor |
| contiguous KV cache | GPU KV storage |
| paged allocator | PagedAttention-style block manager |
| scheduler | continuous batching scheduler |
| workload generator | load-test driver |
| metrics | telemetry / SLO system |
| HTTP boundary | OpenAI-compatible API layer |
| deterministic checkpoint | real model weights |

## Production extension path

1. Replace NumPy matrix operations with PyTorch operators.
2. Add C++ custom operators.
3. Implement GPU-resident KV blocks.
4. Implement fused attention kernels.
5. Add prefix-cache block sharing.
6. Add chunked prefill to the numerical executor.
7. Add quantized weights and KV formats.
8. Add CUDA graph execution.
9. Add asynchronous request streaming.
10. Add tensor parallelism.
11. Add load testing under controlled concurrency.
12. Add hardware counters and GPU profiling.

## Tests

Install dependencies:

```bash
python -m pip install numpy pytest
```

Run:

```bash
pytest -q
```

The tests cover numerical generation, deterministic greedy decoding, request validation, concurrent requests, cancellation, context limits, and metrics snapshots. Existing tests also cover paged KV allocation, scheduler transitions, sampling, and memory accounting.

## Engineering principles

1. Correctness before optimization.
2. Measure before changing the scheduler.
3. Treat KV memory as a first-class resource.
4. Separate scheduling from model execution.
5. Separate sampling policy from numerical execution.
6. Record queueing and model latency separately.
7. Never report hardware performance without running on the target hardware.
