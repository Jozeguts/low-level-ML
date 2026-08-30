# Project 06: LLM Inference Engine

## Scope

Build a realistic CPU reference inference runtime that exposes the systems problems behind production LLM serving before moving the same concepts onto optimized GPU kernels.

The project now covers model execution, request lifecycle, KV-cache memory, paged allocation, continuous batching, chunked prefill budgeting, sampling, cancellation, admission control, metrics, and a serving boundary.

The implementation uses a deterministic tiny decoder-only Transformer checkpoint generated locally. This keeps tests reproducible and makes the systems behavior inspectable without depending on a remote model artifact.

## Architecture

```text
Client
  |
  v
HTTP boundary
  |
  v
Admission control
  |
  v
Request queue
  |
  v
Continuous scheduler
  |                  \
  |                   +--> cancellation
  v
Prefill / Decode work
  |
  +------> Model executor
  |             |
  |             +--> attention
  |             +--> weights
  |             +--> sampler
  |             +--> KV manager
  |                       |
  |                       +--> physical blocks
  |
  v
Streaming / completed result
  |
  v
Metrics
```

## Core execution model

Each Transformer block follows:

```text
x = x + Attention(RMSNorm(x), KV-cache)
x = x + MLP(RMSNorm(x))
```

The runtime separates prompt processing from token-by-token decoding.

### Prefill

Processes prompt tokens and populates the KV cache.

### Decode

Processes new tokens while reusing previous K/V states.

## KV-cache engineering

`runtime.py` contains the straightforward contiguous cache used by the numerical model.

`paged_kv.py` contains a separate reference allocator representing production-style paged KV management. Requests receive logical block tables backed by a shared physical block pool.

The paged cache supports:

- variable-length requests
- block allocation
- block release
- capacity growth
- logical-to-physical mapping
- K/V writes
- K/V reads
- memory accounting
- allocator snapshots

This makes fragmentation and memory pressure explicit instead of hiding them inside a tensor allocation.

## Continuous batching

`scheduler.py` implements a reference continuous-batching scheduler.

Requests move through:

```text
QUEUED -> PREFILL -> DECODING -> FINISHED
                         |           \
                         +---------> CANCELLED
                         +---------> FAILED
```

The scheduler provides:

- request priorities
- admission limits
- token budgets
- decode reservation
- prefill scheduling
- request cancellation
- EOS handling
- maximum generation limits
- request snapshots

The scheduler is intentionally independent of the numerical model. This mirrors production separation between scheduling and execution.

## Serving boundary

`serving.py` provides a small HTTP boundary with:

```text
POST /v1/generate
DELETE /v1/requests/{request_id}
```

It accepts token IDs rather than raw text so tokenization remains a separate concern.

The service submits requests to the scheduler and does not own model execution state.

## Sampling

The existing sampling subsystem supports:

- greedy decoding
- temperature
- top-k
- top-p
- deterministic seeded sampling

Sampling parameters remain outside the numerical Transformer implementation.

## Memory accounting

The engine reports parameter memory and KV memory. The paged cache additionally reports:

- total blocks
- used blocks
- free blocks
- block size
- per-request block tables
- physical cache bytes

For a conventional cache:

```text
layers × sequence × KV_heads × head_dim × 2 × bytes_per_element
```

The factor two represents K and V.

## Performance model

The important inference metrics are:

- queueing delay
- time to first token
- inter-token latency
- end-to-end latency
- generated tokens per second
- prompt tokens per second
- active requests
- queued requests
- KV blocks used

Prefill and decode must be measured separately because their computational characteristics differ.

## Research basis

The design is informed by current inference-system documentation and research:

- Hugging Face KV-cache explanation: https://huggingface.co/docs/transformers/cache_explanation
- Hugging Face cache strategies: https://huggingface.co/docs/transformers/main/kv_cache
- Hugging Face continuous batching: https://huggingface.co/docs/transformers/continuous_batching
- Hugging Face continuous batching architecture: https://huggingface.co/docs/transformers/continuous_batching_architecture
- vLLM documentation: https://docs.vllm.ai/en/stable/
- PagedAttention paper: https://arxiv.org/abs/2309.06180
- FlashAttention paper: https://arxiv.org/abs/2205.14135

The project does not claim to reproduce vLLM's production CUDA implementation. Instead, it implements the underlying systems ideas in a small reference runtime so each optimization is understandable and testable.

## Real-world extension path

The next production-oriented layers are:

1. replace NumPy execution with PyTorch or custom C++ operators
2. replace the reference paged cache with GPU-resident block storage
3. add fused attention kernels
4. add prefix-cache block sharing
5. add chunked prefill to the model executor
6. add quantized weights and KV cache
7. add CUDA graph capture
8. add request streaming
9. add distributed execution
10. add admission control based on measured GPU memory
11. add load testing and SLO dashboards

## Tests

The Day 06 tests cover:

- paged block allocation
- block reuse after release
- non-contiguous logical sequences
- K/V round trips
- cache exhaustion
- continuous request admission
- request replacement after completion
- cancellation
- generation limits
- metrics collection
- KV memory accounting

Run:

```bash
python -m pip install numpy pytest
pytest -q
```

Run the numerical generation example:

```bash
python examples/generate.py
```

Run the service boundary:

```bash
python serving.py
```

## Engineering principle

Correctness and resource accounting come before optimization. A production inference engine is a coordinated system of memory management, scheduling, numerical execution, and observability. Fast kernels without these components do not solve the serving problem.
