# Project 07: Mini vLLM

## Objective

Build a small, inspectable LLM serving runtime that demonstrates the systems ideas behind modern continuous-batching engines.

The implementation is CPU-only and deterministic. It does not pretend to be a GPU implementation. The focus is the runtime architecture: request lifecycle, physical KV memory, block tables, scheduling, continuous batching, and measurable cache behavior.

## Why this project matters

A single LLM forward pass is not the complete serving problem. Real workloads contain requests with different prompt lengths and different generation lengths. KV state grows while requests are active. Short requests finish earlier than long requests.

Static batching waits for a whole batch to finish. Continuous batching changes batch membership as requests finish, keeping available execution capacity useful for other requests. Modern inference systems combine this scheduling model with paged KV memory and token budgets. citeturn1search0turn0search0

## Architecture

```text
Client workload
      |
      v
 Request table
      |
      v
 ContinuousBatchScheduler
      |
      +------ waiting / prefill ------+
      |                                |
      +------ active decode -----------+
                                       v
                                MiniVLLM.step()
                                       |
                                       v
                                PagedKVCache
                               /            \
                         block table       free list
                               |
                               v
                         token KV state
```

## Request lifecycle

```text
WAITING -> PREFILL -> DECODE -> FINISHED
             |                   
             +------> CANCELLED

WAITING -----------------------> REJECTED
```

The scheduler uses a request cap and a token budget. Decode work consumes one logical token per active request. Prefill consumes the selected number of prompt tokens. This mirrors the key scheduling controls exposed by modern continuous-batching runtimes. citeturn0search0turn0search4

## Paged KV cache

`kv_cache.py` implements a fixed-size physical block pool.

For block size `B` and sequence length `L`:

```text
required_blocks = ceil(L / B)
allocated_tokens = required_blocks * B
internal_tail_waste = allocated_tokens - L
```

A request owns a block table. Physical blocks do not need to be adjacent. This is the core distinction between logical sequence order and physical KV placement.

The original PagedAttention work applies virtual-memory-style paging to KV cache management to reduce fragmentation and improve memory utilization for serving. citeturn1academia12

## Implemented components

### `kv_cache.py`

- Fixed-size block allocator
- Free-list management
- Per-request block tables
- Dynamic block growth
- Logical token to physical block mapping
- Token reads across non-contiguous blocks
- Release on completion or cancellation
- Capacity accounting
- Internal fragmentation accounting

### `scheduler.py`

- Request state machine
- FIFO ordering with priority support
- Request admission
- Decode-first scheduling
- Token budget
- Prefill chunking
- Cancellation
- KV capacity rejection
- Scheduler snapshots

### `engine.py`

- End-to-end serving loop
- Deterministic next-token model stand-in
- Continuous batch membership
- Prefill and decode in the same runtime
- TTFT and completion-step accounting
- Completion result collection

### `benchmark.py`

Three deterministic workload classes are included:

- `short`
- `mixed`
- `long`

The benchmark reports prompt tokens, generated tokens, scheduler steps, logical tokens per scheduler step, and TTFT statistics.

This is a logical runtime benchmark. It is not GPU throughput data.

### `tests/test_mini_vllm.py`

The tests cover:

- Non-contiguous physical allocation
- Cross-block reads
- KV capacity exhaustion
- Scheduler priority
- Token-budget enforcement
- Mixed-length completion
- Cache release after cancellation

## Run

From this directory:

```bash
python -m pytest tests/test_mini_vllm.py -q
python benchmark.py --workload mixed
python benchmark.py --workload short --block-size 4 --blocks 64
python benchmark.py --workload long --batch-tokens 16
```

The project has no external model checkpoint and no GPU dependency.

## Experiments

### 1. Block size

Run the same workload with different block sizes. Smaller blocks reduce tail waste but increase block-table and allocator overhead. Larger blocks reduce bookkeeping but can leave more unused capacity in the final block.

### 2. Batch token budget

Run the mixed workload with several `--batch-tokens` values. Observe how a larger budget changes scheduler steps and TTFT.

### 3. Cache pressure

Reduce `--blocks` until requests encounter KV capacity pressure. The runtime must fail admission or stop growth cleanly instead of corrupting another request's state.

### 4. Mixed lengths

The mixed workload is deliberately uneven. This is the important case for studying continuous batching because requests finish at different times.

## What this demonstrates

### Fragmentation

A contiguous allocation model wants a large consecutive region. A paged model only needs enough individual free blocks. The block table becomes the indirection layer.

### Continuous batching

A completed short request can leave the active set while another waiting request enters. The active set therefore changes over time instead of being frozen until the longest request finishes. citeturn0search0turn0search3

### Memory-aware scheduling

The scheduler cannot admit work based only on request count. KV capacity and token budgets also constrain what can run safely.

### Systems separation

The model computation is replaceable. The runtime owns request state, memory ownership, scheduling, admission, and lifecycle accounting.

## Research

See [`research.md`](./research.md) for the detailed design analysis, equations, production mapping, validation strategy, and references.

## Relationship to real vLLM

This project is inspired by the publicly documented systems ideas in vLLM. Current vLLM documentation lists PagedAttention, continuous batching, chunked prefill, prefix caching, CUDA/HIP graphs, optimized attention kernels, quantization, and distributed execution among its serving capabilities. This project implements only a small educational subset. citeturn1search0

It is not a copy of the vLLM implementation and should not be described as production-ready inference software.

## Production extension path

A realistic next sequence would be:

1. Replace the deterministic token generator with a real decoder-only model.
2. Store actual K/V tensors instead of token payloads.
3. Move the block pool to device memory.
4. Implement a GPU block-table representation.
5. Add a CUDA paged-attention kernel.
6. Add prefix-cache hashing and reference counting.
7. Add chunked prefill with explicit per-step budgets.
8. Add streaming output queues.
9. Add request timeouts and backpressure.
10. Add GPU telemetry and kernel-level benchmarks.
11. Add quantized KV storage.
12. Add distributed execution only after the single-device runtime is correct.

GPU implementation should preserve the same logical invariants. CUDA performance work must then account for coalesced memory access, occupancy, block configuration, register pressure, and measured hardware behavior. citeturn0search2

## Success criteria

- Multiple requests execute concurrently.
- Logical token order is preserved across non-contiguous physical blocks.
- KV ownership is explicit and released on terminal states.
- Scheduler limits are measurable.
- Mixed-length workloads are reproducible.
- Cache utilization and tail waste are measurable.
- Tests protect allocator and scheduler invariants.
- Benchmark results are clearly separated from GPU performance claims.
