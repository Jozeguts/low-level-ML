# Day 06 Research: LLM Inference Engine

## 1. Problem definition

Autoregressive LLM inference repeatedly performs a Transformer forward pass to produce one or more next-token logits. A practical inference engine must optimize more than matrix multiplication. It must manage model weights, prompt processing, KV-cache memory, request scheduling, sampling, batching, latency, throughput, cancellation, and resource limits.

The runtime therefore has two major execution regimes.

### Prefill

The prompt is processed as a sequence. Query, key, and value projections have substantial parallel work. The engine populates the KV cache and produces logits for the next token.

### Decode

The engine normally processes one new token per active request per iteration. The new query attends to cached keys and values. Decode is strongly affected by memory traffic, cache layout, scheduling, and batch composition.

Hugging Face documents the same distinction through KV caching: previous keys and values are reused rather than recomputed for every generated token. Their cache representation is organized per layer with key/value tensors indexed by sequence position. See:

- https://huggingface.co/docs/transformers/cache_explanation
- https://huggingface.co/docs/transformers/kv_cache

## 2. KV-cache memory

For a conventional multi-head cache, an approximate byte requirement is:

`layers * batch * sequence * 2 * kv_heads * head_dim * bytes_per_element`

The factor two represents K and V.

For grouped-query attention, `kv_heads` is smaller than the number of query heads. This reduces cache memory and memory traffic while retaining more query heads.

The cache becomes a scheduling resource. A request is not free to enter execution if there is insufficient cache capacity for its expected context.

## 3. Dynamic versus static caches

A dynamic cache grows as generation proceeds. It is simple and flexible, but changing tensor shapes complicates compilation and graph capture.

A static cache preallocates a maximum size and writes into fixed locations. This creates stable shapes at the cost of reserving memory for unused capacity.

Hugging Face documents DynamicCache, StaticCache, quantized cache variants, and cache offloading. Static caches support compilation-oriented workflows while dynamic caches provide flexible growth.

Source:

https://huggingface.co/docs/transformers/main/kv_cache

## 4. Paged KV caching

A production engine should avoid treating each request's KV cache as one contiguous allocation. Requests have different prompt and generation lengths, and allocating large contiguous regions leads to internal fragmentation.

PagedAttention treats KV storage as blocks. A request owns a logical sequence of block references while physical blocks are drawn from a shared pool. The logical sequence therefore does not need to occupy one contiguous physical region.

This resembles virtual-memory paging:

```text
Request A logical tokens
0 1 2 3 4 5 6 7 8 9 ...
 |------|------|------|
    block table
       |
       +--> physical block 12
       +--> physical block 03
       +--> physical block 21
```

The original vLLM PagedAttention work identifies KV-cache fragmentation as a major serving problem and proposes paged management to increase usable memory and batch capacity.

Source:

https://arxiv.org/abs/2309.06180

## 5. Continuous batching

Static batching waits for all requests in a batch to finish. This wastes capacity when requests have different generation lengths.

Continuous batching changes batch membership between generation steps. Finished requests leave. Waiting requests enter. The engine therefore schedules a dynamic set of active sequences.

Current Hugging Face documentation describes this as dynamic rescheduling at every generation step and also exposes scheduling controls, token budgets, KV-cache block sizes, safety margins, prefix caching, CPU offloading, and CUDA-graph options.

Sources:

https://huggingface.co/docs/transformers/continuous_batching
https://huggingface.co/docs/transformers/continuous_batching_architecture

vLLM exposes similar production-oriented concepts including PagedAttention, continuous batching, chunked prefill, prefix caching, optimized kernels, quantization, CUDA/HIP graphs, and distributed execution.

Source:

https://docs.vllm.ai/en/stable/

## 6. Scheduling objectives

An inference scheduler has competing objectives.

### Time to first token

TTFT is primarily influenced by request queueing and prompt prefill work.

### Inter-token latency

ITL measures the interval between generated tokens. Decode scheduling must keep active sequences progressing without allowing large prefills to monopolize execution.

### Throughput

Throughput is commonly measured as generated tokens per second or requests per second.

### Fairness

A scheduler that maximizes throughput without admission controls can starve small or low-priority requests.

A realistic scheduler therefore needs:

- queue limits
- maximum active requests
- token budgets
- per-request generation limits
- cancellation
- admission control
- deterministic request state
- metrics

## 7. Chunked prefill

Long prompts can consume a large execution window and delay decoding for already-active requests. Chunked prefill splits large prompt processing into smaller work units so decode work can interleave with prompt ingestion.

This creates a token-budgeting problem. The scheduler needs to limit the number of prompt tokens admitted into a step while reserving capacity for decode tokens.

Current Transformers continuous-batching documentation exposes `max_batch_tokens` and describes chunking prompts that exceed the budget.

Source:

https://huggingface.co/docs/transformers/continuous_batching

## 8. Prefix caching

Many workloads repeat system prompts, templates, retrieval prefixes, or shared conversation history. If the KV state for an identical prefix is reusable, the engine can skip redundant prefill work.

Prefix caching requires a stable representation of the prefix and safe sharing semantics. Paged KV storage makes block sharing natural because multiple logical request sequences can reference common physical blocks until they diverge.

vLLM lists prefix caching as a core optimization.

Source:

https://docs.vllm.ai/en/stable/

## 9. Attention kernels and IO awareness

The mathematical attention expression is:

`softmax(QK^T / sqrt(d))V`

The naive implementation materializes large intermediate attention matrices. FlashAttention demonstrated that exact attention performance improves when the algorithm is designed around data movement between GPU memory levels. Tiling keeps working sets in fast on-chip memory and reduces high-bandwidth-memory traffic.

Source:

https://arxiv.org/abs/2205.14135

This project uses a NumPy reference implementation so the memory-management architecture remains visible. CUDA kernels belong to later projects where kernel-level optimization is the primary subject.

## 10. Sampling

Sampling belongs after model execution and should remain modular.

Supported policies in this project include:

- greedy decoding
- temperature scaling
- top-k filtering
- nucleus/top-p filtering
- seeded random sampling

A production engine must also support stop conditions and token limits. Sampling must operate independently for each request in a continuous batch.

## 11. Quantization

Quantization reduces weight and cache memory requirements by using lower-precision representations. It introduces accuracy and kernel considerations, so quantization should be treated as a separate subsystem rather than silently changing the numerical model.

Current Transformers serving documentation lists quantization among server optimizations.

Source:

https://huggingface.co/docs/transformers/serve-cli/serving_optims

vLLM documents support for multiple quantization families and optimized kernels.

Source:

https://docs.vllm.ai/en/stable/

## 12. Serving architecture

A realistic inference server separates the network layer from the execution engine:

```text
HTTP / RPC
    |
    v
Admission control
    |
    v
Request queue
    |
    v
Scheduler
    |
    +------> Prefill work
    |
    +------> Decode work
    |
    v
Model executor
    |
    +------> KV manager
    |
    +------> Sampler
    |
    v
Streaming result
```

The network layer should not own model state. The scheduler should not know HTTP details. The executor should consume structured batches.

## 13. Failure modes

A real engine must handle:

- context-window overflow
- KV-cache exhaustion
- malformed token IDs
- empty prompts
- invalid sampling parameters
- cancellation
- request timeout
- maximum generation length
- model execution failure
- scheduler queue overflow

Failure handling is part of inference engineering, not an optional server feature.

## 14. Metrics

At minimum collect:

- request count
- completed requests
- cancelled requests
- prompt tokens
- generated tokens
- TTFT
- decode token latency
- end-to-end latency
- tokens/sec
- active requests
- queued requests
- KV blocks used
- KV bytes
- admission rejections

Metrics should distinguish queueing delay from model execution time.

## 15. Project implementation boundary

Day 06 implements the CPU reference architecture for these production ideas:

1. request state
2. KV-cache allocation
3. paged KV blocks
4. continuous scheduling
5. chunked prefill budgeting
6. per-request sampling
7. cancellation
8. admission control
9. execution metrics
10. deterministic tests

It intentionally does not claim to reproduce vLLM's CUDA kernels or production throughput. The goal is to make the systems architecture executable and measurable before replacing components with optimized GPU implementations.
