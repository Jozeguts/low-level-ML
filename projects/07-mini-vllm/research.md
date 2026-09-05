# Project 07 Research: Mini vLLM

## 1. Problem definition

LLM generation is a serving problem, not only a matrix-multiplication problem. A production server must keep many variable-length requests moving while their KV caches grow token by token.

The central systems constraint is memory. A request's attention state is dynamic, so allocating one large contiguous KV region per request creates fragmentation and makes capacity depend on allocation history. Paged KV storage separates logical sequence position from physical memory location.

## 2. Paged KV cache

PagedAttention applies the virtual-memory idea of pages and page tables to attention state. A logical sequence is divided into fixed-size token blocks. Each logical block maps to a physical block in a shared pool. The attention implementation follows the mapping instead of requiring one contiguous allocation.

This changes the allocator's job from finding a large contiguous region to finding individual free blocks. The last block of a sequence can remain partially filled, so the main predictable internal waste is bounded by the unused tail of the final block.

The original PagedAttention paper identifies KV-cache fragmentation and dynamic growth as important limits on batch size and proposes paging to reduce memory waste. It also describes block sharing as a route to reuse KV state across requests.

## 3. Continuous batching

Static batching waits for every request in a batch to finish before admitting another batch. Generation lengths differ, so short requests can leave unused capacity while the longest request continues.

Continuous batching changes membership at every scheduling iteration. Completed sequences leave the active set and waiting sequences can enter immediately. A scheduler normally enforces both a request-count limit and a token budget so one large prefill cannot consume the whole step.

A practical policy needs to balance decode latency against prompt work. Prioritizing existing decode work protects time between generated tokens. Allowing chunked prefill prevents a long prompt from monopolizing a step.

## 4. Prefill versus decode

Prefill processes the prompt and builds the initial KV state. It is usually compute-heavy and processes many tokens for one request.

Decode generates one new token per active sequence per iteration. Each decode step reads the existing KV state and appends a new KV entry. This creates a memory-bandwidth-sensitive workload for many models.

A serving runtime therefore needs a scheduler that understands both units of work instead of treating every request as a single homogeneous operation.

## 5. Block accounting

For block size B and sequence length L, the number of physical blocks required is:

    ceil(L / B)

If a sequence occupies K blocks, allocated capacity is K*B tokens. Internal tail waste is:

    K*B - L

Across N requests, the allocator can report total allocated blocks, free blocks, logical tokens, capacity tokens, utilization, and internal waste. These metrics are more useful than a simple count of active requests because two workloads with the same request count can have very different memory pressure.

## 6. Scheduler design

Mini vLLM uses these states:

    WAITING -> PREFILL -> DECODE -> FINISHED
                    \-> CANCELLED

A request can also be rejected if the physical cache cannot provide the required blocks.

The scheduler orders requests by priority, then arrival step, then request ID. Active decode work is selected first. Remaining token budget is then used for new and active prefill work.

This is deliberately simpler than a production scheduler. It provides a controlled environment for measuring the effect of policies before introducing more complex heuristics.

## 7. Prefix caching

If requests share an identical prompt prefix, the KV blocks representing completed prefix chunks can be shared. A real implementation needs content hashing and reference counts. The cache cannot return a shared block to the free pool until every request referencing it is finished.

This project does not yet implement prefix hashing. The block table is kept explicit so prefix sharing can be added without changing the request state machine.

## 8. Chunked prefill

Long prompts can be divided into smaller chunks. Chunking lets decode requests continue making progress while another request's prompt is still being processed. It also gives the scheduler a token-level control surface.

The Mini vLLM scheduler exposes `prefill_chunk` and `max_batch_tokens` to make this behavior observable.

## 9. Performance model

The simulator is intentionally CPU-only and deterministic. Its benchmark must not be interpreted as GPU tokens-per-second data.

Useful logical measurements are:

- scheduler steps to completion
- generated tokens per scheduler step
- time to first token in scheduler steps
- maximum TTFT in a mixed workload
- KV block utilization
- internal tail waste
- rejected requests under cache pressure

Hardware measurements belong in a later CUDA implementation. NVIDIA's CUDA guidance emphasizes coalesced global-memory access, adequate occupancy, and empirical block-size tuning. Those concerns become relevant when this logical runtime is connected to real GPU kernels.

## 10. Production mapping

| Mini vLLM | Production inference engine |
| --- | --- |
| `PagedKVCache` | GPU KV-cache manager and block allocator |
| request block table | GPU-resident block table / page mapping |
| `ContinuousBatchScheduler` | scheduler and token-budget admission |
| `MiniVLLM.step()` | engine iteration and model worker execution |
| deterministic `_next_token()` | model forward + logits processing |
| benchmark workloads | load tests and serving traces |

The simulator intentionally stops before CUDA kernels, distributed execution, quantization, speculative decoding, and network backpressure. These are extension points, not hidden assumptions.

## 11. Validation strategy

Correctness is tested at the invariant level:

1. A request can span non-contiguous physical blocks.
2. Reading a request follows its block table rather than physical adjacency.
3. The allocator raises when capacity is exhausted.
4. The scheduler respects request and token budgets.
5. Mixed-length requests finish with exact output lengths.
6. Cancellation releases every physical block owned by the request.
7. Completed requests leave zero live KV allocation.

## References

1. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", https://arxiv.org/abs/2309.06180
2. vLLM documentation, https://docs.vllm.ai/en/stable/
3. Hugging Face Transformers, Continuous Batching Architecture, https://huggingface.co/docs/transformers/continuous_batching_architecture
4. NVIDIA CUDA C Best Practices Guide, https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

The repository implementation is a teaching and systems-engineering reference. It is not a reimplementation of the full vLLM codebase.
