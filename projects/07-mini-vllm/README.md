# Project 07: Mini vLLM

## Purpose

Build a simplified LLM serving engine inspired by the core systems ideas behind vLLM, with emphasis on paged KV-cache management, request scheduling, continuous batching, and memory-aware execution.

## Why this matters

Serving many concurrent generation requests is fundamentally different from running one model forward pass. Memory fragmentation, uneven sequence lengths, scheduling, and KV-cache growth become first-class systems problems.

## Workflow

1. Start with the Project 06 inference engine.
2. Define a request state machine.
3. Represent KV cache blocks independently of request sequences.
4. Build a block allocator and free list.
5. Map logical token positions to physical cache blocks.
6. Implement request admission and scheduling.
7. Add continuous batching.
8. Handle prefill and decode phases together.
9. Stream generated tokens.
10. Measure throughput, latency, queue time, and cache utilization.

## Components

- Request manager
- Scheduler
- KV-cache allocator
- Block table
- Batching engine
- Worker execution loop
- Token streamer
- Metrics collector

## Experiments

- Static batching versus continuous batching
- Contiguous KV cache versus paged cache
- Short and long request mixtures
- Different maximum batch sizes
- Cache pressure and eviction policy experiments
- Throughput versus tail latency

## Deliverables

- Scheduler implementation
- KV block allocator
- Block table
- Continuous batching runtime
- Load generator
- Metrics dashboard or exported metrics
- Benchmark suite
- Architecture documentation

## Success criteria

The system must serve multiple concurrent requests while maintaining correct token order and measurable cache accounting. Performance must be evaluated under mixed request lengths, not a single idealized workload.

## Questions this project should answer

- Why does KV-cache fragmentation matter?
- Why does continuous batching improve accelerator utilization?
- What is the scheduling trade-off between throughput and latency?
- How should physical cache memory be separated from logical sequence state?
