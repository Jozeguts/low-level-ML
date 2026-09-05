"""Correctness tests for Mini vLLM's memory and scheduling invariants."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import EngineConfig, MiniVLLM
from kv_cache import OutOfKVBlocks, PagedKVCache
from scheduler import Request, RequestState, ContinuousBatchScheduler


def test_paged_cache_handles_non_contiguous_request_blocks():
    cache = PagedKVCache(num_blocks=4, block_size=2)
    cache.allocate("a", 0)
    cache.append_many("a", [10])
    cache.allocate("b", 0)
    cache.append_many("b", [20])
    cache.allocate("c", 0)
    cache.append_many("c", [30])
    cache.release("a")
    cache.release("c")
    cache.allocate("d", 0)
    cache.append_many("d", [40, 41, 42])

    assert cache.read("b") == [20]
    assert cache.read("d") == [40, 41, 42]
    assert len(cache.block_table("d")) == 2
    assert cache.block_table("d")[0] != cache.block_table("d")[1] - 1


def test_cache_rejects_capacity_exhaustion():
    cache = PagedKVCache(num_blocks=2, block_size=2)
    cache.allocate("a", 0)
    cache.append_many("a", [1, 2, 3, 4])
    cache.allocate("b", 0)
    try:
        cache.append("b", 5)
    except OutOfKVBlocks:
        pass
    else:
        raise AssertionError("expected KV capacity failure")


def test_scheduler_prioritizes_decode_and_respects_budget():
    cache = PagedKVCache(16, 4)
    scheduler = ContinuousBatchScheduler(cache, max_requests=4, max_batch_tokens=5, prefill_chunk=4)
    scheduler.submit(Request("low", [1, 2], 3, priority=0))
    scheduler.submit(Request("high", [3, 4], 3, priority=10))
    scheduler.start_prefill("low")
    scheduler.apply_prefill("low", 2)
    scheduler.start_prefill("high")
    scheduler.apply_prefill("high", 2)
    plan = scheduler.plan()
    assert plan["decode"] == ["high", "low"]
    assert plan["unused_token_budget"] == 3


def test_continuous_engine_completes_mixed_lengths():
    engine = MiniVLLM(EngineConfig(num_blocks=32, block_size=4, max_requests=3, max_batch_tokens=8, prefill_chunk=4))
    engine.submit("short", [1, 2, 3], 2)
    engine.submit("long", list(range(12)), 5)
    results = engine.run()

    assert [r.request_id for r in results] == ["long", "short"]
    assert len(results[0].output) == 5
    assert len(results[1].output) == 2
    assert engine.cache.stats().used_blocks == 0


def test_cancellation_releases_cache():
    engine = MiniVLLM(EngineConfig(num_blocks=8, block_size=4))
    engine.submit("cancel", list(range(6)), 20)
    engine.step()
    before = engine.cache.stats().used_blocks
    assert before > 0
    assert engine.cancel("cancel")
    assert engine.cache.stats().used_blocks == 0
    assert engine.scheduler.requests["cancel"].state == RequestState.CANCELLED
