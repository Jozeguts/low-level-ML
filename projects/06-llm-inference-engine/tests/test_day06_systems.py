import numpy as np
import pytest

from paged_kv import KVSpec, PagedKVCache
from scheduler import (
    ContinuousBatchScheduler,
    GenerationRequest,
    RequestStatus,
    SchedulerConfig,
)
from metrics import MetricsRegistry


def test_paged_allocator_reuses_released_blocks():
    spec = KVSpec(num_layers=2, num_kv_heads=2, head_dim=4, block_size=4)
    cache = PagedKVCache(spec, num_blocks=4)
    first = cache.allocate("a", tokens=5)
    assert len(first.blocks) == 2
    assert cache.used_blocks == 2
    cache.release("a")
    assert cache.used_blocks == 0
    second = cache.allocate("b", tokens=8)
    assert len(second.blocks) == 2
    assert cache.free_blocks == 2


def test_paged_cache_handles_non_contiguous_logical_sequence():
    spec = KVSpec(num_layers=1, num_kv_heads=1, head_dim=2, block_size=2)
    cache = PagedKVCache(spec, num_blocks=4)
    cache.allocate("r", tokens=0)
    keys = np.arange(10, dtype=np.float32).reshape(1, 5, 2)
    values = keys + 100
    cache.write("r", 0, keys, values, start=0)
    cache.append_tokens("r", 5)
    read_k, read_v = cache.read("r", 0)
    np.testing.assert_array_equal(read_k, keys)
    np.testing.assert_array_equal(read_v, values)
    assert cache.logical_position("r", 4)[1] == 0


def test_paged_allocator_rejects_exhaustion():
    spec = KVSpec(num_layers=1, num_kv_heads=1, head_dim=2, block_size=4)
    cache = PagedKVCache(spec, num_blocks=1)
    cache.allocate("a", tokens=4)
    with pytest.raises(MemoryError):
        cache.allocate("b", tokens=1)


def test_scheduler_continuously_replaces_finished_request():
    scheduler = ContinuousBatchScheduler(SchedulerConfig(max_active_requests=2, max_batch_tokens=32))
    scheduler.submit(GenerationRequest("a", [1, 2], max_new_tokens=2))
    scheduler.submit(GenerationRequest("b", [3], max_new_tokens=2))
    scheduler.submit(GenerationRequest("c", [4, 5], max_new_tokens=1))
    admitted = scheduler.admit()
    assert {r.request_id for r in admitted} == {"a", "b"}
    scheduler.get("a").prompt_cursor = scheduler.get("a").prompt_length
    scheduler.mark_prefill_complete("a")
    scheduler.get("b").prompt_cursor = scheduler.get("b").prompt_length
    scheduler.mark_prefill_complete("b")
    scheduler.record_token("a", 9)
    scheduler.record_token("a", 10)
    assert scheduler.get("a").status == RequestStatus.FINISHED
    scheduler.admit()
    assert "c" in scheduler.active


def test_scheduler_cancellation_releases_logical_request():
    scheduler = ContinuousBatchScheduler()
    scheduler.submit(GenerationRequest("cancel", [1], max_new_tokens=5))
    scheduler.admit()
    assert scheduler.cancel("cancel")
    assert scheduler.get("cancel").status == RequestStatus.CANCELLED
    assert scheduler.snapshot()["active"] == 0


def test_scheduler_enforces_generation_limit():
    scheduler = ContinuousBatchScheduler()
    scheduler.submit(GenerationRequest("limit", [1], max_new_tokens=1))
    scheduler.admit()
    request = scheduler.get("limit")
    request.prompt_cursor = request.prompt_length
    scheduler.mark_prefill_complete("limit")
    scheduler.record_token("limit", 7)
    assert request.generated_tokens == [7]
    assert request.status == RequestStatus.FINISHED
    assert request.finish_reason == "length"


def test_metrics_collect_prompt_and_generation_statistics():
    metrics = MetricsRegistry()
    metric = metrics.start_request("x", prompt_tokens=4)
    metrics.mark_prefill("x")
    metrics.mark_token("x")
    metrics.mark_token("x")
    metrics.mark_complete("x")
    summary = metrics.summary()
    assert summary["requests"] == 1
    assert summary["completed"] == 1
    assert summary["prompt_tokens"] == 4
    assert summary["generated_tokens"] == 2
    assert metric.ttft_ms is not None


def test_kv_spec_accounts_for_keys_and_values():
    spec = KVSpec(num_layers=2, num_kv_heads=4, head_dim=8, block_size=16, dtype="float32")
    expected = 2 * 16 * 4 * 8 * 4 * 2
    assert spec.block_bytes == expected
