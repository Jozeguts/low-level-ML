import numpy as np
import pytest

from engine import EngineConfig, GenerationRequest, InferenceEngine
from runtime import DecoderModel, ModelConfig


@pytest.fixture
def engine():
    cfg = ModelConfig(vocab_size=32, hidden_size=32, num_layers=1,
                      num_heads=4, num_kv_heads=2, intermediate_size=64,
                      max_seq_len=64)
    return InferenceEngine(DecoderModel(cfg), EngineConfig(max_batch_size=4,
                                                            max_batch_tokens=64,
                                                            max_context_tokens=64), seed=7)


def test_request_validation():
    with pytest.raises(ValueError):
        GenerationRequest(np.array([], dtype=np.int64))
    with pytest.raises(ValueError):
        GenerationRequest(np.array([1, 2]), max_new_tokens=0)


def test_single_request_completes(engine):
    req = GenerationRequest(np.array([1, 2, 3], dtype=np.int64), max_new_tokens=4, request_id="r1")
    engine.admit(req)
    result = engine.run_until_complete()
    assert "r1" in result
    assert result["r1"].finished
    assert 1 <= len(result["r1"].generated) <= 4
    assert result["r1"].ttft_ms is not None


def test_multiple_requests_share_engine(engine):
    for i, length in enumerate((2, 5, 7)):
        req = GenerationRequest(np.arange(1, length + 1, dtype=np.int64),
                                max_new_tokens=3, request_id=f"r{i}")
        engine.admit(req)
    engine.run_until_complete()
    assert len(engine.completed) == 3
    assert engine.metrics["submitted"] == 3
    assert engine.metrics["completed"] == 3


def test_cancelled_request(engine):
    req = GenerationRequest(np.array([1, 2, 3], dtype=np.int64), max_new_tokens=8, request_id="cancel")
    engine.admit(req)
    assert engine.cancel("cancel")
    engine.decode_step()
    assert engine.completed["cancel"].cancelled
    assert engine.metrics["cancelled"] == 1


def test_context_limit_is_enforced(engine):
    req = GenerationRequest(np.arange(1, 9, dtype=np.int64), max_new_tokens=60)
    with pytest.raises(ValueError):
        engine.admit(req)
    assert engine.metrics["rejected"] == 1


def test_generation_is_deterministic_for_greedy(engine):
    prompt = np.array([2, 4, 6, 8], dtype=np.int64)
    a = GenerationRequest(prompt, max_new_tokens=4, request_id="a")
    engine.admit(a)
    engine.run_until_complete()

    cfg = engine.model.config
    second = InferenceEngine(DecoderModel(cfg, engine.model.weights),
                             engine.config, seed=999)
    b = GenerationRequest(prompt, max_new_tokens=4, request_id="b")
    second.admit(b)
    second.run_until_complete()
    assert a.generated == b.generated


def test_batched_snapshot(engine):
    req = GenerationRequest(np.array([1, 2]), max_new_tokens=2, request_id="snap")
    engine.admit(req)
    snap = engine.snapshot()
    assert snap["active"] == 1
    assert snap["metrics"]["submitted"] == 1
    assert snap["requests"]["snap"]["prompt_tokens"] == 2
