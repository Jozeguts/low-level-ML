import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime import DecoderModel, KVCache, ModelConfig, Weights, softmax
from sampling import Sampler, greedy, top_k_top_p_sample
from memory import kv_cache_bytes


def model():
    cfg = ModelConfig(vocab_size=32, hidden_size=16, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=32, max_seq_len=32)
    return DecoderModel(cfg, Weights(cfg, seed=11))


def test_softmax_is_normalized():
    p = softmax(np.array([[1.0, 2.0, 3.0]]))
    assert np.isclose(p.sum(), 1.0)
    assert np.all(p > 0)


def test_prefill_populates_cache():
    m = model()
    logits, cache = m.prefill(np.array([[1, 2, 3, 4]]))
    assert logits.shape == (1, 32)
    assert cache.length == 4
    assert cache.keys[0].shape == (1, 2, 32, 4)


def test_decode_grows_cache_by_one():
    m = model()
    _, cache = m.prefill(np.array([[1, 2, 3]]))
    assert cache.length == 3
    logits = m.decode(np.array([[4]]), cache)
    assert logits.shape == (1, 32)
    assert cache.length == 4


def test_prefill_and_incremental_logits_match():
    m = model()
    prompt = np.array([[1, 2, 3, 4]])
    full_cache = KVCache(m.config, 1)
    full = m.forward(prompt, full_cache)[:, -1, :]
    full_cache.advance(prompt.shape[1])

    _, incremental_cache = m.prefill(prompt[:, :3])
    inc = m.decode(prompt[:, 3:], incremental_cache)
    np.testing.assert_allclose(full, inc, rtol=1e-5, atol=1e-5)


def test_cache_overflow_is_rejected():
    m = model()
    _, cache = m.prefill(np.ones((1, 32), dtype=np.int64))
    with pytest.raises(ValueError):
        m.decode(np.array([[1]]), cache)


def test_sampling_is_seeded():
    logits = np.linspace(-2, 2, 32)
    a = Sampler(seed=99)(logits, strategy="sample", temperature=0.8, top_k=8, top_p=0.9)
    b = Sampler(seed=99)(logits, strategy="sample", temperature=0.8, top_k=8, top_p=0.9)
    assert a == b


def test_greedy_selects_maximum():
    assert greedy(np.array([0.0, 5.0, 2.0])) == 1


def test_top_k_limits_candidates():
    logits = np.array([10.0, 9.0, 8.0, -100.0])
    rng = np.random.default_rng(3)
    for _ in range(30):
        token = top_k_top_p_sample(logits, top_k=2, rng=rng)
        assert token in (0, 1)


def test_kv_memory_formula():
    m = model()
    expected = 2 * 1 * 2 * 2 * 10 * 4 * 4
    assert kv_cache_bytes(m.config, 1, 10) == expected
