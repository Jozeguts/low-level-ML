from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory import memory_report
from runtime import DecoderModel, ModelConfig, Weights


def run(prompt_len: int, new_tokens: int) -> dict:
    cfg = ModelConfig(vocab_size=256, hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=128, max_seq_len=512)
    model = DecoderModel(cfg, Weights(cfg, seed=17))
    prompt = np.arange(prompt_len, dtype=np.int64)[None, :] % cfg.vocab_size

    t0 = time.perf_counter()
    logits, cache = model.prefill(prompt)
    prefill_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    for _ in range(new_tokens):
        token = np.argmax(logits, axis=-1).astype(np.int64)[:, None]
        logits = model.decode(token, cache)
    decode_ms = (time.perf_counter() - t1) * 1000

    report = memory_report(model, 1, prompt_len + new_tokens)
    return {
        "prompt_tokens": prompt_len,
        "generated_tokens": new_tokens,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "decode_ms_per_token": decode_ms / new_tokens,
        "decode_tokens_per_second": new_tokens / (decode_ms / 1000.0),
        "kv_cache_bytes": cache.bytes,
        **report,
    }


if __name__ == "__main__":
    results = [run(prompt, 16) for prompt in (8, 32, 128)]
    print(json.dumps(results, indent=2))
