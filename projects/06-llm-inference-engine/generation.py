from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

from runtime import DecoderModel, ModelConfig, KVCache
from sampling import Sampler


@dataclass
class GenerationStats:
    prompt_tokens: int
    generated_tokens: int
    prefill_ms: float
    decode_ms: float
    total_ms: float
    kv_cache_bytes: int

    @property
    def decode_tokens_per_second(self) -> float:
        return self.generated_tokens / (self.decode_ms / 1000.0) if self.decode_ms else 0.0

    @property
    def end_to_end_tokens_per_second(self) -> float:
        total = self.prompt_tokens + self.generated_tokens
        return total / (self.total_ms / 1000.0) if self.total_ms else 0.0


def generate(
    model: DecoderModel,
    prompt_ids: list[int],
    max_new_tokens: int = 16,
    *,
    strategy: str = "greedy",
    sampler_kwargs: dict | None = None,
) -> tuple[list[int], GenerationStats]:
    if not prompt_ids:
        raise ValueError("prompt must contain at least one token")
    if len(prompt_ids) > model.config.max_seq_len:
        raise ValueError("prompt exceeds context window")

    ids = np.asarray(prompt_ids, dtype=np.int64)[None, :]
    t0 = time.perf_counter()
    logits, cache = model.prefill(ids)
    prefill_ms = (time.perf_counter() - t0) * 1000
    sampler = Sampler(seed=1234)
    sampler_kwargs = sampler_kwargs or {}
    generated: list[int] = []

    t1 = time.perf_counter()
    for _ in range(max_new_tokens):
        token = sampler(logits[0], strategy=strategy, **sampler_kwargs)
        generated.append(token)
        logits = model.decode(np.asarray([[token]], dtype=np.int64), cache)
    decode_ms = (time.perf_counter() - t1) * 1000
    total_ms = (time.perf_counter() - t0) * 1000
    stats = GenerationStats(
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(generated),
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        total_ms=total_ms,
        kv_cache_bytes=cache.bytes,
    )
    return generated, stats
