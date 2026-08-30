from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from runtime import DecoderModel, KVCache
from sampling import Sampler


@dataclass
class RequestState:
    request_id: str
    prompt: list[int]
    generated: list[int]
    cache: KVCache | None = None
    finished: bool = False


class BatchRunner:
    """Simple static batch baseline before continuous batching in Project 07."""

    def __init__(self, model: DecoderModel):
        self.model = model

    def prefill(self, requests: list[RequestState]) -> np.ndarray:
        if not requests:
            raise ValueError("batch must not be empty")
        lengths = {len(r.prompt) for r in requests}
        if len(lengths) != 1:
            raise ValueError("baseline runner requires equal prompt lengths")
        ids = np.asarray([r.prompt for r in requests], dtype=np.int64)
        logits, cache = self.model.prefill(ids)
        # Keep one cache per request. A production scheduler would manage
        # blocks and sequences independently instead of copying a batch cache.
        for i, request in enumerate(requests):
            request.cache = self._split_cache(cache, i)
        return logits

    def _split_cache(self, batch_cache: KVCache, index: int) -> KVCache:
        cache = KVCache(self.model.config, batch_size=1)
        cache.length = batch_cache.length
        for layer in range(self.model.config.num_layers):
            cache.keys[layer][0] = batch_cache.keys[layer][index]
            cache.values[layer][0] = batch_cache.values[layer][index]
        return cache

    def decode_once(self, requests: list[RequestState], logits: np.ndarray, sampler: Sampler) -> np.ndarray:
        tokens = []
        active = []
        for i, request in enumerate(requests):
            if request.finished:
                tokens.append(0)
                continue
            token = sampler(logits[i], strategy="greedy")
            request.generated.append(token)
            tokens.append(token)
            active.append(request)
        if len(active) != len(requests):
            raise ValueError("static baseline does not support mixed finished states")
        next_logits = []
        for token, request in zip(tokens, requests):
            next_logits.append(self.model.decode(np.asarray([[token]]), request.cache))
        return np.concatenate(next_logits, axis=0)
