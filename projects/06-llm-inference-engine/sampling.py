from __future__ import annotations

import numpy as np


def _stable_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    x = logits.astype(np.float64) / temperature
    x -= np.max(x)
    p = np.exp(x)
    return p / p.sum()


def greedy(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def top_k_top_p_sample(
    logits: np.ndarray,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    logits = np.asarray(logits, dtype=np.float64).copy()
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if rng is None:
        rng = np.random.default_rng()

    if top_k:
        k = min(top_k, logits.size)
        keep = np.argpartition(logits, -k)[-k:]
        mask = np.ones(logits.size, dtype=bool)
        mask[keep] = False
        logits[mask] = -np.inf

    probs = _stable_probs(logits, temperature)
    if top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        remove = cumulative > top_p
        if remove.size:
            remove[0] = False
        logits[order[remove]] = -np.inf
        probs = _stable_probs(logits, temperature)

    return int(rng.choice(logits.size, p=probs))


class Sampler:
    def __init__(self, seed: int = 1234):
        self.rng = np.random.default_rng(seed)

    def __call__(self, logits: np.ndarray, strategy: str = "greedy", **kwargs) -> int:
        if strategy == "greedy":
            return greedy(logits)
        if strategy in {"sample", "top_k_top_p"}:
            return top_k_top_p_sample(logits, rng=self.rng, **kwargs)
        raise ValueError(f"unknown sampling strategy: {strategy}")
