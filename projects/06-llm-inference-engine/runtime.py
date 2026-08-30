from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import math
import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 128
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 4
    intermediate_size: int = 128
    max_seq_len: int = 256
    dtype: str = "float32"

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must divide evenly across heads")
        return self.hidden_size // self.num_heads


class KVCache:
    def __init__(self, config: ModelConfig, batch_size: int = 1):
        if config.num_heads % config.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.config = config
        self.batch_size = batch_size
        shape = (batch_size, config.num_kv_heads, config.max_seq_len, config.head_dim)
        self.keys = [np.zeros(shape, dtype=config.dtype) for _ in range(config.num_layers)]
        self.values = [np.zeros(shape, dtype=config.dtype) for _ in range(config.num_layers)]
        self.length = 0

    @property
    def bytes(self) -> int:
        return sum(x.nbytes + y.nbytes for x, y in zip(self.keys, self.values))

    def append(self, layer: int, key: np.ndarray, value: np.ndarray) -> None:
        n = key.shape[2]
        end = self.length + n
        if end > self.config.max_seq_len:
            raise ValueError("KV cache capacity exceeded")
        self.keys[layer][:, :, self.length:end, :] = key
        self.values[layer][:, :, self.length:end, :] = value

    def view(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        return self.keys[layer][:, :, : self.length, :], self.values[layer][:, :, : self.length, :]

    def advance(self, tokens: int) -> None:
        if self.length + tokens > self.config.max_seq_len:
            raise ValueError("sequence exceeds context window")
        self.length += tokens


class Weights:
    def __init__(self, config: ModelConfig, seed: int = 7):
        rng = np.random.default_rng(seed)
        h, i, v = config.hidden_size, config.intermediate_size, config.vocab_size
        scale = 1.0 / math.sqrt(h)
        self.token_embedding = (rng.standard_normal((v, h)) * scale).astype(config.dtype)
        self.layers: List[Dict[str, np.ndarray]] = []
        for _ in range(config.num_layers):
            self.layers.append({
                "q": (rng.standard_normal((h, h)) * scale).astype(config.dtype),
                "k": (rng.standard_normal((h, h)) * scale).astype(config.dtype),
                "v": (rng.standard_normal((h, h)) * scale).astype(config.dtype),
                "o": (rng.standard_normal((h, h)) * scale).astype(config.dtype),
                "gate": (rng.standard_normal((h, i)) * scale).astype(config.dtype),
                "up": (rng.standard_normal((h, i)) * scale).astype(config.dtype),
                "down": (rng.standard_normal((i, h)) * scale).astype(config.dtype),
                "norm1": np.ones(h, dtype=config.dtype),
                "norm2": np.ones(h, dtype=config.dtype),
            })
        self.final_norm = np.ones(h, dtype=config.dtype)
        self.lm_head = (rng.standard_normal((h, v)) * scale).astype(config.dtype)

    def bytes(self) -> int:
        total = self.token_embedding.nbytes + self.final_norm.nbytes + self.lm_head.nbytes
        for layer in self.layers:
            total += sum(x.nbytes for x in layer.values())
        return total


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    variance = np.mean(x * x, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(variance + eps)) * weight


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


class DecoderModel:
    def __init__(self, config: ModelConfig, weights: Weights | None = None):
        self.config = config
        self.weights = weights or Weights(config)

    def parameter_bytes(self) -> int:
        return self.weights.bytes()

    def _attention(self, x: np.ndarray, layer: int, cache: KVCache) -> np.ndarray:
        cfg, w = self.config, self.weights.layers[layer]
        b, s, h = x.shape
        hd = cfg.head_dim
        q = x @ w["q"].reshape(h, cfg.num_heads, hd)
        k = x @ w["k"].reshape(h, cfg.num_kv_heads, hd)
        v = x @ w["v"].reshape(h, cfg.num_kv_heads, hd)
        q = np.transpose(q, (0, 2, 1, 3))
        k = np.transpose(k, (0, 2, 1, 3))
        v = np.transpose(v, (0, 2, 1, 3))
        cache.append(layer, k, v)
        keys = np.concatenate([cache.keys[layer][:, :, : cache.length + s, :]], axis=2)
        vals = np.concatenate([cache.values[layer][:, :, : cache.length + s, :]], axis=2)
        # Repeat KV heads for ordinary multi-head attention.
        repeat = cfg.num_heads // cfg.num_kv_heads
        keys = np.repeat(keys, repeat, axis=1)
        vals = np.repeat(vals, repeat, axis=1)
        scores = np.matmul(q, np.transpose(keys, (0, 1, 3, 2))) / math.sqrt(hd)
        causal = np.triu(np.ones((s, keys.shape[2]), dtype=bool), k=keys.shape[2] - s + 1)
        scores = np.where(causal[None, None, :, :], -1e9, scores)
        probs = softmax(scores, axis=-1)
        out = np.matmul(probs, vals)
        out = np.transpose(out, (0, 2, 1, 3)).reshape(b, s, h)
        return out @ w["o"]

    def _block(self, x: np.ndarray, layer: int, cache: KVCache) -> np.ndarray:
        w = self.weights.layers[layer]
        a = rmsnorm(x, w["norm1"])
        x = x + self._attention(a, layer, cache)
        m = rmsnorm(x, w["norm2"])
        hidden = silu(m @ w["gate"]) * (m @ w["up"])
        return x + hidden @ w["down"]

    def forward(self, token_ids: np.ndarray, cache: KVCache) -> np.ndarray:
        token_ids = np.asarray(token_ids, dtype=np.int64)
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if np.any(token_ids < 0) or np.any(token_ids >= self.config.vocab_size):
            raise ValueError("token id outside vocabulary")
        x = self.weights.token_embedding[token_ids]
        for layer in range(self.config.num_layers):
            x = self._block(x, layer, cache)
        x = rmsnorm(x, self.weights.final_norm)
        return x @ self.weights.lm_head

    def prefill(self, token_ids: np.ndarray) -> tuple[np.ndarray, KVCache]:
        cache = KVCache(self.config, batch_size=token_ids.shape[0])
        logits = self.forward(token_ids, cache)
        cache.advance(token_ids.shape[1])
        return logits[:, -1, :], cache

    def decode(self, token_ids: np.ndarray, cache: KVCache) -> np.ndarray:
        logits = self.forward(token_ids, cache)
        cache.advance(token_ids.shape[1])
        return logits[:, -1, :]
