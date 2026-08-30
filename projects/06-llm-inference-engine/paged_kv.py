from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

import numpy as np


@dataclass(frozen=True)
class KVSpec:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    block_size: int = 16
    dtype: str = "float32"

    @property
    def block_bytes(self) -> int:
        values = self.block_size * self.num_kv_heads * self.head_dim
        return 2 * values * np.dtype(self.dtype).itemsize * self.num_layers


@dataclass
class SequenceBlocks:
    request_id: str
    token_count: int = 0
    blocks: list[int] = field(default_factory=list)


class PagedKVCache:
    """CPU reference model of a paged KV allocator.

    Physical blocks are shared by all requests. A request owns a logical
    block table, so variable-length sequences do not need contiguous storage.
    """

    def __init__(self, spec: KVSpec, num_blocks: int):
        if spec.block_size <= 0 or num_blocks <= 0:
            raise ValueError("block_size and num_blocks must be positive")
        self.spec = spec
        self.num_blocks = num_blocks
        shape = (num_blocks, spec.num_kv_heads, spec.block_size, spec.head_dim)
        self.keys = [np.zeros(shape, dtype=spec.dtype) for _ in range(spec.num_layers)]
        self.values = [np.zeros(shape, dtype=spec.dtype) for _ in range(spec.num_layers)]
        self.free: set[int] = set(range(num_blocks))
        self.sequences: Dict[str, SequenceBlocks] = {}

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - len(self.free)

    @property
    def free_blocks(self) -> int:
        return len(self.free)

    @property
    def bytes(self) -> int:
        return self.used_blocks * self.spec.block_bytes

    def required_blocks(self, tokens: int) -> int:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        return (tokens + self.spec.block_size - 1) // self.spec.block_size

    def allocate(self, request_id: str, tokens: int = 0) -> SequenceBlocks:
        if request_id in self.sequences:
            raise ValueError(f"request already allocated: {request_id}")
        blocks_needed = self.required_blocks(tokens)
        if blocks_needed > self.free_blocks:
            raise MemoryError("insufficient KV blocks")
        blocks = [self.free.pop() for _ in range(blocks_needed)]
        state = SequenceBlocks(request_id=request_id, token_count=tokens, blocks=blocks)
        self.sequences[request_id] = state
        return state

    def ensure_capacity(self, request_id: str, tokens: int) -> SequenceBlocks:
        state = self._get(request_id)
        needed = self.required_blocks(tokens)
        extra = needed - len(state.blocks)
        if extra <= 0:
            state.token_count = max(state.token_count, tokens)
            return state
        if extra > self.free_blocks:
            raise MemoryError("insufficient KV blocks")
        state.blocks.extend(self.free.pop() for _ in range(extra))
        state.token_count = max(state.token_count, tokens)
        return state

    def append_tokens(self, request_id: str, tokens: int) -> SequenceBlocks:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        state = self._get(request_id)
        return self.ensure_capacity(request_id, state.token_count + tokens)

    def release(self, request_id: str) -> None:
        state = self.sequences.pop(request_id, None)
        if state is None:
            return
        self.free.update(state.blocks)

    def block_table(self, request_id: str) -> tuple[int, ...]:
        return tuple(self._get(request_id).blocks)

    def logical_position(self, request_id: str, token_index: int) -> tuple[int, int]:
        state = self._get(request_id)
        if token_index < 0 or token_index >= state.token_count:
            raise IndexError("token index outside sequence")
        block_index, offset = divmod(token_index, self.spec.block_size)
        return state.blocks[block_index], offset

    def write(self, request_id: str, layer: int, key: np.ndarray, value: np.ndarray, start: int) -> None:
        if layer < 0 or layer >= self.spec.num_layers:
            raise IndexError("layer outside cache")
        if key.shape != value.shape:
            raise ValueError("key/value shapes must match")
        if key.ndim != 3:
            raise ValueError("key/value must have [kv_heads, tokens, head_dim]")
        if key.shape[0] != self.spec.num_kv_heads or key.shape[2] != self.spec.head_dim:
            raise ValueError("key/value shape does not match KV specification")
        end = start + key.shape[1]
        self.ensure_capacity(request_id, end)
        for local, token in enumerate(range(start, end)):
            block, offset = self.logical_position(request_id, token)
            self.keys[layer][block, :, offset, :] = key[:, local, :]
            self.values[layer][block, :, offset, :] = value[:, local, :]

    def read(self, request_id: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
        state = self._get(request_id)
        k = np.empty((self.spec.num_kv_heads, state.token_count, self.spec.head_dim), dtype=self.spec.dtype)
        v = np.empty_like(k)
        for token in range(state.token_count):
            block, offset = self.logical_position(request_id, token)
            k[:, token, :] = self.keys[layer][block, :, offset, :]
            v[:, token, :] = self.values[layer][block, :, offset, :]
        return k, v

    def _get(self, request_id: str) -> SequenceBlocks:
        try:
            return self.sequences[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown request: {request_id}") from exc

    def snapshot(self) -> dict:
        return {
            "total_blocks": self.num_blocks,
            "used_blocks": self.used_blocks,
            "free_blocks": self.free_blocks,
            "block_size": self.spec.block_size,
            "bytes": self.bytes,
            "requests": {
                key: {"tokens": value.token_count, "blocks": list(value.blocks)}
                for key, value in self.sequences.items()
            },
        }
