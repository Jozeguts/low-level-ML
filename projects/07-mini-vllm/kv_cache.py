"""Paged KV-cache simulator for Mini vLLM.

This module models the memory-management layer of an LLM serving runtime.
It intentionally stores token payloads rather than tensors so the allocator,
block table, fragmentation, and accounting behavior can be tested cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


class KVCacheError(RuntimeError):
    """Base cache error."""


class OutOfKVBlocks(KVCacheError):
    """Raised when the physical block pool cannot satisfy an allocation."""


@dataclass(frozen=True)
class CacheStats:
    total_blocks: int
    free_blocks: int
    used_blocks: int
    block_size: int
    stored_tokens: int
    capacity_tokens: int

    @property
    def utilization(self) -> float:
        return self.used_blocks / self.total_blocks if self.total_blocks else 0.0

    @property
    def internal_waste_tokens(self) -> int:
        return self.used_blocks * self.block_size - self.stored_tokens


class PagedKVCache:
    """Fixed-size physical block pool with per-request block tables.

    A request owns a logical sequence of tokens but its KV state is split over
    arbitrary physical blocks. The last block may be partially occupied.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: List[int] = list(range(num_blocks - 1, -1, -1))
        self._tables: Dict[str, List[int]] = {}
        self._lengths: Dict[str, int] = {}
        self._tokens: Dict[int, List[int]] = {}

    def _required_blocks(self, token_count: int) -> int:
        return (token_count + self.block_size - 1) // self.block_size

    def allocate(self, request_id: str, token_count: int) -> List[int]:
        if request_id in self._tables:
            raise KVCacheError(f"request already allocated: {request_id}")
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        needed = self._required_blocks(token_count)
        if needed > len(self._free):
            raise OutOfKVBlocks(
                f"need {needed} blocks, only {len(self._free)} free"
            )
        blocks = [self._free.pop() for _ in range(needed)]
        self._tables[request_id] = blocks
        self._lengths[request_id] = token_count
        for block in blocks:
            self._tokens[block] = []
        return blocks

    def ensure_capacity(self, request_id: str, token_count: int) -> None:
        if request_id not in self._tables:
            raise KeyError(request_id)
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        needed = self._required_blocks(token_count)
        current = len(self._tables[request_id])
        extra = needed - current
        if extra <= 0:
            return
        if extra > len(self._free):
            raise OutOfKVBlocks(
                f"request {request_id} needs {extra} additional blocks"
            )
        self._tables[request_id].extend(self._free.pop() for _ in range(extra))
        for block in self._tables[request_id][-extra:]:
            self._tokens[block] = []

    def append(self, request_id: str, token: int) -> Tuple[int, int]:
        if request_id not in self._tables:
            raise KeyError(request_id)
        new_length = self._lengths[request_id] + 1
        self.ensure_capacity(request_id, new_length)
        logical_pos = self._lengths[request_id]
        block_index = logical_pos // self.block_size
        offset = logical_pos % self.block_size
        physical_block = self._tables[request_id][block_index]
        payload = self._tokens[physical_block]
        if offset != len(payload):
            raise KVCacheError("block table and payload are inconsistent")
        payload.append(int(token))
        self._lengths[request_id] = new_length
        return physical_block, offset

    def append_many(self, request_id: str, tokens: Iterable[int]) -> None:
        values = list(tokens)
        if not values:
            return
        self.ensure_capacity(request_id, self._lengths[request_id] + len(values))
        for token in values:
            self.append(request_id, token)

    def read(self, request_id: str) -> List[int]:
        if request_id not in self._tables:
            raise KeyError(request_id)
        result: List[int] = []
        for block in self._tables[request_id]:
            result.extend(self._tokens[block])
        return result[: self._lengths[request_id]]

    def block_table(self, request_id: str) -> Tuple[int, ...]:
        if request_id not in self._tables:
            raise KeyError(request_id)
        return tuple(self._tables[request_id])

    def release(self, request_id: str) -> None:
        blocks = self._tables.pop(request_id, None)
        if blocks is None:
            return
        self._lengths.pop(request_id, None)
        for block in blocks:
            self._tokens.pop(block, None)
            self._free.append(block)

    def length(self, request_id: str) -> int:
        return self._lengths[request_id]

    def stats(self) -> CacheStats:
        used = self.num_blocks - len(self._free)
        stored = sum(self._lengths.values())
        return CacheStats(
            total_blocks=self.num_blocks,
            free_blocks=len(self._free),
            used_blocks=used,
            block_size=self.block_size,
            stored_tokens=stored,
            capacity_tokens=self.num_blocks * self.block_size,
        )

    def snapshot(self) -> dict:
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "free_blocks": len(self._free),
            "requests": {
                rid: {"length": self._lengths[rid], "blocks": list(blocks)}
                for rid, blocks in self._tables.items()
            },
        }
