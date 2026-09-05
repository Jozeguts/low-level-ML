"""Simple token streaming interface.

A production server would connect this queue to an async transport such as
SSE or WebSocket. Here the iterator exposes the same per-request contract
without coupling the runtime to a network stack.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterator


@dataclass(frozen=True)
class TokenEvent:
    request_id: str
    token: int
    index: int
    final: bool = False


class TokenStreamer:
    def __init__(self) -> None:
        self._queues: Dict[str, Deque[TokenEvent]] = {}

    def open(self, request_id: str) -> None:
        if request_id in self._queues:
            raise ValueError(f"stream already exists: {request_id}")
        self._queues[request_id] = deque()

    def push(self, event: TokenEvent) -> None:
        if event.request_id not in self._queues:
            raise KeyError(event.request_id)
        self._queues[event.request_id].append(event)

    def drain(self, request_id: str) -> Iterator[TokenEvent]:
        queue = self._queues.get(request_id)
        if queue is None:
            raise KeyError(request_id)
        while queue:
            yield queue.popleft()

    def close(self, request_id: str) -> None:
        self._queues.pop(request_id, None)
