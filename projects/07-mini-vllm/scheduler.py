"""Continuous-batching scheduler for the Mini vLLM simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from kv_cache import OutOfKVBlocks, PagedKVCache


class RequestState(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Request:
    request_id: str
    prompt: List[int]
    max_new_tokens: int
    priority: int = 0
    generated: List[int] = field(default_factory=list)
    state: RequestState = RequestState.WAITING
    prefill_cursor: int = 0
    arrival_step: int = 0
    first_token_step: Optional[int] = None
    finish_step: Optional[int] = None
    cancel_reason: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return len(self.prompt) + len(self.generated)

    @property
    def remaining_prompt(self) -> int:
        return len(self.prompt) - self.prefill_cursor

    @property
    def remaining_generation(self) -> int:
        return self.max_new_tokens - len(self.generated)


class ContinuousBatchScheduler:
    """Token-budgeted scheduler that mixes active decode and waiting prefill."""

    def __init__(
        self,
        cache: PagedKVCache,
        max_requests: int = 8,
        max_batch_tokens: int = 64,
        prefill_chunk: int = 16,
    ) -> None:
        if max_requests <= 0 or max_batch_tokens <= 0 or prefill_chunk <= 0:
            raise ValueError("scheduler limits must be positive")
        self.cache = cache
        self.max_requests = max_requests
        self.max_batch_tokens = max_batch_tokens
        self.prefill_chunk = prefill_chunk
        self.requests: Dict[str, Request] = {}
        self.step_id = 0
        self.rejected = 0

    def submit(self, request: Request) -> bool:
        if request.request_id in self.requests:
            raise ValueError(f"duplicate request id: {request.request_id}")
        if not request.prompt:
            raise ValueError("prompt must not be empty")
        if request.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.requests[request.request_id] = request
        request.arrival_step = self.step_id
        return True

    def cancel(self, request_id: str, reason: str = "client_cancelled") -> bool:
        request = self.requests.get(request_id)
        if request is None or request.state in {
            RequestState.FINISHED,
            RequestState.CANCELLED,
            RequestState.REJECTED,
        }:
            return False
        request.state = RequestState.CANCELLED
        request.cancel_reason = reason
        self.cache.release(request_id)
        return True

    def _active(self) -> List[Request]:
        return [
            r for r in self.requests.values()
            if r.state in {RequestState.PREFILL, RequestState.DECODE}
        ]

    def _waiting(self) -> List[Request]:
        return [r for r in self.requests.values() if r.state == RequestState.WAITING]

    def _ordered(self, requests: List[Request]) -> List[Request]:
        return sorted(requests, key=lambda r: (-r.priority, r.arrival_step, r.request_id))

    def _admit(self, request: Request) -> bool:
        try:
            self.cache.allocate(request.request_id, min(len(request.prompt), self.prefill_chunk))
        except OutOfKVBlocks:
            request.state = RequestState.REJECTED
            request.cancel_reason = "kv_capacity"
            self.rejected += 1
            return False
        request.state = RequestState.PREFILL
        return True

    def plan(self) -> dict:
        """Return work for the next step without mutating token contents."""
        slots = self.max_requests
        budget = self.max_batch_tokens
        decode: List[str] = []
        prefill: List[str] = []

        active = self._ordered(self._active())
        for request in active:
            if slots == 0:
                break
            if request.state == RequestState.DECODE and budget >= 1:
                decode.append(request.request_id)
                budget -= 1
                slots -= 1

        for request in self._ordered(self._waiting()):
            if slots == 0 or budget == 0:
                break
            chunk = min(request.remaining_prompt, self.prefill_chunk, budget)
            if chunk <= 0:
                continue
            prefill.append(request.request_id)
            budget -= chunk
            slots -= 1

        for request in active:
            if request.state == RequestState.PREFILL and slots and budget:
                chunk = min(request.remaining_prompt, self.prefill_chunk, budget)
                if chunk > 0:
                    prefill.append(request.request_id)
                    budget -= chunk
                    slots -= 1

        return {"decode": decode, "prefill": prefill, "unused_token_budget": budget}

    def start_prefill(self, request_id: str) -> Request:
        request = self.requests[request_id]
        if request.state != RequestState.WAITING:
            return request
        self._admit(request)
        return request

    def apply_prefill(self, request_id: str, token_count: int) -> Request:
        request = self.requests[request_id]
        if request.state != RequestState.PREFILL:
            raise ValueError("request is not in prefill")
        token_count = min(token_count, request.remaining_prompt)
        request.prefill_cursor += token_count
        self.cache.ensure_capacity(request_id, request.prefill_cursor + len(request.generated))
        if request.prefill_cursor == len(request.prompt):
            request.state = RequestState.DECODE
        return request

    def append_token(self, request_id: str, token: int) -> Request:
        request = self.requests[request_id]
        if request.state != RequestState.DECODE:
            raise ValueError("request is not decoding")
        self.cache.append(request_id, token)
        request.generated.append(int(token))
        if request.first_token_step is None:
            request.first_token_step = self.step_id
        if len(request.generated) >= request.max_new_tokens:
            request.state = RequestState.FINISHED
            request.finish_step = self.step_id
            self.cache.release(request_id)
        return request

    def tick(self) -> None:
        self.step_id += 1

    def completed(self) -> List[Request]:
        return [r for r in self.requests.values() if r.state == RequestState.FINISHED]

    def snapshot(self) -> dict:
        return {
            "step": self.step_id,
            "states": {
                state.value: sum(r.state == state for r in self.requests.values())
                for state in RequestState
            },
            "cache": self.cache.stats().__dict__,
        }
