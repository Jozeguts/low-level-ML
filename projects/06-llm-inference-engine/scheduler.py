from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Iterable


class RequestStatus(str, Enum):
    QUEUED = "queued"
    PREFILL = "prefill"
    DECODING = "decoding"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class GenerationRequest:
    request_id: str
    prompt_tokens: list[int]
    max_new_tokens: int = 32
    priority: int = 0
    arrival_time: float = field(default_factory=monotonic)
    generated_tokens: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.QUEUED
    prompt_cursor: int = 0
    last_token: int | None = None
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id is required")
        if not self.prompt_tokens:
            raise ValueError("prompt_tokens cannot be empty")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_tokens)

    @property
    def generated_length(self) -> int:
        return len(self.generated_tokens)

    @property
    def total_length(self) -> int:
        return self.prompt_length + self.generated_length

    @property
    def done(self) -> bool:
        return self.status in {
            RequestStatus.FINISHED,
            RequestStatus.CANCELLED,
            RequestStatus.FAILED,
        }


@dataclass(frozen=True)
class SchedulerConfig:
    max_active_requests: int = 8
    max_batch_tokens: int = 256
    decode_reserve_tokens: int = 8
    prefill_first: bool = False


class ContinuousBatchScheduler:
    """Reference scheduler for dynamic prefill/decode batching.

    The scheduler owns request lifecycle and token budgets but deliberately
    delegates model execution to an external executor.
    """

    def __init__(self, config: SchedulerConfig = SchedulerConfig()):
        if config.max_active_requests <= 0 or config.max_batch_tokens <= 0:
            raise ValueError("scheduler limits must be positive")
        self.config = config
        self.pending: list[GenerationRequest] = []
        self.active: dict[str, GenerationRequest] = {}
        self.completed: dict[str, GenerationRequest] = {}

    def submit(self, request: GenerationRequest) -> None:
        if request.request_id in self.active or any(r.request_id == request.request_id for r in self.pending):
            raise ValueError(f"duplicate request id: {request.request_id}")
        self.pending.append(request)
        self.pending.sort(key=lambda r: (-r.priority, r.arrival_time))

    def cancel(self, request_id: str) -> bool:
        for index, request in enumerate(self.pending):
            if request.request_id == request_id:
                request.status = RequestStatus.CANCELLED
                request.finish_reason = "cancelled"
                self.completed[request_id] = self.pending.pop(index)
                return True
        request = self.active.get(request_id)
        if request is None or request.done:
            return False
        request.status = RequestStatus.CANCELLED
        request.finish_reason = "cancelled"
        self.completed[request_id] = self.active.pop(request_id)
        return True

    def admit(self, available_kv_slots: int | None = None) -> list[GenerationRequest]:
        capacity = self.config.max_active_requests - len(self.active)
        if capacity <= 0:
            return []
        admitted: list[GenerationRequest] = []
        for request in list(self.pending):
            if len(admitted) >= capacity:
                break
            if available_kv_slots is not None and request.prompt_length > available_kv_slots:
                continue
            self.pending.remove(request)
            request.status = RequestStatus.PREFILL
            self.active[request.request_id] = request
            admitted.append(request)
            if available_kv_slots is not None:
                available_kv_slots -= request.prompt_length
        return admitted

    def build_step(self, admit: bool = True) -> tuple[list[GenerationRequest], list[GenerationRequest]]:
        """Return `(prefill, decode)` work for one scheduler iteration."""
        if admit:
            self.admit()
        prefills = [r for r in self.active.values() if r.status == RequestStatus.PREFILL]
        decodes = [r for r in self.active.values() if r.status == RequestStatus.DECODING]

        if self.config.prefill_first:
            budget = self.config.max_batch_tokens
            selected_prefill = []
            for request in prefills:
                remaining = request.prompt_length - request.prompt_cursor
                if remaining <= 0:
                    request.status = RequestStatus.DECODING
                    decodes.append(request)
                    continue
                take = min(remaining, budget)
                if take == 0:
                    break
                selected_prefill.append(request)
                request.prompt_cursor += take
                budget -= take
                if budget == 0:
                    break
            return selected_prefill, decodes

        # Decode receives a small reserved token budget so active sequences
        # keep making progress while prompts are admitted.
        selected_decodes = decodes[:]
        decode_budget = min(len(selected_decodes), self.config.decode_reserve_tokens)
        selected_decodes = selected_decodes[:decode_budget] if decode_budget else []
        budget = self.config.max_batch_tokens - len(selected_decodes)
        selected_prefill = []
        for request in prefills:
            remaining = request.prompt_length - request.prompt_cursor
            if remaining <= 0:
                request.status = RequestStatus.DECODING
                selected_decodes.append(request)
                continue
            if budget <= 0:
                break
            take = min(remaining, budget)
            selected_prefill.append(request)
            request.prompt_cursor += take
            budget -= take
        return selected_prefill, selected_decodes

    def mark_prefill_complete(self, request_id: str) -> None:
        request = self.active[request_id]
        if request.prompt_cursor != request.prompt_length:
            raise ValueError("prefill is incomplete")
        request.status = RequestStatus.DECODING

    def record_token(self, request_id: str, token: int, eos_token: int | None = None) -> None:
        request = self.active[request_id]
        if request.status != RequestStatus.DECODING:
            raise ValueError("request is not in decode state")
        request.generated_tokens.append(int(token))
        request.last_token = int(token)
        if eos_token is not None and token == eos_token:
            self.finish(request_id, "eos")
        elif request.generated_length >= request.max_new_tokens:
            self.finish(request_id, "length")

    def finish(self, request_id: str, reason: str) -> GenerationRequest:
        request = self.active.pop(request_id)
        request.status = RequestStatus.FINISHED
        request.finish_reason = reason
        self.completed[request_id] = request
        return request

    def fail(self, request_id: str, reason: str) -> GenerationRequest:
        request = self.active.pop(request_id)
        request.status = RequestStatus.FAILED
        request.finish_reason = reason
        self.completed[request_id] = request
        return request

    def get(self, request_id: str) -> GenerationRequest:
        if request_id in self.active:
            return self.active[request_id]
        if request_id in self.completed:
            return self.completed[request_id]
        for request in self.pending:
            if request.request_id == request_id:
                return request
        raise KeyError(request_id)

    def snapshot(self) -> dict:
        return {
            "queued": len(self.pending),
            "active": len(self.active),
            "completed": len(self.completed),
            "active_ids": list(self.active),
            "queued_ids": [r.request_id for r in self.pending],
        }
