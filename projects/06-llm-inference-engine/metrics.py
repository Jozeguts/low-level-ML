from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class RequestMetrics:
    request_id: str
    queued_at: float = field(default_factory=perf_counter)
    prefill_started_at: float | None = None
    first_token_at: float | None = None
    completed_at: float | None = None
    prompt_tokens: int = 0
    generated_tokens: int = 0
    cancelled: bool = False

    @property
    def queue_ms(self) -> float | None:
        if self.prefill_started_at is None:
            return None
        return (self.prefill_started_at - self.queued_at) * 1000.0

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_at is None:
            return None
        return (self.first_token_at - self.queued_at) * 1000.0

    @property
    def e2e_ms(self) -> float | None:
        if self.completed_at is None:
            return None
        return (self.completed_at - self.queued_at) * 1000.0

    @property
    def decode_tps(self) -> float | None:
        if self.first_token_at is None or self.completed_at is None:
            return None
        seconds = self.completed_at - self.first_token_at
        if seconds <= 0:
            return None
        return self.generated_tokens / seconds


class MetricsRegistry:
    def __init__(self) -> None:
        self.requests: dict[str, RequestMetrics] = {}
        self.rejected = 0
        self.cancelled = 0

    def start_request(self, request_id: str, prompt_tokens: int) -> RequestMetrics:
        if request_id in self.requests:
            raise ValueError(f"duplicate metrics request: {request_id}")
        metric = RequestMetrics(request_id=request_id, prompt_tokens=prompt_tokens)
        self.requests[request_id] = metric
        return metric

    def mark_prefill(self, request_id: str) -> None:
        self.requests[request_id].prefill_started_at = perf_counter()

    def mark_token(self, request_id: str) -> None:
        metric = self.requests[request_id]
        if metric.first_token_at is None:
            metric.first_token_at = perf_counter()
        metric.generated_tokens += 1

    def mark_complete(self, request_id: str) -> None:
        self.requests[request_id].completed_at = perf_counter()

    def mark_cancelled(self, request_id: str) -> None:
        metric = self.requests[request_id]
        metric.cancelled = True
        metric.completed_at = perf_counter()
        self.cancelled += 1

    def summary(self) -> dict:
        completed = [m for m in self.requests.values() if m.completed_at is not None]
        total_generated = sum(m.generated_tokens for m in completed)
        total_prompt = sum(m.prompt_tokens for m in completed)
        ttft = [m.ttft_ms for m in completed if m.ttft_ms is not None]
        e2e = [m.e2e_ms for m in completed if m.e2e_ms is not None]
        tps = [m.decode_tps for m in completed if m.decode_tps is not None]
        return {
            "requests": len(self.requests),
            "completed": len(completed),
            "rejected": self.rejected,
            "cancelled": self.cancelled,
            "prompt_tokens": total_prompt,
            "generated_tokens": total_generated,
            "mean_ttft_ms": sum(ttft) / len(ttft) if ttft else None,
            "mean_e2e_ms": sum(e2e) / len(e2e) if e2e else None,
            "mean_decode_tokens_per_second": sum(tps) / len(tps) if tps else None,
        }
