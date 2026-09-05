"""Runtime metrics for Mini vLLM experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class RequestMetrics:
    request_id: str
    prompt_tokens: int
    generated_tokens: int
    ttft_steps: int | None
    total_steps: int | None


class MetricsCollector:
    def collect(self, engine) -> List[RequestMetrics]:
        records: List[RequestMetrics] = []
        for request in engine.scheduler.requests.values():
            result = engine._completed.get(request.request_id)
            generated = len(result.output) if result else len(request.generated)
            records.append(
                RequestMetrics(
                    request_id=request.request_id,
                    prompt_tokens=len(request.prompt),
                    generated_tokens=generated,
                    ttft_steps=(result.ttft_steps if result else None),
                    total_steps=(result.total_steps if result else None),
                )
            )
        return sorted(records, key=lambda r: r.request_id)

    def summary(self, engine) -> Dict[str, float | int]:
        records = self.collect(engine)
        completed = [r for r in records if r.total_steps is not None]
        generated = sum(r.generated_tokens for r in records)
        prompts = sum(r.prompt_tokens for r in records)
        ttft = [r.ttft_steps for r in completed if r.ttft_steps is not None]
        steps = engine.scheduler.step_id
        cache = engine.cache.stats()
        return {
            "submitted": len(records),
            "completed": len(completed),
            "prompt_tokens": prompts,
            "generated_tokens": generated,
            "scheduler_steps": steps,
            "generated_tokens_per_step": generated / steps if steps else 0.0,
            "ttft_p50_steps": sorted(ttft)[len(ttft) // 2] if ttft else 0,
            "cache_utilization": cache.utilization,
            "internal_waste_tokens": cache.internal_waste_tokens,
        }
