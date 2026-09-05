"""End-to-end Mini vLLM continuous-batching simulator.

The model is deterministic. The engineering target is the serving runtime:
dynamic batch membership, KV block growth, scheduling and latency accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from kv_cache import PagedKVCache
from scheduler import ContinuousBatchScheduler, Request, RequestState


@dataclass(frozen=True)
class EngineConfig:
    num_blocks: int = 128
    block_size: int = 8
    max_requests: int = 8
    max_batch_tokens: int = 32
    prefill_chunk: int = 8
    vocab_size: int = 256


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    output: List[int]
    state: str
    ttft_steps: int | None
    total_steps: int | None


class MiniVLLM:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.cache = PagedKVCache(self.config.num_blocks, self.config.block_size)
        self.scheduler = ContinuousBatchScheduler(
            self.cache,
            max_requests=self.config.max_requests,
            max_batch_tokens=self.config.max_batch_tokens,
            prefill_chunk=self.config.prefill_chunk,
        )
        self._completed: Dict[str, RequestResult] = {}

    def submit(self, request_id: str, prompt: Iterable[int], max_new_tokens: int, priority: int = 0) -> None:
        request = Request(
            request_id=request_id,
            prompt=[int(x) % self.config.vocab_size for x in prompt],
            max_new_tokens=max_new_tokens,
            priority=priority,
        )
        self.scheduler.submit(request)

    def _next_token(self, request: Request) -> int:
        """Deterministic stand-in for a model's next-token logits."""
        previous = request.prompt[-1] if not request.generated else request.generated[-1]
        position = request.total_tokens
        return (previous * 31 + position * 17 + len(request.request_id)) % self.config.vocab_size

    def _finish(self, request: Request) -> None:
        if request.request_id in self._completed:
            return
        self._completed[request.request_id] = RequestResult(
            request_id=request.request_id,
            output=list(request.generated),
            state=request.state.value,
            ttft_steps=(request.first_token_step - request.arrival_step)
            if request.first_token_step is not None else None,
            total_steps=(request.finish_step - request.arrival_step)
            if request.finish_step is not None else None,
        )

    def step(self) -> dict:
        """Execute one continuous-batching scheduler iteration."""
        plan = self.scheduler.plan()

        for item in plan["prefill"]:
            request_id = item["request_id"]
            request = self.scheduler.requests[request_id]
            if request.state == RequestState.WAITING:
                self.scheduler.start_prefill(request_id)
            request = self.scheduler.requests[request_id]
            if request.state == RequestState.PREFILL:
                self.scheduler.apply_prefill(request_id, item["tokens"])

        for request_id in plan["decode"]:
            request = self.scheduler.requests[request_id]
            if request.state != RequestState.DECODE:
                continue
            self.scheduler.append_token(request_id, self._next_token(request))
            if request.state == RequestState.FINISHED:
                self._finish(request)

        self.scheduler.tick()
        return {
            "step": self.scheduler.step_id,
            "decode": list(plan["decode"]),
            "prefill": list(plan["prefill"]),
            "cache": self.cache.stats().__dict__,
        }

    def run(self, max_steps: int = 10000) -> List[RequestResult]:
        for _ in range(max_steps):
            unfinished = [
                r for r in self.scheduler.requests.values()
                if r.state in {RequestState.WAITING, RequestState.PREFILL, RequestState.DECODE}
            ]
            if not unfinished:
                break
            self.step()
        else:
            raise TimeoutError("engine did not finish within max_steps")

        for request in self.scheduler.completed():
            self._finish(request)
        return [self._completed[rid] for rid in sorted(self._completed)]

    def cancel(self, request_id: str) -> bool:
        return self.scheduler.cancel(request_id)

    def snapshot(self) -> dict:
        return {
            "scheduler": self.scheduler.snapshot(),
            "completed": len(self._completed),
            "results": {
                rid: {
                    "generated_tokens": len(result.output),
                    "ttft_steps": result.ttft_steps,
                    "total_steps": result.total_steps,
                }
                for rid, result in self._completed.items()
            },
        }
