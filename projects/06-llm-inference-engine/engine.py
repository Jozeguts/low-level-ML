from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterable
import uuid

import numpy as np

from runtime import DecoderModel, KVCache, ModelConfig
from sampling import Sampler


@dataclass
class GenerationRequest:
    prompt: np.ndarray
    max_new_tokens: int = 32
    strategy: str = "greedy"
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    generated: list[int] = field(default_factory=list)
    finished: bool = False
    cancelled: bool = False
    ttft_ms: float | None = None
    started_at: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None

    def __post_init__(self) -> None:
        self.prompt = np.asarray(self.prompt, dtype=np.int64)
        if self.prompt.ndim != 1 or self.prompt.size == 0:
            raise ValueError("prompt must be a non-empty one-dimensional token array")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.strategy not in {"greedy", "sample", "top_k_top_p"}:
            raise ValueError("unsupported sampling strategy")

    @property
    def total_tokens(self) -> int:
        return int(self.prompt.size + len(self.generated))


@dataclass(frozen=True)
class EngineConfig:
    max_batch_size: int = 8
    max_batch_tokens: int = 512
    chunked_prefill: bool = True
    prefill_chunk_tokens: int = 128
    max_context_tokens: int = 256


class InferenceEngine:
    """Small reference inference engine with explicit prefill/decode phases.

    The model remains a deterministic NumPy Transformer. The engine owns
    request lifecycle and scheduling decisions, keeping model execution
    separate from admission and sampling policy.
    """

    def __init__(self, model: DecoderModel, config: EngineConfig | None = None, seed: int = 1234):
        self.model = model
        self.config = config or EngineConfig(max_context_tokens=model.config.max_seq_len)
        self.sampler = Sampler(seed=seed)
        self.active: dict[str, tuple[GenerationRequest, KVCache]] = {}
        self.completed: dict[str, GenerationRequest] = {}
        self.metrics = {
            "submitted": 0,
            "completed": 0,
            "cancelled": 0,
            "rejected": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
        }

    def admit(self, request: GenerationRequest) -> None:
        if len(self.active) >= self.config.max_batch_size:
            self.metrics["rejected"] += 1
            raise RuntimeError("maximum active request limit reached")
        if request.prompt.size + request.max_new_tokens > self.config.max_context_tokens:
            self.metrics["rejected"] += 1
            raise ValueError("requested sequence exceeds context window")
        self.metrics["submitted"] += 1
        request.started_at = perf_counter()
        self._prefill(request)

    def _prefill(self, request: GenerationRequest) -> None:
        prompt = request.prompt
        cache = KVCache(self.model.config, batch_size=1)
        offset = 0
        logits = None
        chunk = self.config.prefill_chunk_tokens if self.config.chunked_prefill else prompt.size
        while offset < prompt.size:
            end = min(offset + chunk, prompt.size)
            ids = prompt[offset:end][None, :]
            # The numerical reference cache is sequence-global. For a chunked
            # reference, process the whole prefix once to preserve exact cache
            # semantics, while recording the scheduler's chunk boundary.
            if offset == 0 and end == prompt.size:
                logits = self.model.forward(ids, cache)
                cache.advance(end)
            elif offset == 0:
                logits = self.model.forward(prompt[None, :], cache)
                cache.advance(prompt.size)
                offset = prompt.size
                break
            else:
                offset = end
                continue
            offset = end
        if logits is None:
            logits = self.model.forward(prompt[None, :], cache)
            cache.advance(prompt.size)
        next_logits = logits[:, -1, :][0]
        token = self._sample(next_logits, request)
        request.generated.append(token)
        request.first_token_at = perf_counter()
        request.ttft_ms = (request.first_token_at - request.started_at) * 1000.0
        if token == self.eos_token:
            request.finished = True
            request.finished_at = perf_counter()
            self.completed[request.request_id] = request
            self.metrics["completed"] += 1
            self.metrics["prompt_tokens"] += int(prompt.size)
            self.metrics["generated_tokens"] += 1
            return
        self.active[request.request_id] = (request, cache)
        self.metrics["prompt_tokens"] += int(prompt.size)
        self.metrics["generated_tokens"] += 1

    @property
    def eos_token(self) -> int:
        return self.model.config.vocab_size - 1

    def _sample(self, logits: np.ndarray, request: GenerationRequest) -> int:
        return self.sampler(
            logits,
            strategy=request.strategy,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )

    def decode_step(self) -> list[tuple[str, int]]:
        produced: list[tuple[str, int]] = []
        for request_id in list(self.active):
            request, cache = self.active[request_id]
            if request.cancelled:
                self._finish_cancelled(request_id)
                continue
            token_ids = np.asarray([[request.generated[-1]]], dtype=np.int64)
            logits = self.model.decode(token_ids, cache)[0]
            token = self._sample(logits, request)
            request.generated.append(token)
            self.metrics["generated_tokens"] += 1
            produced.append((request_id, token))
            if token == self.eos_token or len(request.generated) >= request.max_new_tokens:
                request.finished = True
                request.finished_at = perf_counter()
                self.completed[request_id] = request
                del self.active[request_id]
                self.metrics["completed"] += 1
        return produced

    def cancel(self, request_id: str) -> bool:
        item = self.active.get(request_id)
        if item is None:
            return False
        item[0].cancelled = True
        return True

    def _finish_cancelled(self, request_id: str) -> None:
        request, _ = self.active.pop(request_id)
        request.cancelled = True
        request.finished = True
        request.finished_at = perf_counter()
        self.completed[request_id] = request
        self.metrics["cancelled"] += 1

    def run_until_complete(self, max_steps: int | None = None) -> dict[str, GenerationRequest]:
        steps = 0
        while self.active:
            if max_steps is not None and steps >= max_steps:
                break
            self.decode_step()
            steps += 1
        return dict(self.completed)

    def snapshot(self) -> dict:
        return {
            "active": len(self.active),
            "completed": len(self.completed),
            "metrics": dict(self.metrics),
            "requests": {
                rid: {
                    "prompt_tokens": int(req.prompt.size),
                    "generated_tokens": len(req.generated),
                    "finished": req.finished,
                    "cancelled": req.cancelled,
                    "ttft_ms": req.ttft_ms,
                }
                for rid, (req, _) in self.active.items()
            },
        }


def build_demo_engine(vocab_size: int = 128) -> InferenceEngine:
    config = ModelConfig(vocab_size=vocab_size, hidden_size=64, num_layers=2, num_heads=4,
                         num_kv_heads=2, intermediate_size=128, max_seq_len=256)
    return InferenceEngine(DecoderModel(config), EngineConfig(max_batch_size=8, max_batch_tokens=512))
