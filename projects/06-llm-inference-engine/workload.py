from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WorkloadCase:
    name: str
    prompt_lengths: tuple[int, ...]
    max_new_tokens: int
    concurrency: int


def make_requests(case: WorkloadCase, vocab_size: int = 128):
    from engine import GenerationRequest

    requests = []
    for i in range(case.concurrency):
        length = case.prompt_lengths[i % len(case.prompt_lengths)]
        rng = np.random.default_rng(1000 + i)
        prompt = rng.integers(0, vocab_size - 1, size=length, dtype=np.int64)
        requests.append(GenerationRequest(prompt=prompt, max_new_tokens=case.max_new_tokens,
                                          strategy="greedy", request_id=f"{case.name}-{i}"))
    return requests


DEFAULT_CASES = (
    WorkloadCase("short", (16, 24, 32), 16, 4),
    WorkloadCase("mixed", (32, 64, 96, 128), 32, 4),
    WorkloadCase("long", (128, 160, 192), 32, 2),
)
