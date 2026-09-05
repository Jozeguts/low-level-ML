from __future__ import annotations

import argparse
import json
from statistics import mean, median
from time import perf_counter

from engine import EngineConfig, InferenceEngine, GenerationRequest
from runtime import DecoderModel, ModelConfig
from workload import DEFAULT_CASES, make_requests


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def run_case(case, seed: int) -> dict:
    model_cfg = ModelConfig(vocab_size=128, hidden_size=64, num_layers=2,
                            num_heads=4, num_kv_heads=2, intermediate_size=128,
                            max_seq_len=256)
    engine = InferenceEngine(DecoderModel(model_cfg),
                             EngineConfig(max_batch_size=case.concurrency,
                                           max_batch_tokens=512,
                                           max_context_tokens=256),
                             seed=seed)
    requests = make_requests(case, model_cfg.vocab_size)
    start = perf_counter()
    for request in requests:
        engine.admit(request)
    engine.run_until_complete()
    elapsed = perf_counter() - start
    completed = list(engine.completed.values())
    ttft = [r.ttft_ms for r in completed if r.ttft_ms is not None]
    generated = sum(len(r.generated) for r in completed)
    prompt = sum(int(r.prompt.size) for r in completed)
    return {
        "case": case.name,
        "concurrency": case.concurrency,
        "prompt_tokens": prompt,
        "generated_tokens": generated,
        "wall_time_s": elapsed,
        "tokens_per_s": generated / elapsed if elapsed else 0.0,
        "ttft_p50_ms": median(ttft) if ttft else 0.0,
        "ttft_p95_ms": percentile(ttft, 0.95),
        "mean_ttft_ms": mean(ttft) if ttft else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = [run_case(case, args.seed) for case in DEFAULT_CASES]
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    for row in rows:
        print(f"{row['case']:>6}  concurrency={row['concurrency']}  "
              f"tokens/s={row['tokens_per_s']:.2f}  "
              f"TTFT p50={row['ttft_p50_ms']:.2f}ms  "
              f"TTFT p95={row['ttft_p95_ms']:.2f}ms")


if __name__ == "__main__":
    main()
