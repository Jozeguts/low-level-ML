"""Deterministic workload benchmark for Mini vLLM.

This measures scheduler work in logical steps, not GPU throughput. It is
reproducible on any machine and separates runtime policy from hardware speed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Dict

from engine import EngineConfig, MiniVLLM


WORKLOADS = {
    "short": [(4, 8), (6, 4), (3, 6), (5, 5), (7, 3), (4, 7)],
    "mixed": [(4, 6), (20, 5), (7, 12), (3, 4), (16, 8), (9, 3)],
    "long": [(24, 16), (32, 12), (28, 20), (20, 10)],
}


def run_case(name: str, config: EngineConfig) -> Dict[str, object]:
    workload = WORKLOADS[name]
    engine = MiniVLLM(config)
    for index, (prompt_len, output_len) in enumerate(workload):
        engine.submit(f"r{index}", range(1, prompt_len + 1), output_len, priority=index % 2)

    start = time.perf_counter()
    results = engine.run()
    wall = time.perf_counter() - start
    steps = engine.scheduler.step_id
    generated = sum(len(r.output) for r in results)
    prompts = sum(prompt_len for prompt_len, _ in workload)
    ttft = [r.ttft_steps for r in results if r.ttft_steps is not None]

    return {
        "workload": name,
        "requests": len(results),
        "prompt_tokens": prompts,
        "generated_tokens": generated,
        "scheduler_steps": steps,
        "logical_tokens_per_step": (generated / steps) if steps else 0.0,
        "wall_seconds": wall,
        "ttft_p50_steps": statistics.median(ttft) if ttft else None,
        "ttft_max_steps": max(ttft) if ttft else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default="mixed")
    parser.add_argument("--blocks", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--batch-tokens", type=int, default=32)
    args = parser.parse_args()
    config = EngineConfig(num_blocks=args.blocks, block_size=args.block_size, max_batch_tokens=args.batch_tokens)
    print(json.dumps(run_case(args.workload, config), indent=2))


if __name__ == "__main__":
    main()
