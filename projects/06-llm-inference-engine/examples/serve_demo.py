from __future__ import annotations

import numpy as np

from engine import GenerationRequest, build_demo_engine


def main() -> None:
    engine = build_demo_engine()
    requests = [
        GenerationRequest(np.array([2, 5, 7, 11], dtype=np.int64), max_new_tokens=8,
                          request_id="user-a"),
        GenerationRequest(np.array([3, 1, 4, 1, 5, 9], dtype=np.int64), max_new_tokens=8,
                          request_id="user-b"),
        GenerationRequest(np.array([6, 8, 6], dtype=np.int64), max_new_tokens=8,
                          request_id="user-c"),
    ]

    for request in requests:
        engine.admit(request)

    engine.run_until_complete()

    for request_id, request in engine.completed.items():
        print({
            "request_id": request_id,
            "prompt_tokens": request.prompt.tolist(),
            "generated_tokens": request.generated,
            "generated_count": len(request.generated),
            "ttft_ms": request.ttft_ms,
        })

    print("engine_snapshot:")
    print(engine.snapshot())


if __name__ == "__main__":
    main()
