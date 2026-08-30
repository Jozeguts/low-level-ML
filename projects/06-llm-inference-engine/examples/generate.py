from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generation import generate
from runtime import DecoderModel, ModelConfig, Weights
from tokenizer import ByteTokenizer


def main() -> None:
    tokenizer = ByteTokenizer()
    # Tokenizer vocabulary is 258, so use a matching controlled model.
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        max_seq_len=256,
    )
    model = DecoderModel(config, Weights(config, seed=42))
    prompt = "low level ML systems"
    prompt_ids = tokenizer.encode(prompt)
    generated, stats = generate(model, prompt_ids, max_new_tokens=24, strategy="sample", sampler_kwargs={"temperature": 0.8, "top_k": 20, "top_p": 0.9})
    print("prompt:", prompt)
    print("generated:", tokenizer.decode(generated))
    print(f"prefill_ms={stats.prefill_ms:.3f}")
    print(f"decode_ms={stats.decode_ms:.3f}")
    print(f"decode_tokens_per_second={stats.decode_tokens_per_second:.2f}")
    print(f"kv_cache_bytes={stats.kv_cache_bytes}")


if __name__ == "__main__":
    main()
