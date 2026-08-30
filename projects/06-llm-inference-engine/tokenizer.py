from __future__ import annotations

import numpy as np


class ByteTokenizer:
    """Deterministic byte tokenizer suitable for a controlled inference lab."""

    bos_id = 0
    eos_id = 1
    offset = 2
    vocab_size = 258

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        ids = [self.bos_id] if add_bos else []
        ids.extend(self.offset + b for b in text.encode("utf-8"))
        return ids

    def decode(self, ids: list[int] | np.ndarray) -> str:
        raw = []
        for token in ids:
            token = int(token)
            if token in (self.bos_id, self.eos_id):
                continue
            if self.offset <= token < self.vocab_size:
                raw.append(token - self.offset)
        return bytes(raw).decode("utf-8", errors="replace")
