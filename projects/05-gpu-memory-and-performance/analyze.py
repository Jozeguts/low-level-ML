from __future__ import annotations

import csv
import sys
from pathlib import Path


def load(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python analyze.py benchmark.csv")
    rows = load(Path(sys.argv[1]))
    best = min(rows, key=lambda r: float(r["kernel_ms"]))
    print("Block-size experiment")
    for row in rows:
        print(f"threads={row['threads']:>4}  blocks={row['blocks']:>6}  "
              f"kernel_ms={float(row['kernel_ms']):.6f}  "
              f"GB/s={float(row['effective_GBps']):.3f}")
    print(f"best_thread_count={best['threads']}")
