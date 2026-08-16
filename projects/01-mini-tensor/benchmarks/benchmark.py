import time

import numpy as np

from mini_tensor.tensor import Tensor


def bench(label, fn, repeats=1000):
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    elapsed = time.perf_counter() - start
    print(f"{label:20s} {elapsed / repeats * 1e6:10.2f} us/op")


a = np.random.randn(128, 128)
b = np.random.randn(128, 128)

bench("NumPy matmul", lambda: a @ b, repeats=100)
bench("MiniTensor matmul", lambda: Tensor(a) @ Tensor(b), repeats=100)
bench("NumPy add", lambda: a + b)
bench("MiniTensor add", lambda: Tensor(a) + Tensor(b))
