"""Small reproducible eager MiniTensor versus NumPy benchmark.

Run with:
    python benchmarks/eager_vs_numpy.py --size 256 --iterations 20
"""

from __future__ import annotations

import argparse

import numpy as np

import minitensor as mt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be positive")

    mt.set_seed(0)
    left = np.random.randn(args.size, args.size).astype(np.float32)
    right = np.random.randn(args.size, args.size).astype(np.float32)
    eager_left, eager_right = mt.tensor(left), mt.tensor(right)

    eager = mt.benchmark(lambda: eager_left @ eager_right, iterations=args.iterations)
    numpy_result = mt.benchmark(lambda: np.matmul(left, right), iterations=args.iterations)
    eager_value = (eager_left @ eager_right).data
    numpy_value = left @ right

    print(f"size={args.size}, iterations={args.iterations}")
    print(f"eager: {eager.mean_milliseconds:.3f} ms")
    print(f"numpy: {numpy_result.mean_milliseconds:.3f} ms")
    print(f"max_abs_error: {np.max(np.abs(eager_value - numpy_value)):.6g}")


if __name__ == "__main__":
    main()
