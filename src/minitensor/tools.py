"""Reproducibility, model inspection, and lightweight benchmarking utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .nn import Module


def set_seed(seed: int) -> None:
    """Seed NumPy, the random source used by MiniTensor modules."""
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")
    np.random.seed(int(seed))


def count_parameters(module: Module, *, trainable_only: bool = True) -> int:
    """Count scalar parameters in a module."""
    return sum(
        parameter.size
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def module_summary(module: Module) -> str:
    """Render a compact module and parameter shape summary."""
    lines = [f"{type(module).__name__}("]
    for name, child in module.named_modules():
        if name:
            lines.append(f"  {name}: {type(child).__name__}")
    lines.append("  parameters:")
    for name, parameter in module.named_parameters():
        lines.append(f"    {name}: shape={parameter.shape}, dtype={parameter.dtype}")
    lines.append(f"  total_parameters: {count_parameters(module)}")
    lines.append(")")
    return "\n".join(lines)


def graph_stats(graph: Any) -> dict[str, int]:
    """Return node and value counts for a static graph."""
    nodes = getattr(graph, "active_nodes", [])
    inputs = getattr(graph, "inputs", [])
    outputs = getattr(graph, "outputs", [])
    return {
        "nodes": len(nodes),
        "inputs": len(inputs),
        "outputs": len(outputs),
    }


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing summary returned by :func:`benchmark`."""

    mean_seconds: float
    min_seconds: float
    iterations: int

    @property
    def mean_milliseconds(self) -> float:
        return self.mean_seconds * 1000


def benchmark(
    function: Callable[[], Any],
    *,
    iterations: int = 10,
    warmup: int = 1,
) -> BenchmarkResult:
    """Measure a callable using wall-clock time after optional warmup."""
    if iterations <= 0 or warmup < 0:
        raise ValueError("iterations must be positive and warmup cannot be negative.")
    for _ in range(warmup):
        function()
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        function()
        durations.append(time.perf_counter() - start)
    return BenchmarkResult(float(np.mean(durations)), float(np.min(durations)), iterations)


__all__ = [
    "BenchmarkResult",
    "benchmark",
    "count_parameters",
    "graph_stats",
    "module_summary",
    "set_seed",
]
