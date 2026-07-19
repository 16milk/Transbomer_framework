"""Dynamic reverse-mode automatic differentiation primitives."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from .tensor import Tensor


_grad_enabled: ContextVar[bool] = ContextVar("minitensor_grad_enabled", default=True)


def is_grad_enabled() -> bool:
    """Return whether eager operations should record a backward graph."""
    return _grad_enabled.get()


@contextmanager
def no_grad():
    """Temporarily disable dynamic graph recording."""
    token = _grad_enabled.set(False)
    try:
        yield
    finally:
        _grad_enabled.reset(token)


@dataclass
class Context:
    """Storage for values needed by an Operation's backward function."""

    saved: list[np.ndarray] = field(default_factory=list)
    values: dict[str, object] = field(default_factory=dict)

    def save_for_backward(self, *values: np.ndarray) -> None:
        self.saved.extend(values)


BackwardFn = Callable[[np.ndarray, Context], tuple[np.ndarray | None, ...]]


@dataclass
class Operation:
    """A node in the dynamic graph and its local backward rule."""

    name: str
    inputs: tuple["Tensor", ...]
    context: Context
    backward_fn: BackwardFn

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        return self.backward_fn(grad_output, self.context)
