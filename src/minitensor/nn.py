"""Neural-network building blocks for MiniTensor."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import numpy as np

from .tensor import Tensor, tensor


class Parameter(Tensor):
    """A floating-point Tensor that is registered as a trainable parameter."""

    def __init__(self, data: Any, dtype: Any = None) -> None:
        super().__init__(data, dtype=dtype, requires_grad=True)

    def __repr__(self) -> str:
        return f"Parameter({self.data!r})"


class Module:
    """Base class for trainable layers and nested models."""

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in {"_parameters", "_modules"}:
            parameters = self.__dict__.get("_parameters")
            modules = self.__dict__.get("_modules")
            if parameters is not None:
                parameters.pop(name, None)
            if modules is not None:
                modules.pop(name, None)
            if isinstance(value, Parameter) and parameters is not None:
                parameters[name] = value
            elif isinstance(value, Module) and modules is not None:
                modules[name] = value
        object.__setattr__(self, name, value)

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def parameters(self) -> Iterator[Parameter]:
        for _, parameter in self.named_parameters():
            yield parameter

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        yield from self._named_parameters(prefix, set())

    def _named_parameters(
        self, prefix: str, seen: set[int]
    ) -> Iterator[tuple[str, Parameter]]:
        for name, parameter in self._parameters.items():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            yield f"{prefix}.{name}" if prefix else name, parameter
        for name, module in self._modules.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from module._named_parameters(child_prefix, seen)

    def children(self) -> Iterator["Module"]:
        return iter(self._modules.values())

    def modules(self) -> Iterator["Module"]:
        yield self
        for module in self._modules.values():
            yield from module.modules()

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, "Module"]]:
        yield prefix, self
        for name, module in self._modules.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_modules(child_prefix)

    def train(self, mode: bool = True) -> "Module":
        self.training = bool(mode)
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()


class Linear(Module):
    """Fully connected layer: ``output = input @ weight + bias``."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Linear feature sizes must be positive.")
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = Parameter(
            np.random.uniform(-limit, limit, size=(in_features, out_features))
        )
        self.bias = (
            Parameter(np.zeros(out_features, dtype=np.float32)) if bias else None
        )

    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs @ self.weight
        return output + self.bias if self.bias is not None else output


class ReLU(Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.relu()


class Sequential(Module):
    """Apply child modules in insertion order."""

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        for index, module in enumerate(modules):
            if not isinstance(module, Module):
                raise TypeError("Sequential accepts Module instances only.")
            setattr(self, str(index), module)

    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs
        for module in self._modules.values():
            output = module(output)
        return output


def mse_loss(prediction: Tensor, target: Tensor | Any) -> Tensor:
    """Mean squared error over all elements."""
    target_tensor = target if isinstance(target, Tensor) else tensor(target, dtype=prediction.dtype)
    return ((prediction - target_tensor) ** 2).mean()


def cross_entropy(logits: Tensor, labels: Tensor | Any) -> Tensor:
    """Numerically stable mean cross-entropy for class logits.

    ``logits`` has shape ``(batch, classes)`` and ``labels`` has shape ``(batch,)``.
    """
    if logits.ndim != 2:
        raise ValueError("cross_entropy expects logits with shape (batch, classes).")
    label_array = labels.data if isinstance(labels, Tensor) else np.asarray(labels)
    if label_array.ndim != 1 or label_array.shape[0] != logits.shape[0]:
        raise ValueError("cross_entropy labels must have shape (batch,).")
    if not np.issubdtype(label_array.dtype, np.integer):
        raise TypeError("cross_entropy labels must be integer class indices.")
    if np.any(label_array < 0) or np.any(label_array >= logits.shape[1]):
        raise ValueError("cross_entropy labels contain an invalid class index.")
    max_values = logits.max(axis=1, keepdims=True).detach()
    shifted = logits - max_values
    log_normalizer = shifted.exp().sum(axis=1).log()
    correct_logits = logits[np.arange(logits.shape[0]), label_array] - max_values.reshape(-1)
    return (log_normalizer - correct_logits).mean()


class MSELoss(Module):
    def forward(self, prediction: Tensor, target: Tensor | Any) -> Tensor:
        return mse_loss(prediction, target)


class CrossEntropyLoss(Module):
    def forward(self, logits: Tensor, labels: Tensor | Any) -> Tensor:
        return cross_entropy(logits, labels)


__all__ = [
    "Linear",
    "CrossEntropyLoss",
    "MSELoss",
    "Module",
    "Parameter",
    "ReLU",
    "Sequential",
    "cross_entropy",
    "mse_loss",
]
