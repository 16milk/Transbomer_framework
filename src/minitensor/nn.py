"""Neural-network building blocks for MiniTensor."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .errors import StateDictError
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

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
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

    def state_dict(self) -> OrderedDict[str, Tensor]:
        """Return detached copies of all registered parameters."""
        return OrderedDict(
            (name, Tensor(parameter.data.copy(), dtype=parameter.dtype))
            for name, parameter in self.named_parameters()
        )

    def load_state_dict(
        self,
        state_dict: Mapping[str, Tensor | np.ndarray | Any],
        *,
        strict: bool = True,
    ) -> None:
        """Load parameters after validating names, shapes, and dtypes."""
        expected = OrderedDict(self.named_parameters())
        received = set(state_dict)
        missing = [name for name in expected if name not in received]
        unexpected = [name for name in received if name not in expected]
        if strict and (missing or unexpected):
            details = []
            if missing:
                details.append(f"missing keys: {missing}")
            if unexpected:
                details.append(f"unexpected keys: {unexpected}")
            raise StateDictError("Incompatible state_dict (" + "; ".join(details) + ").")

        for name, parameter in expected.items():
            if name not in state_dict:
                continue
            value = state_dict[name]
            array = value.data if isinstance(value, Tensor) else np.asarray(value)
            if array.shape != parameter.shape:
                raise StateDictError(
                    f"Parameter {name!r} has shape {array.shape}, expected {parameter.shape}."
                )
            if np.dtype(array.dtype) != parameter.dtype:
                raise StateDictError(
                    f"Parameter {name!r} has dtype {array.dtype}, expected {parameter.dtype}."
                )
            parameter.data[...] = array

    def save_state_dict(self, path: str | Path) -> None:
        """Save parameters and model structure metadata to a compressed NPZ file."""
        from .serialization import save_state_dict

        save_state_dict(self, path)

    def load_state_dict_file(self, path: str | Path, *, strict: bool = True) -> None:
        """Load parameters from a file created by :meth:`save_state_dict`."""
        from .serialization import load_state_dict

        load_state_dict(self, path, strict=strict)


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
    if logits.ndim < 2:
        raise ValueError("cross_entropy expects logits with a final classes dimension.")
    label_array = labels.data if isinstance(labels, Tensor) else np.asarray(labels)
    expected_shape = logits.shape[:-1]
    if label_array.shape != expected_shape:
        raise ValueError(f"cross_entropy labels must have shape {expected_shape}.")
    if not np.issubdtype(label_array.dtype, np.integer):
        raise TypeError("cross_entropy labels must be integer class indices.")
    if np.any(label_array < 0) or np.any(label_array >= logits.shape[-1]):
        raise ValueError("cross_entropy labels contain an invalid class index.")
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = label_array.reshape(-1)
    max_values = flat_logits.max(axis=1, keepdims=True).detach()
    shifted = flat_logits - max_values
    log_normalizer = shifted.exp().sum(axis=1).log()
    correct_logits = flat_logits[np.arange(flat_logits.shape[0]), flat_labels]
    correct_logits = correct_logits - max_values.reshape(-1)
    return (log_normalizer - correct_logits).mean()


def softmax(inputs: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable softmax along ``axis``."""
    normalized_axis = axis if axis >= 0 else inputs.ndim + axis
    if normalized_axis < 0 or normalized_axis >= inputs.ndim:
        raise ValueError(f"Invalid softmax axis {axis} for shape {inputs.shape}.")
    maximum = inputs.max(axis=normalized_axis, keepdims=True).detach()
    exponentials = (inputs - maximum).exp()
    return exponentials / exponentials.sum(axis=normalized_axis, keepdims=True)


def log_softmax(inputs: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable log-softmax along ``axis``."""
    normalized_axis = axis if axis >= 0 else inputs.ndim + axis
    if normalized_axis < 0 or normalized_axis >= inputs.ndim:
        raise ValueError(f"Invalid log_softmax axis {axis} for shape {inputs.shape}.")
    maximum = inputs.max(axis=normalized_axis, keepdims=True).detach()
    shifted = inputs - maximum
    return shifted - shifted.exp().sum(axis=normalized_axis, keepdims=True).log()


class Softmax(Module):
    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, inputs: Tensor) -> Tensor:
        return softmax(inputs, self.axis)


class LogSoftmax(Module):
    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, inputs: Tensor) -> Tensor:
        return log_softmax(inputs, self.axis)


class Embedding(Module):
    """Lookup table mapping integer token ids to trainable vectors."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("Embedding sizes must be positive.")
        self.weight = Parameter(np.random.normal(0.0, 0.02, (num_embeddings, embedding_dim)))
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, indices: Tensor | Any) -> Tensor:
        values = indices.data if isinstance(indices, Tensor) else np.asarray(indices)
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("Embedding indices must be integer values.")
        if np.any(values < 0) or np.any(values >= self.num_embeddings):
            raise IndexError("Embedding index is outside the vocabulary.")
        return self.weight[values]


class LayerNorm(Module):
    """Layer normalization over the last ``normalized_shape`` dimensions."""

    def __init__(self, normalized_shape: int | tuple[int, ...], eps: float = 1e-5) -> None:
        super().__init__()
        shape = (
            (normalized_shape,)
            if isinstance(normalized_shape, int)
            else tuple(normalized_shape)
        )
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("normalized_shape must contain positive dimensions.")
        self.normalized_shape = shape
        self.eps = eps
        self.weight = Parameter(np.ones(shape, dtype=np.float32))
        self.bias = Parameter(np.zeros(shape, dtype=np.float32))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-len(self.normalized_shape) :] != self.normalized_shape:
            raise ValueError("Input trailing dimensions do not match normalized_shape.")
        axis = tuple(range(inputs.ndim - len(self.normalized_shape), inputs.ndim))
        mean_value = inputs.mean(axis=axis, keepdims=True)
        variance = ((inputs - mean_value) ** 2).mean(axis=axis, keepdims=True)
        normalized = (inputs - mean_value) / (variance + self.eps) ** 0.5
        return normalized * self.weight + self.bias


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("Dropout probability must be in [0, 1).")
        self.p = float(p)

    def forward(self, inputs: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return inputs
        mask = (np.random.random(inputs.shape) >= self.p).astype(inputs.dtype)
        return inputs * (mask / (1 - self.p))


class MultiHeadAttention(Module):
    """Scaled dot-product multi-head self/cross attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.query = Linear(embed_dim, embed_dim, bias=bias)
        self.key = Linear(embed_dim, embed_dim, bias=bias)
        self.value = Linear(embed_dim, embed_dim, bias=bias)
        self.output = Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = Dropout(dropout)

    def _split_heads(self, values: Tensor) -> Tensor:
        batch, length, _ = values.shape
        return values.reshape(batch, length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, values: Tensor) -> Tensor:
        batch, _, length, _ = values.shape
        return values.transpose(0, 2, 1, 3).reshape(batch, length, self.embed_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        mask: Tensor | np.ndarray | None = None,
    ) -> Tensor:
        key = query if key is None else key
        value = key if value is None else value
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("Attention inputs must have shape (batch, sequence, embedding).")
        if query.shape[0] != key.shape[0] or key.shape[:2] != value.shape[:2]:
            raise ValueError("Attention batch and sequence dimensions are incompatible.")
        queries = self._split_heads(self.query(query))
        keys = self._split_heads(self.key(key))
        values = self._split_heads(self.value(value))
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            mask_values = mask.data if isinstance(mask, Tensor) else np.asarray(mask)
            if mask_values.dtype == np.bool_:
                additive = np.where(mask_values, -1e9, 0.0).astype(scores.dtype)
                scores = scores + additive
            else:
                scores = scores + mask_values
        weights = self.dropout(softmax(scores, axis=-1))
        return self.output(self._merge_heads(weights @ values))


class PositionalEncoding(Module):
    """Fixed sinusoidal positional encoding added to ``(batch, sequence, dim)``."""

    def __init__(self, embed_dim: int, max_length: int = 512) -> None:
        super().__init__()
        if embed_dim <= 0 or max_length <= 0:
            raise ValueError("embed_dim and max_length must be positive.")
        positions = np.arange(max_length)[:, None]
        frequencies = np.exp(np.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        encoding = np.zeros((max_length, embed_dim), dtype=np.float32)
        encoding[:, 0::2] = np.sin(positions * frequencies)
        encoding[:, 1::2] = np.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        self.encoding = tensor(encoding)
        self.max_length = max_length

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[1] > self.max_length:
            raise ValueError("PositionalEncoding input must fit (batch, sequence, embedding).")
        if inputs.shape[-1] != self.encoding.shape[-1]:
            raise ValueError("Input embedding dimension does not match positional encoding.")
        position = self.encoding[: inputs.shape[1]].reshape(
            1, inputs.shape[1], inputs.shape[2]
        )
        return inputs + position


class FeedForward(Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.network = Sequential(
            Linear(embed_dim, hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, embed_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class TransformerBlock(Module):
    """Pre-norm Transformer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feed_forward_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = feed_forward_dim or embed_dim * 4
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, hidden_dim, dropout=dropout)
        self.norm2 = LayerNorm(embed_dim)
        self.dropout = Dropout(dropout)

    def forward(
        self,
        inputs: Tensor,
        mask: Tensor | np.ndarray | None = None,
    ) -> Tensor:
        attended = self.attention(self.norm1(inputs), mask=mask)
        residual = inputs + self.dropout(attended)
        return residual + self.dropout(self.feed_forward(self.norm2(residual)))


class MSELoss(Module):
    def forward(self, prediction: Tensor, target: Tensor | Any) -> Tensor:
        return mse_loss(prediction, target)


class CrossEntropyLoss(Module):
    def forward(self, logits: Tensor, labels: Tensor | Any) -> Tensor:
        return cross_entropy(logits, labels)


__all__ = [
    "Dropout",
    "Embedding",
    "FeedForward",
    "LayerNorm",
    "Linear",
    "CrossEntropyLoss",
    "LogSoftmax",
    "MultiHeadAttention",
    "MSELoss",
    "Module",
    "Parameter",
    "PositionalEncoding",
    "ReLU",
    "Sequential",
    "Softmax",
    "TransformerBlock",
    "cross_entropy",
    "log_softmax",
    "mse_loss",
    "softmax",
]
