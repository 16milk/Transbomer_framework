"""Tensor data structure and NumPy-backed eager operators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Union

import numpy as np

from .autograd import Context, Operation, is_grad_enabled
from .errors import DTypeError, MiniTensorError, ShapeError

ArrayLike = Union["Tensor", np.ndarray, Sequence[Any], int, float, bool, np.number]
Axis = Union[int, tuple[int, ...], None]
Gradient = np.ndarray[Any, Any]


def _validate_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
        raise DTypeError(
            f"MiniTensor only supports numeric and boolean dtypes; received {dtype}."
        )
    return dtype


def _as_array(data: ArrayLike, dtype: Any = None, *, copy: bool) -> np.ndarray[Any, Any]:
    try:
        if isinstance(data, Tensor):
            array = data._data
            if dtype is not None:
                target_dtype = _validate_dtype(np.dtype(dtype))
                return array.astype(target_dtype, copy=True)
            return array.copy() if copy else array

        if isinstance(data, np.ndarray):
            array = np.array(data, dtype=dtype, copy=copy)
        else:
            array = np.array(data, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise DTypeError(f"Could not create a Tensor with dtype {dtype!r}.") from error

    _validate_dtype(array.dtype)
    return array


def _to_tensor(value: ArrayLike) -> "Tensor":
    return value if isinstance(value, Tensor) else Tensor(value)


def _unbroadcast(gradient: Gradient, shape: tuple[int, ...]) -> Gradient:
    """Sum a broadcast gradient back to an input's original shape."""
    result = np.asarray(gradient)
    while result.ndim > len(shape):
        result = result.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and result.shape[axis] != 1:
            result = result.sum(axis=axis, keepdims=True)
    return result.reshape(shape)


def _inverse_axes(axes: tuple[int, ...]) -> tuple[int, ...]:
    inverse = [0] * len(axes)
    for index, axis in enumerate(axes):
        inverse[axis] = index
    return tuple(inverse)


def _normalize_reduction_axis(axis: Axis, ndim: int) -> tuple[int, ...] | None:
    if axis is None:
        return None
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    normalized = tuple(value + ndim if value < 0 else value for value in axes)
    if any(value < 0 or value >= ndim for value in normalized):
        raise ShapeError(f"Reduction axis {axis!r} is invalid for a {ndim}D Tensor.")
    if len(set(normalized)) != len(normalized):
        raise ShapeError(f"Reduction axis {axis!r} contains duplicates.")
    return normalized


def _safe_log(value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(value)


def _pow_gradients(gradient: Gradient, context: Context) -> tuple[Gradient | None, ...]:
    left, right, output = context.saved
    left_gradient = None
    right_gradient = None
    with np.errstate(divide="ignore", invalid="ignore"):
        if context.values["left_requires_grad"]:
            left_gradient = _unbroadcast(
                gradient * right * np.power(left, right - 1),
                context.values["left_shape"],
            )
        if context.values["right_requires_grad"]:
            right_gradient = _unbroadcast(
                gradient * output * _safe_log(left),
                context.values["right_shape"],
            )
    return left_gradient, right_gradient


def _matmul_gradients(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
    gradient: Gradient,
) -> tuple[Gradient, Gradient]:
    left_was_vector = left.ndim == 1
    right_was_vector = right.ndim == 1
    left_matrix = np.expand_dims(left, axis=-2) if left_was_vector else left
    right_matrix = np.expand_dims(right, axis=-1) if right_was_vector else right
    output_gradient = np.asarray(gradient)
    if left_was_vector and right_was_vector:
        output_gradient = output_gradient.reshape(1, 1)
    elif left_was_vector:
        output_gradient = np.expand_dims(output_gradient, axis=-2)
    elif right_was_vector:
        output_gradient = np.expand_dims(output_gradient, axis=-1)

    left_gradient = np.matmul(output_gradient, np.swapaxes(right_matrix, -1, -2))
    right_gradient = np.matmul(np.swapaxes(left_matrix, -1, -2), output_gradient)
    if left_was_vector:
        left_gradient = np.squeeze(left_gradient, axis=-2)
    if right_was_vector:
        right_gradient = np.squeeze(right_gradient, axis=-1)
    return _unbroadcast(left_gradient, left.shape), _unbroadcast(right_gradient, right.shape)


class Tensor:
    """A numeric, NumPy-backed multidimensional array.

    ``Tensor`` owns a copy of constructor input by default. Use :meth:`from_numpy`
    when deliberate storage sharing with a NumPy array is required. Results of basic
    indexing, ``reshape``, and ``transpose`` preserve NumPy's view-or-copy behavior.
    """

    def __init__(
        self,
        data: ArrayLike,
        dtype: Any = None,
        *,
        copy: bool = True,
        requires_grad: bool = False,
    ) -> None:
        self._data = _as_array(data, dtype=dtype, copy=copy)
        self._requires_grad = bool(requires_grad)
        self._grad: Gradient | None = None
        self._grad_fn: Operation | None = None
        self._parents: tuple[Tensor, ...] = ()
        self._validate_grad_dtype()

    @classmethod
    def _from_array(
        cls,
        array: Any,
        *,
        requires_grad: bool = False,
        grad_fn: Operation | None = None,
    ) -> "Tensor":
        instance = cls.__new__(cls)
        instance._data = np.asarray(array)
        _validate_dtype(instance._data.dtype)
        instance._requires_grad = bool(requires_grad)
        instance._grad = None
        instance._grad_fn = grad_fn
        instance._parents = grad_fn.inputs if grad_fn is not None else ()
        instance._validate_grad_dtype()
        return instance

    @classmethod
    def from_numpy(cls, array: np.ndarray[Any, Any], *, requires_grad: bool = False) -> "Tensor":
        """Create a Tensor that shares storage with ``array``.

        The caller must pass a NumPy array with a numeric or boolean dtype.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("Tensor.from_numpy expects a numpy.ndarray.")
        return cls._from_array(array, requires_grad=requires_grad)

    def _validate_grad_dtype(self) -> None:
        if self._requires_grad and not np.issubdtype(self.dtype, np.inexact):
            raise DTypeError(
                f"requires_grad=True needs a floating or complex dtype; received {self.dtype}."
            )

    @property
    def requires_grad(self) -> bool:
        """Whether this Tensor participates in automatic differentiation."""
        return self._requires_grad

    @property
    def grad(self) -> "Tensor | None":
        """Return the accumulated gradient, if one has been computed."""
        if self._grad is None:
            return None
        return Tensor._from_array(self._grad.copy())

    @property
    def grad_fn(self) -> Operation | None:
        """Return the Operation that created this Tensor, if any."""
        return self._grad_fn

    @property
    def is_leaf(self) -> bool:
        """Return whether this Tensor was created directly by the user."""
        return self._grad_fn is None

    def zero_grad(self) -> None:
        """Clear this Tensor's accumulated gradient."""
        self._grad = None

    def detach(self) -> "Tensor":
        """Return a Tensor sharing data but disconnected from the graph."""
        return Tensor._from_array(self._data)

    def backward(self, gradient: ArrayLike | None = None) -> None:
        """Backpropagate from this Tensor through its dynamic computation graph."""
        if not self.requires_grad:
            raise MiniTensorError("backward() requires a Tensor with requires_grad=True.")
        if gradient is None:
            if self.size != 1:
                raise ShapeError(
                    "backward() requires an explicit gradient for non-scalar Tensors."
                )
            seed = np.ones_like(self._data, dtype=np.result_type(self._data, np.float64))
        else:
            seed = _to_tensor(gradient)._data
            if seed.shape != self.shape:
                raise ShapeError(
                    f"Gradient shape {seed.shape} does not match Tensor shape {self.shape}."
                )
            seed = np.asarray(seed, dtype=np.result_type(seed, np.float64))

        topo: list[Tensor] = []
        visited: set[int] = set()

        def visit(node: Tensor) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._parents:
                visit(parent)
            topo.append(node)

        visit(self)
        for node in topo:
            if node._grad_fn is not None:
                node._grad = None
        self._accumulate_grad(seed)
        for node in reversed(topo):
            if node._grad_fn is None or node._grad is None:
                continue
            gradients = node._grad_fn.backward(node._grad)
            for parent, parent_gradient in zip(node._parents, gradients):
                if parent_gradient is not None and parent.requires_grad:
                    parent._accumulate_grad(parent_gradient)

    def _accumulate_grad(self, gradient: Gradient) -> None:
        value = np.asarray(gradient)
        self._grad = value.copy() if self._grad is None else self._grad + value

    def _with_operation(
        self,
        data: Any,
        inputs: tuple["Tensor", ...],
        operation: Operation | None,
    ) -> "Tensor":
        requires_grad = (
            is_grad_enabled() and any(input_tensor.requires_grad for input_tensor in inputs)
        )
        if not requires_grad:
            return Tensor._from_array(data)
        return Tensor._from_array(data, requires_grad=True, grad_fn=operation)

    @property
    def data(self) -> np.ndarray[Any, Any]:
        """Return the backing ndarray without copying.

        Mutating this array mutates the Tensor. Prefer :meth:`numpy` with its default
        ``copy=True`` when exposing data outside framework code.
        """
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        """Dimensions of this Tensor."""
        return self._data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of this Tensor."""
        return self._data.ndim

    @property
    def size(self) -> int:
        """Number of elements in this Tensor."""
        return self._data.size

    @property
    def dtype(self) -> np.dtype[Any]:
        """NumPy data type of this Tensor."""
        return self._data.dtype

    @property
    def T(self) -> "Tensor":
        """Return a Tensor with dimensions reversed."""
        return self.transpose()

    def numpy(self, *, copy: bool = True) -> np.ndarray[Any, Any]:
        """Return the Tensor data as a NumPy array."""
        return self._data.copy() if copy else self._data

    def item(self) -> Any:
        """Return the Python scalar stored in a single-element Tensor."""
        return self._data.item()

    def tolist(self) -> Any:
        """Return the Tensor data as nested Python lists."""
        return self._data.tolist()

    def astype(self, dtype: Any) -> "Tensor":
        """Return a copy with a new numeric or boolean dtype."""
        try:
            target_dtype = _validate_dtype(np.dtype(dtype))
        except TypeError as error:
            raise DTypeError(f"Unsupported dtype {dtype!r}.") from error
        data = self._data.astype(target_dtype, copy=True)
        if not self.requires_grad or not is_grad_enabled():
            return Tensor._from_array(data)
        context = Context()
        context.values["input_dtype"] = self.dtype
        operation = Operation(
            "astype",
            (self,),
            context,
            lambda gradient, ctx: (gradient.astype(ctx.values["input_dtype"]),),
        )
        return self._with_operation(data, (self,), operation)

    def reshape(self, *shape: Union[int, tuple[int, ...]]) -> "Tensor":
        """Return a Tensor with a new shape using NumPy reshape semantics."""
        target_shape = _normalize_shape_arguments(shape)
        try:
            data = self._data.reshape(target_shape)
        except (TypeError, ValueError) as error:
            raise ShapeError(
                f"Cannot reshape Tensor with shape {self.shape} to {target_shape}."
            ) from error
        if not self.requires_grad or not is_grad_enabled():
            return Tensor._from_array(data)
        context = Context(values={"input_shape": self.shape})
        operation = Operation(
            "reshape",
            (self,),
            context,
            lambda gradient, ctx: (gradient.reshape(ctx.values["input_shape"]),),
        )
        return self._with_operation(data, (self,), operation)

    def transpose(self, *axes: Union[int, tuple[int, ...]]) -> "Tensor":
        """Return a Tensor with dimensions permuted using NumPy transpose semantics."""
        normalized_axes = _normalize_axes_arguments(axes)
        try:
            data = self._data.transpose(normalized_axes)
        except (TypeError, ValueError) as error:
            raise ShapeError(
                f"Cannot transpose Tensor with shape {self.shape} using axes {normalized_axes}."
            ) from error
        if not self.requires_grad or not is_grad_enabled():
            return Tensor._from_array(data)
        actual_axes = (
            tuple(reversed(range(self.ndim)))
            if normalized_axes is None
            else tuple(normalized_axes)
        )
        context = Context(values={"inverse_axes": _inverse_axes(actual_axes)})
        operation = Operation(
            "transpose",
            (self,),
            context,
            lambda gradient, ctx: (gradient.transpose(ctx.values["inverse_axes"]),),
        )
        return self._with_operation(data, (self,), operation)

    def sum(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        """Sum Tensor elements along ``axis``."""
        return self._reduce(np.sum, "sum", axis=axis, keepdims=keepdims)

    def mean(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        """Compute the arithmetic mean along ``axis``."""
        return self._reduce(np.mean, "mean", axis=axis, keepdims=keepdims)

    def max(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        """Compute the maximum value along ``axis``."""
        return self._reduce(np.max, "max", axis=axis, keepdims=keepdims)

    def add(self, other: ArrayLike) -> "Tensor":
        """Add ``other`` elementwise."""
        return self._binary_operation(
            other,
            np.add,
            "add",
            lambda gradient, ctx: (
                _unbroadcast(gradient, ctx.values["left_shape"]),
                _unbroadcast(gradient, ctx.values["right_shape"]),
            ),
        )

    def sub(self, other: ArrayLike) -> "Tensor":
        """Subtract ``other`` elementwise."""
        return self._binary_operation(
            other,
            np.subtract,
            "sub",
            lambda gradient, ctx: (
                _unbroadcast(gradient, ctx.values["left_shape"]),
                _unbroadcast(-gradient, ctx.values["right_shape"]),
            ),
        )

    def mul(self, other: ArrayLike) -> "Tensor":
        """Multiply ``other`` elementwise."""
        return self._binary_operation(
            other,
            np.multiply,
            "mul",
            lambda gradient, ctx: (
                _unbroadcast(gradient * ctx.saved[1], ctx.values["left_shape"]),
                _unbroadcast(gradient * ctx.saved[0], ctx.values["right_shape"]),
            ),
        )

    def div(self, other: ArrayLike) -> "Tensor":
        """Divide by ``other`` elementwise."""
        return self._binary_operation(
            other,
            np.divide,
            "div",
            lambda gradient, ctx: (
                _unbroadcast(gradient / ctx.saved[1], ctx.values["left_shape"]),
                _unbroadcast(
                    -gradient * ctx.saved[0] / (ctx.saved[1] ** 2),
                    ctx.values["right_shape"],
                ),
            ),
        )

    def pow(self, other: ArrayLike) -> "Tensor":
        """Raise elements to ``other`` elementwise."""
        return self._binary_operation(
            other,
            np.power,
            "pow",
            _pow_gradients,
        )

    def matmul(self, other: ArrayLike) -> "Tensor":
        """Multiply this Tensor by ``other`` using NumPy matmul semantics."""
        return self._binary_operation(
            other,
            np.matmul,
            "matmul",
            lambda gradient, ctx: _matmul_gradients(ctx.saved[0], ctx.saved[1], gradient),
        )

    def exp(self) -> "Tensor":
        """Apply the exponential function elementwise."""
        return self._unary_operation(
            np.exp,
            "exp",
            lambda gradient, ctx: (gradient * ctx.saved[1],),
        )

    def log(self) -> "Tensor":
        """Apply the natural logarithm elementwise."""
        return self._unary_operation(
            np.log,
            "log",
            lambda gradient, ctx: (gradient / ctx.saved[0],),
        )

    def relu(self) -> "Tensor":
        """Apply rectified linear unit activation elementwise."""
        return self._unary_operation(
            lambda value: np.maximum(value, 0),
            "relu",
            lambda gradient, ctx: (gradient * (ctx.saved[0] > 0),),
        )

    def sigmoid(self) -> "Tensor":
        """Apply the logistic sigmoid activation elementwise."""
        positive = self._data >= 0
        result = np.empty_like(self._data, dtype=np.result_type(self._data, np.float64))
        result[positive] = 1 / (1 + np.exp(-self._data[positive]))
        exp_values = np.exp(self._data[~positive])
        result[~positive] = exp_values / (1 + exp_values)
        if not self.requires_grad:
            return Tensor._from_array(result)
        context = Context()
        context.save_for_backward(result)
        operation = Operation(
            "sigmoid",
            (self,),
            context,
            lambda gradient, ctx: (gradient * ctx.saved[0] * (1 - ctx.saved[0]),),
        )
        return self._with_operation(result, (self,), operation)

    def __getitem__(self, index: Any) -> "Tensor":
        """Return a Tensor selected with NumPy indexing semantics."""
        try:
            result = self._data[index]
        except (IndexError, TypeError, ValueError) as error:
            raise ShapeError(
                f"Invalid index {index!r} for Tensor with shape {self.shape}."
            ) from error
        data = np.asarray(result)
        if not self.requires_grad:
            return Tensor._from_array(data)
        context = Context(values={"index": index, "input_shape": self.shape})

        def backward(gradient: Gradient, ctx: Context) -> tuple[Gradient]:
            result_gradient = np.zeros(ctx.values["input_shape"], dtype=gradient.dtype)
            np.add.at(result_gradient, ctx.values["index"], gradient)
            return (result_gradient,)

        operation = Operation("getitem", (self,), context, backward)
        return self._with_operation(data, (self,), operation)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Tensor({self._data!r})"

    def __add__(self, other: ArrayLike) -> "Tensor":
        return self.add(other)

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return _to_tensor(other).add(self)

    def __sub__(self, other: ArrayLike) -> "Tensor":
        return self.sub(other)

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return _to_tensor(other).sub(self)

    def __mul__(self, other: ArrayLike) -> "Tensor":
        return self.mul(other)

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return _to_tensor(other).mul(self)

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        return self.div(other)

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return _to_tensor(other).div(self)

    def __pow__(self, other: ArrayLike) -> "Tensor":
        return self.pow(other)

    def __rpow__(self, other: ArrayLike) -> "Tensor":
        return _to_tensor(other).pow(self)

    def __matmul__(self, other: ArrayLike) -> "Tensor":
        return self.matmul(other)

    def __rmatmul__(self, other: ArrayLike) -> "Tensor":
        return _to_tensor(other).matmul(self)

    def __neg__(self) -> "Tensor":
        return self._unary_operation(
            np.negative,
            "neg",
            lambda gradient, ctx: (-gradient,),
        )

    def _unary_operation(
        self,
        operation: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
        name: str,
        backward: Callable[[Gradient, Context], tuple[Gradient | None, ...]],
    ) -> "Tensor":
        try:
            data = operation(self._data)
        except TypeError as error:
            raise MiniTensorError(f"Could not execute {name} on dtype {self.dtype}.") from error
        if not self.requires_grad or not is_grad_enabled():
            return Tensor._from_array(data)
        context = Context()
        if name == "exp":
            context.save_for_backward(self._data, np.asarray(data))
        else:
            context.save_for_backward(self._data)
        operation_node = Operation(name, (self,), context, backward)
        return self._with_operation(data, (self,), operation_node)

    def _binary_operation(
        self,
        other: ArrayLike,
        operation: Callable[[np.ndarray[Any, Any], np.ndarray[Any, Any]], np.ndarray[Any, Any]],
        name: str,
        backward: Callable[[Gradient, Context], tuple[Gradient | None, ...]],
    ) -> "Tensor":
        right = _to_tensor(other)
        try:
            data = operation(self._data, right._data)
        except ValueError as error:
            raise ShapeError(
                f"Cannot {name} Tensors with shapes {self.shape} and {right.shape}."
            ) from error
        except TypeError as error:
            raise MiniTensorError(
                f"Cannot {name} Tensors with dtypes {self.dtype} and {right.dtype}."
            ) from error
        if not (self.requires_grad or right.requires_grad) or not is_grad_enabled():
            return Tensor._from_array(data)
        context = Context(
            values={
                "left_shape": self.shape,
                "right_shape": right.shape,
                "left_requires_grad": self.requires_grad,
                "right_requires_grad": right.requires_grad,
            }
        )
        if name in {"mul", "div", "pow", "matmul"}:
            context.save_for_backward(self._data, right._data)
            if name == "pow":
                context.save_for_backward(np.asarray(data))
        operation_node = Operation(name, (self, right), context, backward)
        return self._with_operation(data, (self, right), operation_node)

    def _reduce(
        self,
        operation: Callable[..., np.ndarray[Any, Any]],
        name: str,
        *,
        axis: Axis,
        keepdims: bool,
    ) -> "Tensor":
        try:
            data = operation(self._data, axis=axis, keepdims=keepdims)
        except (TypeError, IndexError, ValueError) as error:
            raise ShapeError(
                f"Cannot {name} Tensor with shape {self.shape} along axis {axis!r}."
            ) from error
        if not self.requires_grad or not is_grad_enabled():
            return Tensor._from_array(data)
        normalized_axis = _normalize_reduction_axis(axis, self.ndim)
        context = Context(
            values={
                "input_shape": self.shape,
                "axis": normalized_axis,
                "keepdims": keepdims,
                "input_size": self.size,
            }
        )
        context.save_for_backward(self._data, np.asarray(data))

        def backward(gradient: Gradient, ctx: Context) -> tuple[Gradient]:
            axes = ctx.values["axis"]
            expanded_gradient = gradient
            if axes is None:
                expanded_gradient = np.asarray(gradient).reshape(
                    (1,) * len(ctx.values["input_shape"])
                )
            elif not ctx.values["keepdims"]:
                for reduction_axis in axes:
                    expanded_gradient = np.expand_dims(expanded_gradient, reduction_axis)
            if name == "sum":
                result_gradient = np.broadcast_to(
                    expanded_gradient, ctx.values["input_shape"]
                ).copy()
            elif name == "mean":
                divisor = (
                    ctx.values["input_size"]
                    if axes is None
                    else int(np.prod([ctx.values["input_shape"][axis] for axis in axes]))
                )
                result_gradient = np.broadcast_to(
                    expanded_gradient / divisor, ctx.values["input_shape"]
                ).copy()
            else:
                output = ctx.saved[1]
                input_values = ctx.saved[0]
                expanded_output = output
                if axes is None:
                    expanded_output = np.asarray(output).reshape(
                        (1,) * len(ctx.values["input_shape"])
                    )
                elif not ctx.values["keepdims"]:
                    for reduction_axis in axes:
                        expanded_output = np.expand_dims(expanded_output, reduction_axis)
                mask = input_values == expanded_output
                tie_count = mask.sum(axis=axes, keepdims=True) if axes is not None else mask.sum()
                result_gradient = np.broadcast_to(
                    expanded_gradient * mask / tie_count,
                    ctx.values["input_shape"],
                ).copy()
            return (result_gradient,)

        operation_node = Operation(name, (self,), context, backward)
        return self._with_operation(data, (self,), operation_node)


def _normalize_shape_arguments(
    arguments: tuple[Union[int, tuple[int, ...]], ...],
) -> Union[int, tuple[int, ...]]:
    if len(arguments) == 1 and isinstance(arguments[0], (tuple, list)):
        return tuple(arguments[0])
    return arguments  # type: ignore[return-value]


def _normalize_axes_arguments(
    arguments: tuple[Union[int, tuple[int, ...]], ...],
) -> Union[None, tuple[int, ...]]:
    if not arguments:
        return None
    if len(arguments) == 1 and isinstance(arguments[0], (tuple, list)):
        return tuple(arguments[0])
    return arguments  # type: ignore[return-value]


def tensor(
    data: ArrayLike,
    dtype: Any = None,
    *,
    copy: bool = True,
    requires_grad: bool = False,
) -> Tensor:
    """Create a :class:`Tensor` from numeric data."""
    return Tensor(data, dtype=dtype, copy=copy, requires_grad=requires_grad)


def zeros(shape: Union[int, tuple[int, ...]], dtype: Any = np.float32) -> Tensor:
    """Create a Tensor filled with zeros."""
    try:
        return Tensor._from_array(np.zeros(shape, dtype=_validate_dtype(np.dtype(dtype))))
    except (TypeError, ValueError) as error:
        raise ShapeError(f"Cannot create zeros Tensor with shape {shape!r}.") from error


def ones(shape: Union[int, tuple[int, ...]], dtype: Any = np.float32) -> Tensor:
    """Create a Tensor filled with ones."""
    try:
        return Tensor._from_array(np.ones(shape, dtype=_validate_dtype(np.dtype(dtype))))
    except (TypeError, ValueError) as error:
        raise ShapeError(f"Cannot create ones Tensor with shape {shape!r}.") from error


def exp(value: ArrayLike) -> Tensor:
    """Apply the exponential function elementwise."""
    return _to_tensor(value).exp()


def add(left: ArrayLike, right: ArrayLike) -> Tensor:
    """Add two inputs elementwise."""
    return _to_tensor(left).add(right)


def sub(left: ArrayLike, right: ArrayLike) -> Tensor:
    """Subtract ``right`` from ``left`` elementwise."""
    return _to_tensor(left).sub(right)


def mul(left: ArrayLike, right: ArrayLike) -> Tensor:
    """Multiply two inputs elementwise."""
    return _to_tensor(left).mul(right)


def div(left: ArrayLike, right: ArrayLike) -> Tensor:
    """Divide ``left`` by ``right`` elementwise."""
    return _to_tensor(left).div(right)


def pow(left: ArrayLike, right: ArrayLike) -> Tensor:
    """Raise ``left`` to ``right`` elementwise."""
    return _to_tensor(left).pow(right)


def log(value: ArrayLike) -> Tensor:
    """Apply the natural logarithm elementwise."""
    return _to_tensor(value).log()


def relu(value: ArrayLike) -> Tensor:
    """Apply rectified linear unit activation elementwise."""
    return _to_tensor(value).relu()


def sigmoid(value: ArrayLike) -> Tensor:
    """Apply the logistic sigmoid activation elementwise."""
    return _to_tensor(value).sigmoid()


def matmul(left: ArrayLike, right: ArrayLike) -> Tensor:
    """Multiply two inputs using NumPy matmul semantics."""
    return _to_tensor(left).matmul(right)


def sum(value: ArrayLike, axis: Axis = None, keepdims: bool = False) -> Tensor:
    """Sum Tensor elements along ``axis``."""
    return _to_tensor(value).sum(axis=axis, keepdims=keepdims)


def mean(value: ArrayLike, axis: Axis = None, keepdims: bool = False) -> Tensor:
    """Compute the arithmetic mean along ``axis``."""
    return _to_tensor(value).mean(axis=axis, keepdims=keepdims)


def max(value: ArrayLike, axis: Axis = None, keepdims: bool = False) -> Tensor:
    """Compute the maximum value along ``axis``."""
    return _to_tensor(value).max(axis=axis, keepdims=keepdims)
