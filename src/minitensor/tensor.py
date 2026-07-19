"""Tensor data structure and NumPy-backed eager operators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Union

import numpy as np

from .errors import DTypeError, MiniTensorError, ShapeError

ArrayLike = Union["Tensor", np.ndarray, Sequence[Any], int, float, bool, np.number]
Axis = Union[int, tuple[int, ...], None]


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


class Tensor:
    """A numeric, NumPy-backed multidimensional array.

    ``Tensor`` owns a copy of constructor input by default. Use :meth:`from_numpy`
    when deliberate storage sharing with a NumPy array is required. Results of basic
    indexing, ``reshape``, and ``transpose`` preserve NumPy's view-or-copy behavior.
    """

    def __init__(self, data: ArrayLike, dtype: Any = None, *, copy: bool = True) -> None:
        self._data = _as_array(data, dtype=dtype, copy=copy)

    @classmethod
    def _from_array(cls, array: Any) -> "Tensor":
        instance = cls.__new__(cls)
        instance._data = np.asarray(array)
        _validate_dtype(instance._data.dtype)
        return instance

    @classmethod
    def from_numpy(cls, array: np.ndarray[Any, Any]) -> "Tensor":
        """Create a Tensor that shares storage with ``array``.

        The caller must pass a NumPy array with a numeric or boolean dtype.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("Tensor.from_numpy expects a numpy.ndarray.")
        return cls._from_array(array)

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
        return Tensor._from_array(self._data.astype(target_dtype, copy=True))

    def reshape(self, *shape: Union[int, tuple[int, ...]]) -> "Tensor":
        """Return a Tensor with a new shape using NumPy reshape semantics."""
        target_shape = _normalize_shape_arguments(shape)
        try:
            return Tensor._from_array(self._data.reshape(target_shape))
        except ValueError as error:
            raise ShapeError(
                f"Cannot reshape Tensor with shape {self.shape} to {target_shape}."
            ) from error

    def transpose(self, *axes: Union[int, tuple[int, ...]]) -> "Tensor":
        """Return a Tensor with dimensions permuted using NumPy transpose semantics."""
        normalized_axes = _normalize_axes_arguments(axes)
        try:
            return Tensor._from_array(self._data.transpose(normalized_axes))
        except ValueError as error:
            raise ShapeError(
                f"Cannot transpose Tensor with shape {self.shape} using axes {normalized_axes}."
            ) from error

    def sum(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        """Sum Tensor elements along ``axis``."""
        return self._reduce(np.sum, "sum", axis=axis, keepdims=keepdims)

    def mean(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        """Compute the arithmetic mean along ``axis``."""
        return self._reduce(np.mean, "mean", axis=axis, keepdims=keepdims)

    def max(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        """Compute the maximum value along ``axis``."""
        return self._reduce(np.max, "max", axis=axis, keepdims=keepdims)

    def matmul(self, other: ArrayLike) -> "Tensor":
        """Multiply this Tensor by ``other`` using NumPy matmul semantics."""
        return self._binary_operation(other, np.matmul, "matmul")

    def exp(self) -> "Tensor":
        """Apply the exponential function elementwise."""
        return self._unary_operation(np.exp, "exp")

    def log(self) -> "Tensor":
        """Apply the natural logarithm elementwise."""
        return self._unary_operation(np.log, "log")

    def relu(self) -> "Tensor":
        """Apply rectified linear unit activation elementwise."""
        return Tensor._from_array(np.maximum(self._data, 0))

    def sigmoid(self) -> "Tensor":
        """Apply the logistic sigmoid activation elementwise."""
        positive = self._data >= 0
        result = np.empty_like(self._data, dtype=np.result_type(self._data, np.float64))
        result[positive] = 1 / (1 + np.exp(-self._data[positive]))
        exp_values = np.exp(self._data[~positive])
        result[~positive] = exp_values / (1 + exp_values)
        return Tensor._from_array(result)

    def __getitem__(self, index: Any) -> "Tensor":
        """Return a Tensor selected with NumPy indexing semantics."""
        try:
            result = self._data[index]
        except (IndexError, TypeError) as error:
            raise ShapeError(
                f"Invalid index {index!r} for Tensor with shape {self.shape}."
            ) from error
        return Tensor._from_array(np.asarray(result))

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Tensor({self._data!r})"

    def __add__(self, other: ArrayLike) -> "Tensor":
        return self._binary_operation(other, np.add, "add")

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return _to_tensor(other)._binary_operation(self, np.add, "add")

    def __sub__(self, other: ArrayLike) -> "Tensor":
        return self._binary_operation(other, np.subtract, "sub")

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return _to_tensor(other)._binary_operation(self, np.subtract, "sub")

    def __mul__(self, other: ArrayLike) -> "Tensor":
        return self._binary_operation(other, np.multiply, "mul")

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return _to_tensor(other)._binary_operation(self, np.multiply, "mul")

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        return self._binary_operation(other, np.divide, "div")

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return _to_tensor(other)._binary_operation(self, np.divide, "div")

    def __pow__(self, other: ArrayLike) -> "Tensor":
        return self._binary_operation(other, np.power, "pow")

    def __rpow__(self, other: ArrayLike) -> "Tensor":
        return _to_tensor(other)._binary_operation(self, np.power, "pow")

    def __matmul__(self, other: ArrayLike) -> "Tensor":
        return self.matmul(other)

    def __rmatmul__(self, other: ArrayLike) -> "Tensor":
        return _to_tensor(other).matmul(self)

    def __neg__(self) -> "Tensor":
        return self._unary_operation(np.negative, "neg")

    def _unary_operation(
        self,
        operation: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
        name: str,
    ) -> "Tensor":
        try:
            return Tensor._from_array(operation(self._data))
        except TypeError as error:
            raise MiniTensorError(f"Could not execute {name} on dtype {self.dtype}.") from error

    def _binary_operation(
        self,
        other: ArrayLike,
        operation: Callable[[np.ndarray[Any, Any], np.ndarray[Any, Any]], np.ndarray[Any, Any]],
        name: str,
    ) -> "Tensor":
        right = _to_tensor(other)
        try:
            return Tensor._from_array(operation(self._data, right._data))
        except ValueError as error:
            raise ShapeError(
                f"Cannot {name} Tensors with shapes {self.shape} and {right.shape}."
            ) from error
        except TypeError as error:
            raise MiniTensorError(
                f"Cannot {name} Tensors with dtypes {self.dtype} and {right.dtype}."
            ) from error

    def _reduce(
        self,
        operation: Callable[..., np.ndarray[Any, Any]],
        name: str,
        *,
        axis: Axis,
        keepdims: bool,
    ) -> "Tensor":
        try:
            return Tensor._from_array(operation(self._data, axis=axis, keepdims=keepdims))
        except (TypeError, np.AxisError, ValueError) as error:
            raise ShapeError(
                f"Cannot {name} Tensor with shape {self.shape} along axis {axis!r}."
            ) from error


def _normalize_shape_arguments(
    arguments: tuple[Union[int, tuple[int, ...]], ...],
) -> Union[int, tuple[int, ...]]:
    if len(arguments) == 1 and isinstance(arguments[0], tuple):
        return arguments[0]
    return arguments  # type: ignore[return-value]


def _normalize_axes_arguments(
    arguments: tuple[Union[int, tuple[int, ...]], ...],
) -> Union[None, tuple[int, ...]]:
    if not arguments:
        return None
    if len(arguments) == 1 and isinstance(arguments[0], tuple):
        return arguments[0]
    return arguments  # type: ignore[return-value]


def tensor(data: ArrayLike, dtype: Any = None, *, copy: bool = True) -> Tensor:
    """Create a :class:`Tensor` from numeric data."""
    return Tensor(data, dtype=dtype, copy=copy)


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
