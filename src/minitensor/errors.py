"""Exception types shared by the MiniTensor public API."""


class MiniTensorError(Exception):
    """Base exception for errors raised by MiniTensor."""


class DTypeError(MiniTensorError):
    """Raised when a Tensor receives an unsupported data type."""


class ShapeError(MiniTensorError):
    """Raised when Tensor shapes are incompatible for an operation."""
