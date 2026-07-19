"""Public package entry point for MiniTensor."""

from .autograd import no_grad
from .errors import DTypeError, MiniTensorError, ShapeError
from .nn import (
    CrossEntropyLoss,
    Linear,
    Module,
    MSELoss,
    Parameter,
    ReLU,
    Sequential,
    cross_entropy,
    mse_loss,
)
from .optim import SGD, Adam, Momentum
from .tensor import (
    Tensor,
    add,
    div,
    exp,
    log,
    matmul,
    max,
    mean,
    mul,
    ones,
    pow,
    relu,
    sigmoid,
    sub,
    sum,
    tensor,
    zeros,
)

__version__ = "0.1.0"

__all__ = [
    "DTypeError",
    "Adam",
    "CrossEntropyLoss",
    "Linear",
    "MSELoss",
    "MiniTensorError",
    "Momentum",
    "Module",
    "Parameter",
    "ReLU",
    "SGD",
    "Sequential",
    "ShapeError",
    "Tensor",
    "__version__",
    "add",
    "cross_entropy",
    "div",
    "exp",
    "log",
    "matmul",
    "max",
    "mean",
    "mul",
    "mse_loss",
    "no_grad",
    "ones",
    "pow",
    "relu",
    "sigmoid",
    "sub",
    "sum",
    "tensor",
    "zeros",
]
