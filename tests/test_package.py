import unittest

import minitensor


class PackageTest(unittest.TestCase):
    def test_package_imports(self) -> None:
        self.assertEqual(minitensor.__version__, "0.1.0")
        self.assertEqual(
            minitensor.__all__,
            [
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
            ],
        )
