import unittest

import minitensor


class PackageTest(unittest.TestCase):
    def test_package_imports(self) -> None:
        self.assertEqual(minitensor.__version__, "0.1.0")
        self.assertEqual(
            minitensor.__all__,
            [
                "DTypeError",
                "MiniTensorError",
                "ShapeError",
                "Tensor",
                "__version__",
                "exp",
                "log",
                "matmul",
                "max",
                "mean",
                "ones",
                "relu",
                "sigmoid",
                "sum",
                "tensor",
                "zeros",
            ],
        )
