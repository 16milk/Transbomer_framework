import unittest

import minitensor


class PackageTest(unittest.TestCase):
    def test_package_imports(self) -> None:
        self.assertEqual(minitensor.__version__, "0.2.0")
        self.assertEqual(
            minitensor.__all__,
            [
                "DTypeError",
                "Adam",
                "CrossEntropyLoss",
                "Dropout",
                "Embedding",
                "FeedForward",
                "LayerNorm",
                "Linear",
                "LogSoftmax",
                "MSELoss",
                "MiniTensorError",
                "Momentum",
                "Module",
                "MultiHeadAttention",
                "Parameter",
                "PositionalEncoding",
                "ReLU",
                "SGD",
                "Sequential",
                "Softmax",
                "ShapeError",
                "Tensor",
                "TransformerBlock",
                "__version__",
                "add",
                "cross_entropy",
                "div",
                "exp",
                "log",
                "log_softmax",
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
                "softmax",
                "sub",
                "sum",
                "tensor",
                "zeros",
            ],
        )
