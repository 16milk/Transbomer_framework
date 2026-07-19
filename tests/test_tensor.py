import unittest

import numpy as np

import minitensor as mt


class TensorCreationTest(unittest.TestCase):
    def test_metadata_and_factory_functions(self) -> None:
        tensor = mt.Tensor([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        self.assertEqual(tensor.shape, (2, 3))
        self.assertEqual(tensor.ndim, 2)
        self.assertEqual(tensor.size, 6)
        self.assertEqual(tensor.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(mt.zeros((2, 2)).data, np.zeros((2, 2), dtype=np.float32))
        np.testing.assert_array_equal(mt.ones(3).data, np.ones(3, dtype=np.float32))

    def test_constructor_owns_input_by_default(self) -> None:
        source = np.array([1.0, 2.0])
        tensor = mt.Tensor(source)
        source[0] = 99.0

        np.testing.assert_array_equal(tensor.data, [1.0, 2.0])

    def test_explicit_numpy_sharing_and_numpy_export(self) -> None:
        source = np.array([1.0, 2.0])
        tensor = mt.Tensor.from_numpy(source)
        source[0] = 9.0
        self.assertEqual(tensor.data[0], 9.0)

        exported = tensor.numpy()
        exported[0] = 3.0
        self.assertEqual(tensor.data[0], 9.0)

        tensor.numpy(copy=False)[1] = 5.0
        self.assertEqual(tensor.data[1], 5.0)

    def test_rejects_non_numeric_dtypes(self) -> None:
        with self.assertRaises(mt.DTypeError):
            mt.Tensor(["not", "numeric"])

        with self.assertRaises(mt.DTypeError):
            mt.zeros((2, 2), dtype="U8")

        with self.assertRaises(TypeError):
            mt.Tensor.from_numpy([1, 2])  # type: ignore[arg-type]


class TensorShapeAndIndexingTest(unittest.TestCase):
    def test_indexing_and_basic_views(self) -> None:
        tensor = mt.Tensor(np.arange(12).reshape(3, 4))
        sliced = tensor[:, 1:]
        reshaped = tensor.reshape(4, 3)
        transposed = tensor.transpose(1, 0)

        np.testing.assert_array_equal(sliced.data, np.arange(12).reshape(3, 4)[:, 1:])
        self.assertTrue(np.shares_memory(tensor.data, sliced.data))
        self.assertTrue(np.shares_memory(tensor.data, reshaped.data))
        self.assertTrue(np.shares_memory(tensor.data, transposed.data))
        self.assertEqual(tensor[1, 2].item(), 6)
        self.assertEqual(tensor.T.shape, (4, 3))

    def test_advanced_indexing_preserves_numpy_copy_behavior(self) -> None:
        tensor = mt.Tensor(np.arange(6))
        selected = tensor[[1, 3, 5]]

        np.testing.assert_array_equal(selected.data, [1, 3, 5])
        self.assertFalse(np.shares_memory(tensor.data, selected.data))

    def test_shape_errors(self) -> None:
        tensor = mt.Tensor(np.arange(6))
        with self.assertRaises(mt.ShapeError):
            tensor.reshape(4, 4)
        with self.assertRaises(mt.ShapeError):
            tensor.transpose(0, 0)
        with self.assertRaises(mt.ShapeError):
            tensor.sum(axis=2)
        with self.assertRaises(mt.ShapeError):
            _ = tensor[9]


class TensorOperatorsTest(unittest.TestCase):
    def test_elementwise_and_broadcast_operators_match_numpy(self) -> None:
        values = np.array([[1.0], [2.0]])
        offsets = np.array([3.0, 4.0, 5.0])
        tensor = mt.Tensor(values)

        np.testing.assert_allclose((tensor + offsets).data, values + offsets)
        np.testing.assert_allclose((tensor - offsets).data, values - offsets)
        np.testing.assert_allclose((tensor * offsets).data, values * offsets)
        np.testing.assert_allclose((tensor / 2).data, values / 2)
        np.testing.assert_allclose((2 / tensor).data, 2 / values)
        np.testing.assert_allclose((tensor**2).data, values**2)
        np.testing.assert_allclose((2**tensor).data, 2**values)
        np.testing.assert_allclose((-tensor).data, -values)

    def test_incompatible_broadcast_raises_shape_error(self) -> None:
        with self.assertRaises(mt.ShapeError):
            _ = mt.Tensor(np.ones((2, 3))) + mt.Tensor(np.ones((4,)))

    def test_reductions_match_numpy(self) -> None:
        values = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        tensor = mt.Tensor(values)

        np.testing.assert_allclose(tensor.sum().data, np.sum(values))
        np.testing.assert_allclose(tensor.mean(axis=(0, 2)).data, np.mean(values, axis=(0, 2)))
        np.testing.assert_allclose(
            tensor.max(axis=1, keepdims=True).data,
            np.max(values, axis=1, keepdims=True),
        )
        np.testing.assert_allclose(mt.sum(tensor, axis=2).data, np.sum(values, axis=2))
        np.testing.assert_allclose(mt.mean(tensor, axis=0).data, np.mean(values, axis=0))
        np.testing.assert_allclose(mt.max(tensor, axis=0).data, np.max(values, axis=0))

    def test_matmul_matches_numpy_for_matrix_and_batch_inputs(self) -> None:
        left = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        right = np.arange(40, dtype=np.float64).reshape(2, 4, 5)

        np.testing.assert_allclose(
            (mt.Tensor(left) @ mt.Tensor(right)).data,
            np.matmul(left, right),
        )
        np.testing.assert_allclose(mt.matmul(left, right).data, np.matmul(left, right))

        with self.assertRaises(mt.ShapeError):
            _ = mt.Tensor(np.ones((2, 3))) @ mt.Tensor(np.ones((2, 3)))

    def test_math_functions_match_numpy(self) -> None:
        values = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
        tensor = mt.Tensor(values)
        finite_values = values[1:4]
        positive_values = values[3:]

        np.testing.assert_allclose(mt.exp(finite_values).data, np.exp(finite_values))
        np.testing.assert_allclose(mt.log(positive_values).data, np.log(positive_values))
        np.testing.assert_array_equal(tensor.relu().data, np.maximum(values, 0))
        np.testing.assert_allclose(
            tensor.sigmoid().data,
            [0.0, 1 / (1 + np.exp(1)), 0.5, 1 / (1 + np.exp(-1)), 1.0],
        )
        np.testing.assert_array_equal(mt.relu(values).data, np.maximum(values, 0))
