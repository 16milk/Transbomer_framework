import unittest

import numpy as np

import minitensor as mt


def finite_difference(function, values, index, epsilon=1e-5):
    plus = values.copy()
    minus = values.copy()
    plus[index] += epsilon
    minus[index] -= epsilon
    return (function(plus) - function(minus)) / (2 * epsilon)


class AutogradTest(unittest.TestCase):
    def test_scalar_chain_rule_and_grad_accumulation(self) -> None:
        x = mt.tensor(2.0, requires_grad=True)
        output = (x * x + x).sum()

        self.assertTrue(output.requires_grad)
        self.assertIsNotNone(output.grad_fn)
        output.backward()
        self.assertAlmostEqual(x.grad.item(), 5.0)

        output.backward()
        self.assertAlmostEqual(x.grad.item(), 10.0)
        x.zero_grad()
        self.assertIsNone(x.grad)

    def test_shared_subgraph_accumulates_all_paths(self) -> None:
        x = mt.tensor([1.5, -2.0], requires_grad=True)
        shared = x * x
        output = (shared + shared).sum()

        output.backward()
        np.testing.assert_allclose(x.grad.data, 4 * x.data)

    def test_broadcast_gradient_is_reduced_to_input_shape(self) -> None:
        x_values = np.arange(6.0).reshape(2, 3)
        bias_values = np.array([0.5, -1.0, 2.0])
        x = mt.tensor(x_values, requires_grad=True)
        bias = mt.tensor(bias_values, requires_grad=True)

        ((x + bias) ** 2).sum().backward()
        expected_x = 2 * (x_values + bias_values)
        expected_bias = expected_x.sum(axis=0)
        np.testing.assert_allclose(x.grad.data, expected_x)
        np.testing.assert_allclose(bias.grad.data, expected_bias)

    def test_reshape_transpose_reduction_and_matmul_gradients(self) -> None:
        values = np.arange(6.0).reshape(2, 3)
        weights = np.arange(6.0).reshape(3, 2) / 5
        x = mt.tensor(values, requires_grad=True)
        w = mt.tensor(weights, requires_grad=True)
        output = ((x.reshape(3, 2).transpose(1, 0) @ w).mean())

        output.backward()
        expected_x = finite_difference(
            lambda candidate: ((candidate.reshape(3, 2).T @ weights).mean()),
            values,
            (1, 2),
        )
        expected_w = finite_difference(
            lambda candidate: ((values.reshape(3, 2).T @ candidate).mean()),
            weights,
            (1, 0),
        )
        self.assertAlmostEqual(x.grad.data[1, 2], expected_x, places=5)
        self.assertAlmostEqual(w.grad.data[1, 0], expected_w, places=5)

    def test_reduction_and_max_gradients(self) -> None:
        values = np.array([[1.0, 3.0, 2.0], [4.0, 0.5, 2.5]])
        x = mt.tensor(values, requires_grad=True)
        output = x.sum(axis=0).mean() + x.max(axis=1).sum()

        output.backward()
        expected = np.full_like(values, 1 / values.shape[1])
        expected += np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        np.testing.assert_allclose(x.grad.data, expected)

    def test_reduction_and_shape_gradients_match_finite_difference(self) -> None:
        values = np.array([[0.4, 1.2], [1.7, 2.3]], dtype=np.float64)
        operations = [
            (
                lambda value: value.sum(axis=0).mean(),
                lambda candidate: candidate.sum(axis=0).mean(),
            ),
            (
                lambda value: value.mean(axis=1).sum(),
                lambda candidate: candidate.mean(axis=1).sum(),
            ),
            (
                lambda value: value.reshape(4).transpose().sum(),
                lambda candidate: candidate.reshape(4).transpose().sum(),
            ),
        ]
        for eager, reference in operations:
            candidate = mt.tensor(values, requires_grad=True)
            eager(candidate).backward()
            expected = np.array(
                [finite_difference(reference, values, index) for index in np.ndindex(values.shape)]
            ).reshape(values.shape)
            np.testing.assert_allclose(candidate.grad.data, expected, rtol=1e-5, atol=1e-7)

    def test_indexing_gradient_accumulates_repeated_indices(self) -> None:
        x = mt.tensor([1.0, 2.0, 3.0], requires_grad=True)
        x[[0, 0, 2]].sum().backward()

        np.testing.assert_allclose(x.grad.data, [2.0, 0.0, 1.0])

    def test_batched_matmul_gradient_matches_finite_difference(self) -> None:
        left_values = np.arange(8.0).reshape(2, 2, 2) / 4
        right_values = np.arange(4.0).reshape(2, 2) / 3
        left = mt.tensor(left_values, requires_grad=True)
        right = mt.tensor(right_values, requires_grad=True)
        (left @ right).sum().backward()

        expected_left = np.empty_like(left_values)
        expected_right = np.empty_like(right_values)
        for index in np.ndindex(left_values.shape):
            expected_left[index] = finite_difference(
                lambda candidate: np.matmul(candidate, right_values).sum(),
                left_values,
                index,
            )
        for index in np.ndindex(right_values.shape):
            expected_right[index] = finite_difference(
                lambda candidate: np.matmul(left_values, candidate).sum(),
                right_values,
                index,
            )
        np.testing.assert_allclose(left.grad.data, expected_left, rtol=1e-5)
        np.testing.assert_allclose(right.grad.data, expected_right, rtol=1e-5)

    def test_all_basic_unary_and_binary_gradients_against_finite_difference(self) -> None:
        values = np.array([0.4, 1.2], dtype=np.float64)
        other_values = np.array([1.5, 0.8], dtype=np.float64)
        operations = [
            (lambda x, y: (x * y).sum(), lambda x, y: y),
            (lambda x, y: (x / y).sum(), lambda x, y: 1 / y),
            (lambda x, y: (x**y).sum(), lambda x, y: y * x ** (y - 1)),
            (lambda x, y: x.exp().sum(), lambda x, y: np.exp(x)),
            (lambda x, y: x.log().sum(), lambda x, y: 1 / x),
            (lambda x, y: x.relu().sum(), lambda x, y: (x > 0).astype(float)),
            (
                lambda x, y: x.sigmoid().sum(),
                lambda x, y: 1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x))),
            ),
        ]

        for operation, expected_gradient in operations:
            x = mt.tensor(values, requires_grad=True)
            operation(x, other_values).backward()
            np.testing.assert_allclose(
                x.grad.data,
                expected_gradient(values, other_values),
                rtol=1e-5,
            )

    def test_operator_gradients_match_centered_finite_difference(self) -> None:
        values = np.array([0.4, 1.2], dtype=np.float64)
        other_values = np.array([1.5, 0.8], dtype=np.float64)
        operations = [
            (lambda x, y: x + y, lambda x, y: x + y, True),
            (lambda x, y: x - y, lambda x, y: x - y, True),
            (lambda x, y: x * y, lambda x, y: x * y, True),
            (lambda x, y: x / y, lambda x, y: x / y, True),
            (lambda x, y: x**y, lambda x, y: x**y, True),
            (lambda x, y: x.exp(), lambda x, y: np.exp(x), False),
            (lambda x, y: x.log(), lambda x, y: np.log(x), False),
            (lambda x, y: x.relu(), lambda x, y: np.maximum(x, 0), False),
            (
                lambda x, y: x.sigmoid(),
                lambda x, y: 1 / (1 + np.exp(-x)),
                False,
            ),
        ]

        for operation, reference, has_second_input in operations:
            x = mt.tensor(values, requires_grad=True)
            y = mt.tensor(other_values, requires_grad=True) if has_second_input else other_values
            operation(x, y).sum().backward()
            expected_x = np.array(
                [
                    finite_difference(
                        lambda candidate: reference(candidate, other_values).sum(),
                        values,
                        index,
                    )
                    for index in range(values.size)
                ]
            )
            np.testing.assert_allclose(x.grad.data, expected_x, rtol=1e-5, atol=1e-7)
            if has_second_input:
                expected_y = np.array(
                    [
                        finite_difference(
                            lambda candidate: reference(values, candidate).sum(),
                            other_values,
                            index,
                        )
                        for index in range(other_values.size)
                    ]
                )
                np.testing.assert_allclose(y.grad.data, expected_y, rtol=1e-5, atol=1e-7)

    def test_explicit_upstream_gradient_and_scalar_output_rule(self) -> None:
        x = mt.tensor([1.0, 2.0], requires_grad=True)
        with self.assertRaises(mt.ShapeError):
            (x * 2).backward()

        output = x * 2
        output.backward(mt.tensor([3.0, 4.0]))
        np.testing.assert_allclose(x.grad.data, [6.0, 8.0])

    def test_detach_and_no_grad_disconnect_the_graph(self) -> None:
        x = mt.tensor([1.0, 2.0], requires_grad=True)
        detached = (x * 2).detach()
        self.assertFalse(detached.requires_grad)
        self.assertIsNone(detached.grad_fn)

        with mt.no_grad():
            result = x * 3
        self.assertFalse(result.requires_grad)
        self.assertIsNone(result.grad_fn)

        detached_sum = detached.sum()
        with self.assertRaises(mt.MiniTensorError):
            detached_sum.backward()

    def test_integer_tensors_cannot_require_grad(self) -> None:
        with self.assertRaises(mt.DTypeError):
            mt.tensor([1, 2], requires_grad=True)
