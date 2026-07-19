import numpy as np

import minitensor as mt


def test_nested_registration_and_modes():
    model = mt.Sequential(mt.Linear(2, 3), mt.ReLU(), mt.Linear(3, 1))

    assert [name for name, _ in model.named_parameters()] == [
        "0.weight",
        "0.bias",
        "2.weight",
        "2.bias",
    ]
    assert len(list(model.modules())) == 4
    assert model.training
    model.eval()
    assert not any(module.training for module in model.modules())
    model.train()
    assert all(module.training for module in model.modules())


def test_mse_and_cross_entropy_have_expected_gradients():
    prediction = mt.tensor([[1.0], [3.0]], requires_grad=True)
    mt.mse_loss(prediction, [[0.0], [1.0]]).backward()
    np.testing.assert_allclose(prediction.grad.data, [[1.0], [2.0]])

    logits = mt.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    loss = mt.cross_entropy(logits, np.array([0, 1]))
    loss.backward()
    expected = np.array([[-0.05960146, 0.05960146], [0.05960146, -0.05960146]])
    np.testing.assert_allclose(logits.grad.data, expected, rtol=1e-5)


def test_linear_regression_training_updates_and_recovers_parameters():
    np.random.seed(0)
    x_data = np.random.randn(80, 1).astype(np.float32)
    x = mt.tensor(x_data)
    y = mt.tensor(2.5 * x_data - 0.75)
    model = mt.Linear(1, 1)
    optimizer = mt.SGD(model.parameters(), lr=0.1)
    initial_weight = model.weight.data.copy()

    losses = []
    for _ in range(120):
        optimizer.zero_grad()
        loss = mt.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 1e-3
    np.testing.assert_allclose(model.weight.data, [[2.5]], atol=1e-3)
    np.testing.assert_allclose(model.bias.data, [-0.75], atol=1e-3)
    assert not np.array_equal(model.weight.data, initial_weight)
    assert model.weight.grad is not None
    optimizer.zero_grad()
    assert model.weight.grad is None


def test_adam_and_momentum_change_parameters_without_gradients_error():
    parameter = mt.Parameter([1.0])
    parameter._grad = np.array([2.0])
    mt.Momentum([parameter], lr=0.1).step()
    assert parameter.item() < 1.0

    parameter = mt.Parameter([1.0])
    parameter._grad = np.array([2.0])
    mt.Adam([parameter], lr=0.1).step()
    np.testing.assert_allclose(parameter.data, [0.9])
