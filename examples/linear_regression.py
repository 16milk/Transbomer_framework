"""Train a linear model on synthetic data: python examples/linear_regression.py."""

import numpy as np

import minitensor as mt

np.random.seed(7)
x = mt.tensor(np.random.randn(128, 2).astype(np.float32))
true_weight = np.array([[2.0], [-3.0]], dtype=np.float32)
true_bias = np.array([0.5], dtype=np.float32)
y = mt.tensor(x.data @ true_weight + true_bias)

model = mt.Linear(2, 1)
optimizer = mt.SGD(model.parameters(), lr=0.1)
for _ in range(250):
    optimizer.zero_grad()
    loss = mt.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()

print(f"loss={loss.item():.6f}")
print("weight=", model.weight.data.ravel(), "bias=", model.bias.data)
