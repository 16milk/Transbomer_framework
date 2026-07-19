"""Train a small MLP on three Gaussian clusters."""

import numpy as np

import minitensor as mt

np.random.seed(11)
centers = np.array([[2, 2], [-2, 2], [0, -2]], dtype=np.float32)
x_data = np.concatenate([center + 0.35 * np.random.randn(40, 2) for center in centers])
labels = np.repeat(np.arange(3), 40)
x = mt.tensor(x_data.astype(np.float32))
model = mt.Sequential(mt.Linear(2, 12), mt.ReLU(), mt.Linear(12, 3))
optimizer = mt.SGD(model.parameters(), lr=0.08, momentum=0.9)

for _ in range(300):
    optimizer.zero_grad()
    loss = mt.cross_entropy(model(x), labels)
    loss.backward()
    optimizer.step()

accuracy = (np.argmax(model(x).data, axis=1) == labels).mean()
print(f"loss={loss.item():.6f}, accuracy={accuracy:.2f}")
