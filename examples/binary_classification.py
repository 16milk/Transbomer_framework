"""Train an MLP on XOR: python examples/binary_classification.py."""

import numpy as np

import minitensor as mt

np.random.seed(3)
x = mt.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
labels = np.array([0, 1, 1, 0])
model = mt.Sequential(mt.Linear(2, 8), mt.ReLU(), mt.Linear(8, 2))
optimizer = mt.Adam(model.parameters(), lr=0.05)

for _ in range(1200):
    optimizer.zero_grad()
    loss = mt.cross_entropy(model(x), labels)
    loss.backward()
    optimizer.step()

predictions = np.argmax(model(x).data, axis=1)
print(f"loss={loss.item():.6f}, accuracy={(predictions == labels).mean():.2f}")
