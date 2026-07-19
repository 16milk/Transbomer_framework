"""Overfit a tiny sequence classification task with a Transformer block."""

import numpy as np

import minitensor as mt

np.random.seed(12)
tokens = np.array([[0, 1, 1], [0, 1, 2], [3, 4, 4], [3, 4, 5]])
labels = np.array([0, 0, 1, 1])


class SequenceClassifier(mt.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = mt.Embedding(6, 8)
        self.position = mt.PositionalEncoding(8, max_length=3)
        self.block = mt.TransformerBlock(8, 2, feed_forward_dim=16, dropout=0.0)
        self.classifier = mt.Linear(8, 2)

    def forward(self, inputs: mt.Tensor) -> mt.Tensor:
        encoded = self.position(self.embedding(inputs))
        return self.classifier(self.block(encoded).mean(axis=1))


model = SequenceClassifier()
optimizer = mt.Adam(model.parameters(), lr=0.03)
for _ in range(300):
    optimizer.zero_grad()
    loss = mt.cross_entropy(model(tokens), labels)
    loss.backward()
    optimizer.step()

predictions = np.argmax(model(tokens).data, axis=1)
print(f"loss={loss.item():.6f}, accuracy={(predictions == labels).mean():.2f}")
