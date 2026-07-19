"""Run basic eager Tensor operations introduced in development stage 1."""

import minitensor as mt


def main() -> None:
    features = mt.tensor([[1.0, 2.0], [3.0, 4.0]])
    weights = mt.tensor([[0.5, -1.0], [1.5, 2.0]])
    bias = mt.tensor([0.25, -0.25])

    logits = features @ weights + bias
    activations = mt.relu(logits)

    print(f"logits:\n{logits.numpy()}")
    print(f"relu(logits):\n{activations.numpy()}")
    print(f"mean activation: {activations.mean().item():.3f}")


if __name__ == "__main__":
    main()
