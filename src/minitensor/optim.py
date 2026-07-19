"""Parameter optimizers for MiniTensor."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .nn import Parameter


class Optimizer:
    def __init__(self, parameters: Iterable[Parameter], lr: float = 1e-3) -> None:
        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        unique: list[Parameter] = []
        seen: set[int] = set()
        for parameter in parameters:
            if not isinstance(parameter, Parameter):
                raise TypeError("Optimizers accept Parameter instances only.")
            if id(parameter) not in seen:
                unique.append(parameter)
                seen.add(id(parameter))
        self.parameters = unique
        self.lr = float(lr)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 1e-2,
        momentum: float = 0.0,
    ) -> None:
        super().__init__(parameters, lr)
        if momentum < 0 or momentum >= 1:
            raise ValueError("Momentum must be in the range [0, 1).")
        self.momentum = float(momentum)
        self._velocity = {
            id(parameter): np.zeros_like(parameter.data) for parameter in self.parameters
        }

    def step(self) -> None:
        for parameter in self.parameters:
            if parameter._grad is None:
                continue
            gradient = parameter._grad
            if self.momentum:
                velocity = self._velocity[id(parameter)]
                velocity *= self.momentum
                velocity += gradient
                gradient = velocity
            parameter.data[...] -= self.lr * gradient


class Momentum(SGD):
    """Named convenience wrapper for momentum SGD."""

    def __init__(self, parameters: Iterable[Parameter], lr: float = 1e-2, momentum: float = 0.9):
        super().__init__(parameters, lr=lr, momentum=momentum)


class Adam(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(parameters, lr)
        beta1, beta2 = betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Adam betas must be in the range [0, 1).")
        self.betas = betas
        self.eps = eps
        self.step_count = 0
        self._moments = {
            id(parameter): (np.zeros_like(parameter.data), np.zeros_like(parameter.data))
            for parameter in self.parameters
        }

    def step(self) -> None:
        self.step_count += 1
        beta1, beta2 = self.betas
        for parameter in self.parameters:
            if parameter._grad is None:
                continue
            first, second = self._moments[id(parameter)]
            first *= beta1
            first += (1 - beta1) * parameter._grad
            second *= beta2
            second += (1 - beta2) * parameter._grad**2
            first_hat = first / (1 - beta1**self.step_count)
            second_hat = second / (1 - beta2**self.step_count)
            parameter.data[...] -= self.lr * first_hat / (np.sqrt(second_hat) + self.eps)


__all__ = ["Adam", "Momentum", "Optimizer", "SGD"]
