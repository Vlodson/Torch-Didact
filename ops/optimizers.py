"""
Module with implementations of optimizers
"""

import numpy as np

from backend import Optimizer, Parameter


class GradientDescent(Optimizer):
    """
    Implementation of the ordinary gradient descent algorithm
    """

    def __init__(
        self, model_parameters: list[Parameter], learn_rate: np.float32 = 1e-2
    ):
        super().__init__(model_parameters)

        self.learn_rate = learn_rate

    def step(self):
        for parameter in self.parameters:
            parameter.val = parameter.val - self.learn_rate * parameter.grad


class MomentumGradientDescent(Optimizer):
    """
    Implementation of the momentum gradient descent algorithm
    """

    def __init__(
        self,
        model_parameters: list[Parameter],
        learn_rate: np.float32 = 1e-2,
        momentum: np.float32 = 0.9,
    ):
        super().__init__(model_parameters)

        self.learn_rate = learn_rate
        self.momentum = momentum

        self.prev_grads = [np.zeros_like(param.val) for param in self.parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            param.val = param.val - (
                self.learn_rate * param.grad + self.momentum * self.prev_grads[i]
            )

            self.prev_grads[i] = param.grad


class Adam(Optimizer):
    """
    Implements the Adaptive Momentum Estimation optimization algorithm
    """

    def __init__(
        self,
        model_parameters: list[Parameter],
        learn_rate: np.float32 = 1e-2,
        beta1: np.float32 = 0.99,
        beta2: np.float32 = 0.9,
        eps: np.float32 = 1e-8,
    ):
        super().__init__(model_parameters)

        self.learn_rate = learn_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.prev_grad = [np.zeros_like(param.val) for param in self.parameters]
        self.m = [np.zeros_like(param.val) for param in self.parameters]
        self.v = [np.zeros_like(param.val) for param in self.parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * self.prev_grad[i]
            self.v[i] = (
                self.beta2 * self.v[i] + (1 - self.beta2) * self.prev_grad[i] ** 2
            )

            # may have needed to use beta ^ t, t=step of optim
            m_hat = self.m[i] / (1 - self.beta1)
            v_hat = self.v[i] / (1 - self.beta2)

            param.val = param.val - self.learn_rate * m_hat / (v_hat**0.5 + self.eps)

            self.prev_grad[i] = param.grad
