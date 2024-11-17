"""
File with all activations that can be used
"""

import numpy as np
from backend import Operation, Tensor


class ReLU(Operation):
    """
    Implementation of the rectified linear unit function
    """

    def forward(self, x: Tensor) -> Tensor:  # pylint: disable=W0221
        return np.where(x >= 0.0, x, 0.0)

    def backward(self, x: Tensor, grad: Tensor):  # pylint: disable=W0221
        return grad * np.where(x >= 0.0, 1.0, 0.0)


class Tanh(Operation):
    """
    Implementation of the hyperbolic tangent function
    """

    def forward(self, x: Tensor) -> Tensor:  # pylint: disable=W0221
        return np.tanh(x)

    def backward(self, x: Tensor, grad: Tensor) -> Tensor:  # pylint: disable=W0221
        return grad * (1 - np.tanh(x) ** 2)


class Sigmoid(Operation):
    """
    Sigmoid/Logistic function implementation
    """

    def __sigmoid(self, x: Tensor) -> Tensor:
        # clip the input value so that it doesn't explode e^x
        clipped_x = np.clip(x, -500.0, 500.0)

        return 1 / (1 + np.exp(-clipped_x))

    def forward(self, x: Tensor) -> Tensor:  # pylint: disable=W0221
        return self.__sigmoid(x)

    def backward(self, x: Tensor, grad: Tensor) -> Tensor:  # pylint: disable=W0221
        return grad * self.__sigmoid(x) * (1 - self.__sigmoid(x))
