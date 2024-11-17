"""
Implementations of transformations of inputs
"""

import numpy as np

from backend import Tensor, Operation, Parameter
from utils import initialize


class Linear(Operation):
    """
    Implementation of a linear transformation
    """

    def __init__(self, in_dims: int, out_dims: int):
        super().__init__()

        weights = initialize.glorot_init(in_dims, out_dims)
        self.params = [Parameter(val=weights, grad=np.zeros_like(weights))]

    def forward(self, x: Tensor) -> Tensor:  # pylint: disable=W0221
        return x @ self.params[0].val

    def backward(self, x: Tensor, grad: Tensor) -> Tensor:  # pylint: disable=W0221
        self.params[0].grad = x.T @ grad
        return grad @ self.params[0].val.T
