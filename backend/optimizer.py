"""
Module with optimizer abstraction
"""

import numpy as np
from backend.parameter import Parameter


class Optimizer:
    """
    Implementation of the optimizer abstraction
    """

    def __init__(self, model_parameters: list[Parameter]):
        self.parameters = model_parameters

    def step(self):
        """
        Implements a single update cycle for all parameters passed in the initialization
        """

    def reset_grads(self):
        """
        Resets the parameter gradients for between learning steps
        """
        for param in self.parameters:
            param.grad = np.zeros_like(param.grad)
