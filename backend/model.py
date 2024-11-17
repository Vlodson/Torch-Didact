"""
Module with the abstraction of the model
"""

from backend.operation import Operation


class Model:
    """
    Implementation of the model abstraction
    """

    def __init__(self):
        """
        Initialization of all the operations of the model
        All other methods assume that all the operations are present inside __init__
        """

    def forward(self):
        """
        Implementation of the forward pass of the model
        """

    def __call__(self, *args, **kwds):
        return self.forward(*args, *kwds)

    def get_params(self):
        """
        Gets all the parameters of all the operations initialized inside the model
        """
        ops = [op for op in self.__dict__.values() if isinstance(op, Operation)]

        return [param for op in ops if op.params is not None for param in op.params]
