"""
Module with operation and graph node abstraction and global Graph
"""

import typing as t

from backend.parameter import Parameter


class Operation:
    """
    Class of a single operation, that can have parameters
    It implements the forward pass and backward pass calculations
    """

    def __init__(self):
        self.params: t.Optional[list[Parameter]] = None

    def forward(self):
        """
        Forward pass implementation
        """

    def backward(self):
        """
        Backward pass implementation
        """

    def __call__(self, *args, **kwds):
        GRAPH.append(Box(self, *args, *kwds))

        return self.forward(*args, **kwds)


class Box:
    """
    Implementation of a box, which is a node in the operations graph
    """

    def __init__(self, op: Operation, *args, **kwargs):
        self.op = op

        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwds):
        return self.op(self.args, self.kwargs)


GRAPH: list[Box] = []
