"""
Module containing the loss function abstraction and the implementation of the loss ojbect
"""

from backend.types import Tensor
from backend.operation import Operation, Box, GRAPH


class LossFunction(Operation):
    """
    Abstraction of the loss function
    """

    def forward(  # pylint: disable=W0221
        self, targets: Tensor, predictions: Tensor
    ) -> Tensor:
        """
        Defines the way loss is calculated
        """

    def backward(  # pylint: disable=W0221
        self, targets: Tensor, predictions: Tensor
    ) -> Tensor:
        """
        Defines the derivative of the loss function
        """

    def __call__(self, *args, **kwds):
        GRAPH.append(Box(self, *args, **kwds))

        return Loss(self.forward(*args, **kwds))


class Loss:
    """
    Loss object implementation
    """

    def __init__(self, val: Tensor):
        self.val = val

    def backward(self):
        """
        Calculates all the gradients for the parameters of the graph of operations
        Then clears the graph for the next iteration
        """
        assert isinstance(
            GRAPH[-1].op, LossFunction
        ), "Last operator needs to be a Loss Function"

        loss_box = GRAPH[-1]
        grad = loss_box.op.backward(*loss_box.args, **loss_box.kwargs)

        for box in GRAPH[-2::-1]:
            grad = box.op.backward(*box.args, **box.kwargs, grad=grad)

        GRAPH.clear()
