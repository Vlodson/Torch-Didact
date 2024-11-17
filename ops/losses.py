"""
Implementation of different types of losses
"""

import numpy as np

from backend import LossFunction, Tensor


class MSELoss(LossFunction):
    """
    Mean square error loss implementation
    """

    def forward(  # pylint: disable=W0221
        self, targets: Tensor, predictions: Tensor
    ) -> np.float32:
        return 0.5 / targets.shape[0] * np.sum((targets - predictions) ** 2)

    def backward(  # pylint: disable=W0221
        self, targets: Tensor, predictions: Tensor
    ) -> Tensor:
        return (predictions - targets) / targets.shape[0]


class BCELoss(LossFunction):
    """
    Binary cross entropy loss implementation
    """

    def forward(  # pylint: disable=W0221
        self, targets: Tensor, predictions: Tensor
    ) -> np.float32:
        # make the loss stable for vals close to 0 and 1
        clipped_preds = np.clip(predictions, 1e-8, 1 - 1e-8)

        logs = np.where(targets == 1, np.log(clipped_preds), np.log(1 - clipped_preds))

        return -np.sum(logs) / targets.shape[0]

    def backward(  # pylint: disable=W0221
        self, targets: Tensor, predictions: Tensor
    ) -> Tensor:
        # make the loss stable for vals close to 0 and 1
        clipped_preds = np.clip(predictions, 1e-8, 1 - 1e-8)

        dlogs = np.where(targets == 1, 1 / clipped_preds, -1 / (1 - clipped_preds))

        return -dlogs / targets.shape[0]
