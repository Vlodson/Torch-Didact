"""
Module with all metric functions
"""

import numpy as np

from backend import Tensor


def logit_to_cat(logits: Tensor) -> Tensor:
    """
    Transforms a logit tensor into a binary category
    The limit is set to 0.5
    """

    return np.where(logits > 0.5, 1.0, 0.0)


def true_positive(target_cats: Tensor, prediction_cats: Tensor) -> int:
    """
    Calculates the number of true positives for a given set of target and
    prediction categorical tensors

    They need to be one-dimensional tensors
    """

    return np.sum((target_cats == 1) & (prediction_cats == 1))


def true_negative(target_cats: Tensor, prediction_cats: Tensor) -> int:
    """
    Calculates the number of true negatives for a given set of target and
    prediction categorical tensors

    They need to be one-dimensional tensors
    """

    return np.sum((target_cats == 0) & (prediction_cats == 0))


def false_positive(target_cats: Tensor, prediction_cats: Tensor) -> int:
    """
    Calculates the number of false positives for a given set of target and
    prediction categorical tensors

    They need to be one-dimensional tensors
    """

    return np.sum((target_cats == 0) & (prediction_cats == 1))


def false_negative(target_cats: Tensor, prediction_cats: Tensor) -> int:
    """
    Calculates the number of false negatives for a given set of target and
    prediction categorical tensors

    They need to be one-dimensional tensors
    """

    return np.sum((target_cats == 1) & (prediction_cats == 0))


def accuracy(target_cats: Tensor, prediction_cats: Tensor) -> float:
    """
    Calculates the accuracy for a given set of target and
    prediction categorical tensors

    They need to be one-dimensional tensors
    """
    tp = true_positive(target_cats, prediction_cats)
    tn = true_negative(target_cats, prediction_cats)

    return 1.0 * (tp + tn) / target_cats.shape[0]


def precision(target_cats: Tensor, prediction_cats: Tensor) -> float:
    """
    Calculates the precision for a given set of target and
    prediction categorical tensors

    They need to be one-dimensional tensors
    """
    tp = true_positive(target_cats, prediction_cats)
    fp = false_positive(target_cats, prediction_cats)

    return 1.0 * tp / (tp + fp)


def recall(target_cats: Tensor, prediction_cats: Tensor) -> float:
    """
    Calculates the recall for a given set of target and
    prediction categorical tensors

    They need to be one-dimensional tensors
    """
    tp = true_positive(target_cats, prediction_cats)
    fn = false_negative(target_cats, prediction_cats)

    return 1.0 * tp / (tp + fn)


def f1(target_cats: Tensor, prediction_cats: Tensor) -> float:
    """
    Calculates the f1 measure for a given set of target and
    prediction categorical tensors

    They need to be one-dimensional tensors
    """
    prec = precision(target_cats, prediction_cats)
    rec = recall(target_cats, prediction_cats)

    return 2.0 * prec * rec / (prec + rec)
