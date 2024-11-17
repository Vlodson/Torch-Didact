"""
Module for weight initialization techniques
"""

import numpy as np

from backend import Tensor


def glorot_init(fan_in: int, fan_out: int) -> Tensor:
    """
    Glorot weight initialization

    :param fan_in: number of input dimensions
    :param fan_out: number of output dimensions
    :return: weights initialized using glorot uniform initialization
    """
    fan_avg = (fan_in + fan_out) / 2
    factor = np.sqrt(3 / fan_avg)
    return np.random.uniform(-factor, factor, size=(fan_in, fan_out))


def he_init(fan_in: int, fan_out: int) -> Tensor:
    """
    He weight initialization, works better than Glorot for ReLU

    :param fan_in: number of input dimensions
    :return: weights initialized using he uniform initialization
    """
    factor = np.sqrt(6 / fan_in)
    return np.random.uniform(-factor, factor, size=(fan_in, fan_out))
