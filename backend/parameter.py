"""
Abstraction of a parameter of a neural network
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class Parameter:
    """
    Parameter dataclass that holds the value of the parameter and its gradient
    """

    val: npt.NDArray[np.float32]
    grad: npt.NDArray[np.float32]
