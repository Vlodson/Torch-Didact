"""
Module with the abstraction of a dataset class
"""

import typing as t


class Dataset:
    """
    Implementation of the abstraction of a dataset
    """

    def __init__(self):
        """
        User defined dataset loading implementation inside of init
        """

    def __len__(self) -> int:
        """
        User defined len function of the whole dataset
        """

    def __getitem__(self, index) -> t.Any:
        """
        User definition of what it means to get one element of the dataset
        """
