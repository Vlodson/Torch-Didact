"""
Module with the data loader class
"""

import random

from backend import Dataset


class DataLoader:
    """
    Implementation of the data loader class for iterating through datasets
    """

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size

        # assumes you already shuffled your dataset
        # this just shuffles the batches within the dataset
        self.shuffle = shuffle

    def __iter__(self):
        self.index_pairs = [  # pylint: disable=W0201
            (i, min(i + self.batch_size, len(self.dataset)))
            for i in range(0, len(self.dataset), self.batch_size)
        ]

        if self.shuffle:
            random.shuffle(self.index_pairs)

        self.current_index = 0  # pylint: disable=W0201

        return self

    def __next__(self):
        if self.current_index >= len(self.index_pairs):
            raise StopIteration

        start, end = self.index_pairs[self.current_index]
        self.current_index += 1

        # assumes you can slice your dataset :/
        return self.dataset[start:end]
