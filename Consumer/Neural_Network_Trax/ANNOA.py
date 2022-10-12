import pandas as pd
import trax
from trax.fastmath import numpy as np
import abc


class ANNOA(abc.ABC):
    def __init__(self, size: int = 200, dimension: int = 1, hidden_neurons: int = 250, optimizer: str = 'adam',
                 epochs: int = 100, full_classes: bool = False, full_data: bool = False,
                 reference_distribution='Normal'):
        self._size = int(size)
        self._dim = int(dimension)
        self._dtype = float
        self._training = {}

        # Load Reference Distribution
        self._reference_distribution = reference_distribution.capitalize()
        self._theta: np.ndarray = np.empty((1, self._size))

        # Load Training Data
        self._class_set = []
        self._distribution_names = []
        self.full_data = full_data
        self.full_classes = dimension == 1 or full_classes

        # Setting up ANNOA
        # self.train_ratio = 0.80
        # self.validation_ratio = 0.20
        # self.batch_size = 64

