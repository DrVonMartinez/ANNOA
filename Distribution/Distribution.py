#  -*- coding: utf-8 -*-
"""
@author: Benjamin
This file offers a consolidated utility option for Ozturk Algorithm distributions
"""
import numpy as np


class Distribution:
    def __init__(self, dist):
        self.title = ''
        self.name(dist)
        self.rv = [dist(loc=0, scale=1)]

    def name(self, stats):
        if 'scipy.stats' in str(type(stats)):
            full_name = (str(stats).split('.')[3]).split('_')[0]
            self.title = full_name.capitalize()
        else:
            self.title = str(stats)

    def rvs(self, size, samples, dtype):
        larger_size = size * samples
        evaluated = np.reshape(self.rv[0].rvs(size=larger_size), (samples, size, 1))
        for i in range(1, len(self.rv)):
            evaluated = np.dstack((evaluated, np.reshape(self.rv[i].rvs(size=larger_size), (samples, size, 1))))
        return evaluated.astype(dtype=dtype)

    def dimension(self):
        return len(self.rv)

    def __str__(self):
        return self.title

    def __eq__(self, other):
        return str(other) == str(self)

    def __len__(self):
        return len(self.rv)

    def __add__(self, other):
        if not other:
            return self
        self.rv += other.rv
        self.title += ' ' + other.title
        return self

    def __radd__(self, other):
        if not other:
            return self
        self.rv = other.rv + self.rv
        self.title = other.title + self.title
        return self

    def __hash__(self):
        return hash(self.title)
