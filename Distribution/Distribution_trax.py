"""
@author: Benjamin
This file offers a consolidated utility option for Ozturk Algorithm distributions
"""
from trax.fastmath import numpy as np

from Constants.Constants import STATS_SET


class Distribution:
    def __init__(self, dist):
        self.title = ''
        self.name(dist)
        self.rv = [dist(loc=0, scale=1)]

    def name(self, stats):
        self.title = ((str(stats).split('.')[-1]).split('_')[0]).capitalize() \
            if 'scipy.stats' in str(type(stats)) else str(stats)

    def rvs(self, size, samples) -> np.ndarray:
        larger_size = size * samples
        evaluated = [np.reshape(self.rv[i].rvs(size=larger_size), (samples, size, 1)) for i in range(len(self.rv))]
        return np.dstack(tuple(evaluated)).astype(float)

    @property
    def dimension(self):
        return len(self.rv)

    def __str__(self):
        return self.title

    def __eq__(self, other):
        return str(other) == str(self)

    def __add__(self, other):
        if other is None:
            return self
        assert isinstance(other, Distribution), type(other)
        self.rv += other.rv
        self.title += ' ' + other.title
        return self

    def __radd__(self, other):
        if other is None:
            return self
        else:
            return self + other

    def __hash__(self):
        return hash(self.title)


if __name__ == '__main__':
    d = Distribution(STATS_SET[0])
    print(d, d.rvs(1, 1))
    print(hash(d))
    d += Distribution(STATS_SET[1])
    print(d, d.rvs(1, 1))
    print(hash(d))
    d += Distribution(STATS_SET[2])
    print(d, d.rvs(1, 1))
    print(hash(d))
