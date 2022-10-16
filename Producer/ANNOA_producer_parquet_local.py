#  -*- coding: utf-8 -*-
"""
@author: Benjamin
The Goal is to have Ozturk's Algorithm:
    Utilize Data Mining:
        For Classification
        For Cluster Detection
        For Association Learning

"""

import itertools
import warnings

# Artificial Neural Network Ozturk
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale as standardize

from Constants.Constants import STATS_SET, REFERENCE_DICTIONARY, MONTE_CARLO
from Distribution.Distribution import Distribution


class OzturkTrain:
    def __init__(self, monte_carlo=2000, size=200, dimension=1):
        """
        loss function selection following suggestions from https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c
        Optimizer selection following suggestions from https://algorithmia.com/blog/introduction-to-optimizers
        :param monte_carlo:
        :param size:
        :param dimension:
        """
        self.__monte_carlo = int(monte_carlo)
        self.__size = size
        self.__dim = int(dimension)
        self.__theta = None
        self.__theta_distribution = None
        self.__class_set: set[Distribution] = set()
        if self.__monte_carlo < 1000:
            warnings.warn('Warning! Small Number of Monte Carlo Simulations')

    def train(self, distributions):
        """
        :param distributions:
        :return:
        """
        assert self.__theta is not None
        for index in itertools.product(range(len(distributions)), repeat=self.__dim):
            dist = sum([Distribution(distributions[index[i]]) for i in range(len(index))], start=None)
            self.__class_set.add(dist)
        columns = self.__determine_columns()
        for dist in self.__class_set:
            distribution_df = self.__oa_hidden_single(dist, columns).astype(dtype=float).fillna(value=0)
            self.__store_seq(distribution_df, dist)

    def train_single_distribution(self, distribution_set: list):
        distribution_sequence = sum([Distribution(distribution_set[i]) for i in range(self.__dim)], start=None)
        self.__class_set.add(distribution_sequence)
        columns = self.__determine_columns()
        distribution_df = self.__oa_hidden_single(distribution_sequence, columns)
        self.__store_seq(distribution_df, distribution_sequence)

    def __determine_columns(self) -> list[str]:
        columns = ['U', 'V']
        for i in range(self.__size + 1):
            columns.append('U' + str(i))
            columns.append('V' + str(i))
        distribution_set = list(map(str, self.__class_set))
        single_dist_set: set = set(distribution_set + [dist.split(' ')[0] for dist in distribution_set])
        return columns + list(single_dist_set)

    def __oa_hidden_single(self, distribution_sequence, columns):
        oa_producer_df = pd.DataFrame(columns=columns, dtype=float)
        u, v = self.__oa_hidden(distribution_sequence)
        oa_producer_df['U'] = u[:, -1]
        oa_producer_df['V'] = v[:, -1]
        for i in range(self.__size + 1):
            oa_producer_df['U' + str(i)] = u[:, i]
            oa_producer_df['V' + str(i)] = v[:, i]

        ones_column = list(set([str(distribution_sequence)] + [dist for dist in str(distribution_sequence).split(' ')]))
        for values in ones_column:
            oa_producer_df[values] = np.ones((self.__monte_carlo, 1))
        print(oa_producer_df.head())
        return oa_producer_df

    def __oa_hidden(self, distribution_sequence):
        train_ozturk = np.sort(self.__beta_reduction(distribution_sequence), axis=1)
        detrended = standardize(train_ozturk, axis=1)
        u, v = self.__ozturk_function(detrended)
        return u, v

    def __store_seq(self, producer_df, distribution_sequence):
        filename = f'ANNOA_{distribution_sequence}_{self.__size}.parquet_{self.__theta_distribution}_gz'
        producer_df.to_parquet(f'../Data/{filename}', compression='gzip')

    def __beta_reduction(self, stats: Distribution):
        return self.__z_2(stats.rvs(size=self.__size, samples=self.__monte_carlo, dtype=float))

    def __z_2(self, p) -> np.ndarray:
        """
        The Formula for Z^2
        Z**2 = (p-mu).T  *  sigma**-1  * (p-mu)
        """
        if self.__dim == 1:
            return p.reshape(self.__monte_carlo, self.__size)
        p_mean = p.mean(axis=1).reshape((self.__monte_carlo, 1, self.__dim))
        p_cov = np.array([np.cov(p[i, :, :], rowvar=False) for i in range(self.__monte_carlo)])
        inv_p_cov = np.linalg.inv(p_cov)
        try:
            assert not np.isnan(inv_p_cov).any()
        except AssertionError:
            inv_p_cov = np.linalg.pinv(p_cov)

        p_t: np.ndarray = np.subtract(p, p_mean).conj()
        p_new: np.ndarray = np.transpose(np.subtract(p, p_mean), axes=(0, 2, 1))
        z_2 = np.zeros((self.__monte_carlo, self.__size))
        for i in range(self.__size):
            p_t_temp = np.reshape(p_t[:, i, :], (self.__monte_carlo, 1, self.__dim))
            p_new_temp = np.reshape(p_new[:, :, i], (self.__monte_carlo, self.__dim, 1))
            z_2[:, i] = (p_t_temp @ inv_p_cov @ p_new_temp).reshape(self.__monte_carlo)
        return z_2

    def __ozturk_function(self, t):
        initial_u = np.abs(t) * np.cos(self.__theta)
        initial_v = np.abs(t) * np.sin(self.__theta)
        u = np.zeros((self.__monte_carlo, self.__size + 1))
        v = np.zeros((self.__monte_carlo, self.__size + 1))
        for i in range(1, self.__size + 1):
            u[:, i] = np.sum(initial_u[:, :i], axis=1) / i
            v[:, i] = np.sum(initial_v[:, :i], axis=1) / i
        return u, v

    def reference_set(self, reference_distribution='Normal'):
        """
        This finds the angles for the library
        """
        self.__theta_distribution = reference_distribution
        filename = f'../Theta/Reference_Set_{self.size}.parquet_ref'
        reference_df = pd.DataFrame()
        try:
            # Load
            reference_df = pd.read_parquet(filename)[reference_distribution]
            self.__theta = reference_df.to_numpy(dtype=float).reshape((1, self.__size))
            print(f'{reference_distribution} Theta Loaded {self.__theta.shape}', flush=True)
        except FileNotFoundError:
            # Create File
            print('File Not Found', flush=True)
            reference_df = self.__reference_set()
            reference_df.to_parquet(filename)
            print(f'"{filename}" created', flush=True)
        except KeyError:
            # Standardized Null Hypothesis
            print(f'{reference_distribution} Not Found', flush=True)
            new_df = self.__reference_set()
            reference_df = pd.concat([reference_df, new_df], axis=1, ignore_index=True)
            reference_df.to_parquet(filename)
            print(f'{reference_distribution} added to "{filename}"', flush=True)

    def __reference_set(self):
        size = self.__monte_carlo * self.__size
        reference_distribution = REFERENCE_DICTIONARY[self.__theta_distribution]
        reference_set = reference_distribution.rvs(size=size).reshape((self.__size, self.__monte_carlo))
        reference_set.sort(axis=0)
        m = reference_set.mean(axis=1)
        self.__theta = (np.pi * reference_distribution.cdf(m)).astype(float).reshape((1, self.__size))
        return pd.DataFrame(self.__theta.reshape((self.__size, 1)), columns=[self.__theta_distribution],
                            index=range(0, self.__size))

    @property
    def size(self) -> int:
        return self.__size

    @property
    def dim(self) -> int:
        return self.__dim

    def __str__(self):
        classes = ','.join(map(str, self.__class_set))
        return f'gen_Ozturk_[{self.__size}]_[{classes}]'


def main():
    assert (__name__ == "__main__"), "Method not intended to be called if this isn't the main file"
    for dim in [1]:
        for size in [10, 25, 200]:
            for ref_set in ['Normal']:
                training_run = OzturkTrain(monte_carlo=MONTE_CARLO, size=size, dimension=dim)
                training_run.reference_set(ref_set)
                training_run.train(STATS_SET)


if __name__ == "__main__":
    '''
    Optimized for training size ANNOA
    '''
    main()
