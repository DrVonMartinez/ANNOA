#  -*- coding: utf-8 -*-
"""
@author: Benjamin
The Goal is to have Ozturk's Algorithm:
    Utilize Data Mining:
        For Classification
        For Cluster Detection
        For Association Learning

"""
__spec__ = None

import os
import warnings
import itertools

# Artificial Neural Network Ozturk
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale as standardize

from Constants.Constants import STATS_SET, SIZE_SET, REFERENCE_DICTIONARY, MONTE_CARLO
from Constants.Expanded_Constants import REFERENCE_LIST
from Distribution.Distribution import Distribution

warnings.filterwarnings("ignore", category=FutureWarning)


class OzturkTrain:
    def __init__(self, monte_carlo=2000, size=200, dimension=1):
        """
        loss function selection following suggestions from https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c
        Optimizer selection following suggestions from https://algorithmia.com/blog/introduction-to-optimizers
        :param monte_carlo:
        :param size:
        :param dimension:
        """
        if __name__ == "__main__":
            print('Ozturk Algorithm',
                  'Monte Carlo: ' + str(monte_carlo),
                  "Dimension: " + str(dimension),
                  'Size: ' + str(size),
                  flush=True,
                  sep='\n')
        self.__monte_carlo = int(monte_carlo)
        self.__size = size
        self.__dim = int(dimension)
        self.__theta = None
        self.__theta_distribution = ''
        self.__columns = []
        self.__dtype = np.float32
        self.__class_set = []
        if self.__monte_carlo < 1000:
            warnings.warn('Warning! Small Number of Monte Carlo Simulations')

    def train(self, distributions):
        """
        :param distributions:
        :return:
        """
        assert self.__theta is not None
        for index in itertools.product(range(len(distributions)), repeat=self.__dim):
            dist = Distribution(distributions[index[0]])
            for i in range(1, len(index)):
                dist += Distribution(distributions[index[i]])
            assert dist.dimension() == self.__dim, 'Dimension mismatch. Dimension should be ' + str(self.__dim) + ' but is ' + str(dist.dimension())
            self.__class_set.append(dist)
        self.__determine_columns()
        cwd = self.__change_cwd()
        for i in range(len(self.__class_set)):
            distribution_df = self.__oa_hidden_single(self.__class_set[i]).astype(dtype=self.__dtype).fillna(value=0)
            self.__store_seq(distribution_df, self.__class_set[i])
        self.__revert(cwd)

    def train_single_distribution(self, distribution_set: list):
        distribution_sequence = Distribution(distribution_set[0])
        for i in range(1, self.__dim):
            distribution_sequence += Distribution(distribution_set[i])
        self.__class_set = [str(distribution_sequence)]
        self.__determine_columns()
        cwd = self.__change_cwd()
        distribution_df = self.__oa_hidden_single(distribution_sequence)
        self.__store_seq(distribution_df, distribution_sequence)
        self.__revert(cwd)

    def __oa_hidden(self, distribution_sequence):
        train_ozturk = np.sort(self.__beta_reduction(distribution_sequence), axis=1)
        detrended = standardize(train_ozturk, axis=1)
        u, v = self.__ozturk_function(detrended)
        return u, v

    def __determine_columns(self):
        self.__columns += ['U', 'V']
        for i in range(self.__size + 1):
            self.__columns.append('U' + str(i))
            self.__columns.append('V' + str(i))
        distribution_set = list(map(str, self.__class_set))
        single_dist_set = []
        if self.__dim > 1:
            for dim_dist in distribution_set:
                for dist in dim_dist.split(' '):
                    if dist not in single_dist_set:
                        single_dist_set.append(dist)
        self.__columns += distribution_set + single_dist_set

    def __oa_hidden_single(self, distribution_sequence):
        oa_producer_df = pd.DataFrame(columns=self.__columns, dtype=self.__dtype)
        u, v = self.__oa_hidden(distribution_sequence)
        oa_producer_df['U'] = u[:, -1]
        oa_producer_df['V'] = v[:, -1]
        ones_column = [str(distribution_sequence)]
        if self.__dim > 1:
            ones = np.ones((self.__monte_carlo, 1))
            for dist in str(distribution_sequence).split(' '):
                if dist not in ones_column:
                    ones_column.append(dist)
        else:
            ones = np.ones(self.__monte_carlo)
        for values in ones_column:
            oa_producer_df[values] = ones

        for i in range(self.__size + 1):
            oa_producer_df['U' + str(i)] = u[:, i]
            oa_producer_df['V' + str(i)] = v[:, i]
        return oa_producer_df

    def __store_seq(self, oa_producer_df, distribution_sequence):
        oa_producer_df.to_parquet(str(distribution_sequence) + ' Data Set ' + str(self.__size) + '.parquet_' + self.__theta_distribution + '_gz', compression='gzip')

    def __beta_reduction(self, stats: Distribution):
        return self.__z_2(stats.rvs(size=self.__size, samples=self.__monte_carlo, dtype=self.__dtype))

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
        cwd = self.__change_cwd()
        try:
            # Load
            reference_df = pd.read_parquet(reference_distribution + ' Reference Set ' + str(self.__size) + '.parquet_ref')
            self.__theta = reference_df.to_numpy(dtype=self.__dtype).reshape((1, self.__size))
            print(reference_distribution + ' Theta Loaded', self.__theta.shape, flush=True)
        except FileNotFoundError:
            # Create File
            print('File Not Found', flush=True)
            filename = self.__reference_set(reference_distribution)
            print('"' + filename + '" created', flush=True)
        except KeyError:
            # Standardized Null Hypothesis
            print(reference_distribution + ' Not Found', flush=True)
            filename = self.__reference_set(reference_distribution)
            print(reference_distribution + ' added to "' + filename + '"', flush=True)
        self.__revert(cwd)

    def __reference_set(self, reference_distribution):
        reference_set = REFERENCE_DICTIONARY[reference_distribution].rvs(size=self.__monte_carlo * self.__size).reshape((self.__size, self.__monte_carlo))
        reference_set.sort(axis=0)
        m = reference_set.mean(axis=1)
        self.__theta = (np.pi * REFERENCE_DICTIONARY[reference_distribution].cdf(m)).astype(self.__dtype).reshape((1, self.__size))
        reference_df = pd.DataFrame(self.__theta.reshape((self.__size, 1)), columns=[reference_distribution], index=range(0, self.__size))
        filename = reference_distribution + ' Reference Set ' + str(self.__size) + '.parquet_ref'
        reference_df.to_parquet(filename)
        return filename

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

    def size(self) -> int:
        return self.__size

    def dim(self) -> int:
        return self.__dim

    def __adjust_columns(self, df):
        columns = list(df.columns)
        new_order = ['U', 'V']
        if 'U0' in columns:
            for i in range(self.__size + 1):
                new_order.append('U' + str(i))
                new_order.append('V' + str(i))
        stats = list(filter(lambda x: 'U' not in x, columns))
        distribution_sets = list(filter(lambda x: ' ' in x, stats))
        distributions = list(filter(lambda x: ' ' not in x, stats))
        for dist in distribution_sets:
            new_order.append(dist)
        for dist in distributions:
            new_order.append(dist)

        return df.reindex(columns=new_order)

    def __change_cwd(self):
        cwd = os.getcwd()
        os.chdir('F:\\Data\\')
        cwd_dir = os.getcwd()
        new_dim = '\\dim ' + str(self.__dim)
        new_ref = '\\Ref ' + self.__theta_distribution + '\\'
        try:
            os.chdir(cwd_dir + new_dim)
        except FileNotFoundError:
            try:
                os.mkdir(cwd_dir + new_dim)
                os.chdir(cwd_dir + new_dim + new_ref)
            except FileNotFoundError:
                os.mkdir(cwd_dir + new_dim + new_ref)
        if os.getcwd() != cwd_dir + new_dim + new_ref:
            os.chdir(cwd_dir + new_dim + new_ref)
        return cwd

    def __revert(self, cwd):
        os.chdir(cwd)
        return self.__size

    def __str__(self):
        return 'gen_Ozturk_[' + str(self.__size) + ']_[' + ','.join(self.__class_set) + ']'


def main():
    assert (__name__ == "__main__"), "Method not intended to be called if this isn't the main file"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dim in [1, 2]:
            for size in SIZE_SET:
                for ref_set in REFERENCE_LIST:
                    training_run = OzturkTrain(monte_carlo=MONTE_CARLO, size=size, dimension=dim)
                    training_run.reference_set(ref_set)
                    training_run.train(STATS_SET)


if __name__ == "__main__":
    '''
    Optimized for training size ANNOA
    '''
    main()
