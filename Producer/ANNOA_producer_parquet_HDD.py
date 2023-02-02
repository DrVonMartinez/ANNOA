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
from Producer.ANNOA_producer import OzturkTrainGeneral

warnings.filterwarnings("ignore", category=FutureWarning)


class OzturkTrainHDD(OzturkTrainGeneral):
    def __init__(self, monte_carlo=2000, size=200, dimension=1):
        super().__init__(monte_carlo, size, dimension)

    def _store_seq(self, oa_producer_df, distribution_sequence):
        oa_producer_df.to_parquet(
            f'{distribution_sequence} Data Set {self._size}.parquet_{self._theta_distribution}_gz', compression='gzip')


def main():
    assert (__name__ == "__main__"), "Method not intended to be called if this isn't the main file"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dim in [1, 2]:
            for size in SIZE_SET:
                for ref_set in REFERENCE_LIST:
                    training_run = OzturkTrainHDD(monte_carlo=MONTE_CARLO, size=size, dimension=dim)
                    training_run.reference_set(ref_set)
                    training_run.train(STATS_SET)


if __name__ == "__main__":
    '''
    Optimized for training size ANNOA
    '''
    main()
