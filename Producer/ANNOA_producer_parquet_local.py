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
from Producer.ANNOA_producer import OzturkTrainGeneral


class OzturkTrainLocal(OzturkTrainGeneral):
    def __init__(self, monte_carlo=2000, size=200, dimension=1):
        super().__init__(monte_carlo, size, dimension)

    def _store_seq(self, producer_df, distribution_sequence):
        filename = f'ANNOA_{distribution_sequence}_{self._size}.parquet_{self._theta_distribution}_gz'
        producer_df.to_parquet(f'../Data/{filename}', compression='gzip')


def main():
    assert (__name__ == "__main__"), "Method not intended to be called if this isn't the main file"
    for dim in [1]:
        for size in [50, 75, 100]:
            for ref_set in ['Normal']:
                training_run = OzturkTrainLocal(monte_carlo=MONTE_CARLO, size=size, dimension=dim)
                training_run.reference_set(ref_set)
                training_run.train(STATS_SET)


if __name__ == "__main__":
    '''
    Optimized for training size ANNOA
    '''
    main()
