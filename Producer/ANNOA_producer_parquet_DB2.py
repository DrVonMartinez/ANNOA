#  -*- coding: utf-8 -*-
"""
@author: Benjamin
The Goal is to have Ozturk's Algorithm:
    Utilize Data Mining:
        For Classification
        For Cluster Detection
        For Association Learning

"""

# Artificial Neural Network Ozturk
from progressbar import ProgressBar

from Constants.Constants import STATS_SET, MONTE_CARLO
from Producer.ANNOA_producer import OzturkTrainGeneral
from Utilities.DB2_storage import IbmConnection


class OzturkTrain(OzturkTrainGeneral):
    def __init__(self, monte_carlo=2000, size=200, dimension=1):
        super().__init__(monte_carlo, size, dimension)

    def _store_seq(self, producer_df, distribution_sequence):
        ibm = IbmConnection(f'ANNOA_{str(distribution_sequence)}_{self._size}')
        ibm.connect()
        if not ibm.is_connected:
            return
        ibm.drop_table()
        ibm.create_table()
        size = int(25000 / (self._size + 2))
        with ProgressBar(max_value=len(producer_df)) as bar:
            for i in range(0, len(producer_df), size):
                upper = min(i + size - 1, len(producer_df) - 1)
                if upper > len(producer_df) // 4:
                    ibm.close()
                    ibm.connect()
                ibm.insert(producer_df.drop(columns=list(map(str, self._class_set))).loc[i:upper].to_numpy())
                bar.update(i)


def main():
    assert (__name__ == "__main__"), "Method not intended to be called if this isn't the main file"
    for dim in [1]:
        for size in [10]:
            for ref_set in ['Normal']:
                training_run = OzturkTrain(monte_carlo=MONTE_CARLO, size=size, dimension=dim)
                training_run.reference_set(ref_set)
                for stats in STATS_SET:
                    training_run.train_single_distribution([stats])


if __name__ == "__main__":
    '''
    Optimized for training size ANNOA
    '''
    main()
