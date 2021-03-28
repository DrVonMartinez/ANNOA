import glob
import os

import numpy as np
import pandas as pd

from Constants.Constants import DIMENSION_SET

dtype = np.float16


def show_results(dimension, _size_):
        os.chdir('F:\\Data\\dim ' + str(dimension) + '\\')
        dim_cwd = os.getcwd()
        print('Dimension:', dimension)
        for reference in ['Normal', 'Uniform']:
            os.chdir(dim_cwd + '\\Ref ' + reference + '\\')
            # ref_cwd = os.getcwd()
            print('\t' + reference)
            size_files = glob.glob('*Data Set ' + str(_size_) + '.parquet_' + reference + '_gz')
            for file in size_files:
                x = pd.read_parquet(file).astype(dtype)
                print('\t', _size_, x.shape, sep='\t')


for dim in DIMENSION_SET:
    '''
    for size in SIZE_SET:
        show_results(dim, size)
    '''
    show_results(dim, 500)
