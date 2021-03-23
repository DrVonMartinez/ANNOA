import glob
import os
import numpy as np
import pandas as pd

from Utilities.Constants import SIZE_SET

dtype = np.float32
os.chdir('..')
os.chdir('Data\\dim 1\\')
sizes = SIZE_SET
distributions = ['Norm', 'Uniform', 'Expon', 'Cauchy', 'Laplace', 'Logistic', 'Rayleigh']


def show_results(size):
    distribution_files = list(glob.glob('Compressed * Data Set ' + str(size) + '.gz_dist_' + str(size)))
    assert len(distribution_files) != 0
    # column_set = []
    for file in distribution_files:
        try:
            df = pd.read_csv(file, compression='gzip', dtype=dtype)
            print(df.columns)
            print(df)
        except EOFError:
            continue
    '''
        column_set.append(df.columns)
    assert len(column_set) != 0
    print(column_set[0])
    show = True
    for columns in column_set[1:]:
        for i in range(len(columns)):
            assert column_set[0][i] == columns[i]
        if show:
            print('[' + ','.join(columns) + ']')
            show = False
    '''


for s in sizes:
    show_results(s)
