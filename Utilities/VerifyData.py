import glob
import os

import pandas as pd

from Constants.Constants import SIZE_SET, DIMENSION_SET, STATS_SET

for dim in DIMENSION_SET:
    os.chdir('F:\\Data\\dim ' + str(dim) + '\\')
    dim_cwd = os.getcwd()
    print('Dimension:', dim)
    for reference in ['Normal', 'Uniform']:
        os.chdir(dim_cwd + '\\Ref ' + reference + '\\')
        ref_cwd = os.getcwd()
        print('\t' + reference)
        for size in SIZE_SET:
            size_files = glob.glob('*Data Set ' + str(size) + '.parquet_' + reference + '_gz')
            print('\t\t' + str(size) + '\t\t' + str(len(size_files)) + '\t\t' + str(len(STATS_SET) ** dim))
            map(lambda x: pd.read_parquet(x), size_files)
