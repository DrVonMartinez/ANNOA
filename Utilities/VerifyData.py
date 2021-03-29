import glob
import os

import pandas as pd

from Constants.Constants import SIZE_SET, DIMENSION_SET, STATS_SET
from Constants.Expanded_Constants import REFERENCE_LIST
from Constants.Storage_Constants import DATA_PATH

for dim in DIMENSION_SET:
    print('Dimension:', dim)
    for reference in REFERENCE_LIST:
        os.chdir(DATA_PATH.format(dim=dim, reference=reference))
        ref_cwd = os.getcwd()
        print('\t' + reference)
        for size in SIZE_SET:
            size_files = glob.glob('*Data Set ' + str(size) + '.parquet_' + reference + '_gz')
            print('\t\t{0}\t\t{1}\t\t{2}'.format(str(size), str(len(size_files)), str(len(STATS_SET) ** dim)))
            map(lambda x: pd.read_parquet(x), size_files)
