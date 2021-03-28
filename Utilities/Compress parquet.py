import glob
import os

import numpy as np
import pandas as pd

from Constants.Constants import REFERENCE_LIST

cwd = os.getcwd()
for dim in ['dim 1', 'dim 2']:
    for ref in ['Uniform', 'Normal']:
        os.chdir(cwd)
        os.chdir('F:\\Data\\' + dim + '\\Ref ' + ref + '\\')
        uncompressed_files = list(glob.glob('*.parquet_' + ref))
        for file in uncompressed_files:
            print(file)
            df = pd.read_parquet(file, compression='gzip')
            for col in df.columns:
                if col in REFERENCE_LIST:
                    df[[col]] = df[[col]].astype(np.int8)
            df.to_parquet(file + '_gz', compression='gzip')
