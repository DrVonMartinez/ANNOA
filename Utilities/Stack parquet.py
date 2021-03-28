import glob
import os

import pandas as pd

from Constants.Constants import SEED

cwd = os.getcwd()
for dim in ['dim 2']:
    for ref in ['Uniform', 'Normal']:
        os.chdir(cwd)
        os.chdir('F:\\Data\\' + dim + '\\Ref ' + ref + '\\')
        files = list(glob.glob('*500.parquet_' + ref + '_gz'))
        structure = pd.concat(map(pd.read_parquet, files))
        structure.info()
        structure.sample(n=structure.shape[0], random_state=SEED).to_csv('Data Set 500.csv', index=False)
