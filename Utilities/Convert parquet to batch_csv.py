import pandas as pd
import os
import glob
import progressbar

for dim in ['dim 1', 'dim 2']:
    reference_set = ['Normal', 'Uniform']
    for ref in reference_set:
        os.chdir('F:\\Data\\' + dim + '\\Ref ' + ref + '\\')
        compressed_files = list(glob.glob('*500.parquet_' + ref + '_gz'))
        print('\n'.join(compressed_files))
        for i in progressbar.progressbar(range(len(compressed_files))):
            file = compressed_files[i]
            df = pd.read_parquet(file)
            title = file.replace('parquet', 'csv')
            df.to_csv(title, compression='gzip')
