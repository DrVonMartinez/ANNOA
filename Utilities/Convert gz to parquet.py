import pandas as pd
import os
import glob

cwd = os.getcwd()
for dim in ['dim 1', 'dim 2']:
    os.chdir(cwd)
    os.chdir('..\\Data\\' + dim + '\\')
    compressed_files = list(glob.glob('*.gz_dist*'))
    compressed_ref = list(glob.glob('*.gz_ref'))
    for file in compressed_files:
        df = pd.read_csv(file, compression='gzip')
        title = file.split('.')[0].replace('Compressed ', '')
        print(title)
        df.to_parquet('Ref Normal\\' + title + '.parquet_Normal')
    for file in compressed_ref:
        df = pd.read_csv(file, compression='gzip')
        title = file.split('.')[0]
        df.to_parquet('Ref Normal\\Normal ' + title + '.parquet_ref')
        print(title)
