import glob
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Utilities.Constants import SIZE_SET, DIMENSION_SET
from Utilities.Expanded_Constants import REFERENCE_LIST, REFERENCE_DICTIONARY

for size in SIZE_SET:
    print(1, size, 'Complete')
    for reference in REFERENCE_LIST:
        print('\t', reference, sep='')
        os.chdir('F:\\Data\\dim 1\\Ref ' + reference)
        cluster = glob.glob('*Set ' + str(size) + '.parquet_' + reference + '_gz')
        plt.close('all')
        for file in cluster:
            dist = file.split(' ')[:1]
            dist_name = ' '.join(dist)
            file_df = pd.read_parquet(file).astype(dtype=np.float32)
            print('\t\t', file, sep='')
            u = file_df['U'].to_numpy()
            v = file_df['V'].to_numpy()
            plt.plot(u, v, '.', label=dist_name)
        plt.legend()
        plt.xlabel('U')
        plt.ylabel('V')
        plt.title('Dim 1 Size ' + str(size) + ' ' + reference)
        plt.savefig('F:\\Plot\\Dim 1 Size ' + str(size) + ' ' + reference + '.png')
for size in SIZE_SET:
    for class_count_var in ['Limited', 'Complete']:
        print(2, size, class_count_var)
        for reference in REFERENCE_LIST:
            print('\t', reference, sep='')
            os.chdir('F:\\Data\\dim 2\\Ref ' + reference)
            cluster = glob.glob('*Set ' + str(size) + '.parquet_' + reference + '_gz')
            plt.close('all')
            for file in cluster:
                dist = file.split(' ')[:2]
                dist_name = ' '.join(dist)
                file_df = pd.read_parquet(file).astype(dtype=np.float32)
                print('\t\t', file, sep='')
                u = file_df['U'].to_numpy()
                v = file_df['V'].to_numpy()
                if class_count_var == 'Limited' and all(map(lambda x: x == dist[0], dist)):
                    plt.plot(u, v, '.', label=dist_name)
                elif class_count_var == 'Complete':
                    plt.plot(u, v, '.', label=dist_name)
            plt.legend()
            plt.xlabel('U')
            plt.ylabel('V')
            plt.title('Dim 2 Size ' + str(size) + ' ' + reference + ' ' + class_count_var)
            plt.savefig('F:\\Plot\\Dim 2 Size ' + str(size) + ' ' + reference + ' ' + class_count_var + '.png')
