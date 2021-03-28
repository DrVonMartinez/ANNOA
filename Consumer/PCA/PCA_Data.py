import glob
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from Constants.Constants import SIZE_SET, DIMENSION_SET, SEED
from Constants.Expanded_Constants import REFERENCE_LIST

n_components = []
explained_variance_ratio = []
for dim in DIMENSION_SET:
    print('Dimension:', dim)
    size_n_components = []
    size_explained_var = []
    for size in SIZE_SET:
        dist_n_components = []
        dist_explained_var = []
        print('\t' + str(size))
        for reference in REFERENCE_LIST:
            os.chdir('F:\\Data\\dim ' + str(dim) + '\\Ref ' + reference + '\\')
            size_files = glob.glob('*Data Set ' + str(size) + '.parquet_' + reference + '_gz')
            feature_columns = ['U' + str(i) for i in range(size+1)] + ['V' + str(i) for i in range(size+1)]
            size_df = pd.concat(map(lambda x: pd.read_parquet(x), size_files), ignore_index=True)[feature_columns]
            for percent in [0.85, 0.90, 0.95, 0.99, 0.999]:
                print('\t\t' + reference + '\t' + str(percent), end=' \t')
                pca = PCA(n_components=percent, random_state=SEED)
                pca.fit(size_df)
                np_values = pca.transform(size_df)
                print(str(pca.n_components_), flush=True)
                n_components.append(pca.n_components)
                dist_n_components.append(pca.n_components)
                print('\t\t\tExplained Variance:', sum(pca.explained_variance_ratio_), sep='\t')
                explained_variance_ratio.append(sum(pca.explained_variance_ratio_))
                dist_explained_var.append(sum(pca.explained_variance_ratio_))
        arg_max_n_components = np.argmax(dist_n_components) // 5
        arg_max_explained_var = np.argmax(dist_explained_var) // 5
        size_n_components.append((size, REFERENCE_LIST[arg_max_n_components], dist_n_components[arg_max_n_components]))
        size_explained_var.append((size, REFERENCE_LIST[arg_max_explained_var], dist_n_components[arg_max_explained_var]))
    print('\tComponents:', size_n_components, sep='\t', end='\n\t')
    print('Explained Var:', size_explained_var, sep='\t')
# print(n_components)


