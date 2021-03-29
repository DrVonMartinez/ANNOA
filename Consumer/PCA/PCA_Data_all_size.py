import glob
import os

import pandas as pd
from sklearn.decomposition import PCA

from Constants.Constants import SIZE_SET, DIMENSION_SET, SEED, PCA_VAL
from Constants.Storage_Constants import DATA_PATH

n_components = []
explained_variance_ratio = []
for dim in DIMENSION_SET:
    print('Dimension:', dim)
    for reference in ['Normal', 'Uniform']:
        sizes_set = []
        for size in SIZE_SET[:4]:
            print('\t' + str(size))
            os.chdir(DATA_PATH.format(dim=dim, reference=reference))
            size_files = glob.glob('*Data Set ' + str(size) + '.parquet_' + reference + '_gz')
            feature_columns = ['U' + str(i) for i in range(size + 1)] + ['V' + str(i) for i in range(size + 1)]
            sizes_set.append(pd.concat(map(lambda x: pd.read_parquet(x), size_files), ignore_index=True)[feature_columns])
        sizes_df = pd.concat(sizes_set)
        print(sizes_df.info())
        print('\t\t' + reference, end=' \t')
        pca = PCA(n_components=PCA_VAL, random_state=SEED)
        pca.fit(sizes_df)
        np_values = pca.transform(sizes_df)
        columns = []
        for i in range(pca.n_components_):
            pca_series = pd.DataFrame(np_values[:, i], columns=['PCA_Column_' + str(i)])
            for column in sizes_df.columns:
                test_df = pd.concat([pca_series, pd.DataFrame(sizes_df[column], columns=[column])], axis=1)
                if test_df['PCA_Column_' + str(i)].equals(test_df[column]):
                    columns.append(column)
        print(str(columns), end=' \t')
        print('\t\t\tExplained Variance:', pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_), sep='\t')
        explained_variance_ratio.append(sum(pca.explained_variance_ratio_))
