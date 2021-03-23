import glob
import os
import pandas as pd
from Utilities.Constants import SIZE_SET, DIMENSION_SET, SEED, PCA_VAL
from sklearn.decomposition import PCA

n_components = []
explained_variance_ratio = []
for dim in DIMENSION_SET:
    print('Dimension:', dim)
    for size in SIZE_SET[:-3]:
        print('\t' + str(size))
        for reference in ['Normal', 'Uniform']:
            os.chdir('F:\\Data\\dim ' + str(dim) + '\\Ref ' + reference + '\\')
            size_files = glob.glob('*Data Set ' + str(size) + '.parquet_' + reference + '_gz')
            feature_columns = ['U' + str(i) for i in range(size+1)] + ['V' + str(i) for i in range(size+1)]
            size_df = pd.concat(map(lambda x: pd.read_parquet(x), size_files), ignore_index=True)[feature_columns]
            for percent in [0.85, 0.90, 0.95, 0.99, 0.999]:
                print('\t\t' + reference + '\t' + str(percent), end=' \t')
                pca = PCA(n_components=percent, random_state=SEED)
                pca.fit(size_df)
                np_values = pca.transform(size_df)
                columns = []
                for i in range(pca.n_components_):
                    pca_series = pd.DataFrame(np_values[:, i], columns=['PCA_Column_' + str(i)])
                    for column in size_df.columns:
                        test_df = pd.concat([pca_series, pd.DataFrame(size_df[column], columns=[column])], axis=1)
                        if test_df['PCA_Column_' + str(i)].equals(test_df[column]):
                            columns.append(column)
                print(str(columns), end=' \t')
                print(str(pca.n_components_), flush=True)
                print('\t\t\tExplained Variance:', pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_), sep='\t')
                explained_variance_ratio.append(sum(pca.explained_variance_ratio_))



