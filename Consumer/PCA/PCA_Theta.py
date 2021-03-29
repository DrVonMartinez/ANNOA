import glob
import os

import pandas as pd
from sklearn.decomposition import PCA

from Constants.Constants import DIMENSION_SET, SEED, PCA_VAL
from Constants.Expanded_Constants import REFERENCE_LIST
from Constants.Storage_Constants import DATA_PATH

n_components = []
explained_variance_ratio = []
for dim in DIMENSION_SET:
    print('Dimension:', dim)
    for reference in REFERENCE_LIST:
        os.chdir(DATA_PATH.format(dim=dim, reference=reference))
        size_files = glob.glob(reference + ' Reference*')
        for size_file in size_files:
            size_np = pd.read_parquet(size_file).to_numpy()
            size_np = size_np.reshape((size_np.shape[0], -1))
            print(size_np.shape)
            print('\t\t' + reference + '\t' + str(PCA_VAL), end=' \t')
            pca = PCA(n_components=PCA_VAL, random_state=SEED)
            pca.fit(size_np)
            np_values = pca.transform(size_np)
            print(str(pca.n_components_), flush=True)
            print('\t\t\tExplained Variance:', pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_), sep='\t')
            explained_variance_ratio.append(sum(pca.explained_variance_ratio_))
