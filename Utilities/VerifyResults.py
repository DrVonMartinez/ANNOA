import glob
import os
import pandas as pd
from Utilities.Constants import SIZE_SET, DIMENSION_SET, STATS_SET
from Utilities.Expanded_Constants import REFERENCE_LIST


def detail_size(size: int):
    size_files = list(glob.glob(reference + '*' + 'Dim ' + str(dim) + '*Size ' + str(size) + ',*.csv'))
    if len(size_files) == 0 and data == 'Partial Data':
        print('\t\t\t\t' + str(size) + '\t\t0/3')
        incomplete_sizes.append((dim, reference, data, classes, size))
        return
    elif len(size_files) == 0:
        print('\t\t\t\t' + str(size) + '\t\t0/3\t\t0/3')
        incomplete_pca_sizes.append((dim, reference, data, classes, size, 'PCA'))
        incomplete_sizes.append((dim, reference, data, classes, size))
        return
    neural_networks = list(filter(lambda x: 'adam' in x and 'PCA' not in x, size_files))
    pca = list(filter(lambda x: 'adam' in x and 'PCA' in x, size_files))
    if data == 'All Data' and len(pca) == 0:
        incomplete_pca_sizes.append((dim, reference, data, classes, size, 'PCA'))
    # trees = list(filter(lambda x: 'Decision_Tree' in x, size_files))
    # knn = list(filter(lambda x: 'K-NN' in x, size_files))
    # svm = list(filter(lambda x: 'SVM' in x, size_files))
    if data == 'Partial Data':
        print('\t\t\t\t' + str(size) + '\t\t' + str(len(neural_networks)) + '/3')
    else:
        print('\t\t\t\t' + str(size) + '\t\t' + str(len(neural_networks)) + '/3' + '\t\t' + str(len(pca)) + '/3')
    # print('\t\t\t\t' + str(size) + '\t\t' + str(len(neural_networks)) + '\t\t' + str(len(trees)) + '\t\t' + str(len(knn)) + '\t\t' + str(len(svm)))
    # print('\t\t\t\t' + str(neural_networks), '\t\t\t\t' + str(trees), '\t\t\t\t' + str(knn), '\t\t\t\t' + str(svm), sep='\n')


incomplete_sizes = []
incomplete_pca_sizes = []
cwd = os.getcwd()
for dim in [1, 2]:
    print('Dimension:', dim)
    for data in ['All Data', 'Partial Data']:
        print('\t' + data)
        for classes in ['Full Classes', 'Limited Classes']:
            if classes == 'Limited Classes' and dim == 1:
                continue
            print('\t\t' + classes)
            for reference in REFERENCE_LIST:
                print('\t\t\t' + reference)
                os.chdir('C:\\Users\\Auror\\PycharmProjects\\ANNOA\\Results\\Ref ' + reference + '\\' + data + '\\' + classes)
                for size_var in SIZE_SET:
                    detail_size(size_var)
print('\n'.join(map(str, incomplete_sizes)))
print('\n'.join(map(str, incomplete_pca_sizes)))
