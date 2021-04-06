import glob
import os

from Constants.Constants import SIZE_SET
from Constants.Expanded_Constants import REFERENCE_LIST
from Constants.Storage_Constants import RESULT_PATH


def print_files(size_files, size):
    neural_networks = list(filter(lambda x: 'adam' in x and 'PCA' not in x, size_files))
    if len(neural_networks) == 0:
        incomplete_sizes.append((dim, reference, data, classes, size))
    print(str(len(neural_networks)) + '/3', end='\t\t')
    if find_pca and data == 'All Data':
        pca = list(filter(lambda x: 'adam' in x and 'PCA' in x, size_files))
        if len(pca) == 0:
            incomplete_pca_sizes.append((dim, reference, data, classes, size, 'PCA'))
        print(str(len(pca)) + '/3', end='\t\t')
    if find_trees:
        trees = list(filter(lambda x: 'Decision_Tree' in x, size_files))
        if len(trees) == 0:
            incomplete_tree_sizes.append((dim, reference, data, classes, size, 'trees'))
        print(str(len(trees)) + '/3', end='\t\t')
    if find_knn:
        knn = list(filter(lambda x: 'K-NN' in x, size_files))
        if len(knn) == 0:
            incomplete_knn_sizes.append((dim, reference, data, classes, size, 'knn'))
        print(str(len(knn)) + '/3', end='\t\t')
    if find_svm:
        svm = list(filter(lambda x: 'SVM' in x, size_files))
        if len(svm) == 0:
            incomplete_svm_sizes.append((dim, reference, data, classes, size, 'svm'))
        print(str(len(svm)) + '/3', end='\t\t')
    print()


def detail_size(size: int):
    size_files = list(glob.glob(reference + '*' + 'Dim ' + str(dim) + '*Size ' + str(size) + ',*.csv'))
    print('\t\t\t\t' + str(size), end='\t\t')
    print_files(size_files, size)


incomplete_sizes = []
find_pca = True
incomplete_pca_sizes = []
find_trees = False
incomplete_tree_sizes = []
find_knn = False
incomplete_knn_sizes = []
find_svm = False
incomplete_svm_sizes = []
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
                os.chdir(RESULT_PATH.format(dim=dim, reference=reference, data=data, classes=classes))
                for size_var in SIZE_SET:
                    detail_size(size_var)
for dim in [1, 2]:
    print('Dimension:', dim)
    for reference in REFERENCE_LIST:
        print('\t' + reference)
        for data in ['All Data']:
            print('\t\t' + data)
            for classes in ['Full Classes', 'Limited Classes']:
                print('\t\t\t' + classes + '\n\t\t\t\tAll Sizes', end='\t\t')
                os.chdir(RESULT_PATH.format(dim=dim, reference=reference, data=data, classes=classes))
                all_size_files = list(glob.glob(reference + ', Dim ' + str(dim) + '*[*.csv'))
                print(str(len(all_size_files)) + '/3')

incomplete_sizes = sorted(incomplete_sizes, key=lambda x: (x[0], x[-1], x[2], x[1], x[3]))
incomplete_pca_sizes = sorted(incomplete_pca_sizes, key=lambda x: (x[-1], x[0], x[-2], x[1], x[2], x[3]))
print('\n'.join(map(str, incomplete_sizes)))
print('\n'.join(map(str, incomplete_pca_sizes)))
