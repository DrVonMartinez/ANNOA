import glob
import os

from Constants.Constants import SIZE_SET
from Constants.Expanded_Constants import REFERENCE_LIST
from Constants.Storage_Constants import RESULT_PATH


def detail_size(size: int):
    size_files = list(glob.glob(reference + '*' + 'Dim ' + str(dim) + '*Size ' + str(size) + ',*.csv'))
    neural_networks = list(filter(lambda x: 'adam' in x and 'PCA' not in x, size_files))
    pca = list(filter(lambda x: 'adam' in x and 'PCA' in x, size_files))
    if data == 'All Data' and len(pca) == 0:
        incomplete_pca_sizes.append((dim, reference, data, classes, size, 'PCA'))
    if len(neural_networks) == 0:
        incomplete_sizes.append((dim, reference, data, classes, size))
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
incomplete_sizes = sorted(incomplete_sizes, key=lambda x: (x[0], x[-1], x[2], x[1], x[3]))
incomplete_pca_sizes = sorted(incomplete_pca_sizes, key=lambda x: (x[-1], x[0], x[-2], x[1], x[2], x[3]))
print('\n'.join(map(str, incomplete_sizes)))
print('\n'.join(map(str, incomplete_pca_sizes)))
