import argparse

import numpy as np
from sklearn.metrics import precision_score, accuracy_score, recall_score, multilabel_confusion_matrix, \
    mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier

from Constants.Expanded_Constants import REFERENCE_LIST
from Consumer.ANNOA_Model_less import Ozturk
from Consumer.Model import Model


class OA_KNN(Model):
    def __init__(self, k_neighbors=1, epochs=100):
        super().__init__()
        self.__k = k_neighbors
        self.__model = self.__define_model()
        self.__epochs = epochs

    def __define_model(self):
        return KNeighborsClassifier(n_neighbors=self.__k)

    def train(self, input_data, output_data) -> dict:
        train_ratio = int(len(input_data) * self._train_ratio)
        results = {'accuracy': [], 'precision': [], 'recall': [], 'mean_absolute_error': [],
                   'true_positives': [], 'true_negatives': [], 'false_positives': [], 'false_negatives': []}
        np.random.seed(0)
        for i in range(self.__epochs):
            print(f'Epoch {i + 1}')
            indices = np.arange(0, len(input_data))
            np.random.shuffle(indices)
            local_input_data, local_output_data = input_data[indices], output_data[indices]
            trained_model = self.__model.fit(local_input_data[:train_ratio], local_output_data[:train_ratio])
            output_predicted: np.ndarray = trained_model.predict(local_input_data[train_ratio:])
            confusion_matrix = np.sum(multilabel_confusion_matrix(local_output_data[train_ratio:], output_predicted),
                                      axis=0)
            results['true_negatives'].append(confusion_matrix[0, 0])
            results['false_negatives'].append(confusion_matrix[1, 0])
            results['false_positives'].append(confusion_matrix[0, 1])
            results['true_positives'].append(confusion_matrix[1, 1])
            results['accuracy'].append(accuracy_score(local_output_data[train_ratio:], output_predicted))
            results['precision'].append(
                precision_score(local_output_data[train_ratio:], output_predicted, average=None))
            results['recall'].append(
                recall_score(local_output_data[train_ratio:], output_predicted, average=None))
            results['mean_absolute_error'].append(mean_absolute_error(
                local_output_data[train_ratio:], output_predicted))
        return results

    def summary(self) -> None:
        print(f"K Nearest Neighbor: {self.__k} neighbor{'s' if self.__k > 1 else ''}")

    def __str__(self):
        return f'OA_KNN_{self.__k}'


def run_ozturk_annoa(dimension, size, full_data, full_classes):
    for reference_distribution in REFERENCE_LIST[0:1]:
        training_model = Ozturk(size=size,
                                dimension=dimension,
                                reference_distribution=reference_distribution,
                                full_classes=full_classes,
                                full_data=full_data)
        training_model.prepare_training_data()
        for k in range(2, 20, 5):
            model = OA_KNN(k)
            training_model.define_model(model)
            training_model.info()
            training_model.train()
            training_model.store()


def main():
    params = argparse.ArgumentParser(prog='ANNOA_KNN', description='ANNOA for K-Nearest Neighbor')
    params.add_argument('-d', '--dimension', required=True, type=int)
    params.add_argument('-s', '--size', required=True, type=int)
    params.add_argument('full_data', default=False, action='store_true')
    params.add_argument('full_classes', default=True, action='store_false')
    args = params.parse_args()
    run_ozturk_annoa(args.dimension, args.size, args.full_data, args.full_classes)


if __name__ == "__main__":
    '''
    Optimized for training size ANNOA
    '''

    main()
