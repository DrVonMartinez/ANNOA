import argparse
import glob
import os

import keras.callbacks
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Sequential

from Constants.Constants import SEED, label, SIZE_SET
from Constants.Expanded_Constants import NUM_HIDDEN_LAYERS, HIDDEN_NEURONS, NUM_EPOCHS, EXPANDED_METRIC_SET, \
    EXPANDED_HISTORY_KEYS, REFERENCE_LIST
from Constants.Tensor_Constants import OPTIMIZER_SET, EXPANDED_MODEL_METRICS
from Consumer.Model import Model


# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Ozturk:
    def __init__(self, size: int = 200, dimension: int = 1, hidden_neurons: int = 250, optimizer: str = 'adam',
                 epochs: int = 100, full_classes: bool = False, full_data: bool = False,
                 reference_distribution='Normal'):
        """
        loss function selection following suggestions from
        https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c
        Optimizer selection following suggestions from https://algorithmia.com/blog/introduction-to-optimizers
        :param size:
        :param dimension:
        :param hidden_neurons:
        :param optimizer:
        :param epochs:
        :param full_classes:
        :param full_data:
        :param reference_distribution
        """
        self.__size = int(size)
        self.__dim = int(dimension)
        self.__sample_size = 0

        # Load Training Data
        self.__theta: tf.Tensor = tf.zeros((self.__size,))
        self.__class_set = []
        self.__distribution_names = []
        self.__full_data = full_data
        if dimension == 1:
            self.__full_classes = True
        else:
            self.__full_classes = full_classes

        # Setting up ANNOA
        self.__train_ratio = 0.80
        self.__validation_ratio = 0.20
        self.__batch_size = 64

        self.__epochs = epochs
        self.__optimizer = optimizer
        self.__num_hidden_layers = -1
        self.__hidden_neurons = hidden_neurons
        self.__shapes = {}
        self.__model = None
        self.__dtype = np.float32
        self.__history: keras.callbacks.History = keras.callbacks.History()
        self.__training = {}
        self.__reference_distribution = reference_distribution.capitalize()

    def define_network(self, num_hidden_layers):
        self.__num_hidden_layers = num_hidden_layers
        self.__model = Sequential(layers=[Input(shape=self.__shapes['input'], name='UV Input')])
        for i in range(self.__num_hidden_layers):
            self.__model.add(Dense(self.__hidden_neurons, activation='relu', name='ANNOA_Hidden_' + str(i + 1)))
        self.__model.add(Dense(self.__shapes['output'], activation='softmax', name='Output'))
        self.__model.compile(optimizer=self.__optimizer, loss='categorical_crossentropy',
                             metrics=EXPANDED_MODEL_METRICS)
        print(self.__distribution_names)
        self.__model.summary()

    def size(self):
        return self.__size

    def dim(self):
        return self.__dim

    def __str__(self):
        return f'gen_Ozturk_[{self.__size}]_[{self.__full_classes}]_{self.__num_hidden_layers}'

    def train(self):
        batch_size = self.__batch_size
        self.__history = self.__model.fit(self.__training['input'],
                                          self.__training['output'],
                                          batch_size,
                                          self.__epochs,
                                          validation_split=self.__validation_ratio)

    def prepare_training_data(self):
        distribution_files = self.__filter_files()
        print(self.__distribution_names, flush=True)
        if self.__full_data:
            batch_columns = sum([[f'U{i}', f'V{i}'] for i in range(self.__size + 1)], start=[])
        else:
            batch_columns = ['U', 'V']
        batch_input_list = [pd.read_parquet(file).astype(dtype=self.__dtype).fillna(value=0)[batch_columns] for file in
                            distribution_files]
        pd_batch_input = pd.concat(batch_input_list, ignore_index=True)
        pd_batch_input.info()

        batch_output_list = [pd.read_parquet(file).astype(dtype=np.int8).fillna(value=0)[self.__distribution_names] for
                             file in distribution_files]
        pd_batch_output = pd.concat(batch_output_list, ignore_index=True)
        pd_batch_output.info()

        np.random.seed(SEED)
        reindex = np.random.permutation(pd_batch_input.shape[0])
        batch_input = pd_batch_input.to_numpy()[reindex]
        batch_output = pd_batch_output.to_numpy()[reindex]

        self.__training = {'input': batch_input, 'output': batch_output}
        self.__shapes = {'hidden': (self.__hidden_neurons,), 'output': len(self.__distribution_names),
                         'input': ((self.__size + 1) * 2,) if self.__full_data else (2,)}

    def __filter_files(self):
        distribution_files = []
        cwd = '/'.join(os.getcwd().split('\\')[:-2])
        all_files = list(glob.glob(f'{cwd}/Data/*{self.__size}.parquet_{self.__reference_distribution}_gz'))
        for file in all_files:
            dist = (file.split('\\')[-1]).split('_')[1:-3]
            if self.__full_classes:
                self.__distribution_names.append(' '.join(dist))
                distribution_files.append(file)
            elif all(map(lambda x: x == dist[0], dist)):
                self.__distribution_names.append(' '.join(dist))
                distribution_files.append(file)

        return distribution_files

    def store(self):
        cwd = '/'.join(os.getcwd().split('\\')[:-2]) + '/Results'
        history_keys = EXPANDED_HISTORY_KEYS
        history_df = pd.DataFrame(columns=EXPANDED_METRIC_SET)
        for key, col in zip(history_keys, EXPANDED_METRIC_SET):
            history_df[col] = self.__history.history[key]
        history_df.to_csv(f'{cwd}/NN_{self.__size}_HL{self.__num_hidden_layers}_dim{self.__dim}.csv', index=False)

    def info(self):
        for key in self.__training:
            simplified_bytes, in_units = label(self.__training[key].nbytes)
            print(self.__training[key].shape, simplified_bytes, in_units)


def run_ozturk_annoa(dimension, size, full_data, full_classes):
    for reference_distribution in REFERENCE_LIST[0:1]:
        training_model = Ozturk(size=size,
                                dimension=dimension,
                                optimizer=OPTIMIZER_SET[1],
                                hidden_neurons=HIDDEN_NEURONS,
                                epochs=NUM_EPOCHS,
                                reference_distribution=reference_distribution,
                                full_classes=full_classes,
                                full_data=full_data)
        training_model.prepare_training_data()
        for num_hidden_layers in NUM_HIDDEN_LAYERS:
            training_model.define_network(num_hidden_layers)
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
