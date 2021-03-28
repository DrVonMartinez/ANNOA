import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Sequential
from Utilities.Constants import SIZE_SET, SEED, label
from Utilities.Expanded_Constants import NUM_HIDDEN_LAYERS, HIDDEN_NEURONS, NUM_EPOCHS, EXPANDED_METRIC_SET, EXPANDED_HISTORY_KEYS, REFERENCE_LIST
from Utilities.Tensor_Constants import OPTIMIZER_SET, EXPANDED_MODEL_METRICS


# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Ozturk:
    def __init__(self, size: int = 200, dimension: int = 1, hidden_neurons: int = 250, optimizer: str = 'adam', epochs: int = 100, full_classes: bool = False, full_data: bool = False,
                 reference_distribution='Normal'):
        """
        loss function selection following suggestions from https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c
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
        self.__history = {}
        self.__training = {}
        self.__reference_distribution = reference_distribution.capitalize()
        self.__title = self.__set_title()
        print('Ozturk Algorithm', 'Epochs: ' + str(self.__epochs), 'Neurons: ' + str(self.__hidden_neurons), 'Size: ' + str(self.__size), 'Dimension: ' + str(self.__dim), 'Model Optimizer: ' +
              str(self.__optimizer), 'Reference Distribution: ' + str(self.__reference_distribution), 'Full Class Set: ' + str(bool(self.__full_classes)), 'Full Data Set: ' +
              str(bool(self.__full_data)), sep='\n')

    def define_network(self, num_hidden_layers):
        self.__num_hidden_layers = num_hidden_layers
        self.__model = Sequential(layers=[Input(shape=self.__shapes['input'], name='UV Input')])
        for i in range(self.__num_hidden_layers):
            self.__model.add(Dense(self.__hidden_neurons, activation='relu', name='ANNOA_Hidden_' + str(i + 1)))
        self.__model.add(Dense(self.__shapes['output'], activation='softmax', name='Output'))
        self.__model.compile(optimizer=self.__optimizer, loss='categorical_crossentropy', metrics=EXPANDED_MODEL_METRICS)
        print(self.__distribution_names)
        self.__model.summary()
        self.__title = self.__set_title()

    def __define_shape(self):
        if self.__full_data:
            self.__shapes = {'input': ((self.__size + 1) * 2,), 'hidden': (self.__hidden_neurons,)}
        else:
            self.__shapes = {'input': (2,), 'hidden': (self.__hidden_neurons,)}
        self.__shapes['output'] = len(self.__distribution_names)

    def size(self):
        return self.__size

    def dim(self):
        return self.__dim

    def __str__(self):
        return 'gen_Ozturk_[' + str(self.__size) + ']_[' + str(self.__full_classes) + ']' + self.__full_data_label().replace(' ', '_') + '_' + str(self.__num_hidden_layers)

    def train(self):
        batch_size = self.__batch_size
        self.__history = self.__model.fit(self.__training['input'],
                                          self.__training['output'],
                                          batch_size,
                                          self.__epochs,
                                          validation_split=self.__validation_ratio)

    def prepare_training_data(self):
        cwd = self.__change_cwd_data()
        print(os.getcwd())
        distribution_files = self.__filter_files()
        print(self.__distribution_names, flush=True)
        batch_columns = []
        if self.__full_data:
            for i in range(self.__size + 1):
                batch_columns.append('U' + str(i))
                batch_columns.append('V' + str(i))
        else:
            batch_columns = ['U', 'V']
        batch_input_list = [pd.read_parquet(file).astype(dtype=self.__dtype).fillna(value=0)[batch_columns] for file in distribution_files]
        pd_batch_input = pd.concat(batch_input_list)
        pd_batch_input.info()

        batch_output_list = [pd.read_parquet(file).astype(dtype=np.int8).fillna(value=0)[self.__distribution_names] for file in distribution_files]
        pd_batch_output = pd.concat(batch_output_list)
        pd_batch_output.info()

        np.random.seed(seed=SEED)
        reindex = np.random.permutation(pd_batch_input.shape[0])
        batch_input = pd_batch_input.to_numpy()[reindex]
        batch_output = pd_batch_output.to_numpy()[reindex]

        self.__revert(cwd)
        self.__training = {'input': batch_input, 'output': batch_output}
        self.__define_shape()

    def __filter_files(self):
        distribution_files = []
        all_files = list(glob.glob('* ' + str(self.__size) + '.parquet_' + self.__reference_distribution + '_gz'))
        for file in all_files:
            title = file.split('\\')[-1]
            dist = title.split(' ')[:-3]
            if self.__full_classes:
                self.__distribution_names.append(' '.join(dist))
                distribution_files.append(file)
            elif all(map(lambda x: x == dist[0], dist)):
                self.__distribution_names.append(' '.join(dist))
                distribution_files.append(file)

        if self.__full_classes:
            assert len(self.__distribution_names) == len(all_files), str(len(self.__distribution_names)) + ' != ' + str(len(all_files))
        else:
            assert len(self.__distribution_names) == np.round(np.power(len(all_files), 1 / self.__dim), 8), \
                str(len(self.__distribution_names)) + ' != ' + str(np.round(np.power(len(all_files), 1 / self.__dim), 8))
        return distribution_files

    def store(self):
        history_keys = EXPANDED_HISTORY_KEYS
        cwd = self.__change_cwd_results()
        history_df = pd.DataFrame(columns=EXPANDED_METRIC_SET)
        for key, col in zip(history_keys, EXPANDED_METRIC_SET):
            history_df[col] = self.__history.history[key]
        history_df.to_csv(self.__title + '.csv', index=False)
        self.__revert(cwd)

    def __revert(self, cwd):
        os.chdir(cwd)
        return self.__size

    def __change_cwd_data(self):
        cwd = os.getcwd()
        os.chdir('F:\\Data\\')
        cwd_dir = os.getcwd()
        new_dir = '\\dim ' + str(self.__dim) + '\\Ref ' + self.__reference_distribution + '\\'
        os.chdir(cwd_dir + new_dir)
        return cwd

    def __change_cwd_results(self):
        cwd = os.getcwd()
        if self.__full_data:
            data = 'All Data'
        else:
            data = 'Partial Data'
        if self.__full_classes:
            classes = 'Full Classes'
        else:
            classes = 'Limited Classes'
        new_dir = 'Ref ' + self.__reference_distribution + '\\' + data + '\\' + classes + '\\'
        os.chdir('..\\..\\Results\\' + new_dir)
        return cwd

    def __full_data_label(self):
        if self.__full_data:
            data = 'All Data'
        else:
            data = 'Partial Data'
        return data

    def __full_classes_label(self):
        if self.__full_classes:
            classes = 'Complete Classes'
        else:
            classes = 'Limited Classes'
        return classes

    def __set_title(self):
        return '{}, Dim {}, Hidden Layer {}, Size {}, {}'.format(self.__reference_distribution, self.__dim, self.__num_hidden_layers, self.__size, self.__optimizer)

    def info(self):
        for key in self.__training:
            simplified_bytes, in_units = label(self.__training[key].nbytes)
            print(self.__training[key].shape, simplified_bytes, in_units)


def run_ozturk_annoa(dimension, size, full_data, full_classes):
    reference_set = REFERENCE_LIST[2:]
    for reference_distribution in reference_set:
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
    assert (__name__ == "__main__"), "Method not intended to be called if this isn't the main file"
    '''
    for dimension in [1]:
        for size in SIZE_SET:
            for full_data in [True, False]:
                run_ozturk_annoa(dimension, size, full_data, True)
    '''
    for dimension in [2]:
        # for size in SIZE_SET:
        for size in [50, 75, 80, 100, 125, 150, 200]:
            for full_data in [True, False]:
                for full_classes in [True]:
                    run_ozturk_annoa(dimension, size, full_data, full_classes)


if __name__ == "__main__":
    '''
    Optimized for training size ANNOA
    '''
    main()
