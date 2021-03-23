import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Sequential
from Utilities.Constants import SIZE_SET, SEED, label, MONTE_CARLO
from Utilities.Expanded_Constants import NUM_HIDDEN_LAYERS, HIDDEN_NEURONS, NUM_EPOCHS, EXPANDED_METRIC_SET, EXPANDED_HISTORY_KEYS
from Utilities.Tensor_Constants import OPTIMIZER_SET, EXPANDED_MODEL_METRICS

# physical_devices = tf.config.list_physical_devices('GPU')
# print(physical_devices)
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

        self.__batch_columns = []
        self.__distribution_files = []

        self.__epochs = epochs
        self.__steps_per_epoch = np.ceil(MONTE_CARLO * self.__train_ratio/ self.__batch_size)
        self.__optimizer = optimizer
        self.__num_hidden_layers = -1
        self.__hidden_neurons = hidden_neurons
        self.__shapes = {}
        self.__model = None
        self.__dtype = np.float32
        self.__history = {}
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
        cwd = self.__change_cwd_data()
        self.__history = self.__model.fit(self.batch_generator(), epochs=self.__epochs, steps_per_epoch=self.__steps_per_epoch)
        self.__revert(cwd)

    def prepare_training_data(self):
        cwd = self.__change_cwd_data()
        distribution_files = self.__filter_files()
        print(self.__distribution_names, flush=True)
        if self.__full_data:
            for i in range(self.__size + 1):
                self.__batch_columns.append('U' + str(i))
                self.__batch_columns.append('V' + str(i))
        else:
            self.__batch_columns = ['U', 'V']
        self.__distribution_files = distribution_files
        self.__define_shape()
        self.__revert(cwd)

    def batch_generator(self):
        index = np.ones(len(self.__distribution_names), dtype=np.int32)
        # train_df = 'Data Set 500.csv'
        file_index = np.arange(0, len(self.__distribution_names), 1, dtype=np.int32)
        while True:
            batch_index = np.random.choice(a=file_index, size=self.__batch_size)
            unique_files, unique_batch_count = np.unique(batch_index, return_counts=True)
            pd_list = []
            for i in range(len(unique_files)):
                file_number = unique_files[i]
                rows = list(range(index[file_number], index[file_number] + unique_batch_count[i]))
                if index[file_number] + unique_batch_count[i] > (MONTE_CARLO * self.__train_ratio) + 1:
                    rows = list(range(index[file_number], (MONTE_CARLO * self.__train_ratio) + 1)) + list(range(1, index[file_number] + unique_batch_count[i] - ((MONTE_CARLO * self.__train_ratio) + 1)))
                instance = self.__load_instance(rows, file_number)
                index[file_number] += unique_batch_count[i]
                pd_list.append(instance)
            instance_df: pd.DataFrame = pd.concat(pd_list, ignore_index=True)
            training = instance_df[self.__batch_columns].astype(self.__dtype)
            testing = instance_df[self.__distribution_names].astype(np.int16)
            training_shape = (self.__batch_size, len(self.__batch_columns))
            testing_shape = (self.__batch_size, len(self.__distribution_names))
            if training.shape != training_shape:
                str_shape = str(training_shape)
                str_columns = '[' + ', '.join(list(training.columns)) + ']'
                raise ValueError('Training Shape does not follow format (Batch Size, len(Batch Columns))\n' + str_columns + '\n' + str_shape + '!=' + str(training.shape))
            if testing.shape != testing_shape:
                str_columns = '[' + ', '.join(list(testing.columns)) + ']'
                str_shape = str(testing_shape)
                raise ValueError('Testing Shape does not follow format (Batch Size, len(Distribution Names))\n' + str_columns + '\n' + str_shape + '!=' + str(testing.shape))
            yield training.to_numpy(), testing.to_numpy()

    def __load_instance(self, rows: list, file_index: int) -> pd.DataFrame:
        # print(self.__distribution_names[file_index] + '\t\t' + self.__distribution_files[file_index], end='\t\t')
        return pd.read_csv(self.__distribution_files[file_index], skiprows=lambda x: x not in [0] + rows)

    def __filter_files(self):
        distribution_files = []
        all_files = list(glob.glob('* ' + str(self.__size) + '.csv_' + self.__reference_distribution))
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
            assert len(self.__distribution_names) == np.power(len(all_files), 1 / self.__dim), str(len(self.__distribution_names)) + ' != ' + str(np.power(len(all_files), 1 / self.__dim))
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


def run_ozturk_annoa(dimension, size, full_data, full_classes):
    reference_set = ['Normal', 'Uniform']
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
            training_model.train()
            training_model.store()


def main():
    assert (__name__ == "__main__"), "Method not intended to be called if this isn't the main file"
    for dimension in [1]:
        # for size in SIZE_SET[-1:]:
        for size in [500]:
            for full_data in [True, False]:
                run_ozturk_annoa(dimension, size, full_data, True)
    '''                
    for dimension in [2]:
        for size in SIZE_SET:
            for full_data in [True, False]:
                for full_classes in [True, False]:
                    run_ozturk_annoa(dimension, size, full_data, full_classes)
    '''


if __name__ == "__main__":
    '''
    Optimized for training size ANNOA
    '''
    main()
