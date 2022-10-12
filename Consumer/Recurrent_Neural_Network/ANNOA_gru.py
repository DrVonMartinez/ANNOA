import glob
import os

import numpy as np
import pandas as pd
from keras.layers import Dense, GRU, Input
from keras.metrics import mae
from keras.models import Sequential


class Ozturk:
    def __init__(self, size=200, dimension=1, num_hidden_layers=1, hidden_neurons=250, optimizer='adam', epochs=100, full_classes=False):
        """
        loss function selection following suggestions from https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c
        Optimizer selection following suggestions from https://algorithmia.com/blog/introduction-to-optimizers
        :param size:
        :param dimension:
        :param num_hidden_layers:
        :param hidden_neurons:
        :param optimizer:
        :param epochs:
        :param full_classes:
        """
        print('Ozturk Algorithm',
              'Epochs: ' + str(epochs),
              'Neurons: ' + str(hidden_neurons),
              'Size: ' + str(size),
              'Dimension: ' + str(dimension),
              'Hidden Layer Count: ' + str(num_hidden_layers),
              'Model Optimizer: ' + str(optimizer),
              'Full Class Set: ' + str(full_classes),
              sep='\n')
        self.__size = int(size)
        self.__dim = int(dimension)
        self.__sample_size = 0

        # Load Training Data
        self.__class_set = []
        self.__distribution_files = []
        self.__full_data = True
        self.__full_class = full_classes
        self.__load()

        # Setting up ANNOA
        self.__train_ratio = 0.80
        self.__validation_ratio = 0.20
        self.__batch_size = 64
        self.__optimizer = optimizer
        self.__epochs = epochs
        self.__model: Sequential
        self.__num_hidden_layers = num_hidden_layers
        self.__create_network(hidden_neurons)
        self.__title = self.__set_title()
        self.__dtype = np.float32

    def __load(self):
        cwd = self.__change_cwd_data()
        distribution_files = list(glob.glob('*.oa*'))
        filtered = filter(lambda x: 'oa_full' in x, distribution_files)

        def dim_variate(filename):
            distribution_junk = filename.split('//')[-1]
            distribution_junk2 = distribution_junk.split('_')[0]
            distribution_set = distribution_junk2.split(' ')
            is_dim_variate = True
            first_dist = distribution_set[0]
            for dist in distribution_set[1:]:
                if first_dist != dist:
                    is_dim_variate = False
            return is_dim_variate

        if not self.__full_class:
            filtered = filter(dim_variate, filtered)
        self.__distribution_files = list(filtered)
        print('Files loaded', len(self.__distribution_files))
        if len(self.__distribution_files) == 0:
            raise FileNotFoundError("Files Not Found")
        self.__class_set = list(map(lambda x: x.split('_')[0], self.__distribution_files))
        with open(self.__distribution_files[-1], 'r') as line_counter:
            self.__sample_size = len(line_counter.readlines()) - 1
        self.__revert(cwd)

    def __create_network(self, hidden_neurons):
        shapes = self.__define_shape(hidden_neurons)
        self.__model = Sequential(layers=[Input(shape=shapes['input'], name='UV Input')])
        for i in range(self.__num_hidden_layers):
            self.__model.add(GRU(shapes['hidden'], return_sequences=i != (self.__num_hidden_layers - 1), name='ANNOA_Hidden_' + str(i + 1)))
        self.__model.add(Dense(shapes['output'], activation='softmax', name='output'))
        self.__model.compile(optimizer=self.__optimizer, loss='categorical_crossentropy', metrics=[mae, 'accuracy'])
        print(self.__model.summary())

    def __define_shape(self, hidden_neurons):
        shapes = {'input': (2, 1), 'hidden': hidden_neurons, 'output': len(self.__distribution_files)}
        return shapes

    def size(self):
        return self.__size

    def dim(self):
        return self.__dim

    def __str__(self):
        return 'GRU_Ozturk_[' + str(self.__size) + ']_[' + ','.join(self.__class_set) + ']'

    def train(self):
        total_number_samples = len(self.__distribution_files) * self.__sample_size  # number of samples
        training = self.__prepare_training_data(total_number_samples)
        history = self.__model.fit(training['input'], training['output'], self.__batch_size, self.__epochs, validation_split=self.__validation_ratio)
        self.__store(history)

    def __prepare_training_data(self, total_number_samples):
        cwd = self.__change_cwd_data()
        batch_df = pd.DataFrame(dtype=self.__dtype)
        files_df = []
        for file, i in zip(self.__distribution_files, range(len(self.__distribution_files))):
            df = pd.read_csv(file)
            df.drop(columns=['Distribution'], inplace=True)
            df[self.__class_set[i]] = np.ones(self.__sample_size, dtype=self.__dtype)
            df = df.astype(dtype=self.__dtype)
            files_df.append(df)
        batch_df = batch_df.append(files_df, ignore_index=True)
        index = 2 * (self.__size + 1)
        seed = 9
        np.random.seed(seed=seed)
        batch_df = batch_df.reindex(np.random.permutation(batch_df.index))
        train_df = batch_df[batch_df.columns[:index]]
        categorical_df = batch_df[batch_df.columns[index:]].fillna(value=0)
        train_list = train_df.to_numpy(dtype=self.__dtype)
        categorical_list = categorical_df.to_numpy(dtype=self.__dtype)
        shape = (total_number_samples, (self.__size + 1) * 2)
        batch_input = train_list.reshape(shape)
        batch_output = categorical_list.reshape(total_number_samples, len(self.__class_set))
        assert batch_input.shape[0] == total_number_samples, 'Size mismatch'
        assert batch_output.shape[0] == total_number_samples, 'Size mismatch'
        training = {'input': batch_input, 'output': batch_output}
        self.__revert(cwd)
        return training

    def __store(self, history):
        history_keys = ['loss', 'mean_absolute_error', 'accuracy']
        cwd = self.__change_cwd_results()
        history_df = pd.DataFrame(columns=history_keys)
        for key in history_keys:
            history_df[key] = history.history[key]
        history_df.to_csv(self.__title + '.csv', index=False)
        self.__revert(cwd)

    def __revert(self, cwd):
        os.chdir(cwd)
        return self.__size

    def __change_cwd_data(self):
        cwd = os.getcwd()
        os.chdir('../../Data\\')
        cwd_dir = os.getcwd()
        new_dir = '\\dim ' + str(self.__dim) + '\\' + str(self.__size)
        new_dim = '\\dim ' + str(self.__dim)
        new_size = '\\' + str(self.__size)
        try:
            os.chdir(cwd_dir + new_dir)
        except FileNotFoundError:
            try:
                os.mkdir(cwd_dir + new_dim)
            except FileExistsError:
                os.chdir(cwd_dir + new_dim)
            finally:
                cwd_dir = os.getcwd()
            os.mkdir(cwd_dir + new_size)
            os.chdir(cwd_dir + new_size)
        return cwd

    def __change_cwd_results(self):
        cwd = os.getcwd()
        data = 'All Data'
        new_dir = '\\' + data + '\\'
        os.chdir('../../Results\\')
        cwd_dir = os.getcwd()
        try:
            os.chdir(cwd_dir + new_dir)
        except FileNotFoundError:
            os.mkdir(cwd_dir + new_dir)
            os.chdir(cwd_dir + new_dir)
        x = self.__size
        return cwd

    def __full_classes_label(self):
        if self.__full_class:
            classes = 'Complete Classes'
        else:
            classes = 'Limited Classes'
        return classes

    def __set_title(self):
        data = 'All Data'
        classes = self.__full_classes_label()
        return 'GRU Hidden Layer {}; Size {}; {}; {}; {};'.format(self.__num_hidden_layers, self.__size, classes, data, self.__optimizer)


def main():
    assert (__name__ == "__main__"), "Method not intended to be called if this isn't the main file"
    # sizes = [10, 25, 50, 75, 80, 100, 125, 150, 200, 500, 1000]
    sizes = [200]
    dimensions = [2]
    for full_classes in [False]:
        for num_hidden_layers in range(1, 3):
            for dimension in dimensions:
                for size in sizes:
                    hidden_neurons = int(size * 1.25)
                    training_model = Ozturk(size=size, dimension=dimension, num_hidden_layers=num_hidden_layers, hidden_neurons=hidden_neurons, optimizer='adam', epochs=25, full_classes=full_classes)
                    # training_model.train()


if __name__ == "__main__":
    '''
    Optimized for training size ANNOA
    '''
    main()
