import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from Constants.Constants import SEED, label
from Constants.Expanded_Constants import EXPANDED_METRIC_SET, \
    EXPANDED_HISTORY_KEYS
from Consumer import Model


class Ozturk:
    def __init__(self, size: int = 200, dimension: int = 1, full_classes: bool = False, full_data: bool = False,
                 reference_distribution='Normal'):
        """
        loss function selection following suggestions from
        https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c
        Optimizer selection following suggestions from https://algorithmia.com/blog/introduction-to-optimizers
        :param size:
        :param dimension:
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
        self.__batch_size = 64

        self.__model = None
        self.__dtype = float
        self.__history: dict = {}
        self.__training = {}
        self.__reference_distribution = reference_distribution.capitalize()

    def define_model(self, model: Model):
        self.__model = model
        self.__model.summary()

    def size(self):
        return self.__size

    def dim(self):
        return self.__dim

    def __len__(self):
        return len(self.__distribution_names)

    def __str__(self):
        return f'gen_Ozturk_[{self.__size}]_[{self.__full_classes}]_{self.__model}'

    def train(self):
        self.__history = self.__model.train(self.__training['input'], self.__training['output'])

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
            history_df[col] = self.__history[key]
        history_df.to_csv(f'{cwd}/{self.__model}_{self.__size}_{self.__reference_distribution}_dim{self.__dim}.csv', index=False)

    def info(self):
        for key in self.__training:
            simplified_bytes, in_units = label(self.__training[key].nbytes)
            print(self.__training[key].shape, simplified_bytes, in_units)