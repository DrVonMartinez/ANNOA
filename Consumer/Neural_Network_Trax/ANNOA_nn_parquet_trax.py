import glob
import itertools
import random as rnd
import warnings

import pandas as pd
import trax
from markdown.util import deprecated
from sklearn.preprocessing import scale as standardize
from trax import layers as tl
from trax.fastmath import numpy as np
from trax.supervised import training

from Constants.Constants import REFERENCE_DICTIONARY, STATS_SET
from Distribution.Distribution_trax import Distribution
from Utilities.DataGenerator import data_generator


@deprecated
class Ozturk:
    def __init__(self, size: int = 200, dimension: int = 1, hidden_neurons: int = 250, optimizer=trax.optimizers.Adam,
                 full_classes: bool = False, full_data: bool = False, reference_distribution: str = 'Normal',
                 batch_size: int = 64):
        """
                loss function selection following suggestions from
                https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c
                Optimizer selection following suggestions from https://algorithmia.com/blog/introduction-to-optimizers
                :param size:
                :param dimension:
                :param hidden_neurons:
                :param optimizer:
                :param full_classes:
                :param full_data:
                :param reference_distribution
                """
        self.__size = int(size)
        self.__dim = int(dimension)

        # Load Training Data
        self.__theta: np.ndarray = None
        self.__reference_distribution = reference_distribution.capitalize()
        self.__load_reference_set()
        self.__class_set: set[Distribution] = set()
        self.__distribution_names = []
        self.__full_data = full_data
        self.__full_classes = dimension == 1 or full_classes

        # Setting up ANNOA
        self.__train_ratio = 0.80
        self.__batch_size = batch_size
        self.__batch_input = None
        self.__batch_output = None
        self.__optimizer = optimizer
        self.__hidden_neurons = hidden_neurons

    @property
    def size(self) -> int:
        return self.__size

    @property
    def dim(self) -> int:
        return self.__dim

    @property
    def distribution_names(self) -> list:
        return self.__distribution_names

    @property
    def class_set(self) -> set:
        return self.__class_set

    @property
    def reference_distribution(self) -> str:
        return self.__reference_distribution

    def train_model(self, num_hidden_layers=0, seed=0):
        model = self.__create_model(num_hidden_layers, seed)
        length = int(len(self.__batch_input) * self.__train_ratio)

        train_task = training.TrainTask(
            labeled_data=data_generator(self.__batch_input[:length], self.__batch_output[:length], self.__batch_size),
            loss_layer=tl.CrossEntropyLoss(),
            optimizer=trax.optimizers.Adam(0.01),
            lr_schedule=trax.lr.warmup_and_rsqrt_decay(400, 0.01)
        )
        eval_task = training.EvalTask(
            labeled_data=data_generator(self.__batch_input[length:], self.__batch_output[length:], self.__batch_size),
            metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
        )
        return training.Loop(model, tasks=train_task, eval_tasks=[eval_task])

    def __create_model(self, num_hidden_layers=0, seed=0):
        rnd.seed(seed)
        self.__prepare_training_data()
        oa_output = 2 * ((1 + self.__size) if self.__full_data else 1)
        model = tl.Serial(
            tl.Embedding(oa_output, self.__hidden_neurons if num_hidden_layers > 0 else len(self.__distribution_names)),
            # tl.Select([0, 1], name='Receive_Inputs'),
            # tl.Fn(F'Load_{self.__reference_distribution}_Theta', lambda: self.__load_reference_set(), 0),
            # tl.Fn('Ozturk_Algorithm', lambda x: self.__ozturk_function(x), oa_output),
            [tl.Serial(
                tl.Dense(self.__hidden_neurons),
                tl.Relu()
            )] * num_hidden_layers,
            tl.Serial(
                tl.Dense(len(self.__distribution_names)),
                tl.Softmax()
            )
        )
        print(model)
        return model

    def __prepare_training_data(self):
        if self.__full_data:
            batch_columns = sum([[f'U{j}', f'V{j}'] for j in range(self.__size + 1)], start=[])
        else:
            batch_columns = ['U', 'V']
        distribution_files = []
        for file in glob.glob(f'* {self.__size}.parquet_{self.__reference_distribution}_gz'):
            dist = (file.split('\\')[-1]).split(' ')[:-3]
            if self.__full_classes:
                self.__distribution_names.append(' '.join(dist))
                distribution_files.append(file)
            elif all(map(lambda x: x == dist[0], dist)):
                self.__distribution_names.append(' '.join(dist))
                distribution_files.append(file)
        batch_input_list = []
        batch_output_list = []
        for file in distribution_files:
            batch_df = pd.read_parquet(file).fillna(value=0).astype(dtype=float)
            batch_input_temp = batch_df[batch_columns]
            batch_output_temp = batch_df[self.__distribution_names]
            batch_input_list.append(batch_input_temp)
            batch_output_list.append(batch_output_temp)
        self.__batch_input = pd.concat(batch_input_list, axis=0, ignore_index=True)
        batch_output = pd.concat(batch_output_list, axis=0, ignore_index=True)  # .astype(int)
        batch_indexes = [*range(len(self.__batch_input))]
        rnd.shuffle(batch_indexes)
        self.__batch_input = self.__batch_input.loc[batch_indexes]
        self.__batch_input.reset_index(inplace=True, drop=True)
        self.__batch_output = batch_output.loc[batch_indexes]
        # self.__batch_output = (batch_output * np.array([*range(1, len(self.__distribution_names) + 1)])).sum(axis=1)
        # self.__batch_output.reset_index(inplace=True, drop=True)
        # self.__batch_output = pd.Series(self.__batch_output, name='Output')
        print(self.__batch_input.shape, self.__batch_output.shape)

    def __train_generator(self):
        length = int(len(self.__batch_input) * self.__train_ratio)
        batch_input: pd.DataFrame = self.__batch_input[:length]
        batch_output: pd.DataFrame = self.__batch_output[:length]
        input1 = []
        output1 = []
        idx = 0
        batch_indexes = [*range(length)]
        rnd.shuffle(batch_indexes)
        while True:
            if idx >= len(batch_input):
                idx = 0
                rnd.shuffle(batch_indexes)
            idx += 1
            input1.append(np.array(batch_input.loc[batch_indexes[idx]]))
            output1.append(np.array(batch_output.loc[batch_indexes[idx]]))
            if len(input1) == self.__batch_size:
                yield np.array(input1), np.array(output1)
                # yield np.array(input1).T, np.array(output1).reshape((1, len(output1)))
                input1, output1 = [], []

    def __val_generator(self):
        length = int(len(self.__batch_input) * self.__train_ratio)
        batch_input: pd.DataFrame = self.__batch_input[length:]
        batch_output: pd.DataFrame = self.__batch_output[length:]
        input1 = []
        output1 = []
        idx = 0
        batch_indexes = [*range(length, len(self.__batch_input))]
        rnd.shuffle(batch_indexes)
        while True:
            if idx >= len(batch_input):
                idx = 0
                rnd.shuffle(batch_indexes)
            idx += 1
            input1.append(np.array(batch_input.loc[batch_indexes[idx]]))
            output1.append(np.array(batch_output.loc[batch_indexes[idx]]))
            if len(input1) == self.__batch_size:
                yield np.array(input1), np.array(output1)
                # yield np.array(input1).T, np.array(output1).reshape((1, len(output1)))
                input1, output1 = [], []

    def __load_reference_set(self, monte_carlo=2000):
        """
        This finds the angles for the library
        """
        reference_df = None
        try:
            # Load
            reference_df = pd.read_parquet(f'Reference Set {self.__size}.parquet_ref')
            self.__theta = reference_df[self.__reference_distribution].to_numpy().reshape((1, self.__size))
            print(self.__reference_distribution + ' Theta Loaded', self.__theta.shape, flush=True)
        except FileNotFoundError:
            # Create File
            print('File Not Found', flush=True)
            filename = self.__train_reference_set(monte_carlo)
            print('"' + filename + '" created', flush=True)
        except KeyError:
            # Standardized Null Hypothesis
            print(self.__reference_distribution + ' Not Found', flush=True)
            filename = self.__train_reference_set(monte_carlo, reference_df)
            print(self.__reference_distribution + ' added to "' + filename + '"', flush=True)

    def __train_reference_set(self, monte_carlo, reference_df=None):
        reference_set = REFERENCE_DICTIONARY[self.__reference_distribution].rvs(size=monte_carlo * self.__size).reshape(
            (self.__size, monte_carlo))
        reference_set.sort(axis=0)
        m = reference_set.mean(axis=1)
        self.__theta = (np.pi * REFERENCE_DICTIONARY[self.__reference_distribution].cdf(m)).reshape((1, self.__size))
        df = pd.DataFrame(self.__theta.reshape((self.__size, 1)), columns=[self.__reference_distribution])
        reference_df = pd.merge(reference_df, df, left_index=True, right_index=True) if reference_df else df
        filename = f'Reference Set {self.__size}.parquet_ref'
        reference_df.to_parquet(filename)
        return filename

    def train_ozturk_distributions(self, distributions: list, monte_carlo=2000):
        if monte_carlo < 1000:
            warnings.warn('Warning! Small Number of Monte Carlo Simulations')
        elif monte_carlo < 0:
            raise ValueError("There must be at least 1 Monte Carlo run")
        dist_iter = list(itertools.product(range(len(distributions)), repeat=self.__dim))
        for index in dist_iter:
            dist = sum([Distribution(distributions[index[j]]) for j in range(len(index))], start=None)
            if dist.dimension != self.__dim:
                raise ValueError(f'Dimension mismatch. Dimension should be {self.__dim} but is {dist.dimension}')
            self.__class_set.add(dist)
        if len(self.__class_set) != len(dist_iter):
            raise ValueError(f'Class Set mismatch. Expected should be {len(dist_iter)} but is {len(self.class_set)}')

        columns = self.__determine_columns()
        for dist_class in self.__class_set:
            distribution_df = self.__oa_hidden_single(columns, dist_class, monte_carlo)
            self.__store_seq(distribution_df.astype(dtype=float).fillna(value=0), dist_class)

    def train_single_distribution(self, distribution_set, monte_carlo=2000):
        if monte_carlo < 1000:
            warnings.warn('Warning! Small Number of Monte Carlo Simulations')
        elif monte_carlo < 0:
            raise ValueError("There must be at least 1 Monte Carlo run")
        distribution_set = [distribution_set] if not isinstance(distribution_set, list) else distribution_set
        distribution_sequence = sum([Distribution(distribution_set[j]) for j in range(self.dim)], start=None)
        self.__class_set.add(distribution_sequence)
        columns = self.__determine_columns()
        distribution_df = self.__oa_hidden_single(columns, distribution_sequence, monte_carlo)
        self.__store_seq(distribution_df, distribution_sequence)

    def __determine_columns(self) -> list[str]:
        columns: list = sum([['U', 'V'] if j < 0 else [f'U{j}', f'V{j}'] for j in range(-1, self.__size + 1)], start=[])
        distribution_set = list(map(str, self.__class_set))
        single_dist_set: list = list(set([dist for dist in sum([d.split(' ') for d in distribution_set], start=[])]))
        return columns + distribution_set + (single_dist_set if self.__dim > 1 else [])

    def __oa_hidden_single(self, columns: list[str], distribution_sequence: Distribution, monte_carlo: int):
        oa_producer_df = pd.DataFrame(columns=columns, dtype=float)
        u, v = self.__predict(distribution_sequence.rvs(size=self.__size, samples=monte_carlo), monte_carlo)
        oa_producer_df['U'] = u[:, -1]
        oa_producer_df['V'] = v[:, -1]
        ones_column: set = {str(distribution_sequence)}
        ones = np.ones(monte_carlo)  # if self.__dim == 1 else np.ones((monte_carlo, 1))
        if self.__dim > 1:
            ones_column = ones_column.union({dist for dist in str(distribution_sequence).split(' ')})

        for values in ones_column:
            oa_producer_df[values] = ones

        for j in range(self.__size + 1):
            oa_producer_df['U' + str(j)] = u[:, j]
            oa_producer_df['V' + str(j)] = v[:, j]
        return oa_producer_df

    def predict(self, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.__predict(prediction)

    def __predict(self, prediction, monte_carlo=1):
        oa = np.sort(self.__beta_reduction(prediction, monte_carlo), axis=1)
        detrended = standardize(oa, axis=1)
        return self.__ozturk_function(detrended, monte_carlo)

    def __ozturk_function(self, t, monte_carlo=1) -> tuple[np.ndarray, np.ndarray]:
        initial_u = np.abs(t) * np.cos(self.__theta)
        initial_v = np.abs(t) * np.sin(self.__theta)
        u = np.zeros((monte_carlo, self.__size + 1))
        v = np.zeros((monte_carlo, self.__size + 1))
        for j in range(1, self.__size + 1):
            u = u.at[:, j].set(np.sum(initial_u[:, :j], axis=1) / j)
            v = v.at[:, j].set(np.sum(initial_v[:, :j], axis=1) / j)
        return u, v

    def __beta_reduction(self, prediction: np.ndarray, monte_carlo=1) -> np.ndarray:
        """
        The Formula for Z^2
        Z**2 = (p-mu).T  *  sigma**-1  * (p-mu)
        """
        if self.__dim == 1:
            return prediction.reshape(monte_carlo, self.__size)
        p_mean = prediction.mean(axis=0).reshape((monte_carlo, 1, self.__dim))
        p_cov = np.array([np.cov(prediction[j, :, :], rowvar=False) for j in range(monte_carlo)])
        inv_p_cov = np.linalg.inv(p_cov)
        try:
            assert not np.isnan(inv_p_cov).any()
        except AssertionError:
            inv_p_cov = np.linalg.pinv(p_cov)

        p_t: np.ndarray = np.subtract(prediction, p_mean).conj()
        p_new: np.ndarray = np.transpose(np.subtract(prediction, p_mean), axes=(0, 2, 1))
        z_2 = np.zeros((monte_carlo, self.__size))
        for j in range(self.__size):
            p_t_temp = np.reshape(p_t[:, j, :], (monte_carlo, 1, self.__dim))
            p_new_temp = np.reshape(p_new[:, :, j], (monte_carlo, self.__dim, 1))
            z_2[:, j] = (p_t_temp @ inv_p_cov @ p_new_temp).reshape(monte_carlo)
        return z_2

    def __store_seq(self, oa_producer_df, distribution_sequence):
        filename: str = f'{distribution_sequence} Data Set {self.size}.parquet_{self.__reference_distribution}_gz'
        oa_producer_df.to_parquet(filename, compression='gzip')


def train_ozturk():
    assert (__name__ == "__main__"), "Method not intended to be called if this isn't the main file"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dim in [1]:
            for size in [10]:
                for ref_set in ['Normal']:
                    training_run = Ozturk(size=size, dimension=dim, reference_distribution=ref_set)
                    training_run.train_ozturk_distributions(STATS_SET)


if __name__ == "__main__":
    '''
    Optimized for training size ANNOA
    '''
    # train_ozturk()

    train_steps = 10
    testing_run = Ozturk(10, batch_size=20)
    for i in range(2):
        print('#1')
        training_loop = testing_run.train_model(i)
        print('#2')
        training_loop.run(train_steps)
        print('#3')
        assert False
