import glob
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale as standardize

from Constants.Constants import SEED, label, PCA_VAL
from Constants.Storage_Constants import DATA_PATH, MODEL_PATH, RESULT_PATH


class GeneralizedOzturk:
    def __init__(self, size: int = 200, dimension: int = 1, full_classes: bool = False, full_data: bool = False, reference_distribution='Normal'):
        """
        loss function selection following suggestions from https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c
        Optimizer selection following suggestions from https://algorithmia.com/blog/introduction-to-optimizers
        :param size:
        :param dimension:
        :param full_classes:
        :param full_data:
        :param reference_distribution
        """
        self.__size = int(size)
        self.__dim = int(dimension)
        self.__dtype = np.float32
        self.__training = {}

        # Load Reference Distribution
        self.__reference_distribution = reference_distribution.capitalize()
        self.__theta: np.ndarray = np.empty((1, self.__size))

        # Load Training Data
        self.__class_set = []
        self.__distribution_names = []
        self.full_data = full_data
        self.full_classes = dimension == 1 or full_classes

        # Setting up ANNOA
        # self.train_ratio = 0.80
        # self.validation_ratio = 0.20
        # self.batch_size = 64
        self.title = 'Generalized Ozturk Network'

    def size(self):
        return self.__size

    def dim(self):
        return self.__dim

    def distribution_names(self):
        return self.__distribution_names

    def class_set(self):
        return self.__class_set

    def reference_distribution(self):
        return self.__reference_distribution

    def get_training_data(self) -> [pd.DataFrame, pd.DataFrame]:
        return self.__training['input'], self.__training['output']

    def __str__(self):
        return 'gen_Ozturk_[' + str(self.size) + ']_[' + str(self.full_classes) + ']' + self.full_data_label().replace(' ', '_')

    def prepare_training_data(self, multilabel=False, result_column=False, pca_data=False):
        cwd = self.change_cwd_data()
        self.__theta = self.__reference_set()
        distribution_files = self.__filter_files()
        print(self.__distribution_names, flush=True)
        if multilabel:
            pd_batch_input, pd_batch_output = self.__multilabel_prepare_training_data(distribution_files)
            pd_batch_input.info()
            pd_batch_output.info()
            np.random.seed(seed=SEED)
            reindex = np.random.permutation(pd_batch_input.shape[0])
            batch_input = pd_batch_input.to_numpy()[reindex]
            batch_output = pd_batch_output.to_numpy()[reindex]
        elif result_column:
            pd_batch_input, pd_batch_output = self.__column_form_prepare_training_data(distribution_files)
            pd_batch_input.info()
            pd.DataFrame(pd_batch_output).info()
            np.random.seed(seed=SEED)
            reindex = np.random.permutation(pd_batch_input.shape[0])
            batch_input = pd_batch_input.to_numpy()[reindex]
            batch_output = pd_batch_output.to_numpy()[reindex]
        elif pca_data:
            if not self.full_data:
                raise ValueError('PCA needs more than the 2 features')
            pd_batch_input, pd_batch_output = self.__prepare_training_data(distribution_files)
            pca = PCA(n_components=PCA_VAL)
            pca.fit(pd_batch_input)
            pd_batch_input = pca.transform(pd_batch_input)
            pd.DataFrame(pd_batch_input, columns=['PCA_COL_' + str(i) for i in range(PCA_VAL)]).info()
            pd_batch_output.info()
            np.random.seed(seed=SEED)
            reindex = np.random.permutation(pd_batch_input.shape[0])
            batch_input = pd_batch_input[reindex]
            batch_output = pd_batch_output.to_numpy()[reindex]
        else:
            pd_batch_input, pd_batch_output = self.__prepare_training_data(distribution_files)
            pd_batch_input.info()
            pd_batch_output.info()
            np.random.seed(seed=SEED)
            reindex = np.random.permutation(pd_batch_input.shape[0])
            batch_input = pd_batch_input.to_numpy()[reindex]
            batch_output = pd_batch_output.to_numpy()[reindex]

        self.revert(cwd)
        self.__training = {'input': batch_input, 'output': batch_output}

    def __multilabel_prepare_training_data(self, distribution_files):
        for distribution in self.__distribution_names:
            for dist in distribution.split(' '):
                if dist not in self.__class_set:
                    self.__class_set.append(dist)
        batch_columns = []
        if self.full_data:
            for i in range(self.__size + 1):
                batch_columns.append('U' + str(i))
                batch_columns.append('V' + str(i))
        else:
            batch_columns = ['U', 'V']
        batch_input_list = []
        batch_output_list = []
        for file in distribution_files:
            batch_df = pd.read_parquet(file).astype(dtype=self.__dtype).fillna(value=0)
            batch_input_temp = batch_df[batch_columns]
            batch_output_temp = batch_df[self.__class_set]
            batch_input_list.append(batch_input_temp)
            batch_output_list.append(batch_output_temp)
        pd_batch_input = pd.concat(batch_input_list, ignore_index=True)
        pd_batch_output = pd.concat(batch_output_list, ignore_index=True).astype(np.int8)
        return pd_batch_input, pd_batch_output

    def __prepare_training_data(self, distribution_files):
        batch_columns = []
        if self.full_data:
            for i in range(self.__size + 1):
                batch_columns.append('U' + str(i))
                batch_columns.append('V' + str(i))
        else:
            batch_columns = ['U', 'V']
        batch_input_list = []
        batch_output_list = []
        for file in distribution_files:
            batch_df = pd.read_parquet(file).astype(dtype=self.__dtype).fillna(value=0)
            batch_input_temp = batch_df[batch_columns]
            batch_output_temp = batch_df[self.__distribution_names]
            batch_input_list.append(batch_input_temp)
            batch_output_list.append(batch_output_temp)
        pd_batch_input = pd.concat(batch_input_list)
        pd_batch_output = pd.concat(batch_output_list).astype(np.int8)
        return pd_batch_input, pd_batch_output

    def __column_form_prepare_training_data(self, distribution_files):
        batch_columns = []
        if self.full_data:
            for i in range(self.__size + 1):
                batch_columns.append('U' + str(i))
                batch_columns.append('V' + str(i))
        else:
            batch_columns = ['U', 'V']
        batch_input_list = []
        batch_output_list = []
        for file in distribution_files:
            batch_df = pd.read_parquet(file).astype(dtype=self.__dtype).fillna(value=0)
            batch_input_temp = batch_df[batch_columns]
            batch_output_temp = batch_df[self.__distribution_names]
            batch_input_list.append(batch_input_temp)
            batch_output_list.append(batch_output_temp)
        pd_batch_input = pd.concat(batch_input_list)
        pd_batch_output = pd.concat(batch_output_list)
        pd_batch_output['Class'] = pd_batch_output[self.__distribution_names[0]]
        for i in range(1, len(self.__distribution_names)):
            pd_batch_output['Class'] += pd_batch_output[self.__distribution_names[i]] * (i + 1)
        return pd_batch_input, pd_batch_output['Class'].astype(np.int8)

    def __filter_files(self):
        distribution_files = []
        glob_cluster = '* ' + str(self.__size) + '.parquet_' + self.__reference_distribution + '_gz'
        all_files = list(glob.glob(glob_cluster))
        for file in all_files:
            title = file.split('\\')[-1]
            dist = title.split(' ')[:-3]
            if self.full_classes:
                self.__distribution_names.append(' '.join(dist))
                distribution_files.append(file)
            elif all(map(lambda x: x == dist[0], dist)):
                self.__distribution_names.append(' '.join(dist))
                distribution_files.append(file)

        if self.full_classes:
            assert len(self.__distribution_names) == len(all_files), str(len(self.__distribution_names)) + ' != ' + str(len(all_files))
        else:
            assert len(self.__distribution_names) == np.power(len(all_files), 1 / self.__dim), str(len(self.__distribution_names)) + ' != ' + str(np.power(len(all_files), 1 / self.__dim))
        return distribution_files

    def __reference_set(self):
        """
        This finds the angles for the library
        """
        # Load
        reference_df = pd.read_parquet(self.__reference_distribution + ' Reference Set ' + str(self.__size) + '.parquet_ref')
        theta = reference_df.to_numpy(dtype=self.__dtype).reshape((1, self.__size))
        print(self.__reference_distribution + ' Theta Loaded', theta.shape, flush=True)
        return theta

    def revert(self, cwd):
        os.chdir(cwd)
        return self.size

    def change_cwd_data(self):
        cwd = os.getcwd()
        os.chdir(DATA_PATH.format(dim=self.__dim, reference=self.__reference_distribution))
        return cwd

    def change_cwd_results_local(self):
        cwd = os.getcwd()
        if self.full_data:
            data = 'All Data'
        else:
            data = 'Partial Data'
        if self.full_classes:
            classes = 'Full Classes'
        else:
            classes = 'Limited Classes'
        new_dir = 'Ref ' + self.__reference_distribution + '\\' + data + '\\' + classes + '\\'
        os.chdir('..\\..\\Results\\' + new_dir)
        return cwd

    def change_cwd_results(self):
        cwd = os.getcwd()
        os.chdir(RESULT_PATH.format(reference=self.__reference_distribution, data=self.full_data_label(), classes=self.full_classes_label()))
        return cwd

    def change_cwd_model(self):
        cwd = os.getcwd()
        os.chdir(MODEL_PATH.format(reference=self.__reference_distribution, data=self.full_data_label(), classes=self.full_classes_label()))
        return cwd

    def full_data_label(self):
        if self.full_data:
            data = 'All Data'
        else:
            data = 'Partial Data'
        return data

    def full_classes_label(self):
        if self.full_classes:
            classes = 'Full Classes'
        else:
            classes = 'Limited Classes'
        return classes

    def info(self):
        for key in self.__training:
            simplified_bytes, in_units = label(self.__training[key].nbytes)
            print(self.__training[key].shape, simplified_bytes, in_units)

    # Methods for a trained OA Network
    def predict(self, prediction: np.ndarray):
        train_ozturk = np.sort(self.__beta_reduction(prediction), axis=1)
        detrended = standardize(train_ozturk, axis=1)
        u, v = self.__ozturk_function(detrended)
        return u, v

    def __beta_reduction(self, prediction: np.ndarray):
        return self.__z_2(prediction)

    def __ozturk_function(self, t):
        initial_u = np.abs(t) * np.cos(self.__theta)
        initial_v = np.abs(t) * np.sin(self.__theta)
        u = np.zeros(self.__size + 1)
        v = np.zeros(self.__size + 1)
        for i in range(1, self.__size + 1):
            u[:, i] = np.sum(initial_u[:, :i], axis=1) / i
            v[:, i] = np.sum(initial_v[:, :i], axis=1) / i
        return u, v

    def __z_2(self, p):
        """
        The Formula for Z^2
        Z**2 = (p-mu).T  *  sigma**-1  * (p-mu)
        """
        if self.__dim == 1:
            return p.reshape(1, self.__size)
        p_mean = p.mean(axis=0).reshape((1, self.__dim))
        p_cov = np.array(np.cov(p, rowvar=False))
        inv_p_cov = np.linalg.inv(p_cov)
        try:
            assert not np.isnan(inv_p_cov).any()
        except AssertionError:
            inv_p_cov = np.linalg.pinv(p_cov)

        p_t = np.subtract(p, p_mean).conj()
        p_new = np.transpose(np.subtract(p, p_mean), axes=(2, 1))
        z_2 = np.zeros(self.__size)
        for i in range(self.__size):
            p_t_temp = np.reshape(p_t[i, :], (1, self.__dim))
            p_new_temp = np.reshape(p_new[:, i], (self.__dim, 1))
            z_2 = (p_t_temp @ inv_p_cov @ p_new_temp)
        return z_2


assert __name__ != "__main__"
