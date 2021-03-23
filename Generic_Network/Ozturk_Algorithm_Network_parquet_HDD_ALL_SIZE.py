import glob
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale as standardize

from Utilities.Constants import SEED, label, PCA_VAL


class GeneralizedOzturk:
    def __init__(self, sizes=None, dimension: int = 1, full_classes: bool = False, reference_distribution='Normal'):
        """
        loss function selection following suggestions from https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c
        Optimizer selection following suggestions from https://algorithmia.com/blog/introduction-to-optimizers
        :param sizes:
        :param dimension:
        :param full_classes:
        :param reference_distribution
        """
        if sizes is None:
            sizes = []
        self.__sizes: list = sizes
        self.__dim: int = int(dimension)
        self.__dtype = np.float32
        self.__training = {}
        self.__in_size_set = {}
        self.__out_size_set = {}

        # Load Reference Distribution
        self.__reference_distribution: str = reference_distribution.capitalize()
        self.__theta_set = []

        # Load Training Data
        self.__class_set = []
        self.__distribution_names = []
        if dimension == 1:
            self.full_classes = True
        else:
            self.full_classes = full_classes
        self.title = 'Generalized Ozturk Network'

    def sizes(self) -> list:
        return self.__sizes

    def dim(self) -> int:
        return self.__dim

    def distribution_names(self):
        return self.__distribution_names

    def class_set(self):
        return self.__class_set

    def reference_distribution(self):
        return self.__reference_distribution

    def get_training_data(self) -> [np.ndarray, np.ndarray]:
        return self.__training['input'], self.__training['output']

    def get_training_data_validate(self, size) -> [np.ndarray, np.ndarray, tuple]:
        in_validation_size = self.__in_size_set[str(size)].to_numpy()
        out_validation_size = self.__out_size_set[str(size)].to_numpy()
        in_size_set = []
        out_size_set = []
        for s in self.__sizes:
            if s == size:
                continue
            in_size_set.append(self.__in_size_set[str(s)])
            out_size_set.append(self.__out_size_set[str(s)])
        pd_batch_input = pd.concat(in_size_set, ignore_index=True)
        pd_batch_output = pd.concat(out_size_set, ignore_index=True)
        pd_batch_input.info()
        pd_batch_output.info()
        np.random.seed(seed=SEED)
        reindex = np.random.permutation(pd_batch_input.shape[0])
        np.random.seed(seed=SEED)
        reindex2 = np.random.permutation(in_validation_size.shape[0])
        batch_input = pd_batch_input.to_numpy()[reindex]
        batch_output = pd_batch_output.to_numpy()[reindex]
        return batch_input, batch_output, (in_validation_size[reindex2], out_validation_size[reindex2])

    def __str__(self):
        return 'gen_Ozturk_' + str(self.__sizes) + '_[' + str(self.full_classes) + ']_All_Data'

    def prepare_training_data(self, multilabel=False, result_column=False):
        cwd = self.change_cwd_data()
        in_size_set = []
        out_size_set = []
        for size in self.__sizes:
            self.__theta_set.append(self.__reference_set(size))
            distribution_files = self.__filter_files(size)
            if multilabel:
                pd_batch_input, pd_batch_output = self.__multilabel_prepare_training_data(distribution_files, size)
            elif result_column:
                pd_batch_input, pd_batch_output = self.__column_form_prepare_training_data(distribution_files, size)
            else:
                pd_batch_input, pd_batch_output = self.__prepare_training_data(distribution_files, size)
            pca = PCA(n_components=PCA_VAL)
            pca.fit(pd_batch_input)
            pd_batch_input = pca.transform(pd_batch_input)
            in_size_set.append(pd.DataFrame(pd_batch_input, columns=['PCA_COL_' + str(i) for i in range(PCA_VAL)]))
            out_size_set.append(pd_batch_output)
        print(self.__distribution_names, flush=True)
        pd_batch_input = pd.concat(in_size_set, ignore_index=True)
        pd_batch_output = pd.concat(out_size_set, ignore_index=True)
        pd_batch_input.info()
        pd_batch_output.info()
        np.random.seed(seed=SEED)
        reindex = np.random.permutation(pd_batch_input.shape[0])
        batch_input = pd_batch_input.to_numpy()[reindex]
        batch_output = pd_batch_output.to_numpy()[reindex]
        self.revert(cwd)
        self.__training = {'input': batch_input, 'output': batch_output}
        for i in range(len(self.__sizes)):
            size = self.__sizes[i]
            self.__in_size_set[str(size)] = in_size_set[i]
            self.__out_size_set[str(size)] = out_size_set[i]

    def __multilabel_prepare_training_data(self, distribution_files, size):
        for distribution in self.__distribution_names:
            for dist in distribution.split(' '):
                if dist not in self.__class_set:
                    self.__class_set.append(dist)
        batch_columns = []
        for i in range(size + 1):
            batch_columns.append('U' + str(i))
            batch_columns.append('V' + str(i))
        batch_input_list = []
        batch_output_list = []
        for file in distribution_files:
            batch_df = pd.read_parquet(file).astype(dtype=self.__dtype).fillna(value=0)
            batch_input_temp = batch_df[batch_columns]
            batch_output_temp = batch_df[self.__class_set]
            batch_input_list.append(batch_input_temp)
            batch_output_list.append(batch_output_temp)
        pd_batch_input = pd.concat(batch_input_list)
        pd_batch_output = pd.concat(batch_output_list).astype(np.int8)
        return pd_batch_input, pd_batch_output

    def __prepare_training_data(self, distribution_files, size):
        batch_columns = []
        for i in range(size + 1):
            batch_columns.append('U' + str(i))
            batch_columns.append('V' + str(i))
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

    def __column_form_prepare_training_data(self, distribution_files, size):
        batch_columns = []
        for i in range(size + 1):
            batch_columns.append('U' + str(i))
            batch_columns.append('V' + str(i))
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

    def __filter_files(self, size):
        distribution_files = []
        glob_cluster = '* ' + str(size) + '.parquet_' + self.__reference_distribution + '_gz'
        all_files = list(glob.glob(glob_cluster))
        first_size = len(self.__distribution_names) == 0
        for file in all_files:
            title = file.split('\\')[-1]
            dist = title.split(' ')[:-3]
            if self.full_classes:
                if first_size:
                    self.__distribution_names.append(' '.join(dist))
                distribution_files.append(file)
            elif all(map(lambda x: x == dist[0], dist)):
                if first_size:
                    self.__distribution_names.append(' '.join(dist))
                distribution_files.append(file)
        return distribution_files

    def __reference_set(self, size):
        """
        This finds the angles for the library
        """
        # Load
        reference_df = pd.read_parquet(self.__reference_distribution + ' Reference Set ' + str(size) + '.parquet_ref')
        theta = reference_df.to_numpy(dtype=self.__dtype).reshape((1, size))
        print(self.__reference_distribution + ' Theta Loaded', theta.shape, flush=True)
        return theta

    def revert(self, cwd):
        os.chdir(cwd)
        return self.sizes()

    def change_cwd_data(self):
        cwd = os.getcwd()
        os.chdir('F:\\Data\\')
        cwd_dir = os.getcwd()
        new_dir = '\\dim ' + str(self.__dim) + '\\Ref ' + self.__reference_distribution + '\\'
        os.chdir(cwd_dir + new_dir)
        return cwd

    def change_cwd_results(self):
        cwd = os.getcwd()
        if self.full_classes:
            classes = 'Full Classes'
        else:
            classes = 'Limited Classes'
        new_dir = 'Ref ' + self.__reference_distribution + '\\All Data\\' + classes + '\\'
        os.chdir('..\\..\\Results\\' + new_dir)
        return cwd

    def change_cwd_model(self):
        cwd = os.getcwd()
        if self.full_classes:
            classes = 'Full Classes'
        else:
            classes = 'Limited Classes'
        new_dir = 'Ref ' + self.__reference_distribution + '\\All Data\\' + classes + '\\'
        os.chdir('F:\\Model\\' + new_dir)
        return cwd

    def full_classes_label(self):
        if self.full_classes:
            classes = 'Complete Classes'
        else:
            classes = 'Limited Classes'
        return classes

    def info(self):
        for key in self.__training:
            simplified_bytes, in_units = label(self.__training[key].nbytes)
            print(self.__training[key].shape, simplified_bytes, in_units)

    '''
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
        u = np.zeros(self.__sizes + 1)
        v = np.zeros(self.__sizes + 1)
        for i in range(1, self.__sizes + 1):
            u[:, i] = np.sum(initial_u[:, :i], axis=1) / i
            v[:, i] = np.sum(initial_v[:, :i], axis=1) / i
        return u, v

    def __z_2(self, p):
        """
        The Formula for Z^2
        Z**2 = (p-mu).T  *  sigma**-1  * (p-mu)
        """
        if self.__dim == 1:
            return p.reshape(1, self.__sizes)
        p_mean = p.mean(axis=0).reshape((1, self.__dim))
        p_cov = np.array(np.cov(p, rowvar=False))
        inv_p_cov = np.linalg.inv(p_cov)
        try:
            assert not np.isnan(inv_p_cov).any()
        except AssertionError:
            inv_p_cov = np.linalg.pinv(p_cov)

        p_t = np.subtract(p, p_mean).conj()
        p_new = np.transpose(np.subtract(p, p_mean), axes=(2, 1))
        z_2 = np.zeros(self.__sizes)
        for i in range(self.__sizes):
            p_t_temp = np.reshape(p_t[i, :], (1, self.__dim))
            p_new_temp = np.reshape(p_new[:, i], (self.__dim, 1))
            z_2 = (p_t_temp @ inv_p_cov @ p_new_temp)
        return z_2
'''


assert __name__ != "__main__"
