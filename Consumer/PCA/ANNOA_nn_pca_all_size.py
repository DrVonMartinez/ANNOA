import pandas as pd
from keras.layers import Dense, Input
from keras.models import Sequential

from Constants.Constants import PCA_VAL, SIZE_SET
from Constants.Tensor_Constants import NUM_HIDDEN_LAYERS, HIDDEN_NEURONS, NUM_EPOCHS, OPTIMIZER_SET, EXPANDED_METRIC_SET, EXPANDED_MODEL_METRICS, EXPANDED_HISTORY_KEYS
from Generic_Network.Ozturk_Algorithm_Network_parquet_HDD_ALL_SIZE import GeneralizedOzturk


# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Ozturk:
    def __init__(self, sizes=None, dimension: int = 1, hidden_neurons: int = 250, optimizer: str = 'adam', epochs: int = 100, full_classes: bool = False, reference_distribution='Normal'):
        """
        loss function selection following suggestions from https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c
        Optimizer selection following suggestions from https://algorithmia.com/blog/introduction-to-optimizers
        :param sizes:
        :param dimension:
        :param hidden_neurons:
        :param optimizer:
        :param epochs:
        :param full_classes:
        :param reference_distribution
        """
        if sizes is None:
            sizes = []
        # self.__generalizedOzturk_set = [GeneralizedOzturk(size, dimension, full_classes, True, reference_distribution) for size in sizes]
        self.__generalizedOzturk = GeneralizedOzturk(sizes, dimension, full_classes, reference_distribution)

        # Setting up ANNOA
        self.__train_ratio = 0.80
        self.__validation_ratio = 0.20
        self.__batch_size = 64

        self.__model = None
        self.__validate_size = None
        self.__epochs = epochs
        self.__optimizer = optimizer
        self.__num_hidden_layers = -1
        self.__hidden_neurons = hidden_neurons
        self.__shapes = {}
        self.__history = {}
        self.__title = "ANNOA PCA ALL SIZE"
        print('Ozturk Algorithm Neural Network PCA ALL SIZE',
              'Epochs: ' + str(self.__epochs),
              'Neurons: ' + str(self.__hidden_neurons),
              'Size: ' + str(self.sizes()),
              'Dimension: ' + str(self.dim()),
              'Model Optimizer: ' + str(self.__optimizer),
              'Reference Distribution: ' + str(self.reference_distribution()),
              'Full Class Set: ' + str(self.full_classes()),
              'Full Data Set: ' + str(self.full_data),
              sep='\n', flush=True)

    def define_network(self, num_hidden_layers):
        self.__num_hidden_layers = num_hidden_layers
        self.__model = Sequential(layers=[Input(shape=self.__shapes['input'], name='UV Input')])
        for i in range(self.__num_hidden_layers):
            self.__model.add(Dense(self.__hidden_neurons, activation='relu', name='ANNOA_Hidden_' + str(i + 1)))
        self.__model.add(Dense(self.__shapes['output'], activation='softmax', name='Output'))
        self.__model.compile(optimizer=self.__optimizer, loss='categorical_crossentropy', metrics=EXPANDED_MODEL_METRICS)
        # print(self.__generalizedOzturk.distribution_names())
        self.__model.summary()
        self.__title = self.__set_title()

    def __define_shape(self):
        self.__shapes = {'input': (PCA_VAL,), 'hidden': (self.__hidden_neurons,), 'output': (len(self.__generalizedOzturk.distribution_names()))}

    def sizes(self) -> list:
        return self.__generalizedOzturk.sizes()

    def dim(self) -> int:
        return self.__generalizedOzturk.dim()

    def full_classes(self) -> bool:
        return self.__generalizedOzturk.full_classes

    @property
    def full_data(self) -> bool:
        return True

    def reference_distribution(self) -> str:
        return self.__generalizedOzturk.reference_distribution()

    def distribution_names(self) -> list:
        return self.__generalizedOzturk.distribution_names()

    def __str__(self):
        return 'gen_Ozturk_[' + str(self.sizes()) + ']_[' + str(self.full_classes()) + ']' + '_' + str(self.__num_hidden_layers)

    def train(self, validate_against=None):
        if validate_against:
            input_data, output_data, validate_data = self.__generalizedOzturk.get_training_data_validate(validate_against)
            self.__history = self.__model.fit(input_data, output_data, self.__batch_size, self.__epochs, validation_data=validate_data)
        else:
            input_data, output_data = self.__generalizedOzturk.get_training_data()
            self.__history = self.__model.fit(input_data, output_data, self.__batch_size, self.__epochs, validation_split=self.__validation_ratio)
        '''
        results = self.__generalizedOzturk.get_training_data()
        
        input_data_set, output_data_set = zip(*results)
        np.random.seed(seed=SEED)
        input_data: ndarray = np.vstack(input_data_set)
        x = np.arange(input_data.shape[0])
        reindex = np.random.permutation(input_data.shape[0])
        print(reindex)
        output_data: ndarray = np.vstack(output_data_set)[reindex]
        input_data = input_data[reindex]
        print(input_data.shape)
        print(output_data.shape)
        '''

    def prepare_training_data(self):
        self.__generalizedOzturk.prepare_training_data(False, False)
        self.__define_shape()

    def store(self):
        history_keys = EXPANDED_HISTORY_KEYS
        cwd = self.__generalizedOzturk.change_cwd_results()
        history_df = pd.DataFrame(columns=EXPANDED_METRIC_SET)
        for key, col in zip(history_keys, EXPANDED_METRIC_SET):
            history_df[col] = self.__history.history[key]
        history_df.to_csv(self.__title + '.csv', index=False)
        self.__generalizedOzturk.revert(cwd)

    def __set_title(self):
        return '{}, Dim {}, Hidden Layer {}, PCA, Validate_Size {}, Size {}, {}'.format(self.reference_distribution(), self.dim(), self.__num_hidden_layers, self.__validate_size, self.sizes(),
                                                                                        self.__optimizer)


def run_ozturk_annoa(dimension, sizes, full_classes=True):
    reference_set = ['Normal', 'Uniform']
    for reference_distribution in reference_set:
        training_model = Ozturk(sizes=sizes,
                                dimension=dimension,
                                optimizer=OPTIMIZER_SET[1],
                                hidden_neurons=HIDDEN_NEURONS,
                                epochs=NUM_EPOCHS,
                                reference_distribution=reference_distribution,
                                full_classes=full_classes)
        training_model.prepare_training_data()
        # for validate_size in [None] + sizes:
        for validate_size in [None]:
            for num_hidden_layers in NUM_HIDDEN_LAYERS:
                training_model.define_network(num_hidden_layers)
                training_model.train(validate_size)
                training_model.store()


def main():
    assert (__name__ == "__main__"), "Method not intended to be called if this isn't the main file"
    for dimension in [1, 2]:
        if dimension == 1:
            run_ozturk_annoa(dimension=dimension, sizes=SIZE_SET[:-2])
            # run_ozturk_annoa(dimension=dimension, sizes=SIZE_SET[0:2])
        else:
            for full_classes in [True, False]:
                run_ozturk_annoa(dimension=dimension, sizes=SIZE_SET[:-2], full_classes=full_classes)


if __name__ == "__main__":
    '''
    Optimized for training size ANNOA
    '''
    assert isinstance(Ozturk, type(GeneralizedOzturk))

    main()
