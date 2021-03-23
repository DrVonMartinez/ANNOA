import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Input
from keras.metrics import mae
from keras.models import Sequential

from Generic_Network.Ozturk_Algorithm_Network_parquet_HDD import GeneralizedOzturk
from Utilities.Constants import SIZE_SET, NUM_HIDDEN_LAYERS, HIDDEN_NEURONS, NUM_EPOCHS, OPTIMIZER_SET, label, METRIC_SET

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


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
        self.__generalizedOzturk = GeneralizedOzturk(size, dimension, full_classes, full_data, reference_distribution)

        # Setting up ANNOA
        self.__train_ratio = 0.80
        self.__validation_ratio = 0.20
        self.__batch_size = 64

        self.__model = None
        self.__epochs = epochs
        self.__optimizer = optimizer
        self.__num_hidden_layers = -1
        self.__hidden_neurons = hidden_neurons
        self.__shapes = {}
        self.__history = {}
        self.__title = self.__generalizedOzturk.title
        print('Ozturk Algorithm Multilabel Neural Network',
              'Epochs: ' + str(self.__epochs),
              'Neurons: ' + str(self.__hidden_neurons),
              'Size: ' + str(self.__generalizedOzturk.size()),
              'Dimension: ' + str(self.__generalizedOzturk.dim()),
              'Model Optimizer: ' + str(self.__optimizer),
              'Reference Distribution: ' + str(self.__generalizedOzturk.reference_distribution()),
              'Full Class Set: ' + str(bool(self.__generalizedOzturk.full_classes)),
              'Full Data Set: ' + str(bool(self.__generalizedOzturk.full_data)),
              sep='\n', flush=True)

    def define_network(self, num_hidden_layers):
        self.__num_hidden_layers = num_hidden_layers
        self.__model = Sequential(layers=[Input(shape=self.__shapes['input'], name='UV Input')])
        for i in range(self.__num_hidden_layers):
            self.__model.add(Dense(self.__hidden_neurons, activation='relu', name='ANNOA_Hidden_' + str(i + 1)))
        self.__model.add(Dense(self.__shapes['output'], activation='sigmoid', name='Output'))
        self.__model.compile(optimizer=self.__optimizer, loss='binary_crossentropy', metrics=[mae, 'accuracy'])
        print(self.__generalizedOzturk.class_set())
        self.__model.summary()
        self.__title = self.__set_title()

    def __define_shape(self):
        if self.__generalizedOzturk.full_data:
            self.__shapes = {'input': ((self.__generalizedOzturk.size() + 1) * 2,), 'hidden': (self.__hidden_neurons,)}
        else:
            self.__shapes = {'input': (2,), 'hidden': (self.__hidden_neurons,)}
        self.__shapes['output'] = len(self.__generalizedOzturk.class_set())

    def size(self):
        return self.__generalizedOzturk.size()

    def dim(self):
        return self.__generalizedOzturk.dim()

    def __str__(self):
        data = self.__generalizedOzturk.full_data_label().replace(' ', '_')
        return 'Dim_Hot_Encoded_Ozturk_[' + str(self.__generalizedOzturk.size()) + ']_[' + str(self.__generalizedOzturk.full_classes) + ']' + '_' + data + '_' + str(self.__num_hidden_layers)

    def train(self):
        input_data, output_data = self.__generalizedOzturk.get_training_data()
        self.__history = self.__model.fit(input_data,
                                          output_data,
                                          self.__batch_size,
                                          self.__epochs,
                                          validation_split=self.__validation_ratio)

    def prepare_training_data(self):
        self.__generalizedOzturk.prepare_training_data(multilabel=True)
        self.__define_shape()

    def store(self):
        history_keys = ['loss', 'mean_absolute_error', 'accuracy']
        cwd = self.__generalizedOzturk.change_cwd_results()
        history_df = pd.DataFrame(columns=METRIC_SET)
        for key, col in zip(history_keys, METRIC_SET):
            history_df[col] = self.__history.history[key]
        history_df.to_csv(self.__set_title() + '.csv', index=False)
        self.__generalizedOzturk.revert(cwd)

    def save_model(self):
        cwd = self.__generalizedOzturk.change_cwd_model()
        self.__model.save(self.__set_title(), save_format='h5')
        self.__generalizedOzturk.revert(cwd)

    def __set_title(self):
        return '{}, Dim {}, Hidden Layer {}, Size {}, Multilabel, {}'.format(self.__generalizedOzturk.reference_distribution(), self.__generalizedOzturk.dim(), self.__num_hidden_layers,
                                                                             self.__generalizedOzturk.size(), self.__optimizer)

    def info(self):
        self.__generalizedOzturk.info()


def run_ozturk_annoa(dimension, size, full_data, full_classes):
    reference_set = ['Normal']
    # reference_set = ['Normal', 'Uniform']
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
    for dimension in [2]:
        for size in SIZE_SET:
            for full_data in [True, False]:
                if dimension == 1:
                    run_ozturk_annoa(dimension, size, full_data, True)
                else:
                    for full_classes in [True, False]:
                        run_ozturk_annoa(dimension, size, full_data, full_classes)


if __name__ == "__main__":
    '''
    Optimized for training size ANNOA
    '''
    main()
