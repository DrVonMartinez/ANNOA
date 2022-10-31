from keras.layers import Dense, Input
from keras.layers import Dense, Input
from keras.models import Sequential
from sklearn.decomposition import PCA

from Constants.Constants import PCA_VAL, SIZE_SET
from Constants.Expanded_Constants import NUM_HIDDEN_LAYERS, HIDDEN_NEURONS, NUM_EPOCHS, REFERENCE_LIST
from Constants.Tensor_Constants import EXPANDED_MODEL_METRICS
from Consumer.ANNOA_Model_less import Ozturk
from Consumer.Model import Model
from Generic_Network.Ozturk_Algorithm_Network_parquet_HDD import GeneralizedOzturk


# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

class OA_PCA_NN(Model):
    def __init__(self, shape=None, hidden_neurons: int = 250, hidden_layers: int = 0, optimizer: str = 'adam',
                 epochs: int = 100):
        super().__init__()
        if shape is None:
            self.__shapes = {}
        else:
            self.__shapes = shape
        self.__hidden_neurons = hidden_neurons
        self.__num_hidden_layers = hidden_layers
        self.__optimizer = optimizer
        self.__epochs = epochs
        self.__model = self.__define_model()

    def __define_model(self):
        model = Sequential(layers=[Input(shape=self.__shapes['input'], name='UV Input')])
        for i in range(self.__num_hidden_layers):
            model.add(Dense(self.__hidden_neurons, activation='relu', name='ANNOA_Hidden_' + str(i + 1)))
        model.add(Dense(self.__shapes['output'], activation='softmax', name='Output'))
        model.compile(optimizer=self.__optimizer, loss='categorical_crossentropy', metrics=EXPANDED_MODEL_METRICS)
        return model

    def train(self, input_data, output_data) -> dict:
        pca = PCA(n_components=PCA_VAL)
        pca_batch_input = pca.fit_transform(input_data)
        return self.__model.fit(pca_batch_input, output_data, self._batch_size, self.__epochs,
                                validation_split=self._validation_ratio).history

    def summary(self) -> None:
        self.__model.summary()

    def __str__(self):
        i = self.__shapes["input"][0]
        o = self.__shapes["output"]
        hn = [str(self.__hidden_neurons) for _ in range(self.__num_hidden_layers)]
        print(hn)
        model = f'{i}-{"-".join(hn)}{"-" if len(hn) > 0 else ""}{o}'
        return 'OA_PCA_NN_[' + str(model) + ']'


def run_ozturk_annoa(dimension, size, full_data, full_classes):
    for reference_distribution in REFERENCE_LIST[0:1]:
        training_model = Ozturk(size=size,
                                dimension=dimension,
                                reference_distribution=reference_distribution,
                                full_classes=full_classes,
                                full_data=full_data)
        training_model.prepare_training_data()
        model_shape = {'input': (PCA_VAL,), 'hidden': (HIDDEN_NEURONS,), 'output': len(training_model)}
        for num_hidden_layers in NUM_HIDDEN_LAYERS:
            model = OA_PCA_NN(shape=model_shape, hidden_neurons=HIDDEN_NEURONS, hidden_layers=num_hidden_layers,
                              epochs=NUM_EPOCHS)
            training_model.define_model(model)
            training_model.info()
            training_model.train()
            training_model.store()


def main():
    assert (__name__ == "__main__"), "Method not intended to be called if this isn't the main file"
    for dimension in [2]:
        # for size in [150, 200]:
        for size in SIZE_SET:
            if dimension == 1:
                run_ozturk_annoa(dimension=dimension, size=size, full_classes=True, full_data=True)
            else:
                for full_classes in [True, False]:
                    run_ozturk_annoa(dimension=dimension, size=size, full_classes=full_classes, full_data=True)


if __name__ == "__main__":
    '''
    Optimized for training size ANNOA
    '''
    assert isinstance(Ozturk, type(GeneralizedOzturk))

    main()
