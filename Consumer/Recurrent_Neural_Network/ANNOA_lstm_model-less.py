import argparse

from keras.layers import Dense, Input, LSTM
from keras.models import Sequential

from Constants.Expanded_Constants import NUM_HIDDEN_LAYERS, HIDDEN_NEURONS, NUM_EPOCHS, REFERENCE_LIST
from Constants.Tensor_Constants import EXPANDED_MODEL_METRICS
from Consumer.ANNOA_Model_less import Ozturk
from Consumer.Model import Model


# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

class OA_NeuralNetwork(Model):
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
            self.__model.add(LSTM(self.__shapes['hidden'], return_sequences=i != (self.__num_hidden_layers - 1),
                                  name='ANNOA_Hidden_' + str(i + 1)))
        model.add(Dense(self.__shapes['output'], activation='softmax', name='Output'))
        model.compile(optimizer=self.__optimizer, loss='categorical_crossentropy', metrics=EXPANDED_MODEL_METRICS)
        return model

    def train(self, input_data, output_data) -> dict:
        return self.__model.fit(input_data, output_data, self._batch_size, self.__epochs,
                                validation_split=self._validation_ratio).history

    def summary(self) -> None:
        self.__model.summary()

    def __str__(self):
        i = self.__shapes["input"][0]
        o = self.__shapes["output"]
        hn = [str(self.__hidden_neurons) for _ in range(self.__num_hidden_layers)]
        model = f'{i}-{"-".join(hn)}{"-" if len(hn) > 0 else ""}{o}'
        return 'OA_LSTM_[' + str(model) + ']'


def run_ozturk_annoa(dimension, size, full_data, full_classes):
    for reference_distribution in REFERENCE_LIST[0:1]:
        training_model = Ozturk(size=size,
                                dimension=dimension,
                                reference_distribution=reference_distribution,
                                full_classes=full_classes,
                                full_data=full_data)
        training_model.prepare_training_data()
        model_shape = {'hidden': (HIDDEN_NEURONS,), 'output': len(training_model),
                       'input': ((size + 1) * 2,) if full_data else (2,)}
        for num_hidden_layers in NUM_HIDDEN_LAYERS:
            model = OA_NeuralNetwork(shape=model_shape, hidden_neurons=HIDDEN_NEURONS, hidden_layers=num_hidden_layers,
                                     epochs=NUM_EPOCHS)
            training_model.define_model(model)
            training_model.info()
            training_model.train()
            training_model.store()


def main():
    params = argparse.ArgumentParser(prog='ANNOA_KNN', description='ANNOA for K-Nearest Neighbor')
    params.add_argument('-d', '--dimension', required=True, type=int)
    params.add_argument('-s', '--size', required=True, type=int)
    params.add_argument('full_data', default=False, action='store_true')
    params.add_argument('full_classes', default=True, action='store_false')
    args = params.parse_args()
    run_ozturk_annoa(args.dimension, args.size, args.full_data, args.full_classes)


if __name__ == "__main__":
    '''
    Optimized for training size ANNOA
    '''
    main()
