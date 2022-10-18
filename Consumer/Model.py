import abc


class Model(abc.ABC):
    def __init__(self):
        self._train_ratio = 0.80
        self._validation_ratio = 0.20
        self._batch_size = 64

    @abc.abstractmethod
    def train(self, input_data, output_data) -> dict:
        ...

    @abc.abstractmethod
    def summary(self) -> None:
        ...
