from abc import ABC, abstractmethod
from tensorflow import keras
from tensorflow.python.keras.layers.core import Flatten


class Model(ABC):
    @abstractmethod
    def build_model(self, input_shape, nb_classes, nb_components):
        pass

class MLP(Model):
    def __init__(self, n_filters=32) -> None:
        self.n_filters = n_filters

    def get_name(self):
        return 'MLP'

    def build_model(self, input_shape, nb_classes, nb_components):
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=input_shape),
            keras.layers.Dense(nb_components, activation='linear'),
            keras.layers.Dense(512, activation='relu'), # 128 256 512 
            keras.layers.Dense(nb_classes, activation='softmax')
        ])
        return model
