from abc import ABC, abstractmethod
from tensorflow import keras
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.keras import backend as K
import numpy as np

class Model(ABC):
    @abstractmethod
    def build_model(self, input_shape, nb_classes, nb_components, dense_size, mean, sigma):
        pass
class NormalizingLayer(keras.layers.Layer):

    def __init__(self,mean,sigma, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs) 
        self.mean = K.constant(mean, dtype=K.floatx())
        self.sigma = K.constant(sigma, dtype=K.floatx())
        
    def get_config(self):
        base_conf = super().get_config()
        return {**base_conf,
                'mean': np.asfarray(self.mean),
                'sigma': np.asfarray(self.sigma)
                }

    def call(self, inputs, **kwargs):
        print('input:',inputs.shape)
        out = (inputs - self.mean) / self.sigma
        return out

class MLP(Model):
    def __init__(self, n_filters=32) -> None:
        self.n_filters = n_filters

    def get_name(self):
        return 'MLP'

    def build_model(self, input_shape, nb_classes, nb_components, dense_size, mean, sigma):
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=input_shape),
            keras.layers.Dense(nb_components, activation='linear'),
            keras.layers.Dense(60000, activation='relu'),
            NormalizingLayer(mean, sigma),
            keras.layers.Dense(nb_classes ) #, activation='softmax') #
        ])
        return model


class SampleCNN(Model):
    def __init__(self, n_filters=32) -> None:
        self.n_filters = n_filters

    def get_name(self):
        return 'SampleCNN'

    def build_model(self, input_shape, nb_classes):
        model = keras.models.Sequential([
            keras.layers.Conv2D(self.n_filters, (5, 5), padding='same', input_shape=input_shape, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(self.n_filters * 2, (5, 5), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dense(nb_classes)
        ])
        return model