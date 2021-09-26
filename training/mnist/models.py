from abc import ABC, abstractmethod
from tensorflow import keras
from tensorflow.python.keras.constraints import MinMaxNorm
from tensorflow.python.keras.layers.core import Dense, Flatten
from tensorflow.keras import backend as K
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import Sequential

class Model(ABC):
    @abstractmethod
    def build_model(self, input_shape, nb_classes, nb_components, dense_size, mean1, sigma1, mean2, sigma2, normalize, freeze):
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
        out = (inputs - self.mean) / self.sigma   # standarization
        return out

class MLP(Model):
    def __init__(self, n_filters=32) -> None:
        self.n_filters = n_filters 

    def get_name(self):
        return 'MLP'
    
    def build_model(self, input_shape, nb_classes, nb_components, dense_size, mean1, sigma1, mean2, sigma2, normalize, freeze):
        layers=[]
        #0
        layers.append(Flatten(input_shape=input_shape))
        if normalize: # applied on the original dataset   
            #1
            layers.append(NormalizingLayer(mean1, sigma1))
        #1/2
        layers.append(Dense(nb_components, activation='linear'))
        if normalize and freeze: # applied in the projected space
            layers.append(NormalizingLayer(mean2, sigma2))
        layers.append(Dense(nb_classes ))
        model=Sequential(layers)
        
        return model


class SampleCNN(Model):
    def __init__(self, n_filters=50) -> None:
        self.n_filters = n_filters

    def get_name(self):
        return 'SampleCNN'

    def build_model(self, input_shape, nb_classes, nb_components, dense_size, mean, sigma):
        model = keras.models.Sequential([
            keras.layers.Conv2D(self.n_filters, (28, 28), padding='same', input_shape=input_shape, activation='linear'),
            #keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(self.n_filters * 2, (28, 28), padding='same', activation='relu'),
            #keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dense(nb_classes)
        ])
        return model