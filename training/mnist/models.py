from abc import ABC, abstractmethod
from tensorflow import keras
from tensorflow.python.keras.constraints import MinMaxNorm
from tensorflow.python.keras.layers.core import Dense, Flatten
from tensorflow.keras import backend as K
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import Sequential
import tensorflow as tf

class Model(ABC):
    @abstractmethod
    def build_model(self, input_shape, nb_classes, nb_components, freeze, denses, dense_size):
        pass

class MLP(Model):
    def __init__(self, n_filters=32) -> None:
        self.n_filters = n_filters

    def get_name(self):
        return 'MLP'
    def V_regularizer(self,weights):
        return tf.reduce_sum(0.02 * tf.square(weights))
    # remove dense  layer
    def build_model(self, input_shape, nb_classes, nb_components, freeze, denses, dense_size,reconstruct, regularizer):
        layers=[]
        #0
        layers.append(Flatten(input_shape=input_shape))
        
        #1/2
        if regularizer:
            layers.append(Dense(nb_components,activation="linear" , kernel_regularizer=self.V_regularizer))   
        else :
            layers.append(Dense(nb_components,activation="linear" )) 
        if reconstruct==1:
            layers.append(Dense(np.prod(input_shape), activation='linear'))
            layers.append(Dense(nb_components, activation='linear'))
        elif reconstruct==2:
            layers.append(Dense(np.prod(input_shape), activation='linear'))

        #128,256
        #nr_of_layers=1,2
        if denses == 1:
            layers.append(Dense(dense_size, activation='relu' ))
        if denses == 2:
            layers.append(Dense(dense_size , activation='relu' ))
            layers.append(Dense(dense_size , activation='relu' ))
        if denses == 3:
            layers.append(Dense(dense_size , activation='relu' ))
            layers.append(Dense(dense_size , activation='relu' ))
            layers.append(Dense(dense_size , activation='relu' ))
        if denses == 4:
            layers.append(Dense(dense_size , activation='relu' ))
            layers.append(Dense(dense_size , activation='relu' ))
            layers.append(Dense(dense_size , activation='relu' ))
            layers.append(Dense(dense_size , activation='relu' ))
        if denses == 5:
            layers.append(Dense(dense_size , activation='relu' ))
            layers.append(Dense(dense_size , activation='relu' ))
            layers.append(Dense(dense_size , activation='relu' ))
            layers.append(Dense(dense_size , activation='relu' ))
            layers.append(Dense(dense_size , activation='relu' ))
        if denses == 6:
            layers.append(Dense(dense_size , activation='relu' ))
            layers.append(Dense(dense_size , activation='relu' ))
            layers.append(Dense(dense_size , activation='relu' ))
            layers.append(Dense(dense_size , activation='relu' ))
            layers.append(Dense(dense_size , activation='relu' ))
            layers.append(Dense(dense_size , activation='relu' ))

        layers.append(Dense(nb_classes, activation='linear' ))  # no activation for
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