from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

class DataSet(ABC):
    @abstractmethod
    def get_train(self):
        pass

    @abstractmethod
    def get_test(self):
        pass

    @abstractmethod
    def get_val(self):
        pass


class MNIST(DataSet):

    def __init__(self, original="true", reconstruct='true', comps=10, val_size=1000, seed=9) -> None:
        self.rnd = np.random.RandomState(seed)
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # SVD DECOMPOSITION 
        if(original!="true"):
            x_train=x_train.reshape((60000,-1))
            num_components = comps
            U, s, V = np.linalg.svd(x_train[:10000]) 
            if(reconstruct=="true"):
                x_proj=np.dot(x_train,V[:num_components,:].T)
                x_train=np.dot(x_proj,V[:num_components,:])  
                x_test=x_test.reshape((10000,-1))
                x_proj=np.dot(x_test,V[:num_components,:].T)
                x_test=np.dot(x_proj,V[:num_components,:])  

        x_train, x_test = np.array(x_train / 255.0, np.float32), np.array(x_test / 255.0, np.float32)

        self.x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        self.y_train = np.array(y_train, np.int64)
        self.y_test = np.array(y_test, np.int64)
        self.x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
        self.x_train, self.y_train, self.x_val, self.y_val = self.split_data(self.rnd, val_size // 10, self.x_train, self.y_train)

        if(original!="true"):
            if(reconstruct=="true"):
                self.x_val=self.x_val.reshape((1000,-1))
                x_proj=np.dot(self.x_val,V[:num_components,:].T)
                self.x_val=np.dot(x_proj,V[:num_components,:])

        self.V = []
        self.components = comps
        if(original!="true"):
            self.V = V
            self.components = num_components


    def split_data(self, rnd, sample_per_class, x, y):
        x_equalized = ()
        x_remained = ()
        y_equalized = ()
        y_remained = ()
        for i in np.unique(y):
            idxs = rnd.permutation(np.sum(y == i))
            x_i = x[y == i]
            y_i = y[y == i]
            x_equalized = x_equalized + (x_i[idxs[:sample_per_class]],)
            y_equalized = y_equalized + (y_i[idxs[:sample_per_class]],)
            x_remained = x_remained + (x_i[idxs[sample_per_class:]],)
            y_remained = y_remained + (y_i[idxs[sample_per_class:]],)
        return np.concatenate(x_remained, axis=0), np.concatenate(y_remained, axis=0), \
               np.concatenate(x_equalized, axis=0), np.concatenate(y_equalized, axis=0)

    def get_bound(self):
        return (0., 1.)

    def get_input_shape(self):
        return self.x_train.shape[1:]

    def get_nb_classes(self):
        return np.unique(self.y_train).shape[0]
    
    def get_nb_components(self):
        return self.components

    def get_v(self):
        return self.V

    def get_train(self):
        return self.x_train, self.y_train

    def get_test(self):
        return self.x_test, self.y_test

    def get_val(self):
        return self.x_val, self.y_val

    def get_name(self):
        return 'MNIST'
