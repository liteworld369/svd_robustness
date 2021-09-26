from abc import ABC, abstractmethod
import numpy as np
from numpy.core.numerictypes import ScalarType
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


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
        self.m1 = 0
        self.sigma1 =1
        self.m2 = 0
        self.sigma2 =1
        self.V = []
        self.components = comps
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
        # SVD DECOMPOSITION 
        if(original!="true"):
            #print('x:', x_train[1])
            #x_train = self.TrimOutliers(x_train )
            #x_test = self.TrimOutliers(x_test)

            #x_train = self.SVD(x_train , comps)
            #x_test = self.SVD(x_test, comps)
            
            #print('v:', x_train[1])
            
            x_train=x_train.reshape((60000,-1))
            x_test=x_test.reshape((10000,-1))
            
             # standarization
            #scaler = StandardScaler(with_mean=True, with_std=True)
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            #scaler.fit(x_test)
            x_test = scaler.transform(x_test)
            self.m1,self.sigma1=scaler.mean_,scaler.var_   ## for the first normal layer
        
            U, s, V = np.linalg.svd(x_train[:10000])   
            self.V = V #.reshape((V.shape[0], 784))
             
            ##  to obtain new mean and sigma for the second normal layer
            x_proj=np.dot(x_train,self.V[:self.components,:].T)
            self.m2=  np.mean(x_proj,axis=0)
            self.sigma2= np.std(x_proj,axis=0)
            ##         
                
            if(reconstruct=="true"):
                x_proj=np.dot(x_train,self.V[:self.components,:].T)
                self.m1=  np.mean(x_proj,axis=0)
                self.sigma1= np.std(x_proj,axis=0)
                print('shapes sigma m', self.m1.shape, self.sigma1.shape)
                x_train=np.dot(x_proj,self.V[:self.components,:])   
                x_proj=np.dot(x_test,self.V[:self.components,:].T) 
                x_test=   np.dot(x_proj,self.V[:self.components,:])

        x_train, x_test = np.array(x_train / 255.0, np.float32), np.array(x_test / 255.0, np.float32)
        
        
        self.x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        self.y_train = np.array(y_train, np.int64)
        self.y_test = np.array(y_test, np.int64)
        self.x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
        self.x_train, self.y_train, self.x_val, self.y_val = self.split_data(self.rnd, val_size // 10, self.x_train, self.y_train)

        if(original!="true"):
            if(reconstruct=="true"):
                self.x_val=self.x_val.reshape((1000,-1))
                x_proj=np.dot(self.x_val,V[:self.components,:].T)
                self.x_val=np.dot(x_proj,V[:self.components,:])


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
    
    def get_mean1(self):
        return self.m1
    def get_sigma1(self):
        return self.sigma1
    def get_mean2(self):
        return self.m2
    def get_sigma2(self):
        return self.sigma2
    
    def get_train(self):
        return self.x_train, self.y_train

    def get_test(self):
        return self.x_test, self.y_test

    def get_val(self):
        return self.x_val, self.y_val

    def get_name(self):
        return 'MNIST'

    def SVD(self, D, comps):
        print('...generating SVDs')
        for i in range(0,len(D),1):
            num_components = comps  # add more hps for comps
            U, s, V = np.linalg.svd(D[i])
            #min_matrix2 = np.dot(U[:,:10],np.dot(np.diag(s[:10]),V[:10,:]))
            # standarization
            x_proj = np.dot(D[i],V[:3,:].T)
            min_matrix =  np.dot(x_proj,V[:3,:])
            #D[i]= np.dot(x_proj,V[:num_components,:])
            x_proj = np.dot(D[i],V[:num_components,:].T)
            #img1
            #img2
            #(img1+img2)/2-->[0,512]
            # [0,2]
            D[i]  = np.dot(x_proj,V[:num_components,:]) + self.TrimOutliers(min_matrix) 
        return D

    def TrimOutliers(self, D):
        D[D>190] = 0
        D[D<80] = 0
        return D



## CNN 
""" class MNISTPatches():
    def init():
        x_train=..
        x_train_patches=[]
        for x_i in x_train:
            #scikit learn
            #pick random patches
            x_i_patches=extract_patrches(x_i,(7,7))
            x_train_patches.append(x_train_patches)
        #x_train_patches.shape=2400000x7x7x1
        s,d,v=np.linalg.svd(x_train_patches)
        #V.shape=n_componentsx49
        #28x28
        #CNN((7x7))=V
        #model=sequentila()
        #model.add(CONv2d(n_components,(7,7))) """

