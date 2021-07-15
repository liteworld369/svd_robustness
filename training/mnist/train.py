from argparse import ArgumentParser
import os
import tensorflow as tf
import numpy as np
from dataset import MNIST
from models import MLP
import util


def main(params):
    original = params.original
    reconstruct = params.reconstruct
    comps = params.comps
    dense_size = params.dense_size
    ds = MNIST(original, reconstruct, comps)
    x_train, y_train = ds.get_train()
    x_val, y_val = ds.get_val()
    print(x_train.shape, np.bincount(y_train), x_val.shape, np.bincount(y_val))
    model_holder = MLP()
    model = model_holder.build_model(ds.get_input_shape(), ds.get_nb_classes(), ds.get_nb_components(), dense_size)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['sparse_categorical_accuracy']
    
    #Update Model Weights from SVD
    if(original!="true"):
        ws=model.get_weights() 
        V = ds.get_v()
        ws[0]=V[:ds.get_nb_components(),:].T
        model.set_weights(ws)
        model.layers[1].trainable=False
    ##
    
    model.compile(tf.keras.optimizers.Adam(1e-4), loss_fn, metrics)
    m_path = os.path.join(params.save_dir, model_holder.get_name())
    util.mk_parent_dir(m_path)
    if(original!="true"):
        if(reconstruct=='true'):
            label = '_svd_reconstruct_comps_' + str(ds.get_nb_components()) + '_dense_' + str(dense_size)
        else:
            label = '_svd_comps_' + str(ds.get_nb_components()) + '_dense_' + str(dense_size) 
    else:
        label = '_origin_dense_n_' + str(ds.get_nb_components()) + '_dense_' + str(dense_size) 
    callbacks = [tf.keras.callbacks.ModelCheckpoint(m_path + label + '_{epoch:03d}.h5'),
                 tf.keras.callbacks.CSVLogger(os.path.join(params.save_dir, label + '.csv'))]
    model.fit(x_train, y_train, epochs=params.epoch, validation_data=(x_val, y_val),
              batch_size=params.batch_size,
              callbacks=callbacks)


if __name__ == '__main__':
    parser = ArgumentParser(description='Main entry point')
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--original", type=str, default='true')
    parser.add_argument("--reconstruct", type=str, default='true')
    parser.add_argument("--comps", type=int, default=10)
    parser.add_argument("--dense_size", type=int, default=128)
    parser.add_argument("--memory_limit", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default=os.path.join('saved_models'))
    FLAGS = parser.parse_args()
    np.random.seed(9)
    if FLAGS.gpu is not None:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        selected = gpus[FLAGS.gpu]
        tf.config.experimental.set_visible_devices(selected, 'GPU')
        tf.config.experimental.set_memory_growth(selected, True)
        tf.config.experimental.set_virtual_device_configuration(
            selected,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=FLAGS.memory_limit)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        l_gpu = logical_gpus[0]
        with tf.device(l_gpu.name):
            main(FLAGS)
    else:
        main(FLAGS)
