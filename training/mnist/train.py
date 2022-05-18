from argparse import ArgumentParser
import os
from re import M
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.gen_array_ops import where
from dataset import MNIST, FMNIST
from models import MLP, SampleCNN
import util


def main(params):
    comps = params.comps
    v2 = params.v2
    dataset = params.dataset
    if dataset == 'MNIST':
        ds = MNIST( params.method, comps, v2) #, patch_size_for_v=(5,5))
    elif dataset == 'FMNIST':
        ds = FMNIST( params.method, comps, v2) #, patch_size_for_v=(5,5))
    x_train, y_train = ds.get_train()
    x_val, y_val = ds.get_val()
    #print(x_train.shape, np.bincount(y_train), x_val.shape, np.bincount(y_val))

    model_holder = MLP()
    #model_holder = SampleCNN()
    model = model_holder.build_model(ds.get_input_shape(), ds.get_nb_classes(), ds.get_nb_components(),  params.freeze, params.denses, params.dense_size,params.reconstruct, params.regularizer)
    model.summary()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['sparse_categorical_accuracy']

    #Update Model Weights from V
    ws=model.get_weights()
    #range of V is [-inf,+inf]
    #min,max
    # np.median(V) as threshold --> set the half of V to zero
    V = ds.get_v()
    scaler=0
    ws[0][:]=  V[:ds.get_nb_components(),:].T
    if params.reconstruct>=1:
        ws[2][:]=  V[:ds.get_nb_components(),:]

    #ws[0]=  U[:ds.get_nb_components(),:].T

    """  print(ws[0].shape)
    import sys
    sys.exit(0) """
    if params.freeze:  # svd
     model.set_weights(ws)

    if params.freeze: # for dense layer // freeze = non-trainable
        model.layers[1].trainable=False
        if params.reconstruct>=1:
            #Freeze the reconstruction layer
            model.layers[2].trainable=False


    model.compile(tf.keras.optimizers.Adam(1e-4), loss_fn, metrics)

    m_path = os.path.join(params.save_dir, model_holder.get_name())
    util.mk_parent_dir(m_path)
    label = '_model_comps_' + str(ds.get_nb_components()) + '_dataset_' +  params.dataset +  '_method_' + params.method +  '_v2_' + str(params.v2) +'_freezed_' +  str(params.freeze) + '_denses_' +  str(params.denses) + '_dense-size_' +  str(params.dense_size)+'_recon_'+str(params.reconstruct)+'_regularizer_'+str(params.regularizer)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(m_path + label  + '_{epoch:03d}.h5', monitor='val_sparse_categorical_accuracy', save_best_only=True),
                 tf.keras.callbacks.CSVLogger(os.path.join(params.save_dir, label + '.csv'))]
    #print(model.predict(x_train[:10]))
    #import sys
    #sys.exit(0)

    # adverasial training
    model.fit(x_train, y_train, epochs=params.epoch, validation_data=(x_val, y_val),
              batch_size=params.batch_size,
              callbacks=callbacks)


if __name__ == '__main__':
    parser = ArgumentParser(description='Main entry point')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--comps", type=int, default=32)
    parser.add_argument("--dataset", type=str, default='MNIST')
    parser.add_argument("--method", type=str, default='svd')
    parser.add_argument("--reconstruct", type=int, default=0)
    parser.add_argument("--regularizer", type=int, default=0)
    parser.add_argument("--v2", type=int, default=0)
    parser.add_argument("--freeze", type=int, default=0)  # freeze training on dense layer
    parser.add_argument("--denses", type=int, default=1)  # number of extra dense layers
    parser.add_argument("--dense_size", type=int, default=256)
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
