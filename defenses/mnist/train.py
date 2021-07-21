from argparse import ArgumentParser
import os
import tensorflow as tf
import numpy as np
from dataset import DataSet, MNIST
from models import SampleCNN
import util
import time
from foolbox.attacks import LinfPGD
from foolbox.models import TensorFlowModel
from sklearn.feature_extraction import image

def batch_attack(imgs, labels, attack, fmodel, eps, batch_size):
    adv = []
    for i in range(int(np.ceil(imgs.shape[0] / batch_size))):
        x_adv, _, success = attack(fmodel, imgs[i * batch_size:(i + 1) * batch_size],
                                   criterion=labels[i * batch_size:(i + 1) * batch_size], epsilons=eps)
        adv.append(x_adv)
    return np.concatenate(adv, axis=0)


def svd_attack(fmodel, x_batch, criterion, epsilons):
    print('svd attack')
    num_components = 10
    x_adv_batch = []
    U, s, V = np.linalg.svd(x_batch)
    pe = image.PatchExtractor(patch_size=(7, 7))
    for x in x_batch: 
        #print('Patches shape: {}'.format(pe_trans.shape))
        
        x_proj = np.dot(x_adv_batch,V[:num_components,:].T)
        x_adv_batch =  np.dot(x_proj,V[:num_components,:]) 
        
    
    
    
    
    x_adv_batch = x_adv_batch[:] / .255 * .8
    
    success = False
    return x_adv_batch, 1, success

def main(params):
    ds = MNIST()
    x_train, y_train = ds.get_train()
    U, s, V = np.linalg.svd(x_train)
    x_val, y_val = ds.get_val()
    print(x_train.shape, np.bincount(y_train), x_val.shape, np.bincount(y_val))
    model_holder = SampleCNN()
    model = model_holder.build_model(ds.get_input_shape(), ds.get_nb_classes())
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['sparse_categorical_accuracy']
    model.compile(tf.keras.optimizers.Adam(1e-4), loss_fn, metrics)
    m_path = os.path.join(params.save_dir, model_holder.get_name())
    util.mk_parent_dir(m_path)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(m_path + '_{epoch:03d}-{val_loss:.2f}.h5'),
                 tf.keras.callbacks.CSVLogger(os.path.join(params.save_dir, 'metrics.csv'))]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(seed=9, buffer_size=x_train.shape[0], reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(params.batch_size)

    if params.attack == 'pgd':
        attack = LinfPGD(abs_stepsize=params.step_size, steps=params.steps, random_start=True)
    else:
        attack = lambda fmodel, x_batch, criterion, epsilons: svd_attack(fmodel, x_batch, criterion, epsilons)
    fmodel = TensorFlowModel(model, bounds=(0, 1), device='/device:GPU:0')
    imgs_val, labels_val = tf.convert_to_tensor(x_val), tf.convert_to_tensor(y_val)
    for cb in callbacks:
        cb.set_model(model)
        cb.set_params(
            {'batch_size': params.batch_size, 'epochs': params.epoch, 'steps': x_train.shape[0] // params.batch_size,
             'samples': x_train.shape[0], 'verbose': 0,
             'do_validation': True,
             'metrics': ['loss', 'accuracy', 'val_loss', 'val_accuracy']})
        cb.on_train_begin()
    for i in range(params.epoch):
        for cb in callbacks:
            cb.on_epoch_begin(i)
        delta0 = time.time()
        a_loss, a_acc = 0, 0
        for b_idx, (x_batch, y_batch) in enumerate(train_dataset):
            print('\r', b_idx, end=' ')
            x_adv_batch, _, success = attack(fmodel, x_batch,
                                             criterion=y_batch, epsilons=params.eps)
            batch_eval = model.train_on_batch(x_adv_batch, y_batch)
            a_loss = a_loss + batch_eval[0]
            a_acc = a_acc + batch_eval[1]
        x_adv_val = batch_attack(imgs_val, labels_val, attack, fmodel, params.eps, params.batch_size)
        train_eval = [a_loss / (b_idx + 1), a_acc / (b_idx + 1)]
        val_eval = model.evaluate(x_adv_val, y_val, verbose=0)
        stats = {'loss': train_eval[0], 'accuracy': train_eval[1], 'val_loss': val_eval[0],
                 'val_accuracy': val_eval[1]}
        print(i, time.time() - delta0, 's', stats)
        for cb in callbacks:
            cb.on_epoch_end(i, stats)
            if i == (params.epoch - 1) or model.stop_training:
                cb.on_train_end()
        if model.stop_training:
            break


if __name__ == '__main__':
    parser = ArgumentParser(description='Main entry point')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--attack", type=str, default='pgd')
    parser.add_argument("--memory_limit", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--save_dir", type=str, default=os.path.join('saved_models'))
    parser.add_argument('--step_size', type=float, default=0.01)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--eps', type=float, default=0.3)
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
