import os
import sys
import numpy as np
import tensorflow as tf
import logging
from model import DistrSystem

tf.app.flags.DEFINE_string('labels', './labels.csv', 'path to labels file')
tf.app.flags.DEFINE_integer('num_classes', 2, 'number of classes')
tf.app.flags.DEFINE_string('train', None, 'train data path (None to not train)')
tf.app.flags.DEFINE_integer('num_inst', 4, 'number of training institutions')
tf.app.flags.DEFINE_string('train_split', None, 'path to training institution split file')
tf.app.flags.DEFINE_integer('num_epochs', 1, 'number of epochs to train at an institution each cycle')
tf.app.flags.DEFINE_integer('max_cycles', 200, 'number of cycles of epochs to train')
tf.app.flags.DEFINE_string('val', None, 'val data path (None if no val set)')
tf.app.flags.DEFINE_integer('val_freq', 1, 'validates every val_freq cycles')
tf.app.flags.DEFINE_integer('val_size', 128, 'val set sample size when validating')
# tf.app.flags.DEFINE_integer('early_stop', 20, 'terminates training if early_stop consecutive epochs without val loss improvement')
tf.app.flags.DEFINE_string('test', None, 'test data path (None to not test)')
tf.app.flags.DEFINE_string('load', None, 'path to saved weights (None to train from scratch)')
tf.app.flags.DEFINE_string('save', None, 'path to save weights (None to not save')
tf.app.flags.DEFINE_string('log', 'log.txt', 'path to save log')
tf.app.flags.DEFINE_string('loss_curve', None, 'path to save loss curves (None for no graph')
tf.app.flags.DEFINE_string('acc_curve', None, 'path to save accuracy curves (None for no graph')

tf.app.flags.DEFINE_integer('img_height', 256, 'image height in pixels')
tf.app.flags.DEFINE_integer('img_width', 256, 'image width in pixels')
tf.app.flags.DEFINE_integer('img_channels', 3, 'number of image channels')

tf.app.flags.DEFINE_string('model', None, 'Type of model to use from model.py')
tf.app.flags.DEFINE_integer("batch_size", 32, 'training batch size')
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.app.flags.DEFINE_float('decay', 1.0, 'learning rate decay after each epoch')
tf.app.flags.DEFINE_float('dropout', 0.0, 'dropout probability')
tf.app.flags.DEFINE_boolean('augment', True, 'whether or not to perform data augmentation')

tf.app.flags.DEFINE_boolean('weighted_loss', False, 'whether or not to weight loss function')
tf.app.flags.DEFINE_boolean('weighted_samples', False, 'whether or not to weight training samples in random batch selection')
tf.app.flags.DEFINE_boolean('prop_iter', False, 'whether or not to train at each instutition for iterations proportional to number of data samples')
tf.app.flags.DEFINE_boolean('prop_lr', False, 'whether or not to use lr at each instutition proportional to number of data samples')

FLAGS = tf.app.flags.FLAGS
logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(FLAGS.log), logging.StreamHandler()])

def initialize_model(session, model):
    if FLAGS.load is not None:
        ckpt = tf.train.get_checkpoint_state(FLAGS.load)
        v2_path = ckpt.model_checkpoint_path + '.index' if ckpt else ''
        if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
            logging.info('Reading model parameters from %s' % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info('Created model with fresh parameters.')
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))

def main(_):
    labels = {line.strip().split(',')[0]: int(line.strip().split(',')[1]) for line in open(FLAGS.labels)}
    institutions = {line.strip().split(',')[0]: int(line.strip().split(',')[1]) for line in open(FLAGS.train_split)}

    classifier = DistrSystem(FLAGS, labels, institutions)

    with tf.Session() as sess:
        initialize_model(sess, classifier)
        if FLAGS.train is not None:
            val = None if FLAGS.val is None else set(os.listdir(FLAGS.val))
            train = {i: set() for i in range(FLAGS.num_inst)}
            for i in institutions:
                train[institutions[i]].add(i)
            classifier.train(sess, train, val) 
        if FLAGS.test is not None:
            test = [os.listdir(FLAGS.test)]
            test_loss, test_acc = classifier.test(sess, test, 'test') 
            logging.info('test loss: %.3f, test acc: %.3f' % (test_loss, test_acc))


if __name__ == "__main__":
    tf.app.run()
