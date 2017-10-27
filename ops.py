import numpy as np
import math
import tensorflow as tf


def xavier_initializer(n_inputs, n_outputs):
    stddev = math.sqrt(2.0 / (n_inputs + n_outputs))
    return tf.truncated_normal_initializer(stddev=stddev)


def fc_variable(shape, name):
    W = tf.get_variable(name='w_' + name, shape=shape, dtype=tf.float32,
                        initializer=xavier_initializer(shape[0], shape[1]))
    b = tf.get_variable(name='b_' + name, dtype=tf.float32, initializer=tf.zeros(shape[1]))
    return W, b


def conv_variable(shape, name):
    W = tf.get_variable(name='w_' + name, shape=shape, dtype=tf.float32,
                        initializer=xavier_initializer(shape[0] * shape[1] * shape[2], shape[3]))
    b = tf.get_variable(name='b_' + name, dtype=tf.float32, initializer=tf.zeros(shape[3]))
    return W, b


def conv2d(x, w, b, stride=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.bias_add(tf.nn.conv2d(x, w, stride, padding), b)


def relu(x):
    return tf.nn.relu(x)


def fc(x, w, b):
    return tf.matmul(x, w) + b


def max_pool(x, ksize, stride, padding='SAME'):
    return tf.nn.max_pool(x, ksize, stride, padding)


def _phase_shift(I, r):
    # Helper function with main phase shift operation
    a = 32
    b = 32
    c = 4
    bsize = tf.shape(I)[0]# Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a * r, b * r, 1))


def _phase_shift_test(I, r):
    bsize, a, b, c = I.shape

    X = np.reshape(I, [bsize, a, b, r, r])
    X = np.transpose(X, [0, 1, 2, 4, 3])
    X = np.split(X, a, 1)
    X = np.concatenate([np.squeeze(x, axis=1) for x in X], 2)
    X = np.split(X, b, 1)
    X = np.concatenate([np.squeeze(x, axis=1) for x in X], 2)
    return np.reshape(X, [bsize, a * r, b * r, 1])

def PS(h, r):
    # input = CbCr channel concat tensor
    [Cb_split, Cr_split] = tf.split(h, 2, 3)
    Cb = _phase_shift(Cb_split, r)
    Cr = _phase_shift(Cr_split, r)
    CbCr = tf.concat([Cb, Cr], 3)
    return CbCr

def PS_test(h, r):
    # input = CbCr channel concat tensor
    [Cb_split, Cr_split] = np.split(h, 2, 3)
    Cb = _phase_shift_test(Cb_split, r)
    Cr = _phase_shift_test(Cr_split, r)
    CbCr = np.concatenate([Cb, Cr], 3)
    return CbCr
