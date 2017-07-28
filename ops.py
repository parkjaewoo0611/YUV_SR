import tensorflow as tf


def conv2d(input, w, b, stride=1, padding="SAME", activation=None):
    output = tf.nn.conv2d(input=input, filter=w, strides=stride, padding=padding, use_cudnn_on_gpu=1)
    if activation == "tanh":
        output = tf.nn.tanh(output + b)
    else:
        output = output + b
    return output



def _phase_shift(h2, r):
    # Helper function with main phase shift operation
    batchsize, w, h, c = tf.shape(h2)
    X = tf.reshape(h2, (batchsize, w, h, r, r))
    X = tf.split(1, w, X)  # a, [bsize, b, r, r]
    X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
    X = tf.split(1, h, X)  # b, [bsize, a*r, r]
    X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, a * r, b * r
    return tf.reshape(X, (batchsize, w * r, h * r, 1))


def PS(h2, r):
    [N, _, _] = len(h2)
    num_feature_map = N/2
    [Cb_split, Cr_split] = tf.split(h2, [num_feature_map, num_feature_map*2], 0)
    Cb = [_phase_shift(x, r) for x in Cb_split]
    Cr = [_phase_shift(x, r) for x in Cr_split]
    CbCr = tf.concat([Cb, Cr], 0)
    return CbCr