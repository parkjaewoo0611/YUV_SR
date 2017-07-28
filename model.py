import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

class CbCrSRNN:
    # 5x5 64개, 3x3 32개, 3x3 (2^2)x2개 - 2
    W = 1920
    H = 1080
    r = 2
    learning_rate = 0.1
    def __init__(self):
        rand_uni = tf.truncated_normal(stddev=0.1)
        self.x_input = tf.placeholder(tf.float32, shape=[None, self.W, self.H], name="x-input")
        self.y_target = tf.placeholder(tf.float32, shape=[None, self.W*self.r, self.H*self.r], name="y-target")
        #NCHW
        self.w0 = tf.Variable(rand_uni([5, 5, 2, 64]))
        self.b0 = tf.Variable(rand_uni([64]))

        self.w1 = tf.Variable(rand_uni([3, 3, 64, 32]))
        self.b1 = tf.Variable(rand_uni([32]))

        self.w2 = tf.Variable(rand_uni([3, 3, 32, 8]))
        self.b2 = tf.Variable(rand_uni([8]))

        self.h0 = tf.nn.tanh(tf.nn.conv2d(self.x_input, self.w0, [1, 1, 1, 1], padding='SAME') + self.b0)
        self.h1 = tf.nn.tanh(tf.nn.conv2d(self.h0, self.w0, [1, 1, 1, 1], padding='SAME') + self.b1)
        self.h2 = tf.nn.conv2d(self.h1, self.w1, [1, 1, 1, 1], padding='SAME') + self.b2

        self.y = self.PS(self.h2, self.r)

        self.loss = tf.losses.mean_squared_error(self.y, self.y_target)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        #TODO: subpixel shuffler

    # def subixel_shuffler(self, x, h2, r):
        # [bsize, a, b] = tf.shape(h2)
        #
        # [Xcb, Xcr] = tf.split(h2, [bsize/2, bsize/2], 0)
        # [Xcb1, Xcb2, Xcb3, Xcb4] = tf.squeeze(tf.split(Xcb, [1, 1, 1, 1], 0))
        # [Xcb1_col, Xcb2_col, Xcb3_col, Xcb4_col] = tf.split([Xcb1, Xcb2, Xcb3, Xcb4], b, 2)
        # for i in 2:

        # [_, h, w] = tf.shape(h2)
        # y = np.ndarray(shape = [2, h*r, w*r])
        # for i in range(h):
        #     for j in range(w):
        #         y[1, 2 * i, 2 * j] = x[2, i, j]
        #         y[1, 2 * i, 2 * j + 1] = h2[1, i, j]
        #         y[1, 2 * i + 1, 2 * j] = h2[2, i, j]
        #         y[1, 2 * i, 2 * j + 1] = h2[3, i, j]
        #
        #         y[2, 2 * i, 2 * j] = x[3, i, j]
        #         y[2, 2 * i, 2 * j + 1] = h2[4, i, j]
        #         y[2, 2 * i + 1, 2 * j] = h2[5, i, j]
        #         y[2, 2 * i, 2 * j + 1] = h2[6, i, j]
        # return y

    def _phase_shift(self, h2, r):
        # Helper function with main phase shift operation
        batchsize, w, h, c = tf.shape(h2)
        X = tf.reshape(h2, (batchsize, w, h, r, r))
        X = tf.split(1, w, X)  # a, [bsize, b, r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
        X = tf.split(1, h, X)  # b, [bsize, a*r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, a * r, b * r
        return tf.reshape(X, (batchsize, w * r, h * r, 1))

    def PS(self, h2, r, color=False):
        # Main OP that you can arbitrarily use in you tensorflow code
        Xc = tf.split(h2, [2, 2], 0)
        X = tf.concat([self._phase_shift(x, r) for x in Xc], 0)
        return X

    def train(self, x_i, y_i):
        error, _ = sess.run([self.loss, self.opt], feed_dict={self.x_input: x_i, self.y_target: y_i})
        return error
