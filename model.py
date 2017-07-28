import numpy as np
from ops import *

sess = tf.InteractiveSession()

init = tf.initialize_all_variables()
sess.run(init)

class CbCrSRNN():
    # 5x5 64개, 3x3 32개, 3x3 (2^2)x2개 - 2
    W = 1920
    H = 1080
    r = 2
    learning_rate = 0.1
    def __init__(self):
        trunc_normal = tf.truncated_normal(stddev=0.1)
        self.x_input = tf.placeholder(tf.float32, shape=[None, self.W, self.H], name="x-input")
        self.y_target = tf.placeholder(tf.float32, shape=[None, self.W*self.r, self.H*self.r], name="y-target")
        #NHWC
        self.w0 = tf.Variable(trunc_normal([2, 5, 5, 64]))
        self.b0 = tf.Variable(trunc_normal([64]))

        self.w1 = tf.Variable(trunc_normal([64, 3, 3, 32]))
        self.b1 = tf.Variable(trunc_normal([32]))

        self.w2 = tf.Variable(trunc_normal([32, 3, 3, 8]))
        self.b2 = tf.Variable(trunc_normal([8]))


    def train(self, x_i, y_i):
        self.h0 = conv2d(self.x_input, self.w0, self.b0, activation="tanh")
        self.h1 = conv2d(self.h0, self.w1, self.b1, activation="tanh")
        self.h2 = conv2d(self.h1, self.w2, self.b2)

        self.y = PS(self.h2, self.r)

        self.loss = tf.losses.mean_squared_error(self.y, self.y_target)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        error, _ = sess.run([self.loss, self.opt], feed_dict={self.x_input: x_i, self.y_target: y_i})
        return error

