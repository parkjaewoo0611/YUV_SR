from model import *
import cv2
import numpy as np
import os
sess = tf.InteractiveSession()

def train():
    # load X_train and X_val can't load in one time(different image size)
    UV420_path = 'Training/UV_420'
    UV444_path = 'Training/UV_444'

    X_train = np.load('train_x.npy')
    Y_train = np.load('train_y.npy')
    X_val = np.load('val_x.npy')
    Y_val = np.load('val_y.npy')

    # Clear old variables
    tf.reset_default_graph()

    # Declare out model
    model = CbCrSRNN(learning_rate=0.001)
    model.build()

    with tf.Session() as sess:
        # initialize variables
        tf.global_variables_initializer().run()
        print("Initialized")

        model.training(sess, X_train, Y_train, X_val, Y_val, epochs=3000000, batch_size=1024)

if __name__ =='__main__':
    train()