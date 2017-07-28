from model import *
import cv2
import numpy as np

sess = tf.InteractiveSession()

def main():
    NN = CbCrSRNN
    im = cv2.imread('t1.bmp', 1)
    im2 = cv2.imread('small_t1.bmp',1)
    cv2.imshow('window', im)
    cv2.waitKey(0)

    # Training = np.load("Training.npy")
    # Training_shape = np.shape(Training)

    # Training = np.divide(Training, 255.0)

    # Training_1 = Training(1, None, None, None)
    # print(Training)
    init = tf.initialize_all_variables()
    sess.run(init)
    print(NN.train(NN, ))



if __name__ =='__main__':
    main()