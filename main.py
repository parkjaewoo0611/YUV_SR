from model import *
import cv2
import numpy as np
#TODO: data input

# sess = tf.InteractiveSession()

def main():
    # NN = CbCrSRNN
    im = cv2.imread('Training/t1.bmp', 1)
    cv2.imshow('window', im)
    cv2.waitKey(0)

if __name__ =='__main__':
    main()