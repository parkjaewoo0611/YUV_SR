import numpy as np
import cv2
import sys


def psnr(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    psnr = 10 * np.log(255 * 255 / err)

    return psnr

original = cv2.imread(sys.argv[1])
result = cv2.imread(sys.argv[2])

m = psnr(original, result)

print(m)
