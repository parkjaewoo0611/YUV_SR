import os
import sys
import numpy as np
import math
import cv2


def UV444_to_UV420(uv444):
    H, W, C = uv444.shape

    if H % 2 == 1 and W % 2 == 1:
        temp = np.pad(uv444, ((0, 1), (0, 1), (0, 0)), 'edge')
        uv420 = np.ndarray([int((H+1)/2), int((W+1)/2), C])

    elif H % 2 == 1:
        temp = np.pad(uv444, ((0, 1), (0, 0), (0, 0)), 'edge')
        uv420 = np.ndarray([int((H+1)/2), int(W/2), C])

    elif W % 2 == 1:
        temp = np.pad(uv444, ((0, 0), (0, 1), (0, 0)), 'edge')
        uv420 = np.ndarray([int(H/2), int((W+1)/2), C])

    else:
        temp = uv444
        uv420 = np.ndarray([int(H/2), int((W+1)/2), C])

    HH, WW, _ = uv420.shape
    for i in range(HH):
        for j in range(WW):
            uv420[i, j, :] = temp[2*i, 2*j, :]

    return uv420

if __name__ == '__main__':
    RGB_path = sys.argv[1]
    UV420_path = sys.argv[2]
    UV444_path = sys.argv[3]
    Y_path = sys.argv[4]
    input_names = []
    list = os.listdir(RGB_path)

    for name in list:
        input_names.append(os.path.join(RGB_path, name))

    for i in range(len(list)):
        current_name = input_names[i]
        im = cv2.imread(current_name)
        if im.shape[2] != 3:
            print('convert 1 channel to 3 channel : ', list[i])
            tmp = np.ndarray(shape=[im.shape[0], im.shape[1], 3])
            tmp[0] = im
            tmp[1] = im
            tmp[2] = im
            im = tmp

        W, H, C = im.shape
        print(W, H, current_name)

        im = im / 255.0
        im = np.float32(im)

        yuv_444 = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)
        uv_444 = yuv_444[:, :, 1:3]
        uv_420 = UV444_to_UV420(uv_444)

        np.save(os.path.join(UV420_path, os.path.splitext(list[i])[0]), uv_420)
        np.save(os.path.join(UV444_path, os.path.splitext(list[i])[0]), uv_444)
        np.save(os.path.join(Y_path, os.path.splitext(list[i])[0]), yuv_444[:, :, 0])



