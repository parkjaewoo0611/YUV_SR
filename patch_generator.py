import numpy as np
import os


def im2patch(im, shape=[32, 32], stride=[14, 14]):
    H, W, C = im.shape
    patchs = np.ndarray([1, shape[0], shape[1], C])
    i = 0
    j = 0
    while stride[0] * i + shape[0] < H:
        while stride[1] * j + shape[1] < W:
            current_patch = im[stride[0] * i: stride[0] * i + shape[0], stride[1] * j: stride[1] * j + shape[1], :]
            current_patch = np.reshape(current_patch, [1, shape[0], shape[1], C])
            patchs = np.append(patchs, current_patch, 0)
            j = j + 1
        i = i + 1

    return patchs[1:, :, :, :]


if __name__ == '__main__':
    UV420_path = 'Test/Set14/UV_420'
    UV444_path = 'Test/Set14/UV_444'

    X_list = os.listdir(UV420_path)
    Y_list = os.listdir(UV444_path)

    X_train = []
    Y_train = []

    X_shape = [32, 32]
    Y_shape = [X_shape[0] * 2, X_shape[1] * 2]

    X_patchs = np.ndarray([1, X_shape[0], X_shape[1], 2])
    Y_patchs = np.ndarray([1, Y_shape[0], Y_shape[1], 2])

    for i in range(len(X_list)):
        X_train = np.load(os.path.join(UV420_path, X_list[i]))
        Y_train = np.load(os.path.join(UV444_path, Y_list[i]))

        X_patchs = np.append(X_patchs, im2patch(X_train, shape=X_shape, stride=[14, 14]), 0)
        Y_patchs = np.append(Y_patchs, im2patch(Y_train, shape=Y_shape, stride=[28, 28]), 0)

    X_patchs = X_patchs[1:, :, :, :]
    Y_patchs = Y_patchs[1:, :, :, :]

    np.save('test_x.npy', X_patchs)
    np.save('test_y.npy', Y_patchs)