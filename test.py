from model import *
import cv2
import numpy as np
import os
sess = tf.InteractiveSession()

def test():
    # load X_train and X_val can't load in one time(different image size)
    UV420_path = 'Test/Set14/UV_420'
    Y_path = 'Test/Set14/Y'
    result_path = 'Test/Set14/result_with_420'

    name = os.listdir(UV420_path)

    # Clear old variables
    tf.reset_default_graph()

    # declare model
    model = CbCrSRNN()
    model.build()
    with tf.Session() as sess:
        model.load(sess, 'model_checkpoints')

        for i in range(len(name)):
            UV420 = np.load(os.path.join(UV420_path, name[i]))
            UV420 = np.reshape(UV420, [1, UV420.shape[0], UV420.shape[1], UV420.shape[2]])

            Y = np.load(os.path.join(Y_path, name[i]))
            Y = np.reshape(Y, [Y.shape[0], Y.shape[1], 1])

            UV420_result = model.test(sess, UV420)

            UV444_result = PS_test(UV420_result, 2)

            UV444_result = np.reshape(UV444_result, [UV444_result.shape[1], UV444_result.shape[2], 2])
            for k in range(UV420.shape[1]):
                for j in range(UV420.shape[2]):
                    UV444_result[2 * k, 2 * j, :] = UV420[0, k, j, :]

            result = cv2.cvtColor(np.concatenate([Y, UV444_result], 2), cv2.COLOR_YUV2RGB) * 255.0
            cv2.imwrite(os.path.join(result_path, name[i]) + '.jpg', result)

if __name__ =='__main__':
    test()