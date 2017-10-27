import os
import sys
import numpy as np
import math
from PIL import Image, ImageOps
from six.moves import cPickle as pickle

if __name__ == '__main__':
    path = sys.argv[1]
    resizing = float(sys.argv[2])
    output_path = sys.argv[3]
    input_names = []
    list = os.listdir(path)

    for name in list:
        input_names.append(os.path.join(path, name))

    for i in range(len(list)):
        current_name = input_names[i]
        im = Image.open(current_name)
        if im.mode != 'RGB':
            print('Mode: ', im.mode)
            tmp = im.convert('RGB')
            im.close()
            im = tmp
        W, H = im.size
        print(W, H, current_name)
        downsampled_im = ImageOps.fit(im, (math.ceil(resizing * W), math.ceil(resizing * H)), method=Image.BICUBIC)
        downsampled_im.save(os.path.join(output_path, list[i]))
        downsampled_im.close()
        im.close()

