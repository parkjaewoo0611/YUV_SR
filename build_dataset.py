import os
import sys
import numpy as np
from PIL import Image, ImageOps

# make low resolution image dataset from high resolution image dataset
def image_resizing_bicubic(filename, xsize, ysize):
    im = Image.open(filename)

    if im.mode != 'RGB':
        print('Mode: ', im.mode)
        tmp = im.convert('RGB')
        im.close()
        im = tmp

    downsampled_im = ImageOps.fit(im, (xsize, ysize), method=Image.BICUBIC)
    norm_im = np.array(downsampled_im, dtype=np.float32)

    downsampled_im.close()
    im.close()
    return norm_im

if __name__ == '__main__':
    pathA = sys.argv[1]
    namesA = []
    resizing = 0.25
    for name in os.listdir(pathA):
        namesA.append(os.path.join(pathA, name))


    dataset_A = np.zeros((len(namesA), 256, 256, 3))

    for i in range(len(namesA)):
        dataset_A[i] = image_resizing_bicubic(namesA[i], 256, 256)
        print(namesA[i])

    np.save('%s.npy' % sys.argv[2], dataset_A)
