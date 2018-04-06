import numpy as np
from glob import glob
from keras.preprocessing.image import load_img, img_to_array

filenames = glob('female_ventral/*.png')
outpath   = 'ndarrays/fvent277.npy'

image_shape = 227, 227, 3

def image_to_theano_np(filenames, image_shape):

    pics = []
    percent_done = [10 * i for i in range(1, 11)]
    j = 0
    dropped = 0
    for i, filename in enumerate(filenames):
        arr = img_to_array(load_img(filename))
        if arr.shape == image_shape: pics.append(arr)
        else: dropped += 1

        if (i + 1) / len(filenames) * 100 >= percent_done[j]:
            print(percent_done[j],'%  complete.')
            j += 1

    return np.array(pics, dtype=np.uint8), dropped

pics, dropped = image_to_theano_np(filenames, image_shape)
np.save(outpath, pics)
print(str(dropped / len(filenames)) + '% incompatible size')
print('Saved flies:', len(pics))
