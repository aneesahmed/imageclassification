# coding=utf-8
import numpy as np
from PIL import Image

#from scipy import misc
#fimg = misc.imread("teeth.png")

def image_to_vector(name):
    try:
        img = Image.open(name)
        img = img.resize((50,50), Image.ANTIALIAS)
        img_array = np.array(img, dtype=np.uint8)
        # img = Image.open('orig.png').convert('RGBA')
        shape = img_array.shape
        #randint(0, 9))
        vector = img_array.ravel()
        return  shape, vector

    except IOError as e:

        return 0,[-1]

def vector_to_image(vector, shape):
    matrix = np.matrix(vector)

    # do something to the vector
    matrix[:, ::10] = 128
    # reform a numpy array of the original shape
    arr2 = np.asarray(vector).reshape(shape)

    # make a PIL image
    img2 = Image.fromarray(arr2, 'RGBA')
    img2.show()