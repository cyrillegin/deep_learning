from random import randint
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as pyplt
from scipy.ndimage import interpolation


def displayImages(mnist, imageSet):
    fig = pyplt.figure()
    for i in range(0, 10):
        nextImage = False
        while nextImage is False:
            labelIndex = randint(0, len(imageSet) - 1)
            if mnist.train.labels[labelIndex][i] == 1.0:
                print(mnist.train.labels[labelIndex], i)
                image = imageSet[i]
                image = np.array(image, dtype='float')
                data = image.reshape((28, 28))
                pyplt.subplot(2, 5, (i+1))
                pyplt.imshow(data)
                nextImage = True

    pyplt.SubplotTool
    pyplt.show()


def displayWeights(weightSet):
    fig = pyplt.figure()
    for i in range(0, 10):
        data = np.reshape(weightSet[:, i], [28, 28])
        pyplt.subplot(2, 5, (i+1))
        pyplt.imshow(data)
        nextImage = True

    pyplt.SubplotTool
    pyplt.show()

    


def doRotation(mnist):
    rotation = randint(0, 360)
    for i in range(0, len(mnist.train.images)):
        img = np.reshape(mnist.train.images[i, :], [28, 28])
        img = rotate(img, rotation, reshape=False)
        mnist.train.images[i, :] = np.reshape(img, (784,))
    return mnist


# Adapted from:
# https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
def clipped_zoom(img, zoom_factor):

    h, w = img.shape[:2]

    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))

    # for multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # zooming out
    if zoom_factor < 1:
        # bounding box of the clip region within the output array
        top = (h - zh) // 2
        left = (w - zw) // 2
        # zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = interpolation.zoom(img, zoom_tuple)

    # if zoom_factor == 1, just return the input array
    else:
        out = img
    return out


def doScale(mnist):
    scale = randint(5, 10) * 0.1
    for i in range(0, len(mnist.train.images)):
        img = np.reshape(mnist.train.images[i, :], [28, 28])
        zm = clipped_zoom(img, scale)
        mnist.train.images[i, :] = np.reshape(zm, (784,))
    return mnist
