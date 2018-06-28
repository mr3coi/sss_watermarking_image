import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image
from image import read
from os import remove

def crop(image: Image, topleft: tuple, botright: tuple):
    """
    Produces a rectangular crop of the input image, with edges parallel
        to those of the input image.

    :param image: input image
    :type image: PIL.Image
    :param topleft: the coordinate of the top-left corner of the desired crop
    :type topleft: tuple(int,int) :param botright: the coordinate of the bottom-right corner of the desired crop
    :type botright: tuple(int,int)

    :return: the modified image
    :rtype: PIL.Image
    """
    top, left = topleft
    bottom, right = botright

    MSG = "Cannot crop outside the given image"
    assert top >= 0, MSG
    assert left >= 0, MSG
    assert bottom <= image.size[1], MSG
    assert right <= image.size[0], MSG

    return image.crop(box=(left,top,right,bottom))

def rotate(image: Image, angle: int):
    """
    :param image: input image
    :type image: PIL.Image
    :param angle: the degrees by which to rotate the input image (CCW)
    :type angle: int 

    :return: the modified image
    :rtype: PIL.Image
    """
    return image.rotate(angle)

def scale(image: Image, ratio: float):
    """
    Scales the given input image by the given ratio (either enlarge or reduce).

    :param image: input image
    :type image: PIL.Image
    :param ratio: the ratio by which to scale the input image
    :type ratio: float (>0)

    :return: the modified image
    :rtype: PIL.Image
    """
    assert ratio > 0, "Negative scale ratio is not supported."
    new_size = tuple([int(val*ratio) for val in image.size])
    return image.resize(new_size, resample=Image.BILINEAR)

def compress_jpeg(image, qf):
    """
    :param image: input image
    :type image: PIL.Image
    :param qf: quality factor of the new .jpeg output
    :type qf: int (1~95)

    :return: the modified image
    :rtype: PIL.Image
    """
    TMP_PATH = './tmp.jpg'
    image.save(TMP_PATH, 'jpeg', quality=qf)
    out = read(TMP_PATH)
    remove(TMP_PATH)
    return out

