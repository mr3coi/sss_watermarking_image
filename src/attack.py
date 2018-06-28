import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

def crop(image, topleft, botright):
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
    return image.crop(box=(left,top,right,bottom))

def rotate(image, angle):
    """
    :param image: input image
    :type image: PIL.Image
    :param angle: the angle by which to rotate the input image
    :type angle: int 

    :return: the modified image
    :rtype: PIL.Image
    """
    pass

def scale(image, ratio):
    """
    :param image: input image
    :type image: PIL.Image
    :param : 
    :type : 

    :return: the modified image
    :rtype: PIL.Image
    """
    pass

def compress_jpeg(image, qf):
    """
    :param image: input image
    :type image: PIL.Image
    :param qf: quality factor of the new .jpeg output
    :type qf: int (1~100)

    :return: the modified image
    :rtype: PIL.Image
    """
    pass

