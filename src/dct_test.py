from PIL import Image
from scipy import misc
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import numpy as np
import os, sys

HOME_DIR='../'
SRC_NAME='corel_bavarian_couple.jpg'

def show_img(image, gray=False):
    if gray:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()

def dctII(image: np.array):
    return dct(dct(image,axis=1),axis=0)

'''
def sigmoid(arr):
    def sigmoid_element(val):
        exp = np.exp(val)
        return exp / (1+exp)
    return np.array([[sigmoid_element(x) for x in row] for row in arr])

def shift_and_log(arr):
    min_val = np.min(arr)
    if min_val < 0:
        arr = arr - min_val + 1
    return np.log(arr)
'''

def visualize_dct(image, alpha=20000):
    #print(np.percentile(image_dct,5), np.percentile(image_dct,95))
    clipped = np.clip(image,-1*alpha,alpha)   # approx. 5th, 95th percentiles
    out = (clipped + alpha) / alpha * 128
    return np.floor(out)

if __name__ == "__main__":
    # Load image data
    IMG_PATH = os.path.join(HOME_DIR, 'data', SRC_NAME)
    image = misc.imread(IMG_PATH,flatten=True,mode='L').astype(float)

    # Conduct 2D-DCT(II) on the input image
    image_dct = dctII(image)
    '''
    plt.contour(image_dct)
    plt.show()
    '''

    # Visualize DCT output
    viz = visualize_dct(image_dct)
    show_img(viz, gray=True)
