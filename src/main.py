import matplotlib.pyplot as plt
import numpy as np
import os, sys
import argparse
from sss_watermark import SSSW

HOME_DIR='../'
SRC_NAME='corel_bavarian_couple.jpg'

def show_img(image, gray=True):
    if gray:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()

def visualize_dct(image, alpha=590):
    '''
    clipped = np.clip(image,-1*alpha,alpha)   # approx. 5th, 95th percentiles
    out = (clipped + alpha) / alpha * 128
    '''
    out = np.log(np.abs(image))
    return np.floor(out)

if __name__ == "__main__":
    # Load image data
    IMG_PATH = os.path.join(HOME_DIR, 'data', SRC_NAME)
    sssw = SSSW(img_path=IMG_PATH)

    output = sssw.insert()
    show_img(sssw.original)
    show_img(output)

    '''
    # Conduct 2D-DCT(II) on the input image
    image_dct = SSSW.dctII(sssw.original)
    print(np.percentile(image_dct,5), np.percentile(image_dct,95))
    reverse = SSSW.idctII(image_dct)

    # Visualize DCT output
    viz = visualize_dct(image_dct)
    show_img(viz)
    show_img(reverse)
    '''
