import numpy as np
import os, sys
import argparse
from sss_watermark import SSSW
from attack import crop, scale, rotate, compress_jpeg
from image import read, show_img

from PIL import Image

HOME_DIR='../'
SRC_NAME='corel_bavarian_couple.jpg'

def visualize_dct(image, alpha=590):
    '''
    clipped = np.clip(image,-1*alpha,alpha)   # approx. 5th, 95th percentiles
    out = (clipped + alpha) / alpha * 128
    '''
    out = np.log(np.abs(image))
    return np.floor(out)

if __name__ == "__main__":
    #np.random.seed(seed=61854)

    # Load image data
    IMG_PATH = os.path.join(HOME_DIR, 'data', SRC_NAME)
    sssw = SSSW(img_path=IMG_PATH)
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

    '''
    # Test SSSW class
    output = sssw.insert()
    show_img(sssw.original)
    show_img(output)
    print(sssw.detect(output))
    '''

    '''
    modified_image = scale(sssw.original, 0.3); print(modified_image.size)
    modified_image = crop(sssw.original, (150,200), (300,300))
    modified_image = compress_jpeg(sssw.original, 10)
    '''
    modified_image = rotate(sssw.original, 400)
    show_img(modified_image)

    '''
    recovered_image = sssw.recover_scale(modified_image)
    recovered_image = sssw.recover_crop(modified_image, (150,200))
    '''
    recovered_image = sssw.recover_rotate(modified_image, 400)
    show_img(recovered_image)
    show_img(sssw.original)
