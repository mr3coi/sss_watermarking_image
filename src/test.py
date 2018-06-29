import os
from sss_watermark import SSSW
from attack import crop, scale, rotate, compress_jpeg
from image import read, show_img
from itertools import chain, product
import argparse

HOME_DIR='../'
SRC_NAME='corel_bavarian_couple.jpg'

if __name__ == "__main__":
    IMG_PATH = os.path.join(HOME_DIR, 'data', SRC_NAME)

    # Arguments for testing
    topleft = (30,48); botright = (300,500)     # Crop
    degrees = 273                               # Rotate
    ratio = 1.4                                 # Scale
    qf = 10                                     # Recompress

    # Generate watermarked images
    sssws = [SSSW(img_path=IMG_PATH) for _ in range(4)]
    marked_images = [sssw.insert() for sssw in sssws]

    # Present the watermarked images
    for image in marked_images:
        show_img(image)

    # Generate attacked versions of the watermarked images
    args = [(topleft, botright), (degrees,), (ratio,), (qf,)]
    args = [tuple([img]) + arg for img, arg in zip(marked_images, args)]
    attacks = [crop,rotate,scale,compress_jpeg]
    attack_outputs = [attack(*arg) for attack, arg in zip(attacks, args)]

    # Present the attacked versions
    for image in attack_outputs:
        show_img(image)

    # Recover the attacked images
    recover_args = [(topleft,), (degrees,), ()]
    recover_args = [tuple([sssw, img]) + arg \
            for sssw, img, arg in zip(sssws, attack_outputs, recover_args)]
    recover_outputs = [recover_fn(*arg) \
            for recover_fn, arg in zip(SSSW.recover_flist, recover_args)]
    recover_outputs.append(attack_outputs[-1])      # recompressed image is used as-is

    # Present the recovered versions
    for image in recover_outputs:
        show_img(image)

    # Criss-cross detect watermarks (both not-attacked and attacked)
    targets = marked_images + recover_outputs
    detect_own = [ ((i+1,j+1), sssws[i].detect(targets[j])) \
                    for i,j in product(range(len(sssws)),range(len(targets))) ]

    # Present test results
    for pair in detect_own:
        print(pair)
