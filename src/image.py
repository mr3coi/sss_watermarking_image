import matplotlib.pyplot as plt
from PIL import Image

def read(img_path):
    """
    Reads the input image file into an PIL.Image object for modification.
    Converts the input image into grayscale

    :param img_path: datapath to the desired image file
    :type img_path: string

    :return: pixel information of the image designated by the given datapath
    :rtype: PIL.Image
    """
    #return misc.imread(img_path, flatten=True, mode='L').astype(float)     # deprecated
    return Image.open(img_path).convert('L')

def show_img(image: Image, gray=True):
    if gray:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()

def save(image: Image):
    pass
