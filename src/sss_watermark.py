from scipy import misc
from scipy.fftpack import dct, idct
import numpy as np

class SSSW(object):
    def __init__(self, img_path, alpha=0.1, pat_size=100):
        self.original = misc.imread(img_path,
                                    flatten=True,
                                    mode='L').astype(float)
        self.alpha = alpha
        self.mark = self.gaussian_vector(pat_size)

    def insert(self):
        image_dct = SSSW.dctII(sssw.original)
        image_rev = SSSW.idctII(image_dct)
        return image_rev

    def detect(self, image):
        pass

    @staticmethod
    def gaussian_vector(size: int):
        return np.random.randn(size)

    @staticmethod
    def similarity(X,X_star):
        return np.sum(np.multiply(X,X_star)) / np.sqrt(np.sum(np.multiply(X_star,X_star)))

    @staticmethod
    def dctII(image: np.array):
        return dct(dct(image,axis=1),axis=0)

    @staticmethod
    def idctII(image: np.array):
        return idct(idct(image,axis=1),axis=0)
