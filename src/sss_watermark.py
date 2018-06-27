from scipy import misc
from scipy.fftpack import dct, idct
import numpy as np

class SSSW(object):
    def __init__(self, img_path, alpha=0.1, pat_size=100):
        self.original = misc.imread(img_path,
                                    flatten=True,
                                    mode='L').astype(float)
        self.alpha = alpha
        self.mark_size = pat_size
        self.mark = self.gaussian_vector(pat_size)

    def insert(self):
        """
        :return: the new version of the input image w/ watermark inserted
        :rtype: numpy.ndarray
        """
        # Compute DCT of the original image
        image_dct = SSSW.dctII(self.original)

        # Compute the regions most perceptually significant
        locations = np.argsort(-image_dct,axis=None)
        ROW_SIZE = self.original.shape[-1]
        locations = [(val//ROW_SIZE, val%ROW_SIZE) for val in locations]

        # Embed the watermark into the regions
        # (Using the formula (2) described in the paper)
        for idx,(loc,mark_val) in enumerate(zip(locations,self.mark)):
            image_dct[loc] *= 1 + self.alpha * mark_val

        # Convert the DCT result back to image
        image_rev = SSSW.idctII(image_dct)

        return image_rev

    def detect(self, image):
        """
        :param image: the image whose watermark is to be matched
        :type image: numpy.ndarray
        :return: whether the watermark is detected or not
        :rtype: bool
        """
        def extract(self, image):
            """
            :param image: the image whose watermark is to be matched
            :type image: numpy.ndarray
            :return: whether the watermark is detected or not
            """
            pass
        pass

    @staticmethod
    def gaussian_vector(size: int):
        return np.random.randn(size)

    @staticmethod
    def similarity(X,X_star):
        return np.sum(np.multiply(X,X_star)) / np.sqrt(np.sum(np.multiply(X_star,X_star)))

    @staticmethod
    def dctII(image: np.array):
        return dct(dct(image,axis=0),axis=1, norm='ortho')

    @staticmethod
    def idctII(image: np.array):
        return idct(idct(image,axis=1),axis=0, norm='ortho')
