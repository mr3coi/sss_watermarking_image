from scipy import misc
from scipy.fftpack import dct, idct
import numpy as np

class SSSW(object):
    """
    Given an input image,
    (i)     randomly generates a watermark pattern,
    (ii)    inserts the pattern into the given image (as specified in paper),
    (iii)   stores intermediate data for detection, and
    (iv)    conducts detection process when given a modified version of the input image
    """
    def __init__(self, img_path, alpha=0.1, pat_size=100):
        """
        :param img_path: the datapath to the source image
        :type img_path: const string
        :param alpha: scaling parameter
        :type alpha: float
        :param pat_size: size of the watermark to be inserted
        :type pat_size: int
        """
        self.original   = misc.imread(img_path,
                                    flatten=True,
                                    mode='L').astype(float)
        self.alpha      = alpha
        self.mark_size  = pat_size
        self.mark       = self.gaussian_vector(pat_size)
        self.locations  = None
        self.mark_2d    = None
        self.ori_dct    = None

    def insert(self):
        """
        Method for inserting a generated watermark into the input image.

        :return: the new version of the input image w/ watermark inserted
        :rtype: numpy.ndarray(float)
        """
        # Compute DCT of the original image
        self.ori_dct = SSSW.dctII(self.original)

        # Compute the regions most perceptually significant
        locations = np.argsort(-self.ori_dct,axis=None)
        ROW_SIZE = self.original.shape[-1]
        self.locations = [(val//ROW_SIZE, val%ROW_SIZE) for val in locations]

        # Generate 2-D watermark
        # (Using the formula (2) described in the paper)
        self.mark_2d = np.zeros(shape=self.original.shape, dtype=float)
        for idx,(loc,mark_val) in enumerate(zip(self.locations,self.mark)):
            self.mark_2d[loc] += self.alpha * mark_val
        self.mark_2d *= self.ori_dct

        # Embed the watermark into the regions & convert the DCT back to image
        image_rev = SSSW.idctII(self.ori_dct + self.mark_2d)

        return image_rev

    def extract(self,new_image):
        """
        Extracts the watermark from the presumed location of the new image.

        :param new_image: the image whose watermark is to be matched
        :type new_image: numpy.ndarray
        :return: the extracted watermark
        :rtype: numpy ndarray(float)
        """
        return SSSW.dctII(new_image) - self.ori_dct

    def detect(self, image, threshold=6):
        """
        Given an image, determines whether its supposed watermark matches its
            original counterpart.

        :param image: the image whose watermark is to be matched
        :type image: numpy.ndarray(float)
        :param threshold: threshold for concluding the similarity of the watermarks 
        :type threshold: float
        :return: whether the watermark is detected or not
        :rtype: bool
        """
        def similarity(X,X_star):
            """
            Computes the similarity measure between the original and the new watermarks.

            :param X: the image whose watermark is to be matched
            :type X: numpy.ndarray(float, 2-D)
            :param X_star: the image whose watermark is to be matched
            :type X: numpy.ndarray(float, 2-D)
            :return: the degree of similarity
            :rtype: float
            """
            return np.sum(np.multiply(X,X_star)) / np.sqrt(np.sum(np.multiply(X_star,X_star)))

        target_mark = self.extract(image)
        sim = similarity(self.mark_2d, target_mark)
        return sim, (sim > threshold)

    @staticmethod
    def gaussian_vector(size: int):
        """
        :param size: the size of the generated vector
        :type size: int
        :return: the generated vector
        :rtype: numpy.ndarray(float)
        """
        return np.random.randn(size)

    @staticmethod
    def dctII(image: np.array):
        """
        :param image: the input for 2-D DCT
        :type image: numpy.ndarray(float, 2-D)
        :return: the DCT result
        :rtype: numpy ndarray(float, 2-D)
        """
        return dct(dct(image,axis=0, norm='ortho'),axis=1, norm='ortho')

    @staticmethod
    def idctII(image: np.array):
        """
        :param image: the input for 2-D Inverse-DCT
        :type image: numpy.ndarray(float, 2-D)
        :return: the Inverse-DCT result
        :rtype: numpy ndarray(float, 2-D)
        """
        return idct(idct(image,axis=1, norm='ortho'),axis=0, norm='ortho')
