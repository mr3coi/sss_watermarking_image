from scipy import misc
from scipy.fftpack import dct, idct
import numpy as np
from PIL import Image
from PIL.ImageOps import invert
from itertools import product

class SSSW(object):
    """
    Given an input image (specifically, a path to the input image file),
    (i)     randomly generates a watermark pattern,
    (ii)    inserts the pattern into the given image (as specified in paper),
    (iii)   stores intermediate data for detection, and
    (iv)    conducts detection process when given a modified version of the input image

    Also includes functions that reconstruct "attacked" versions of the images \
            to match the original image.
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
        ''' deprecated
        self.original   = misc.imread(img_path,
                                    flatten=True,
                                    mode='L').astype(float)
        '''
        self.original   = Image.open(img_path).convert("L")
        self.alpha      = alpha     # Scaling parameter
        self.mark_size  = pat_size  # Size of the watermark
        self.mark       = self.gaussian_vector(pat_size)    # Watermark for the current instance
        self.mark_2d    = None      # Mark scattered in 2-D image (computed in 'insert')
        self.ori_dct    = None      # Storage of DCT result of input image

    def insert(self):
        """
        Method for inserting a generated watermark into the input image.

        :return: the new version of the input image w/ watermark inserted
        :rtype: numpy.ndarray(float)
        """
        # Compute DCT of the original image
        self.ori_dct = SSSW.dctII(self.to_float_array(self.original))

        # Compute the regions most perceptually significant
        locations = np.argsort(-self.ori_dct,axis=None)
        ROW_SIZE = self.original.size[0]
        locations = [(val//ROW_SIZE, val%ROW_SIZE) for val in locations]

        # Generate 2-D watermark
        # (Using the formula (2) described in the paper)
        self.mark_2d = np.zeros(shape=self.ori_dct.shape, dtype=float)
        for idx,(loc,mark_val) in enumerate(zip(locations,self.mark)):
            self.mark_2d[loc] += self.alpha * mark_val
        self.mark_2d *= self.ori_dct

        # Embed the watermark into the regions & convert the DCT back to image
        image_rev = self.from_float_array(SSSW.idctII(self.ori_dct + self.mark_2d))

        return image_rev

    def extract(self,new_image):
        """
        Extracts the watermark from the presumed location of the new image.

        :param new_image: the image whose watermark is to be matched
        :type new_image: PIL.Image

        :return: the extracted watermark
        :rtype: numpy ndarray(float)
        """
        return SSSW.dctII(np.array(new_image)) - self.ori_dct

    def detect(self, image, threshold=6):
        """
        Given an image, determines whether its supposed watermark matches its
            original counterpart.

        :param image: the image whose watermark is to be matched
        :type image: numpy.ndarray(float)
        :param threshold: threshold for concluding the similarity of the watermarks 
        :type threshold: float

        :return: (similarity value, whether the watermark is detected or not)
        :rtype: tuple(float, bool)
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

    def recover_crop(self, image, topleft):
        """
        Assumes that the coordinate of the topleft corner w.r.t. \
            the original image is known.
        Also assumes that no other modification has been made to the input.

        :param image: input image
        :type image: PIL.Image
        :param topleft: the coordinates of the topleft corner of the crop \
                        in the original image (row, column)
        :type topleft: tuple(int,int)

        :return: the input image w/ cropped parts filled in \
                 w/ data from the orignal image
        :rtype: PIL.Image
        """
        out = self.original.copy()
        out.paste(image, box=tuple(reversed(topleft)))
        return out

    def recover_rotate(self, image: Image, angle: int):
        """
        Rotates the input image back to fit the original image, \
            and fills in the parts that are missing due to rotation.
        Assumes that the angle of rotation is provided, and that the image \
            has not been expanded when rotated (i.e. corners have been cut off).
        Also ssumes that no other modification has been made to the input.

        :param image: input image
        :type image: PIL.Image
        :param angle: the degree that the image was rotated (CCW)
        :type angle: int

        :return: the input image modified to match the original image
        :rtype: PIL.Image
        """
        out = image.rotate(-angle)
        '''
        for coord in product(range(self.original.size[1]), range(self.original.size[0])):
            if out.getpixel(coord) != 0:
                out.putpixel(coord, self.original.getpixel(coord))
        '''
        #mask = out.convert(mode="1",dither=None).convert('L')  # insufficient masking
        mask = out.point(lambda x: 255*int(x>0))
        mask = invert(mask).convert('1')
        out.paste(self.original,mask=mask)

        return out

    def recover_scale(self, image: Image):
        """
        Resizes the input image to fit the size of the original image.
        Assumes that no other modification has been made to the input.

        :param image: input image
        :type image: PIL.Image

        :return: the input image resized to match the original image
        :rtype: PIL.Image
        """
        return image.resize(size=self.original.size)

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

    @staticmethod
    def to_array(image: Image):
        """
        Converts the given PIL.Image to a pixel array (0-255).

        :param image: input image
        :type image: PIL.Image

        :return: pixel array
        :rtype: numpy ndarray
        """
        return np.array(image)

    @staticmethod
    def to_float_array(image: Image):
        """
        Converts the given PIL.Image to a pixel array of float values in range 0-1.

        :param image: input image
        :type image: PIL.Image

        :return: pixel array
        :rtype: numpy ndarray(float, 2-D)
        """
        return np.array(image).astype(float) / 256

    @staticmethod
    def from_float_array(array: np.ndarray):
        """
        Converts the given PIL.Image to a pixel array of float values in range 0-1.
        Ignores rounding error wihle converting pixel values from float \
                to integers in 0-255 range.

        :param array: pixel array (w/ values in float, 0-1)
        :type array: numpy ndarray(float, 2-D)

        :return: output image
        :rtype: PIL.Image
        """
        array *= 255
        array = np.around(array)
        return Image.fromarray(array).convert(mode="L")

    # Class variable (list of recovery functions - used in 'test.py')
    recover_flist = [recover_crop, recover_rotate, recover_scale]
