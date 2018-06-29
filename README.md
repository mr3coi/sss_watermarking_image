# sss_watermarking_image

Python 3.6 implementation of 'Secure Spread Spectrum Watermarking for Multimedia' (Cox et al., 1997) using Numpy, Scipy.

### Contents

- `sss_watermarking.py`: code for `SSSW` (Secure Spread Spectrum Watermarking) class

- `attack.py`: implementation of "attacks" that can be conducted on an image

- `test.py` : a simple test to check the correctness and robustness of the watermarking technique

- `image.py` : personal interface module to `PIL.Image`, includes functions for reading/displaying/saving PIL.Image images

- `main.py` : a bunch of codes for testing the above two modules (deprecated)

- `dct_test.py` : codes for testing DCT (deprecated)

### Required Libraries

- `Numpy`
- `Scipy`
- `PIL`

### Data

Bavarian couple image (true name unknown, source: Corel Stock Photo Library)

### Run

run `python src/test.py`
