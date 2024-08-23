import numpy as np


class ToFloatUKBB:
    """Convert ndarrays in sample values to integers."""

    def __call__(self, image):
        try:
            image = image.astype('f8')
            maxv = np.max(image)
            minv = np.min(image)
            return ((image - minv) / maxv).astype('f4')
        except:
            return image


class MeanSub:
    def __init__(self, mean_img):
        self.mean_img = mean_img

    def __call__(self, img):
        try:
            img = img - self.mean_img
        except:
            print('Error Occured in subtracting mean image')

        return img
