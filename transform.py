# -*- coding: utf-8 -*-

__all__ = ['RandomGaussianNoise', 'RandomPoissonNoise', 'RandomTextOverlay', 'RandomCrop']

from torchtoolbox.transform import RandomPoissonNoise
from torchtoolbox.transform import functional as F
from string import ascii_letters
import cv2
import numpy as np
import numbers
import random


def gaussian_noise(img: np.ndarray, mean, std):
    imgtype = img.dtype
    gauss = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = np.clip(gauss + img.astype(np.float32), 0, 255)
    return noisy.astype(imgtype)


def crop(img, i, j, h, w):
    """Crop the given CV Image.

    Args:
        img (PIL Image): Image to be cropped.
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped image.
        w (int): Width of the cropped image.

    Returns:
        PIL Image: Cropped image.
    """

    return img[j:j + w, i:i + h, ...].copy()


class RandomCrop(object):
    """Crop the given CV Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (CV Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h, _ = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (CV Image): Image to be cropped.

        Returns:
            CV Image: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.shape[1] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.shape[1], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.shape[0] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.shape[0]), self.fill, self.padding_mode)
        i, j, h, w = self.get_params(img, self.size)
        img = crop(img, i, j, h, w)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomGaussianNoise(object):
    """Applying gaussian noise on the given CV Image randomly with a given probability.
        Args:
            p (float): probability of the image being noised. Default value is 0.5
        """

    def __init__(self, p=0.5, mean=0, std=0.1, fixed_distribution=True):
        assert isinstance(mean, numbers.Number) and mean >= 0, 'mean should be a positive value'
        assert isinstance(std, numbers.Number) and std >= 0, 'std should be a positive value'
        assert isinstance(p, numbers.Number) and p >= 0, 'p should be a positive value'
        self.p = p
        self.mean = mean
        self.std = std
        self.fixed_distribution = fixed_distribution

    @staticmethod
    def get_params(mean, std):
        """Get parameters for gaussian noise
        Returns:
            sequence: params to be passed to the affine transformation
        """
        mean = random.uniform(0, mean)
        std = random.uniform(0, std)

        return mean, std

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Image to be noised.
        Returns:
            np.ndarray: Randomly noised image.
        """
        if random.random() < self.p:
            if self.fixed_distribution:
                mean, std = self.mean, self.std
            else:
                mean, std = self.get_params(self.mean, self.std)
            return gaussian_noise(img, mean=mean, std=std)
        return img


class RandomTextOverlay(object):
    def __init__(self, p, max_occupancy, length=(10, 25), font=1, text_scale=(0.1, 1.5)):
        self.p = p
        self.length = length
        self.font = font
        self.text_scale = text_scale
        self.max_occupancy = max_occupancy

    def _overlay_once(self, img, length=(10, 25), font=cv2.FONT_HERSHEY_PLAIN, text_scale=(0.1, 1.5)):
        h, w, c = img.shape
        length = np.random.randint(*length)
        text_scale = np.random.uniform(*text_scale)
        text = ''.join(random.choice(ascii_letters) for _ in range(length))
        color = np.random.randint(0, 255, c).tolist()
        pos = (random.randint(0, w), random.randint(0, h))
        img = cv2.putText(img, text, pos, font, text_scale, color)
        return img

    def __call__(self, img):
        assert img.dtype == np.uint8
        if random.random() < self.p:
            for _ in range(random.randint(0, self.max_occupancy)):
                img = self._overlay_once(img)
        return img
