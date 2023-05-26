"""Gaussian noise augmentation."""
from typing import Optional, Tuple

import numpy as np


def random_noise(img: np.ndarray, mean: float = 0, std: float = 0.001) -> np.ndarray:
    """
    Apply random Gaussian noise to the input image.

    Args:
        img (np.ndarray): Input image to apply noise.
        mean (float, optional): Mean value of the Gaussian noise. Defaults to 0.
        std (float, optional): Standard deviation of the Gaussian noise.
            Defaults to 0.001.

    Returns:
        np.ndarray: Image with random Gaussian noise.
    """
    noise = np.random.normal(mean, std, img.shape)
    return img + noise


class GaussianNoise(object):
    def __init__(self, mean: float = 0, std: float = 0.001):
        self.mean = mean
        self.std = std

    def __call__(
        self, img: np.ndarray, label: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply Gaussian noise to the input image.

        Args:
            img(np.ndarray): Image to apply noise.
            label (np.ndarray, optional): Label segmentation map. Defaults to None.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Tuple containing the image with
                applied noise and the label.
        """
        return random_noise(img, self.mean, self.std), label
