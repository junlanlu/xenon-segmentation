"""Random zoom augmentation module."""
from typing import Optional, Tuple

import numpy as np
import scipy.ndimage as ndimage


def random_zoom(
    img_numpy: np.ndarray, min_percentage: float = 0.8, max_percentage: float = 1.1
) -> np.ndarray:
    """
    Apply a random zoom transformation to the input image.

    Args:
        img_numpy (np.ndarray): Input image.
        min_percentage (float, optional): Minimum zoom percentage. Defaults to 0.8.
        max_percentage (float, optional): Maximum zoom percentage. Defaults to 1.1.

    Returns:
        np.ndarray: Zoomed image.
    """
    z = np.random.sample() * (max_percentage - min_percentage) + min_percentage
    zoom_matrix = np.array([[z, 0, 0, 0], [0, z, 0, 0], [0, 0, z, 0], [0, 0, 0, 1]])
    return ndimage.interpolation.affine_transform(img_numpy, zoom_matrix)


class RandomZoom(object):
    def __init__(self, min_percentage: float = 0.8, max_percentage: float = 1.1):
        self.min_percentage = min_percentage
        self.max_percentage = max_percentage

    def __call__(
        self, img_numpy: np.ndarray, label: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply a random zoom transformation to the input image and label.

        Args:
            img_numpy (np.ndarray): Input image.
            label (np.ndarray, optional): Label segmentation map. Defaults to None.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Zoomed image and label.
        """
        img_numpy = random_zoom(img_numpy, self.min_percentage, self.max_percentage)
        if label is not None:
            label = random_zoom(label, self.min_percentage, self.max_percentage)
        return img_numpy, label
