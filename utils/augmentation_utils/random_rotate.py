"""Random rotation augmentation module."""

from typing import Optional, Tuple

import numpy as np
import scipy.ndimage as ndimage


def random_rotate3D(
    img_numpy: np.ndarray, min_angle: int, max_angle: int
) -> np.ndarray:
    """
    Apply a random 3D rotation to the input array.

    Args:
        img_numpy (np.ndarray): 3D numpy array to be rotated.
        min_angle (int): Minimum rotation angle in degrees.
        max_angle (int): Maximum rotation angle in degrees.

    Returns:
        np.ndarray: Rotated 3D array.
    """
    assert img_numpy.ndim == 3, "Provide a 3D numpy array"
    assert min_angle < max_angle, "Min angle should be less than max angle"
    assert min_angle > -360 or max_angle < 360
    all_axes = [(1, 0), (1, 2), (0, 2)]
    angle = np.random.randint(low=min_angle, high=max_angle + 1)
    axes_random_id = np.random.randint(low=0, high=len(all_axes))
    axes = all_axes[axes_random_id]
    return ndimage.rotate(img_numpy, angle, axes=axes)


class RandomRotation(object):
    def __init__(self, min_angle: int = -10, max_angle: int = 10):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(
        self, img_numpy: np.ndarray, label: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply a random rotation to the input image and label.

        Args:
            img_numpy (np.ndarray): Image to be rotated.
            label (np.ndarray, optional): Label segmentation map to be rotated.
                Defaults to None.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Rotated image and label.
        """
        img_numpy = random_rotate3D(img_numpy, self.min_angle, self.max_angle)
        if label is not None:
            label = random_rotate3D(label, self.min_angle, self.max_angle)
        return img_numpy, label
