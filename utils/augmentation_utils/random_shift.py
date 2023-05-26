"""Random shift augmentation."""
from typing import Optional, Tuple, Union

import numpy as np
import scipy.ndimage as ndimage


def transform_matrix_offset_center_3d(
    matrix: np.ndarray, x: int, y: int, z: int
) -> np.ndarray:
    """
    Apply a transformation matrix to shift the matrix's center.

    Args:
        matrix (np.ndarray): Input matrix to be transformed.
        x (int): Offset along the x-axis.
        y (int): Offset along the y-axis.
        z (int): Offset along the z-axis.

    Returns:
        np.ndarray: Transformed matrix.
    """
    offset_matrix = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    return ndimage.interpolation.affine_transform(matrix, offset_matrix)


def random_shift(img: np.ndarray, max_percentage: float = 0.2) -> np.ndarray:
    """
    Apply a random shift to the input image.

    Args:
        img (np.ndarray): Input image to be shifted.
        max_percentage (float, optional): Maximum percentage of shift along each
            dimension. Defaults to 0.2.

    Returns:
        np.ndarray: Shifted image.
    """
    dim1, dim2, dim3 = img.shape
    m1, m2, m3 = (
        int(dim1 * max_percentage / 2),
        int(dim2 * max_percentage / 2),
        int(dim3 * max_percentage / 2),
    )
    d1 = np.random.randint(-m1, m1)
    d2 = np.random.randint(-m2, m2)
    d3 = np.random.randint(-m3, m3)
    return transform_matrix_offset_center_3d(img, d1, d2, d3)


class RandomShift(object):
    def __init__(self, max_percentage: float = 0.2):
        self.max_percentage = max_percentage

    def __call__(
        self, img: np.ndarray, label: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply a random shift to the input image and label.

        Args:
            img (np.ndarray): Image to be shifted.
            label (np.ndarray, optional): Label segmentation map to be shifted.
                Defaults to None.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Shifted image and label.
        """
        img = random_shift(img, self.max_percentage)
        if label is not None:
            label = random_shift(label, self.max_percentage)
        return img, label
