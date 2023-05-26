"""Random flip augmentation."""

from typing import Optional, Tuple, Union

import numpy as np


def random_flip(
    img: np.ndarray, label: Optional[np.ndarray] = None, axis_for_flip: int = 0
) -> Union[Tuple[np.ndarray, None], Tuple[np.ndarray, ...]]:
    """Apply random flip to the input image.

    Args:
        img (np.ndarray): Input image to be flipped.
        label (np.ndarray, optional): Label segmentation map to be flipped. Defaults to
            None.
        axis_for_flip (int, optional): Axis along which the flip is applied. Defaults
            to 0.

    Returns:
        np.ndarray: Flipped image.
    """
    axes = [0, 1, 2]

    img = flip_axis(img, axes[axis_for_flip])
    img = np.squeeze(img)

    if label is None:
        return img, label
    else:
        y = flip_axis(label, axes[axis_for_flip])
        y = np.squeeze(y)
        return img, y


def flip_axis(img: np.ndarray, axis: int) -> np.ndarray:
    """Flip the input image along the specified axis.

    Args:
        img (np.ndarray): Input image to be flipped.
        axis (int): Axis along which the flip is applied.

    Returns:
        np.ndarray: Flipped image.
    """
    img = np.asarray(img).swapaxes(axis, 0)
    img = img[::-1, ...]
    img = img.swapaxes(0, axis)
    return img


class RandomFlip(object):
    def __init__(self):
        self.axis_for_flip = np.random.randint(0, 3)

    def __call__(
        self, img_numpy: np.ndarray, label: Optional[np.ndarray] = None
    ) -> Union[Tuple[np.ndarray, None], Tuple[np.ndarray, ...]]:
        """
        Apply random flip to the input image and label.

        Args:
            img_numpy (np.ndarray): Image to be flipped.
            label (np.ndarray, optional): Label segmentation map to be flipped. Defaults
                to None.

        Returns:
            Union[Tuple[np.ndarray, None], Tuple[np.ndarray, ...]]: Flipped image and
                label.
        """
        return random_flip(img_numpy, label, self.axis_for_flip)
