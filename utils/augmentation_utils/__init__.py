"""Utility functions and classes for image transformations and augmentation."""
import random
from typing import List

import numpy as np

from .elastic_deform import ElasticTransform
from .gaussian_noise import GaussianNoise
from .random_crop import RandomCropToLabels
from .random_flip import RandomFlip
from .random_rescale import RandomZoom
from .random_rotate import RandomRotation
from .random_shift import RandomShift

functions = [
    "elastic_deform",
    "random_crop",
    "random_flip",
    "random_rescale",
    "random_rotate",
    "random_shift",
]


class RandomChoice(object):
    """Choose a random transform from a list and apply it.

    Args:
        transforms (list, optional): List of transforms to apply. Defaults to [].
        p (float, optional): Probability of applying the chosen transform.
        Defaults to 0.5.
    """

    def __init__(self, transforms: list = [], p: float = 0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, img_tensors: List[np.ndarray], label: np.ndarray):
        """Apply a composition of transforms to the input images and label.

        Args:
            img_tensors (list): List of input image tensors.
            label: Label to be transformed.

        Returns:
            tuple: Transformed image tensors and label.
        """
        augment = np.random.random(1) < self.p
        if not augment:
            return img_tensors, label
        t = random.choice(self.transforms)

        for i in range(len(img_tensors)):
            if i == (len(img_tensors) - 1):
                ### do only once the augmentation to the label
                img_tensors[i], label = t(img_tensors[i], label)
            else:
                img_tensors[i], _ = t(img_tensors[i], label)
        return img_tensors, label


class ComposeTransforms(object):
    """Composes several transforms together.

    Args:
        transforms (list, optional): List of transforms to compose. Defaults to [].
        p (float, optional): Probability of applying the composed transforms.
        Defaults to 0.9.
    """

    def __init__(self, transforms=[], p=0.9):
        self.transforms = transforms
        self.p = p

    def __call__(self, img_tensors, label):
        """Apply a composition of transforms to the input images and label.

        Args:
            img_tensors (List[np.ndarray]): List of input image tensors.
            label (np.ndarray): Label to be transformed.

        Returns:
            Tuple[List[np.ndarray], np.ndarray]: Transformed image tensors and label.
        """
        augment = np.random.random(1) < self.p
        if not augment:
            return img_tensors, label

        for i in range(len(img_tensors)):
            for t in self.transforms:
                if i == (len(img_tensors) - 1):
                    ### do only once augmentation to the label
                    img_tensors[i], label = t(img_tensors[i], label)
                else:
                    img_tensors[i], _ = t(img_tensors[i], label)
        return img_tensors, label
