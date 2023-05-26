"""Elastic deformation of images.

 As described in
 Simard, Steinkraus and Platt, "Best Practices for
 Convolutional Neural Networks applied to Visual
 Document Analysis", in
 Proc. of the International Conference on Document Analysis and
 Recognition, 2003.

 Modified from:
 https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
 https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62

 Modified to take 3D inputs
 Deforms both the image and corresponding label file
 Label volumes are interpolated via nearest neighbour 
 """
from typing import Optional, Tuple, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.filters import gaussian_filter


def elastic_transform_3d(
    img: np.ndarray,
    labels: Optional[np.ndarray] = None,
    alpha: float = 1,
    sigma: float = 20,
    c_val: float = 0.0,
    method: str = "linear",
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Apply 3D elastic transformation to the input image.

    Args:
        img (np.ndarray): 3D medical image.
        labels (np.ndarray, optional): 3D medical image labels. Defaults to None.
        alpha (float, optional): Scaling factor of the Gaussian filter. Defaults to 1.
        sigma (float, optional): Standard deviation of the random Gaussian filter.
            Defaults to 20.
        c_val (float, optional): Fill value. Defaults to 0.0.
        method (str, optional): Interpolation method.
            Supported methods: ("linear", "nearest"). Defaults to "linear".

    Returns:
        np.ndarray: Deformed image and/or label.
    """
    assert img.ndim == 3, "Wrong img shape, provide 3D img"
    if labels is not None:
        assert img.shape == labels.shape, "Shapes of img and label do not match!"
    shape = img.shape

    # Define 3D coordinate system
    coords = np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2])

    # Interpolated img
    im_intrps = RegularGridInterpolator(
        coords, img, method=method, bounds_error=False, fill_value=c_val
    )

    # Get random elastic deformations
    dx = (
        gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0.0
        )
        * alpha
    )
    dy = (
        gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0.0
        )
        * alpha
    )
    dz = (
        gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0.0
        )
        * alpha
    )

    # Define sample points
    x, y, z = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    indices = (
        np.reshape(x + dx, (-1, 1)),
        np.reshape(y + dy, (-1, 1)),
        np.reshape(z + dz, (-1, 1)),
    )

    # Interpolate 3D image
    img = im_intrps(indices).reshape(shape)

    # Interpolate labels
    if labels is not None:
        lab_intrp = RegularGridInterpolator(
            coords, labels, method="nearest", bounds_error=False, fill_value=0
        )

        labels = lab_intrp(indices).reshape(shape).astype(labels.dtype)
        return img, labels

    return img


class ElasticTransform(object):
    """Apply elastic transformation to the input image.

    Args:
        alpha (float, optional): Scaling factor of the Gaussian filter.
        sigma (float, optional): Standard deviation of the random Gaussian filter.
        c_val (float, optional): Fill value. Defaults to 0.0.
        method (str, optional): Interpolation method. Supported methods:
            ("linear", "nearest").
    """

    def __init__(
        self,
        alpha: float = 1,
        sigma: float = 20,
        c_val: float = 0.0,
        method: str = "linear",
    ):
        self.alpha = alpha
        self.sigma = sigma
        self.c_val = c_val
        self.method = method

    def __call__(
        self, img: np.ndarray, label: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Apply the elastic transformation to the input image.

        Args:
            img (np.ndarray): Input image to be transformed.
            label (np.ndarray, optional): Corresponding label image. Defaults to None.

        Returns:
            np.ndarray or tuple: Transformed image if label is None, or tuple of
                transformed image and label.
        """
        return elastic_transform_3d(
            img, label, self.alpha, self.sigma, self.c_val, self.method
        )
