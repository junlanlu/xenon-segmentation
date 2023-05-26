"""Preprocessing functions."""

from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_to_output
from PIL import Image
from scipy import ndimage


def load_medical_image(
    path: str,
    is_label: bool = False,
    resample: Optional[Tuple[float, float, float]] = None,
    viz3d: bool = False,
    to_canonical: bool = False,
    rescale: Optional[Tuple[float, float]] = None,
    normalization: str = "full_volume_mean",
    clip_intensity: bool = True,
    crop_size: Tuple[int, int, int] = (0, 0, 0),
    crop: Tuple[int, int, int] = (0, 0, 0),
) -> torch.Tensor:
    """
    Loads a medical image from the given path.

    Args:
        path (str): Path to the medical image file.
        type (str, optional): Type of the image. Defaults to None.
        resample (Tuple[float, float, float], optional): Resampling voxel sizes.
            Defaults to None.
        viz3d (bool, optional): Whether to return a 3D visualization tensor.
            Defaults to False.
        to_canonical (bool, optional): Whether to transform to the canonical
            orientation. Defaults to False.
        rescale (Tuple[float, float], optional): Rescaling factors. Defaults to None.
        normalization (str, optional): Intensity normalization method.
            Defaults to "full_volume_mean".
        clip_intensity (bool, optional): Whether to clip intensity outliers.
            Defaults to True.
        crop_size (Tuple[int, int, int], optional): Size of the crop.
            Defaults to (0, 0, 0).
        crop (Tuple[int, int, int], optional): Crop indices. Defaults to (0, 0, 0).

    Returns:
        torch.Tensor: Loaded medical image tensor.
    """
    img_nii = nib.load(path)

    if to_canonical:
        img_nii = nib.as_closest_canonical(img_nii)
    if resample is not None:
        img_nii = resample_to_output(img_nii, voxel_sizes=resample)

    img_np = np.squeeze(img_nii.get_fdata(dtype=np.float32))

    if viz3d:
        return torch.from_numpy(img_np)

    # 1. Intensity outlier clipping
    if clip_intensity and not is_label:
        img_np = percentile_clip(img_np)

    # 2. Rescale to specified output shape
    if rescale is not None:
        rescale_data_volume(img_np, rescale)

    # 3. Intensity normalization
    img_tensor = torch.from_numpy(img_np)

    mean, std, max_val, min_val = 0.0, 1.0, 1.0, 0.0
    if not is_label:
        mean, std = img_tensor.mean(), img_tensor.std()
        max_val, min_val = img_tensor.max(), img_tensor.min()

    if not is_label:
        img_tensor = normalize(
            img_tensor,
            method=normalization,
            norm_values=(mean, std, max_val, min_val),
        )

    img_tensor = crop_img(img_tensor, crop_size, crop)
    return img_tensor


def medical_image_transform(
    img: torch.Tensor,
    is_label: bool = False,
    normalization: str = "full_volume_mean",
    norm_values: tuple = (0.0, 1.0, 1.0, 0.0),
) -> torch.Tensor:
    """
    Transform a medical image tensor.

    Args:
        img (torch.Tensor): Input image tensor.
        is_label (bool, optional): Indicates if the image is a label. Defaults to False.
        normalization (str, optional): Normalization method. Choices: "full_volume_mean",
            "max", "mean", "brats", "max_min", None. Defaults to "full_volume_mean".
        norm_values (tuple, optional): Normalization values (MEAN, STD, MAX, MIN).
            Defaults to (0.0, 1.0, 1.0, 0.0).

    Returns:
        torch.Tensor: Transformed image tensor.
    """
    MEAN, STD, MAX, MIN = norm_values
    if not is_label:
        MEAN, STD = img.mean(), img.std()
        MAX, MIN = img.max(), img.min()

    if not is_label:
        img = normalize(img, method=normalization, norm_values=(MEAN, STD, MAX, MIN))

    return img


def crop_img(img: torch.Tensor, crop_size: tuple, crop: tuple) -> torch.Tensor:
    """
    Crop an image tensor based on the specified crop size and crop coordinates.

    Args:
        img (torch.Tensor): Input image tensor.
        crop_size (tuple): Size of the crop (dim1, dim2, dim3).
        crop (tuple): Crop coordinates (slices_crop, w_crop, h_crop).

    Returns:
        torch.Tensor: Cropped image tensor.
    """
    if crop_size[0] == 0:
        return img
    slices_crop, w_crop, h_crop = crop
    dim1, dim2, dim3 = crop_size
    inp_img_dim = img.dim()
    assert inp_img_dim >= 3
    if img.dim() == 3:
        full_dim1, full_dim2, full_dim3 = img.shape
    elif img.dim() == 4:
        _, full_dim1, full_dim2, full_dim3 = img.shape
        img = img[0, ...]

    if full_dim1 == dim1:
        img = img[:, w_crop : w_crop + dim2, h_crop : h_crop + dim3]
    elif full_dim2 == dim2:
        img = img[slices_crop : slices_crop + dim1, :, h_crop : h_crop + dim3]
    elif full_dim3 == dim3:
        img = img[slices_crop : slices_crop + dim1, w_crop : w_crop + dim2, :]
    else:
        img = img[
            slices_crop : slices_crop + dim1,
            w_crop : w_crop + dim2,
            h_crop : h_crop + dim3,
        ]

    if inp_img_dim == 4:
        return img.unsqueeze(0)

    return img


def load_affine_matrix(path: str) -> np.ndarray:
    """
    Reads a path to a NIfTI file and returns the affine matrix as a numpy array (4x4).

    Args:
        path (str): Path to the NIfTI file.

    Returns:
        np.ndarray: Affine matrix (4x4) as a numpy array.
    """
    img = nib.load(path)
    return img.affine


def rescale_data_volume(img_numpy: np.ndarray, out_dim: tuple) -> np.ndarray:
    """
    Rescale the 3D numpy array to the specified dimensions.

    Args:
        img_numpy (np.ndarray): Input 3D numpy array.
        out_dim (tuple): New dimensions (depth, height, width).

    Returns:
        np.ndarray: Rescaled 3D numpy array.
    """
    depth, height, width = img_numpy.shape
    scale = [
        out_dim[0] * 1.0 / depth,
        out_dim[1] * 1.0 / height,
        out_dim[2] * 1.0 / width,
    ]
    return ndimage.interpolation.zoom(img_numpy, scale, order=0)


def normalize(
    img: torch.Tensor,
    method: str = "full_volume_mean",
    norm_values: tuple = (0, 1, 1, 0),
) -> torch.Tensor:
    """
    Normalizes an input image tensor using the specified method.

    Args:
        img (torch.Tensor): Input image tensor.
        method (str): Normalization method. Choices: "mean", "max", "brats",
            "full_volume_mean", "max_min", None.
        norm_values (tuple): Normalization values used in the "brats" method.

    Returns:
        torch.Tensor: Normalized image tensor.
    """
    if method == "mean":
        mask = img.ne(0.0)
        desired = img[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img = (img - mean_val) / std_val
    elif method == "max":
        max_val, _ = torch.max(img)
        img = img / max_val
    elif method == "brats":
        normalized_tensor = (img.clone() - norm_values[0]) / norm_values[1]
        final_tensor = torch.where(img == 0.0, img, normalized_tensor)
        final_tensor = (
            100.0
            * (
                (final_tensor.clone() - norm_values[3])
                / (norm_values[2] - norm_values[3])
            )
            + 10.0
        )
        x = torch.where(img == 0.0, img, final_tensor)
        return x
    elif method == "full_volume_mean":
        img = (img.clone() - norm_values[0]) / norm_values[1]
    elif method == "max_min":
        img = (img - norm_values[3]) / (norm_values[2] - norm_values[3])
    elif method is None:
        img = img
    else:
        raise ValueError("Normalization method not recognized.")
    return img


def one_hot_to_label(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a one-hot encoded image tensor to a label map.

    Args:
        input_tensor (torch.Tensor): An NCHWD tensor where N is the batch size,
                                     C is the number of channels (classes),
                                     H is the height, W is the width,
                                     and D is the depth.

    Returns:
        torch.Tensor: An NHWD tensor where each value is the index of the class
                      that had the highest value in the input tensor.
    """
    return torch.argmax(input_tensor, dim=1).float()


def clip_range(img: np.ndarray) -> np.ndarray:
    """Clips the range of an image by cutting off outliers related to detected black
        areas (e.g., air).

    Args:
        img (np.ndarray): Input image.

    Returns:
        np.ndarray: Clipped image.
    """
    zero_value = (
        img[0, 0, 0]
        + img[-1, 0, 0]
        + img[0, -1, 0]
        + img[0, 0, -1]
        + img[-1, -1, -1]
        + img[-1, -1, 0]
        + img[0, -1, -1]
        + img[-1, 0, -1]
    ) / 8.0
    non_zeros_idx = np.where(img >= zero_value)
    [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
    [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
    clipped_img = img[min_z:max_z, min_h:max_h, min_w:max_w]
    return clipped_img


def percentile_clip(
    img: np.ndarray, min_val: float = 0.1, max_val: float = 99.8
) -> np.ndarray:
    """
    Clips the intensity range of an image based on percentiles.

    Args:
        img (np.ndarray): Input image.
        min_val (float, optional): Minimum percentile value.
            Should be in the range [0, 100]. Defaults to 0.1.
        max_val (float, optional): Maximum percentile value.
            Should be in the range [0, 100]. Defaults to 99.8.

    Returns:
        np.ndarray: Intensity normalized image.
    """
    low = np.percentile(img, min_val)
    high = np.percentile(img, max_val)

    img[img < low] = low
    img[img > high] = high
    return img
