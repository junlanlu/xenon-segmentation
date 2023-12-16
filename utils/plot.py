"""Plotting functions for the project."""

import os
import pdb
import sys
from typing import Dict, List, Optional, Tuple

sys.path.append("..")
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torchio as tio

from utils import constants, io_utils


def map_grey_to_rgb(image: np.ndarray, cmap: Dict[int, np.ndarray]) -> np.ndarray:
    """Map a greyscale image to a RGB image using a colormap.

    Args:
        image (np.ndarray): greyscale image of shape (x, y, z)
        cmap (Dict[int, np.ndarray]): colormap mapping integers to RGB values.
    Returns:
        RGB image of shape (x, y, z, 3)
    """
    rgb_image = np.zeros((image.shape[0], image.shape[1], image.shape[2], 3))
    for key in cmap.keys():
        rgb_image[image == key] = cmap[key]
    return rgb_image


def get_biggest_island_indices(arr: np.ndarray) -> Tuple[int, int]:
    """Get the start and stop indices of the biggest island in the array.

    Args:
        arr (np.ndarray): binary array of 0s and 1s.
    Returns:
        Tuple of start and stop indices of the biggest island.
    """
    # intitialize count
    cur_count = 0
    cur_start = 0

    max_count = 0
    pre_state = 0

    index_start = 0
    index_end = 0

    for i in range(0, np.size(arr)):
        if arr[i] == 0:
            cur_count = 0
            if (pre_state == 1) & (cur_start == index_start):
                index_end = i - 1
            pre_state = 0

        else:
            if pre_state == 0:
                cur_start = i
                pre_state = 1
            cur_count += 1
            if cur_count > max_count:
                max_count = cur_count
                index_start = cur_start

    return index_start, index_end


def get_plot_indices(image: np.ndarray, n_slices: int = 16) -> Tuple[int, int]:
    """Get the indices to plot the image.

    Args:
        image (np.ndarray): binary image.
        n_slices (int, optional): number of slices to plot. Defaults to 16.
    Returns:
        Tuple of start and interval indices.
    """
    sum_line = np.sum(np.sum(image, axis=0), axis=0)
    index_start, index_end = get_biggest_island_indices(sum_line > 300)
    flt_inter = (index_end - index_start) // n_slices

    # threshold to decide interval number
    if np.modf(flt_inter)[0] > 0.4:
        index_skip = np.ceil(flt_inter).astype(int)
    else:
        index_skip = np.floor(flt_inter).astype(int)

    return index_start, index_skip


def overlay_mask_on_image(
    image: np.ndarray, mask: np.ndarray, percentile_rescale: float = 100
) -> np.ndarray:
    """Overlay the border of a binary mask on a greyscale image in red.

    Args:
        image (np.ndarray): Greyscale image of shape (x, y, z)
        mask (np.ndarray): Binary mask of shape (x, y, z)
        percentile_rescale (float, optional): Percentile to rescale the image by.

    Returns:
        np.ndarray: Overlaid image of shape (x, y, z, 3)
    """
    image = image / np.percentile(image[mask.astype(bool)], percentile_rescale)
    image[image > 1] = 1

    def border_mask(mask: np.ndarray) -> np.ndarray:
        mask_dilated = np.zeros_like(mask)
        for i in range(mask.shape[2]):
            mask_dilated[:, :, i] = cv2.dilate(
                mask[:, :, i].astype(np.uint8), np.ones((3, 3)), iterations=1
            )
        return mask_dilated - mask

    border = border_mask(mask)

    image_out = np.zeros((image.shape[0], image.shape[1], image.shape[2], 3))
    for i in range(image.shape[2]):
        image_slice = np.repeat(image[:, :, i][:, :, np.newaxis], 3, axis=2)
        border_slice = border[:, :, i][:, :, np.newaxis]
        image_slice[border_slice[..., 0] == 1] = [1, 0, 0]
        image_out[:, :, i, :] = image_slice

    return image_out


def make_montage(image: np.ndarray, n_slices: int = 16) -> np.ndarray:
    """Make montage of the image.

    Makes montage of the image.
    Assumes the image is of shape (x, y, z, 3).

    Args:
        image (np.ndarray): image to make montage of.
        n_slices (int, optional): number of slices to plot. Defaults to 16.
    Returns:
        Montaged image array.
    """
    # get the shape of the image
    x, y, z, _ = image.shape
    # get the number of rows and columns
    n_rows = 1 if n_slices < 8 else 2
    n_cols = np.ceil(n_slices / n_rows).astype(int)
    # get the shape of the slices
    slice_shape = (x, y)
    # make the montage array
    montage = np.zeros((n_rows * slice_shape[0], n_cols * slice_shape[1], 3))
    # iterate over the slices
    for slice in range(n_slices):
        # get the row and column
        row = slice // n_cols
        col = slice % n_cols
        # get the slice
        slice = image[:, :, slice, :]
        # add to the montage
        montage[
            row * slice_shape[0] : (row + 1) * slice_shape[0],
            col * slice_shape[1] : (col + 1) * slice_shape[1],
            :,
        ] = slice
    return montage


def plot_montage_color(
    image: np.ndarray,
    path: str,
    index_start: int,
    index_skip: int = 1,
    n_slices: int = 16,
):
    """Plot a montage of the image in RGB.

    Will make a montage of default (2x8) of the image in RGB and save it to the path.
    Assumes the image is of shape (x, y, z) where there are at least n_slices.
    Otherwise, will plot all slices.

    Args:
        image (np.ndarray): RGB image to plot of shape (x, y, z, 3).
        path (str): path to save the image.
        index_start (int): index to start plotting from.
        index_skip (int, optional): indices to skip. Defaults to 1.
        n_slices (int, optional): number of slices to plot. Defaults to 16.
    """
    # plot the montage
    index_end = index_start + index_skip * n_slices
    montage = make_montage(
        image[:, :, index_start:index_end:index_skip, :], n_slices=n_slices
    )
    plt.figure()
    plt.imshow(montage, cmap="gray")
    plt.axis("off")
    plt.savefig(path, transparent=True, bbox_inches="tight", pad_inches=-0.05, dpi=300)
    plt.clf()
    plt.close()


def plot_slice_color(image: np.ndarray, path: str, index: int):
    """Plot a single slice of the image in RGB.

    Assumes the image is of shape (x, y, z, 3) where there are at least n_slices.
    Otherwise, will plot all slices.

    Args:
        image (np.ndarray): RGB image to plot of shape (x, y, z, 3).
        path (str): path to save the image.
        index (int): index to plot.
    """
    # plot the montage
    plt.figure()
    plt.imshow(image[:, :, index, :], cmap="gray")
    plt.axis("off")
    plt.savefig(path, transparent=True, bbox_inches="tight", pad_inches=-0.05, dpi=300)
    plt.clf()
    plt.close()


def plot_slice_grey(
    image: np.ndarray, path: str, index: int, percentile_rescale: float = 100
):
    """Plot a montage of the image in RGB.

    Assumes the image is of shape (x, y, z) where there are at least n_slices.
    Otherwise, will plot all slices.

    Args:
        image (np.ndarray): gray scale image to plot of shape (x, y, z)
        path (str): path to save the image.
        index (int): index to plot.
        percentile_rescale (float, optional): percentile to rescale the image by.
    """
    # divide by the maximum value
    image = image / np.percentile(image, percentile_rescale)
    image[image > 1] = 1
    # stack the image to make it 4D (x, y, z, 3)
    image = np.stack((image, image, image), axis=-1)
    # plot the montage
    plt.figure()
    plt.imshow(image[:, :, index, :], cmap="gray")
    plt.axis("off")
    plt.savefig(path, transparent=True, bbox_inches="tight", pad_inches=-0.05, dpi=300)
    plt.clf()
    plt.close()


def plot_subject(subject_dict: dict):
    """
    Plot the 'gas', 'mask', and 'proton' volumes from a subject dictionary.

    Args:
        subject (dict): Dictionary containing the volumes with keys
            ['gas', 'mask', 'proton', 'name'].

    Raises:
        ValueError: If the shapes of the volumes do not match.

    """
    gas_volume = subject_dict["gas"][tio.DATA]
    mask_volume = subject_dict["mask"][tio.DATA]
    proton_volume = subject_dict["proton"][tio.DATA]

    if (
        gas_volume.shape[0] != mask_volume.shape[0]
        or gas_volume.shape[0] != proton_volume.shape[0]
    ):
        raise ValueError("Shapes of volumes do not match.")

    num_volumes = gas_volume.shape[0]
    save_dir = "tmp"

    for i in range(num_volumes):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(
            gas_volume[i, 0, :, :, gas_volume.shape[-1] // 2].cpu().numpy(), cmap="gray"
        )
        axs[0].set_title("Gas Volume")
        axs[1].imshow(
            mask_volume[i, 0, :, :, mask_volume.shape[-1] // 2].cpu().numpy(),
            cmap="gray",
        )
        axs[1].set_title("Mask Volume")
        axs[2].imshow(
            proton_volume[i, 0, :, :, proton_volume.shape[-1] // 2].cpu().numpy(),
            cmap="gray",
        )
        axs[2].set_title("Proton Volume")

        plt.suptitle(f"Subject: {subject_dict['name'][0]}, Volume: {i+1}")
        save_path = os.path.join(
            save_dir, f"subject_{subject_dict['name'][0]}_volume_{i+1}.png"
        )
        plt.savefig(save_path)
        plt.close(fig)
