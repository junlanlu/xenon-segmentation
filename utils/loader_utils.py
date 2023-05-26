"""Loader utilities."""

import logging
import pdb
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from utils import img_utils


def get_viz_set(
    *ls,
    dataset_name: str,
    test_subject: int = 0,
    save: bool = False,
    sub_vol_path: Optional[str] = None,
):
    """
    Returns total 3D input volumes (T1 and T2 or more) and segmentation maps.

    Args:
        *ls: Variable number of lists containing paths to the images.
        dataset_name (str): Name of the dataset.
        test_subject (int, optional): Index of the test subject. Defaults to 0.
        save (bool, optional): Whether to save sub-volumes as separate files.
            Defaults to False.
        sub_vol_path (str, optional): Path to save the sub-volumes.
            Required if `save` is True.

    Returns:
        torch.Tensor: Total 3D input volumes stacked along the 0th dimension.
    """
    modalities = len(ls)
    total_volumes = []

    for i, modality_paths in enumerate(ls):
        path_img = modality_paths[test_subject]
        img_tensor = img_utils.load_medical_image(path_img, viz3d=True)

        if i == modalities - 1:
            img_tensor = fix_seg_map(img_tensor, dataset=dataset_name)

        total_volumes.append(img_tensor)

    if save:
        total_subvolumes = total_volumes[0].shape[0]

        for i in range(total_subvolumes):
            filename = f"{sub_vol_path}id_{test_subject}_VIZ_{i}_modality_"

            for j in range(modalities):
                filename_j = f"{filename}{j}.npy"
                np.save(filename_j, total_volumes[j][i])
    else:
        return torch.stack(total_volumes, dim=0)


def fix_seg_map(segmentation_map, dataset="iseg2019"):
    if dataset == "iseg2019":
        label_values = [0, 10, 150, 250]
        for c, j in enumerate(label_values):
            segmentation_map[segmentation_map == j] = c
    else:
        raise ValueError("Dataset not supported.")
    return segmentation_map


def create_sub_volumes(
    *ls,
    dataset_name: str,
    mode: str,
    samples: int,
    full_vol_dim: tuple,
    crop_size: tuple,
    sub_vol_path: str,
    normalization: str = "max_min",
    th_percent: float = 0.1,
) -> list:
    """
    Create sub-volumes from the input modalities and segmentation maps.

    Args:
        ls (Tuple): List of modality paths, where the last path is the segmentation map.
        dataset_name (str): Name of the dataset used.
        mode (str): Training mode ('train' or 'val').
        samples (int): Number of sub-volume samples to generate.
        full_vol_dim (tuple): Full image size.
        crop_size (tuple): Train volume size.
        sub_vol_path (str): Path for the particular patient.
        normalization (str, optional): Normalization method. Defaults to "max_min".
        th_percent (float, optional): The percentage of the cropped dimension that
            corresponds to non-zero labels. Defaults to 0.1.

    Returns:
        list: List of saved sub-volume paths.
    """
    total = len(ls[0])
    assert total != 0, "Problem reading data. Check the data paths."
    modalities = len(ls)
    saved_paths = []

    logging.info(
        "Mode: {} - Subvolume samples to generate: {} - Volumes: {}".format(
            mode, samples, total
        )
    )

    for i in range(samples):
        random_index = np.random.randint(total)
        sample_paths = []

        tensor_images = []
        for j in range(modalities):
            sample_paths.append(ls[j][random_index])

        while True:
            label_path = sample_paths[-1]
            crop = find_random_crop_dim(full_vol_dim, crop_size)
            full_segmentation_map = img_utils.load_medical_image(
                label_path, viz3d=True, is_label=True, crop_size=crop_size, crop=crop
            )
            full_segmentation_map = fix_seg_map(full_segmentation_map, dataset_name)

            if find_non_zero_labels_mask(
                full_segmentation_map, th_percent, crop_size, crop
            ):
                segmentation_map = img_utils.load_medical_image(
                    label_path, is_label=True, crop_size=crop_size, crop=crop
                )
                segmentation_map = fix_seg_map(segmentation_map, dataset_name)

                for j in range(modalities - 1):
                    img_tensor = img_utils.load_medical_image(
                        sample_paths[j],
                        is_label=False,
                        normalization=normalization,
                        crop_size=crop_size,
                        crop=crop,
                    )
                    tensor_images.append(img_tensor)

                break

        filename = (
            sub_vol_path + "id_" + str(random_index) + "_s_" + str(i) + "_modality_"
        )
        list_saved_paths = []

        for j in range(modalities - 1):
            f_t1 = filename + str(j) + ".npy"
            list_saved_paths.append(f_t1)
            np.save(f_t1, tensor_images[j])

        f_seg = filename + "seg.npy"
        np.save(f_seg, segmentation_map)
        list_saved_paths.append(f_seg)
        saved_paths.append(tuple(list_saved_paths))

    return saved_paths


def get_all_sub_volumes(
    *ls,
    dataset_name,
    mode,
    samples,
    full_vol_dim,
    crop_size,
    sub_vol_path,
    normalization="max_min",
):
    # TODO
    # 1.) gia ola tas subject fortwnwn image kai target
    # 2.) call generate_non_overlapping_volumes gia na kanw to image kai target sub_volumnes patches
    # 3.) apothikeuw tensors
    total = len(ls[0])
    assert total != 0, "Problem reading data. Check the data paths."
    modalities = len(ls)
    list = []

    for vol_id in range(total):
        tensor_images = []
        for modality_id in range(modalities - 1):
            img_tensor = img_utils.medical_image_transform(
                img_utils.load_medical_image(ls[modality_id][vol_id]),
                normalization=normalization,
            )

            img_tensor = generate_padded_subvolumes(img_tensor, kernel_dim=crop_size)

            tensor_images.append(img_tensor)
        segmentation_map = img_utils.medical_image_transform(
            img_utils.load_medical_image(
                ls[modalities - 1][vol_id], viz3d=True, is_label=True
            )
        )
        segmentation_map = generate_padded_subvolumes(
            segmentation_map, kernel_dim=crop_size
        )

        filename = (
            sub_vol_path + "id_" + str(vol_id) + "_s_" + str(modality_id) + "_modality_"
        )

        list_saved_paths = []
        # print(len(tensor_images[0]))
        for k in range(len(tensor_images[0])):
            for j in range(modalities - 1):
                f_t1 = filename + str(j) + "_sample_{}".format(str(k).zfill(8)) + ".npy"
                list_saved_paths.append(f_t1)
                # print(f_t1,tensor_images[j][k].shape)
                np.save(f_t1, tensor_images[j])

            f_seg = filename + "seg_sample_{}".format(str(k).zfill(8)) + ".npy"
            # print(f_seg)
            np.save(f_seg, segmentation_map)
            list_saved_paths.append(f_seg)
            list.append(tuple(list_saved_paths))

    return list


def roundup(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Round up x to the nearest multiple of y.

    Args:
        x (torch.Tensor): The number to round up.
        y (torch.Tensor): The multiple to round up to.
    """
    return ((x + y - 1) // y) * y


def generate_padded_subvolumes(full_volume, kernel_dim=(32, 32, 32)):
    x = full_volume.detach()

    modalities, D, H, W = x.shape
    kc, kh, kw = kernel_dim
    dc, dh, dw = kernel_dim  # stride
    # Pad to multiples of kernel_dim
    a = (
        (roundup(W, kw) - W) // 2 + W % 2,
        (roundup(W, kw) - W) // 2,
        (roundup(H, kh) - H) // 2 + H % 2,
        (roundup(H, kh) - H) // 2,
        (roundup(D, kc) - D) // 2 + D % 2,
        (roundup(D, kc) - D) // 2,
    )
    # print('padding ', a)
    x = F.pad(x, a)
    # print('padded shape ', x.shape)
    assert x.size(3) % kw == 0
    assert x.size(2) % kh == 0
    assert x.size(1) % kc == 0
    patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    unfold_shape = list(patches.size())

    patches = patches.contiguous().view(-1, modalities, kc, kh, kw)

    return patches


def find_random_crop_dim(full_vol_dim: tuple, crop_size: tuple) -> tuple:
    """
    Find random crop dimensions based on the full volume dimensions and crop size.

    Args:
        full_vol_dim (tuple): Full volume dimensions (depth, height, width).
        crop_size (tuple): Desired crop size (depth, height, width).

    Returns:
        tuple: Random crop dimensions (slices, w_crop, h_crop).
    """
    assert full_vol_dim[0] >= crop_size[0], "Crop size is too big."
    assert full_vol_dim[1] >= crop_size[1], "Crop size is too big."
    assert full_vol_dim[2] >= crop_size[2], "Crop size is too big."

    if full_vol_dim[0] == crop_size[0]:
        slices = crop_size[0]
    else:
        slices = np.random.randint(full_vol_dim[0] - crop_size[0])

    if full_vol_dim[1] == crop_size[1]:
        w_crop = crop_size[1]
    else:
        w_crop = np.random.randint(full_vol_dim[1] - crop_size[1])

    if full_vol_dim[2] == crop_size[2]:
        h_crop = crop_size[2]
    else:
        h_crop = np.random.randint(full_vol_dim[2] - crop_size[2])

    return (slices, w_crop, h_crop)


def find3Dlabel_boundaries(segmentation_map):
    target_indexs = np.where(segmentation_map > 0)
    maxs = np.max(np.array(target_indexs), axis=1)
    mins = np.min(np.array(target_indexs), axis=1)
    diff = maxs - mins
    labels_voxels = diff[0] * diff[1] * diff[2]
    return labels_voxels


def find_non_zero_labels_mask(
    segmentation_map: torch.Tensor, th_percent: float, crop_size: tuple, crop: tuple
) -> bool:
    """
    Check if the cropped segmentation map contains a sufficient percentage of non-zero
    labels.

    Args:
        segmentation_map (torch.Tensor): Segmentation map array.
        th_percent (float): Threshold percentage for non-zero labels.
        crop_size (tuple): Crop size (depth, height, width).
        crop (tuple): Crop dimensions (slices, w_crop, h_crop).

    Returns:
        bool: True if the label percentage is greater than or equal to the threshold,
            False otherwise.
    """
    d1, d2, d3 = segmentation_map.shape
    segmentation_map[segmentation_map > 0] = 1
    total_voxel_labels = segmentation_map.sum()

    cropped_segm_map = img_utils.crop_img(segmentation_map, crop_size, crop)
    crop_voxel_labels = cropped_segm_map.sum()

    label_percentage = crop_voxel_labels / total_voxel_labels

    if label_percentage >= th_percent:
        return True
    else:
        return False
