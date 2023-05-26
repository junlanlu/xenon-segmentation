"""Utility functions for input/output."""
import os
import pickle
import shutil
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchio as tio

from config import base_config


def import_nii(path: str) -> np.ndarray:
    """Import image as np.ndarray.

    Args:
        path: str file path of nifti file
    Returns:
        np.ndarray loaded from nifti file
    """
    return nib.load(path).get_fdata()


def make_dirs(path: str):
    """Create a directory if it does not exist.

    Args:
        str: Path to the directory.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.makedirs(path)


def save_list_to_txt(file_path: str, data: list) -> None:
    """
    Save a list to a text file.

    Args:
        file_path (str): Path to the output text file.
        data (list): List of elements to be saved.

    Returns:
        None
    """
    with open(file_path, "w") as file:
        for item in data:
            file.write(str(item) + "\n")


def prepare_input(
    input_subject: tio.Subject,
    inModalities: int = -1,
    inChannels: int = -1,
    use_cuda: bool = False,
    config: Optional[base_config.Config] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare input tensors for training or evaluation.

    Args:
        input_subject (tuple): Torch IO subject
        inModalities (int, optional): Number of modalities. Defaults to -1.
        inChannels (int, optional): Number of channels. Defaults to -1.
        use_cuda (bool, optional): Flag indicating whether to use CUDA. Defaults to False.
        config (base_config.Config, optional): Configuration object. Defaults to None.

    Returns:
        tuple: Tuple of input tensor of shape (n, c, h, w, d) and target tensor
            (n, h, w, d).
    """
    if config is not None:
        modalities = config.data.n_modalities
        channels = config.data.n_channels
        in_cuda = config.use_cuda
    else:
        modalities = inModalities
        channels = inChannels
        in_cuda = use_cuda

    if modalities == 2:
        if channels == 2:
            img_gas = input_subject["gas"][tio.DATA]
            img_proton = input_subject["proton"][tio.DATA]
            img_mask = input_subject["mask"][tio.DATA]

            input_tensor = torch.cat((img_gas, img_proton), dim=1)
            target = img_mask.squeeze(1)
        elif channels == 1:
            img_gas = input_subject["gas"][tio.DATA]
            img_mask = input_subject["mask"][tio.DATA]

            input_tensor = img_gas
            target = img_mask.squeeze(1)
        else:
            raise ValueError("Wrong number of modalities or channels.")
    else:
        raise ValueError("Wrong number of modalities or channels.")

    if in_cuda:
        input_tensor, target = input_tensor.cuda(), target.cuda()

    return input_tensor, target


def export_nii(image: np.ndarray, path: str, fov: Optional[float] = None):
    """Export image as nifti file.

    Args:
        image: np.ndarray 3D image to be exported
        path: str file path of nifti file
        fov: float field of view in cm
    """
    nii_imge = nib.Nifti1Image(image, np.eye(4))
    if fov:
        nii_imge.header["pixdim"][1:4] = [
            fov / np.shape(image)[0] / 10,
            fov / np.shape(image)[0] / 10,
            fov / np.shape(image)[0] / 10,
        ]
    nib.save(nii_imge, path)


def save_tensor_to_nii(image: torch.Tensor, path: str):
    """Save a torch tensor as a nifti file.

    Args:
        image: torch.Tensor 3D image to be exported
        path: str file path of nifti file
    """
    image = torch.squeeze(image).detach().cpu().numpy()
    export_nii(image, path)
