"""Dataset for simple thoracic cavity segmentation of xenon MRI."""
import glob
import os
import pdb

import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset

import utils.augmentation_utils as aug
from utils import constants, img_utils, io_utils, loader_utils


class XenonSimple(tio.SubjectsDataset):
    def __init__(
        self,
        mode: str = constants.DatasetMode.TRAIN,
        dataset_path: str = "datasets/xenon",
        crop_dim: tuple = (32, 32, 32),
        augmentation: bool = True,
    ):
        subjects_list = self.get_subjects_list(mode, dataset_path)
        transforms = self.get_transforms(augmentation)
        super().__init__(subjects_list, transform=tio.Compose(transforms))

    @staticmethod
    def get_subjects_list(mode: str, dataset_path: str):
        subjects_list = []
        if mode == constants.DatasetMode.TRAIN:
            mode_str = "train"
        elif mode == constants.DatasetMode.VALIDATION:
            mode_str = "val"
        elif mode == constants.DatasetMode.TEST:
            mode_str = "test"
        else:
            raise ValueError("Invalid mode: {}".format(mode))
        mode_path = os.path.join(dataset_path, mode_str)
        for root, dirs, files in os.walk(mode_path):
            if any(file.endswith(".nii") for file in files):
                gas_path = os.path.join(root, "gas.nii")
                mask_path = os.path.join(root, "mask.nii")
                proton_path = os.path.join(root, "proton.nii")
                gas_img = torch.FloatTensor(
                    io_utils.import_nii(gas_path).copy()
                ).unsqueeze(0)
                mask_img = torch.FloatTensor(
                    io_utils.import_nii(mask_path).copy()
                ).unsqueeze(0)
                proton_img = torch.FloatTensor(
                    io_utils.import_nii(proton_path).copy()
                ).unsqueeze(0)

                subject = tio.Subject(
                    gas=tio.ScalarImage(tensor=gas_img),
                    mask=tio.LabelMap(tensor=mask_img),
                    proton=tio.ScalarImage(tensor=proton_img),
                    name=root.split("/")[-2].split("/")[-1],  # type: ignore
                    mode=root.split("/")[-1],
                )
                subjects_list.append(subject)
        return subjects_list

    @staticmethod
    def get_transforms(augmentation: bool):
        transforms = [tio.RescaleIntensity(out_min_max=(0, 1))]
        if augmentation:
            transforms.extend(
                [
                    tio.RandomNoise(std=0.01, p=0.5),
                    tio.RandomAffine(
                        scales=0,
                        degrees=0,
                        translation=3,
                        p=0.5,
                        default_pad_value="otsu",
                    ),
                    # tio.RandomElasticDeformation(max_displacement=4, p=1),
                    tio.RandomBiasField(coefficients=0.2, p=0.5),
                    # tio.RandomSwap(p=1),
                    tio.RandomGamma(p=0.2),
                ]  # type: ignore
            )
        return transforms
