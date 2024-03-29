"""Dataset for reading the infant brain MRI dataset of ISEG 2019 challenge."""
import glob
import os
import pdb

import numpy as np
import torch
from torch.utils.data import Dataset

import utils.augmentation_utils as aug
from utils import img_utils, io_utils, loader_utils


class MRIDatasetISEG2019(Dataset):
    """Dataset for reading the infant brain MRI dataset of ISEG 2019 challenge."""

    def __init__(
        self,
        mode: str = "train",
        dataset_path: str = "datasets",
        crop_dim: tuple = (32, 32, 32),
        split_id: int = 1,
        samples: int = 1000,
    ):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        :param crop_dim: subvolume tuple
        :param fold_id: 1 to 10 values
        :param samples: number of sub-volumes that you want to create
        """
        self.mode = mode
        self.root = str(dataset_path)
        self.training_path = os.path.join(self.root, "iseg_2019/iSeg-2019-Training")
        self.testing_path = os.path.join(self.root, "/iseg_2019/iSeg-2019-Validation/")
        self.n_classes = 4
        self.full_vol_dim = (144, 192, 256)  # slice, width, height
        self.crop_size = crop_dim
        self.threshold = 0.0001
        self.augmentation = True
        self.list_subvolume_paths = []
        self.samples = samples
        self.full_volume = None
        self.save_name = (
            self.root
            + "/iseg_2019/iseg2019-list-"
            + mode
            + "-samples-"
            + str(samples)
            + ".txt"
        )
        if self.augmentation:
            self.transform = aug.RandomChoice(
                transforms=[
                    aug.GaussianNoise(mean=0, std=0.01),
                    aug.RandomFlip(),
                    aug.ElasticTransform(),
                ],
                p=0.5,
            )
        subvol = (
            "_vol_" + str(crop_dim[0]) + "x" + str(crop_dim[1]) + "x" + str(crop_dim[2])
        )
        self.sub_vol_path = self.root + "/iseg_2019/generated/" + mode + subvol + "/"
        io_utils.make_dirs(self.sub_vol_path)
        list_IDsT1 = sorted(glob.glob(os.path.join(self.training_path, "*T1.img")))
        list_IDsT2 = sorted(glob.glob(os.path.join(self.training_path, "*T2.img")))
        labels = sorted(glob.glob(os.path.join(self.training_path, "*label.img")))
        self.affine = img_utils.load_affine_matrix(list_IDsT1[0])
        if self.mode == "train":
            list_IDsT1 = list_IDsT1[:split_id]
            list_IDsT2 = list_IDsT2[:split_id]
            labels = labels[:split_id]
            self.list_subvolume_paths = loader_utils.create_sub_volumes(
                list_IDsT1,
                list_IDsT2,
                labels,
                dataset_name="iseg2019",
                mode=mode,
                samples=samples,
                full_vol_dim=self.full_vol_dim,
                crop_size=self.crop_size,
                sub_vol_path=self.sub_vol_path,
                th_percent=self.threshold,
            )
        elif self.mode == "val":
            list_IDsT1 = list_IDsT1[split_id:]
            list_IDsT2 = list_IDsT2[:split_id:]
            labels = labels[split_id:]
            self.list_subvolume_paths = loader_utils.create_sub_volumes(
                list_IDsT1,
                list_IDsT2,
                labels,
                dataset_name="iseg2019",
                mode=mode,
                samples=samples,
                full_vol_dim=self.full_vol_dim,
                crop_size=self.crop_size,
                sub_vol_path=self.sub_vol_path,
                th_percent=self.threshold,
            )

            self.full_volume = loader_utils.get_viz_set(
                list_IDsT1, list_IDsT2, labels, dataset_name="iseg2019"
            )
        elif self.mode == "test":
            self.list_IDsT1 = sorted(
                glob.glob(os.path.join(self.testing_path, "*T1.img"))
            )
            self.list_IDsT2 = sorted(
                glob.glob(os.path.join(self.testing_path, "*T2.img"))
            )
            self.labels = None
            # todo inference here
        else:
            raise ValueError("Invalid mode {}.".format(mode))

        io_utils.save_list_to_txt(self.save_name, self.list_subvolume_paths)

    def __len__(self):
        return len(self.list_subvolume_paths)

    def __getitem__(self, index):
        t1_path, t2_path, seg_path = self.list_subvolume_paths[index]
        t1, t2, s = np.load(t1_path), np.load(t2_path), np.load(seg_path)
        if self.mode == "train" and self.augmentation:
            [augmented_t1, augmented_t2], augmented_s = self.transform([t1, t2], s)

            return (
                torch.FloatTensor(augmented_t1.copy()).unsqueeze(0),
                torch.FloatTensor(augmented_t2.copy()).unsqueeze(0),
                torch.FloatTensor(augmented_s.copy()),
            )

        return (
            torch.FloatTensor(t1).unsqueeze(0),
            torch.FloatTensor(t2).unsqueeze(0),
            torch.FloatTensor(s),
        )
