"""Get data loaders for the specified dataset."""

import logging

from ml_collections import config_flags
from torch.utils.data import DataLoader

from config import base_config
from utils import constants

from .iseg2019 import MRIDatasetISEG2019
from .xenon_simple import XenonSimple
from .xenon_trachea import XenonTrachea


def generate_trainval_datasets(config: base_config.Config, path: str = "datasets"):
    """Return data loaders for the specified dataset.

    Args:
        config: Config object.
        path: Path to the dataset folder.
    """
    params = {"batch_size": config.data.batch_size, "shuffle": True, "num_workers": 2}
    samples_train = config.data.n_samples_train
    samples_val = config.data.n_samples_test
    split_percent = config.data.train_val_split

    if config.data.name == constants.DatasetName.ISEG2019:
        total_data = 10
        split_idx = int(split_percent * total_data)
        train_loader = MRIDatasetISEG2019(
            "train",
            dataset_path=path,
            split_id=split_idx,
            samples=samples_train,
        )

        val_loader = MRIDatasetISEG2019(
            "val",
            dataset_path=path,
            split_id=split_idx,
            samples=samples_val,
        )
    elif config.data.name == constants.DatasetName.XENONSIMPLE:
        train_loader = XenonSimple(mode=constants.DatasetMode.TRAIN)
        val_loader = XenonSimple(mode=constants.DatasetMode.VALIDATION)
    elif config.data.name == constants.DatasetName.XENONTRACHEA:
        train_loader = XenonTrachea(mode=constants.DatasetMode.TRAIN)
        val_loader = XenonTrachea(mode=constants.DatasetMode.VALIDATION)
    else:
        raise ValueError("Dataset: {} not supported.".format(config.data.name))

    logging.info("Data samples successfully generated.")
    return DataLoader(train_loader, **params), DataLoader(val_loader, **params)


def generate_test_datasets(config: base_config.Config):
    """Return data loaders for the specified dataset.

    Args:
        config: Config object.
        path: Path to the dataset folder.
    """
    params = {"batch_size": 1, "shuffle": True, "num_workers": 2}

    if config.data.name == constants.DatasetName.XENONSIMPLE:
        test_loader = XenonSimple(mode=constants.DatasetMode.TEST, augmentation=False)
    elif config.data.name == constants.DatasetName.XENONTRACHEA:
        test_loader = XenonTrachea(mode=constants.DatasetMode.TEST, augmentation=False)
    else:
        raise ValueError("Dataset: {} not supported.".format(config.data.name))

    logging.info("Data samples successfully generated.")
    return DataLoader(test_loader, **params)
