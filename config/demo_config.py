"""Common settings among multiple config files."""

import sys

import torch
from ml_collections import config_dict

from config import base_config

# parent directory
sys.path.append("..")

from utils import constants


class Config(base_config.Config):
    """Base config file of experiment settings.

    Attributes:
        TODO
    """

    def __init__(self):
        """Initialize config parameters."""
        super().__init__()
        self.data = Data()
        self.loss = Loss()
        self.model = Model()
        self.n_epochs = 100


class Data(base_config.Data):
    """Define the dataset processes.

    Attributes:
        TODO
    """

    def __init__(self):
        """Initialize the dataset parameters."""
        super().__init__()
        self.augmentation = True
        self.batch_size = 1
        self.name = constants.DatasetName.XENONSIMPLE
        self.n_classes = 2
        self.n_modalities = 2
        self.n_channels = 1
        self.ckpt_dir = ""
        self.crop_size = (128, 128, 128)
        self.normalization_method = "full_volume_mean"


class Loss(base_config.Loss):
    """Define the loss hyperparameters."""

    def __init__(self):
        super().__init__()
        self.name = constants.LossName.DICELOSS
        self.weight = torch.tensor([0.1, 1])


class Model(base_config.Model):
    """Define the model hyperparameters.

    Attributes:
        TODO
    """

    def __init__(self):
        """Initialize the model parameters."""
        super().__init__()
        self.name = constants.ModelName.UNET3D


def get_config() -> config_dict.ConfigDict:
    """Return the config dict. This is a required function.

    Returns:
        a ml_collections.config_dict.ConfigDict
    """
    return Config()
