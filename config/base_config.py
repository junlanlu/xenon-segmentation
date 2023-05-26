"""Common settings among multiple config files."""

import os
import sys

import torch
from ml_collections import config_dict

# parent directory
sys.path.append("..")

from utils import constants


class Config(config_dict.ConfigDict):
    """Base config file of experiment settings.

    Attributes:
        TODO
    """

    def __init__(self):
        """Initialize config parameters."""
        super().__init__()
        self.data = Data()
        self.model = Model()
        self.optimizer = Optimizer()
        self.loss = Loss()
        self.n_epochs = 250
        self.seed = 1777777
        self.use_cuda = True
        self.log_frequency = 50
        self.log_dir = os.path.join("results", "logs")
        self.save_dir = None


class Data(object):
    """Define the dataset processes.

    Attributes:
        TODO
    """

    def __init__(self):
        """Initialize the dataset parameters."""
        self.augmentation = True
        self.batch_size = 4
        self.name = constants.DatasetName.ISEG2019
        self.n_classes = 4
        self.n_modalities = 2
        self.n_channels = 2
        self.n_samples_train = 10
        self.n_samples_test = 10
        self.ckpt_dir = ""
        self.crop_size = (32, 32, 32)
        self.normalization_method = "full_volume_mean"
        self.train_val_split = 0.8


class Model(object):
    """Define the model hyperparameters.

    Attributes:
        TODO
    """

    def __init__(self):
        """Initialize the model parameters."""
        self.name = constants.ModelName.UNET3D


class Optimizer(object):
    """Define the optimizer hyperparameters."""

    def __init__(self):
        """Initialize the model parameters."""
        self.name = constants.OptimizerName.ADAM
        self.lr = 1e-3
        self.weight_decay = 1e-5


class Loss(object):
    """Define the loss hyperparameters."""

    def __init__(self):
        self.name = constants.LossName.DICELOSS
        self.weight = torch.tensor([0.1, 1, 1, 1])
        self.ignore_index = None
        self.pos_weight = None


def get_config() -> config_dict.ConfigDict:
    """Return the config dict. This is a required function.

    Returns:
        a ml_collections.config_dict.ConfigDict
    """
    return Config()
