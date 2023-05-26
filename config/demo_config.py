"""Common settings among multiple config files."""

import os
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


def get_config() -> config_dict.ConfigDict:
    """Return the config dict. This is a required function.

    Returns:
        a ml_collections.config_dict.ConfigDict
    """
    return Config()
