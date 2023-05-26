"""General utility functions for training/evaluation."""
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def reproducibility(use_cuda: bool = False, seed: int = 0):
    """Set seed for reproducibility.

    Args:
        use_cuda (bool, optional): Use CUDA. Defaults to False.
        seed (int, optional): Seed. Defaults to 0.
    """
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    # for reproducibility and faster training
    cudnn.benchmark = True


def datestr() -> str:
    """Return a string with the current date and time."""
    now = time.gmtime()
    return "{:02}_{:02}_{:02}_{:02}".format(
        now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min
    )
