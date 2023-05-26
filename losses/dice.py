"""Dice Loss.

Code was adapted and modified from
https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py
"""
import pdb
from typing import Optional

import torch

from losses.base_dice import _AbstractDiceLoss
from utils import loss_utils


class DiceLoss(_AbstractDiceLoss):
    """
    Dice Loss.

    Computes Dice Loss. For multi-class segmentation `weight` parameter can be used to
    assign different weights per class.

    Source: https://arxiv.org/abs/1606.04797
    """

    def __init__(
        self,
        classes: int = 4,
        skip_index_after=None,
        weight: Optional[torch.Tensor] = None,
        sigmoid_normalization: bool = True,
    ):
        super().__init__(weight, sigmoid_normalization)
        self.classes = classes
        self.weight = weight
        if skip_index_after is not None:
            self.skip_index_after = skip_index_after

    def dice(self, input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor):
        """Compute the per channel dice loss.

        Args:
            input (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.
            weight (torch.Tensor): The weight tensor.

        Returns:
            torch.Tensor: The Dice coefficient.
        """
        return loss_utils.compute_per_channel_dice(input, target, weight=self.weight)
