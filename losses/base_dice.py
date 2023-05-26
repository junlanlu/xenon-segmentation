""" Base class for different implementations of Dice loss. 

Code was adapted and modified from
https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py
"""
from typing import Optional

import torch
from torch import nn as nn

from utils.loss_utils import expand_as_one_hot


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(
        self, weight: Optional[torch.Tensor] = None, sigmoid_normalization=True
    ):
        """
        Initialize the _AbstractDiceLoss.

        Args:
            weight: Weight tensor for the loss.
            sigmoid_normalization: Whether to use sigmoid normalization or softmax
                normalization.
        """
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer("weight", weight)
        self.classes = -1
        self.skip_index_after = None

        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def dice(self, input, target, weight):
        """
        Compute the Dice score.

        This method needs to be implemented by the subclass.

        Args:
            input: Input tensor.
            target: Target tensor.
            weight: Weight tensor.

        Returns:
            Dice score tensor.
        """
        raise NotImplementedError

    def skip_target_channels(self, target, index):
        """
        Skip the target channels after the specified index.

        Args:
            target: Target tensor.
            index: Index to skip channels after.

        Returns:
            Skipped target tensor.
        """
        assert index >= 2
        return target[:, 0:index, ...]

    def forward(self, input, target):
        """
        Compute the forward pass of the loss.

        Args:
            input: Input tensor.
            target: Target tensor.

        Returns:
            Tuple containing the loss tensor and per-channel Dice scores.
        """
        target = expand_as_one_hot(target.long(), self.classes)

        assert (
            input.dim() == target.dim() == 5
        ), "'input' and 'target' have different number of dimensions"

        if self.skip_index_after is not None:
            before_size = target.size()
            target = self.skip_target_channels(target, self.skip_index_after)
            print("Target {} after skip index {}".format(before_size, target.size()))

        assert (
            input.size() == target.size()
        ), "'input' and 'target' must have the same shape"

        input = self.normalization(input)
        per_channel_dice = self.dice(input, target, weight=self.weight)

        loss = 1.0 - torch.mean(per_channel_dice)
        per_channel_dice = per_channel_dice.detach().cpu().numpy()

        return loss, per_channel_dice
