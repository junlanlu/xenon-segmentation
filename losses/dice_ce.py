"""Combination of BCE and Dice losses for 3D segmentation."""
import torch
import torch.nn as nn
from monai.losses import DiceCELoss

from losses.dice import DiceLoss
from utils import loss_utils
from utils.loss_utils import expand_as_one_hot


class CEDiceLoss(nn.Module):
    """TODO"""

    def __init__(self, n_classes: int):
        super().__init__()
        self.cedice = DiceCELoss()
        self.n_classes = n_classes

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """

        Args:
            input (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The Dice loss and channel score.
        """
        target_expanded = expand_as_one_hot(target.long(), self.n_classes)
        assert (
            input.size() == target_expanded.size()
        ), "'input' and 'target' must have the same shape"

        return (
            self.cedice(input, target_expanded),
            loss_utils.compute_per_channel_dice(input, target_expanded)
            .detach()
            .cpu()
            .numpy(),
        )
