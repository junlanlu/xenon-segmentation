"""Combination of BCE and Dice losses for 3D segmentation."""
import torch
import torch.nn as nn

from losses.dice import DiceLoss
from utils.loss_utils import expand_as_one_hot


class BCEDiceLoss(nn.Module):
    """
    Linear combination of BCE and Dice losses3D.

    This loss combines Binary Cross Entropy (BCE) and Dice Loss. It is used for 3D
    segmentation tasks.
    The loss consists of two terms: BCE loss and Dice loss. The contribution of each
    term is controlled
    by the alpha and beta coefficients.

    Args:
        alpha (float): Coefficient for BCE loss.
        beta (float): Coefficient for Dice loss.
        classes (int): Number of classes for Dice loss.

    """

    def __init__(self, alpha: float = 1, beta: float = 1, classes: int = 4):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta
        self.dice = DiceLoss(classes=classes)
        self.classes = classes

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Calculate the BCEDiceLoss.

        Args:
            input (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The BCEDiceLoss and channel score.
        """
        target_expanded = expand_as_one_hot(target.long(), self.classes)
        assert (
            input.size() == target_expanded.size()
        ), "'input' and 'target' must have the same shape"
        loss_1 = self.alpha * self.bce(input, target_expanded)
        loss_2, channel_score = self.beta * self.dice(input, target_expanded)
        return (loss_1 + loss_2), channel_score
