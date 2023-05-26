"""Generalized Dice Loss (GDL).

Source: https://arxiv.org/pdf/1707.03237.pdf"""
import torch

from losses.base_dice import _AbstractDiceLoss
from utils.loss_utils import flatten


class GeneralizedDiceLoss(_AbstractDiceLoss):
    """
    Generalized Dice Loss (GDL).

    Source: https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(
        self,
        classes=4,
        sigmoid_normalization=True,
        skip_index_after=None,
        epsilon=1e-6,
    ):
        """
        Initialize the GeneralizedDiceLoss.

        Args:
            classes (int): Number of classes.
            sigmoid_normalization (bool): Flag to indicate whether to use sigmoid normalization.
            skip_index_after (int): Index to skip after during loss calculation.
            epsilon (float): Small value for numerical stability.
        """
        super().__init__(weight=None, sigmoid_normalization=sigmoid_normalization)
        self.epsilon = epsilon
        self.classes = classes
        if skip_index_after is not None:
            self.skip_index_after = skip_index_after

    def dice(self, input, target, weight):
        """
        Compute the Generalized Dice Loss (GDL).

        Args:
            input (torch.Tensor): Input tensor of shape (NxCxDxHxW).
            target (torch.Tensor): Target tensor of shape (NxDxHxW).
            weight (torch.Tensor): Weight tensor of shape (NxDxHxW).

        Returns:
            torch.Tensor: Computed loss value.
        """
        assert input.size() == target.size()

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # For GDL to make sense we need at least 2 channels
            # Put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse
        # of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())
