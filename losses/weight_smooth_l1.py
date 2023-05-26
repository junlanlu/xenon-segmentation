"""Weighted Smooth L1 Loss."""

import torch
from torch import nn

from utils.loss_utils import expand_as_one_hot


class WeightedSmoothL1Loss(nn.SmoothL1Loss):
    def __init__(
        self,
        threshold: float = 0,
        initial_weight: float = 0.1,
        apply_below_threshold: bool = True,
        classes: int = 4,
    ):
        super().__init__(reduction="none")
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold
        self.weight = initial_weight
        self.classes = classes

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate the weighted smooth L1 loss.

        Args:
            input (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The weighted smooth L1 loss.

        Raises:
            AssertionError: If input and target have different shapes.
        """
        target = expand_as_one_hot(target, self.classes)
        assert (
            input.size() == target.size()
        ), "'input' and 'target' must have the same shape"
        l1 = super().forward(input, target)

        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold

        l1[mask] = l1[mask] * self.weight

        return l1.mean()
