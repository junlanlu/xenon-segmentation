"""Weighted Cross Entropy Loss.

Code as described in https://arxiv.org/pdf/1707.03237.pdf
"""
import torch
from torch import nn

from utils.loss_utils import flatten


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss (WCE)"""

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate the weighted cross entropy loss.

        Args:
            input (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The weighted cross entropy loss.
        """
        weight = self._class_weights(input)
        return nn.functional.cross_entropy(
            input, target, weight=weight, ignore_index=self.ignore_index
        )

    @staticmethod
    def _class_weights(input: torch.Tensor) -> torch.Tensor:
        """
        Calculate the class weights.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The class weights.
        """
        # Normalize the input first
        input = nn.functional.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1.0 - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = torch.autograd.Variable(
            nominator / denominator, requires_grad=False
        )
        return class_weights
