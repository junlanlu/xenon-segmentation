"""Pixelwise Cross Entropy Loss."""

import torch
import torch.nn as nn

from utils.loss_utils import expand_as_one_hot


class PixelWiseCrossEntropyLoss(nn.Module):
    """
    Pixelwise Cross Entropy Loss.
    """

    def __init__(self, class_weights=None, ignore_index=None):
        """
        Initialize the PixelWiseCrossEntropyLoss.

        Args:
            class_weights (torch.Tensor): Class weights for weighting the loss.
            ignore_index (int): Index to ignore during loss calculation.
        """
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer("class_weights", class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        """
        Compute the Pixelwise Cross Entropy Loss.

        Args:
            input (torch.Tensor): Input tensor of shape (NxCxDxHxW).
            target (torch.Tensor): Target tensor of shape (NxDxHxW).
            weights (torch.Tensor): Weights tensor of shape (NxDxHxW).

        Returns:
            torch.Tensor: Computed loss value.
        """
        assert target.size() == weights.size()

        # Normalize the input
        log_probabilities = self.log_softmax(input)

        # Standard CrossEntropyLoss requires the target to be (NxDxHxW),
        # so we need to expand it to (NxCxDxHxW)
        target = expand_as_one_hot(
            target, C=input.size()[1], ignore_index=self.ignore_index
        )

        # Expand weights
        weights = weights.unsqueeze(0)
        weights = weights.expand_as(input)

        # Create default class_weights if None
        if self.class_weights is None:
            class_weights = torch.ones(input.size()[1]).float().to(input.device)
        else:
            class_weights = self.class_weights

        # Resize class_weights to be broadcastable into the weights
        class_weights = class_weights.view(1, -1, 1, 1, 1)  # type: ignore

        # Multiply weights tensor by class weights
        weights = class_weights * weights

        # Compute the losses
        result = -weights * target * log_probabilities

        # Average the losses
        return result.mean()
