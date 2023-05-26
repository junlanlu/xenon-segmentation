"""Utilities for loss functions."""

from typing import Optional

import torch


def compute_per_channel_dice(
    input: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes Dice Coefficient given a multi-channel input and target.
    Assumes the input is a normalized probability, e.g. a result of the Sigmoid or
    Softmax function.

    Args:
        input: NxCxSpatial input tensor
        target: NxCxSpatial target tensor
        epsilon: Prevents division by zero
        weight: Cx1 tensor of weight per channel/class

    Returns:
        Tensor: Per channel Dice Coefficient
    """
    # input and target shapes must match
    assert (
        input.size() == target.size()
    ), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


def flatten(tensor: torch.Tensor) -> torch.Tensor:
    """
    Flattens a given tensor such that the channel axis is first.

    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)

    Args:
        tensor: Input tensor to be flattened.

    Returns:
        Flattened tensor with channel axis first.
    """
    # Number of channels
    C = tensor.size(1)
    # New axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def expand_as_one_hot(input: torch.Tensor, C: int, ignore_index=None) -> torch.Tensor:
    """Convert label image to one-hot encoding.
    Converts NxDxHxW label image to NxCxDxHxW where each label gets converted to its
    corresponding one-hot vector

    Args:
        input: 4D input image (NxDxHxW)
        C: Number of channels/labels
        ignore_index: Ignore index to be kept during the expansion

    Returns:
        5D output image (NxCxDxHxW)
    """
    if input.dim() == 5:
        return input
    assert input.dim() == 4

    # Expand the input tensor to Nx1xDxHxW before scattering
    input = input.unsqueeze(1)
    # Create result tensor shape (NxCxDxHxW)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # Create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # Clone the input tensor and zero out ignore_index
        input = input.clone()
        input[input == ignore_index] = 0
        # Scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # Bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # Scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)
