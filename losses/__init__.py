""" Loss functions.

Adapted from:
https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py
"""
import monai
import torch
import torch.nn as nn
from torch.nn import L1Loss, MSELoss, SmoothL1Loss

from config import base_config
from utils import constants

from .dice import DiceLoss
from .dice_bce import BCEDiceLoss
from .dice_ce import CEDiceLoss
from .generalized_dice import GeneralizedDiceLoss
from .pixelwise_cross_entropy import PixelWiseCrossEntropyLoss
from .tags_angular_loss import TagsAngularLoss
from .weight_cross_entropy import WeightedCrossEntropyLoss
from .weight_smooth_l1 import WeightedSmoothL1Loss


def create_loss(config: base_config.Config) -> nn.Module:
    """
    Create a loss function based on the configuration.

    Args:
        config: ml_collections.ConfigDict containing the configuration.

    Returns:
        nn.Module: The created loss function.

    Raises:
        RuntimeError: If the given loss function name is unsupported.
    """

    if config.loss.name == constants.LossName.BCEWITHLOGITSLOSS:
        return nn.BCEWithLogitsLoss(pos_weight=config.loss.pos_weight)
    elif config.loss.name == constants.LossName.BCEDICELOSS:
        return BCEDiceLoss(alpha=1, beta=1, classes=config.data.n_classes)
    elif config.loss.name == constants.LossName.CEDICELOSS:
        return CEDiceLoss(n_classes=config.data.n_classes)
    elif config.loss.name == constants.LossName.CROSSENTROPYLOSS:
        if config.loss.ignore_index is None:
            config.loss.ignore_index = (  # type: ignore
                -100
            )  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return nn.CrossEntropyLoss(
            weight=config.loss.weight, ignore_index=config.loss.ignore_index
        )
    elif config.loss.name == constants.LossName.DICELOSS:
        return DiceLoss(
            weight=config.loss.weight,
            sigmoid_normalization=False,
            classes=config.data.n_classes,
        )
    elif config.loss.name == constants.LossName.GENERALIZEDDICELOSS:
        return GeneralizedDiceLoss(sigmoid_normalization=False)
    elif config.loss.name == constants.LossName.L1LOSS:
        return L1Loss()
    elif config.loss.name == constants.LossName.MSELoss:
        return MSELoss()
    elif config.loss.name == constants.LossName.PIXELWISECROSSENTROPYLOSS:
        return PixelWiseCrossEntropyLoss(
            class_weights=config.loss.weight, ignore_index=config.loss.ignore_index
        )
    elif config.loss.name == constants.LossName.SMOOTHL1LOSS:
        return SmoothL1Loss()
    elif config.loss.name == constants.LossName.TAGSANGULARLOSS:
        return TagsAngularLoss(classes=config.data.n_classes)
    elif config.loss.name == constants.LossName.WEIGHTEDCROSSENTROPYLOSS:
        if config.loss.ignore_index is None:
            config.loss.ignore_index = (  # type: ignore
                -100
            )  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return WeightedCrossEntropyLoss(ignore_index=config.loss.ignore_index)
    elif config.loss.name == constants.LossName.WEIGHTEDSMOOTHL1LOSS:
        return WeightedSmoothL1Loss(classes=config.data.n_classes)
    else:
        raise RuntimeError(f"Unsupported loss function: '{config.loss.name}'.")


class SkipLastTargetChannelWrapper(nn.Module):
    """
    Loss wrapper which removes additional target channel
    """

    def __init__(self, loss, squeeze_channel=False):
        super(SkipLastTargetChannelWrapper, self).__init__()
        self.loss = loss
        self.squeeze_channel = squeeze_channel

    def forward(self, input, target):
        assert (
            target.size(1) > 1
        ), "Target tensor has a singleton channel dimension, cannot remove channel"

        # skips last target channel if needed
        target = target[:, :-1, ...]

        if self.squeeze_channel:
            # squeeze channel dimension if singleton
            target = torch.squeeze(target, dim=1)
        return self.loss(input, target)


class _MaskingLossWrapper(nn.Module):
    """
    Loss wrapper which prevents the gradient of the loss to be computed where target is equal to `ignore_index`.
    """

    def __init__(self, loss, ignore_index):
        super(_MaskingLossWrapper, self).__init__()
        assert ignore_index is not None, "ignore_index cannot be None"
        self.loss = loss
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = target.clone().ne_(self.ignore_index)
        mask.requires_grad = False

        # mask out input/target so that the gradient is zero where on the mask
        input = input * mask
        target = target * mask

        # forward masked input and target to the loss
        return self.loss(input, target)
