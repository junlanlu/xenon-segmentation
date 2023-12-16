"""Define important constants used throughout the pipeline."""

import enum


class ModelName(enum.Enum):
    """Model names."""

    ATTENTIONUNET = "attention_unet"
    CNN = "cnn"
    COVIDNET1 = "covidnet1"
    COVIDNET2 = "covidnet2"
    DENSENET1 = "densenet1"
    DENSENET2 = "densenet2"
    DENSENET3 = "densenet3"
    DENSEVOXELNET = "densevoxelnet"
    HIGHRESNET = "highresnet"
    HYPERDENSENET = "hyperdensenet"
    RESNET3DVAE = "resnet3dvae"
    RESNETMED3D = "resnetmed3d"
    SKIPDENSENET3D = "skipdensenet3d"
    UNET = "unet"
    UNET2D = "unet2d"
    UNET3D = "unet3d"
    UNETR = "unetr"
    NNUNET = "nnunet"
    VNET = "vnet"
    VNETLIGHT = "vnetlight"


class OptimizerName(enum.Enum):
    """Optimizer names."""

    ADAM = "adam"
    ADAMAX = "adamax"
    SGD = "sgd"
    RMSprop = "rmsprop"


class DatasetName(enum.Enum):
    """Dataset names."""

    ISEG2019 = "iseg2019"
    XENONSIMPLE = "xenonsimple"
    XENONTRACHEA = "xenontrachea"


class LossName(enum.Enum):
    """Loss function names."""

    BCEWITHLOGITSLOSS = "BCEWithLogitsLoss"
    BCEDICELOSS = "BCEDiceLoss"
    CROSSENTROPYLOSS = "CrossEntropyLoss"
    CEDICELOSS = "CEDiceLoss"
    DICELOSS = "DiceLoss"
    GENERALIZEDDICELOSS = "GeneralizedDiceLoss"
    L1LOSS = "L1Loss"
    MSELoss = "MSELoss"
    PIXELWISECROSSENTROPYLOSS = "PixelWiseCrossEntropyLoss"
    SMOOTHL1LOSS = "SmoothL1Loss"
    TAGSANGULARLOSS = "TagsAngularLoss"
    WEIGHTEDCROSSENTROPYLOSS = "WeightedCrossEntropyLoss"
    WEIGHTEDSMOOTHL1LOSS = "WeightedSmoothL1Loss"


class DatasetMode(object):
    """Dataset modes."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class LabelName(object):
    """Label names."""

    ISEG2019 = ["Air", "CSF", "GM", "WM"]
    XENONSIMPLE = ["Background", "Lung"]
    XENONTRACHEA = ["Background", "Trachea"]
