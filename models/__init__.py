"""Model factory module."""
import logging
import pdb

import torch.optim as optim

from config import base_config
from utils import constants

from .attention_unet import AttentionUnet
from .COVIDNet import CNN, CovidNet
from .Densenet3D import DualPathDenseNet, DualSingleDenseNet, SinglePathDenseNet
from .DenseVoxelNet import DenseVoxelNet
from .HighResNet3D import HighResNet3D
from .HyperDensenet import HyperDenseNet, HyperDenseNet_2Mod
from .nnunet import DynUNet
from .ResNet3D_VAE import ResNet3dVAE
from .ResNet3DMedNet import generate_resnet3d
from .SkipDenseNet3D import SkipDenseNet3D
from .unet import UNet
from .unet_3d import UNet3D
from .unetr import UNETR
from .vnet import VNet, VNetLight


def create_model(config: base_config.Config):
    """Create model and optimizer.

    Args:
        config (base_config.Config): ml_collections config object.
    """
    logging.info("Building model: {}".format(config.model.name.value))
    if config.model.name == constants.ModelName.ATTENTIONUNET:
        model = AttentionUnet(
            spatial_dims=3,
            in_channels=config.data.n_channels,
            out_channels=config.data.n_classes,
            channels=[8, 16, 32, 64, 128, 256],
            strides=[2, 2, 2, 2, 2],
        )
    elif config.model.name == constants.ModelName.CNN:
        model = CNN(config.data.n_classes, "resnet18")
    elif config.model.name == constants.ModelName.COVIDNET1:
        model = CovidNet("small", config.data.n_classes)
    elif config.model.name == constants.ModelName.COVIDNET2:
        model = CovidNet("large", config.data.n_classes)
    elif config.model.name == constants.ModelName.DENSENET1:
        model = SinglePathDenseNet(
            in_channels=config.data.n_channels, classes=config.data.n_classes
        )
    elif config.model.name == constants.ModelName.DENSENET2:
        model = DualPathDenseNet(
            in_channels=config.data.n_channels, classes=config.data.n_classes
        )
    elif config.model.name == constants.ModelName.DENSENET3:
        model = DualSingleDenseNet(
            in_channels=config.data.n_channels,
            drop_rate=0.1,
            classes=config.data.n_classes,
        )
    elif config.model.name == constants.ModelName.DENSEVOXELNET:
        model = DenseVoxelNet(
            in_channels=config.data.n_channels, classes=config.data.n_classes
        )
    elif config.model.name == constants.ModelName.HIGHRESNET:
        model = HighResNet3D(
            in_channels=config.data.n_channels, classes=config.data.n_classes
        )
    elif config.model.name == constants.ModelName.HYPERDENSENET:
        if config.data.n_channels == 2:
            model = HyperDenseNet_2Mod(classes=config.data.n_classes)
        elif config.data.n_channels == 3:
            model = HyperDenseNet(classes=config.data.n_classes)
        else:
            raise NotImplementedError

    elif config.model.name == constants.ModelName.NNUNET:
        model = DynUNet(
            spatial_dims=3,
            in_channels=config.data.n_channels,
            out_channels=config.data.n_classes,
            kernel_size=[
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
            ],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            upsample_kernel_size=[
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
            ],
        )
    elif config.model.name == constants.ModelName.RESNET3DVAE:
        model = ResNet3dVAE(
            in_channels=config.data.n_channels,
            classes=config.data.n_classes,
            dim=config.data.crop_size,
        )
    elif config.model.name == constants.ModelName.RESNET3DVAE:
        depth = 18
        model = generate_resnet3d(
            in_channels=config.data.n_channels,
            classes=config.data.n_classes,
            model_depth=depth,
        )
    elif config.model.name == constants.ModelName.SKIPDENSENET3D:
        model = SkipDenseNet3D(
            growth_rate=16,
            num_init_features=32,
            drop_rate=0.1,
            classes=config.data.n_classes,
        )
    elif config.model.name == constants.ModelName.UNET:
        model = UNet(
            spatial_dims=3,
            in_channels=config.data.n_channels,
            out_channels=config.data.n_classes,
            channels=[8, 16, 32, 64, 128, 256],
            strides=[2, 2, 2, 2, 2],
        )
    elif config.model.name == constants.ModelName.UNETR:
        model = UNETR(
            in_channels=config.data.n_channels,
            out_channels=config.data.n_classes,
            img_size=config.data.crop_size,
            num_heads=8,
        )
    elif config.model.name == constants.ModelName.UNET3D:
        model = UNet3D(
            in_channels=config.data.n_channels,
            n_classes=config.data.n_classes,
            base_n_filter=8,
        )
    elif config.model.name == constants.ModelName.VNET:
        model = VNet(
            in_channels=config.data.n_channels, elu=False, classes=config.data.n_classes
        )
    elif config.model.name == constants.ModelName.VNETLIGHT:
        model = VNetLight(
            in_channels=config.data.n_channels, elu=False, classes=config.data.n_classes
        )
    else:
        raise NotImplementedError

    logging.info(
        "Number of params: {}".format(
            sum([p.data.nelement() for p in model.parameters()])
        )
    )

    if config.optimizer.name == constants.OptimizerName.SGD:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.optimizer.lr,
            momentum=0.5,
            weight_decay=config.optimizer.weight_decay,
        )
    elif config.optimizer.name == constants.OptimizerName.ADAM:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )
    elif config.optimizer.name == constants.OptimizerName.RMSprop:
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )
    else:
        raise NotImplementedError

    return model, optimizer
