"""Inference script.

Example usage:
python inference.py --config config/xenon_unet_trachea.py --image_file \
    datasets/xenon/test/006-113_s2/gx/gas.nii --model_dir \
    results/unet3d_checkpoints/10_30_17_29_xenontrachea_good/
"""

import glob
import logging
import os
import pdb

import torch
from absl import app, flags, logging
from ml_collections import config_flags

import architectures
import dataloaders
from tester import Tester
from utils import general, img_utils, io_utils, metrics

_CONFIG = config_flags.DEFINE_config_file(
    "config", "config/xenon_unet_trachea.py", "config file."
)

MODEL_DIR = flags.DEFINE_string(
    "model_dir",
    "results/unet3d_checkpoints/10_30_17_29_xenontrachea_good/",
    "Directory to save model.",
)
IMAGE_FILE = flags.DEFINE_string(
    "image_file", None, "Image file to test.", required=True
)
CUDA = flags.DEFINE_boolean("use_cuda", False, "Whether to use CUDA.")


def main(unused_argv):
    """Run the tests."""
    config = _CONFIG.value
    general.reproducibility(config.use_cuda, config.seed)
    # Load the trained model
    model, _ = architectures.create_model(config=config)
    model.load_state_dict(
        torch.load(glob.glob(os.path.join(MODEL_DIR.value, "*_best.pth"))[0])[
            "model_state_dict"
        ]
    )
    if CUDA.value:
        model = model.cuda()
        logging.info("Model transferred to GPU...")

    model.eval()
    with torch.no_grad():
        input_tensor = io_utils.import_nii_to_input_tensor(IMAGE_FILE.value)
        input_tensor.requires_grad = False

        if CUDA.value:
            input_tensor = input_tensor.cuda()
        output = model(input_tensor)
        output_onehot = img_utils.one_hot_to_label(output)

        # Here you can add any evaluation metrics you need
        # For example computing the dice score between output and target
        io_utils.save_tensor_to_nii(
            input_tensor, os.path.join(os.path.dirname(IMAGE_FILE.value), "input.nii")
        )
        io_utils.save_tensor_to_nii(
            output_onehot, os.path.join(os.path.dirname(IMAGE_FILE.value), "output.nii")
        )


if __name__ == "__main__":
    app.run(main)
