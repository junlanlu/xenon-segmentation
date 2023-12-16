"""Inference script.

Example usage:
python inference_batch.py --model_dir \
    results/unet3d_checkpoints/10_30_17_29_xenontrachea_good --cohort_dir datasets/xenon/test \
    --use_cuda=False --config config/xenon_unet_trachea.py 
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
from utils import general, img_utils, io_utils

_CONFIG = config_flags.DEFINE_config_file(
    "config", "config/base_config.py", "config file."
)

MODEL_DIR = flags.DEFINE_string("model_dir", "tmp", "Directory to save model.")
COHORT_DIR = flags.DEFINE_string(
    "cohort_dir", "datasets/xenon/healthy", "Image file to test."
)
CUDA = flags.DEFINE_boolean("use_cuda", True, "Whether to use CUDA.")


def main(unused_argv):
    """Run the tests."""
    image_files = glob.glob(
        os.path.join(COHORT_DIR.value, "**", "gas.nii"), recursive=True
    )
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
        for image_file in image_files:
            logging.info("Processing image: {}".format(image_file))
            input_tensor = io_utils.import_nii_to_input_tensor(image_file)
            input_tensor.requires_grad = False

            if CUDA.value:
                input_tensor = input_tensor.cuda()
            output = model(input_tensor)
            output_onehot = img_utils.one_hot_to_label(output)

            # Here you can add any evaluation metrics you need
            # For example computing the dice score between output and target
            io_utils.save_tensor_to_nii(
                input_tensor, os.path.join(os.path.dirname(image_file), "input.nii")
            )
            io_utils.save_tensor_to_nii(
                output_onehot,
                os.path.join(os.path.dirname(image_file), "output_tcv.nii"),
            )


if __name__ == "__main__":
    app.run(main)
