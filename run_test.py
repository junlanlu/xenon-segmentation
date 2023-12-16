"""Train and eval script."""
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
from utils import general, io_utils

_CONFIG = config_flags.DEFINE_config_file(
    "config", "config/base_config.py", "config file."
)

MODEL_DIR = flags.DEFINE_string("model_dir", "tmp", "Directory to save model.")


def run_tests(unused_argv):
    """Run the tests."""
    config = _CONFIG.value
    general.reproducibility(config.use_cuda, config.seed)

    # Create the test dataset
    test_generator = dataloaders.generate_test_datasets(config=config)

    # Load the trained model
    model, _ = architectures.create_model(config=config)
    model.load_state_dict(
        torch.load(glob.glob(os.path.join(MODEL_DIR.value, "*_best.pth"))[0])[
            "model_state_dict"
        ]
    )
    if config.use_cuda:
        model = model.cuda()
        logging.info("Model transferred to GPU...")

    # Create a Tester and run tests
    tester = Tester(
        config,
        model,
        test_data_loader=test_generator,
    )
    logging.info("Started testing...")
    tester.testing()
    logging.info("Finished testing...")


if __name__ == "__main__":
    app.run(run_tests)
