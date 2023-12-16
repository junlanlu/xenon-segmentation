"""Train and eval script."""
import logging
import os
import pdb

import torch
from absl import app, flags, logging
from ml_collections import config_flags

import architectures
import dataloaders
from losses import create_loss
from trainer import Trainer
from utils import general, io_utils

_CONFIG = config_flags.DEFINE_config_file(
    "config", "config/base_config.py", "config file."
)


def main(unused_argv):
    """Main function."""
    config = _CONFIG.value
    general.reproducibility(config.use_cuda, config.seed)
    # create checkpoint directory
    if config.save_dir is None:
        config.save_dir = (
            "results/"
            + config.model.name.value.lower()
            + "_checkpoints/"
            + "{}_{}_".format(general.datestr(), config.data.name.value.lower())
        )
    io_utils.make_dirs(path=config.save_dir)
    # get the dataloaders
    (training_generator, val_generator) = dataloaders.generate_trainval_datasets(
        config=config, path="datasets"
    )
    # get the loss criterions
    criterion = create_loss(config=config)

    # get the models
    model, optimizer = architectures.create_model(config=config)
    if config.use_cuda:
        criterion = criterion.cuda()
        model = model.cuda()
        logging.info("Model transferred to GPU...")
    trainer = Trainer(
        config,
        model,
        criterion,
        optimizer,
        train_data_loader=training_generator,
        valid_data_loader=val_generator,
        lr_scheduler=None,
    )
    logging.info("Started training...")
    trainer.training()
    logging.info("Finished training...")


if __name__ == "__main__":
    """Run the main function."""
    app.run(main)
