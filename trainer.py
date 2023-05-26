"""Trainer class for training a model."""

import pdb
from typing import Optional

import numpy as np
import torch
import torchio as tio
from torch.utils.data.dataloader import DataLoader

from config import base_config
from utils.io_utils import prepare_input
from utils.writer.base_writer import TensorboardWriter


class Trainer:
    """Trainer class for training a model."""

    def __init__(
        self,
        config: base_config.Config,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_data_loader: DataLoader,
        valid_data_loader: Optional[DataLoader] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        Initialize the Trainer class.

        Args:
            config (base_config.Config): Configuration object.
            model (torch.nn.Module): Model to be trained.
            criterion (torch.nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer.
            train_data_loader (torch.utils.data.DataLoader): Training data loader.
            valid_data_loader (torch.utils.data.DataLoader, optional):
                Validation data loader. Defaults to None.
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional):
                Learning rate scheduler. Defaults to None.
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_data_loader
        # epoch-based training
        self.len_epoch = len(self.train_dataloader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))  # type: ignore
        self.writer = TensorboardWriter(config)

        self.save_frequency = 10
        self.terminal_show_freq = config.log_frequency
        self.start_epoch = 1

    def training(self):
        for epoch in range(self.start_epoch, self.config.n_epochs + 1):
            self.train_epoch(epoch)

            if self.do_validation:
                self.validate_epoch(epoch)

            val_loss = (
                self.writer.data["val"]["loss"] / self.writer.data["val"]["count"]
            )
            if self.config.save_dir is not None and ((epoch + 1) % self.save_frequency):
                self.model.save_checkpoint(
                    self.config.save_dir, epoch, val_loss, optimizer=self.optimizer
                )  # type: ignore

            self.writer.write_end_of_epoch(epoch)
            self.writer.reset("train")
            self.writer.reset("val")

    def train_epoch(self, epoch: int):
        """
        Perform one training epoch.

        Args:
            epoch (int): Current epoch number.
        """
        self.model.train()

        # for batch_idx, input_tuple in enumerate(self.train_data_loader):
        for batch_idx, input_subject in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()

            input_tensor, target = prepare_input(
                input_subject=input_subject, config=self.config
            )

            input_tensor.requires_grad = True

            output = self.model(input_tensor)

            loss_dice, per_ch_score = self.criterion(output, target)
            loss_dice.backward()
            self.optimizer.step()

            self.writer.update_scores(
                batch_idx,
                loss_dice.item(),
                per_ch_score,
                "train",
                epoch * self.len_epoch + batch_idx,
            )
            if (batch_idx + 1) % self.terminal_show_freq == 0:
                partial_epoch = epoch + batch_idx / self.len_epoch - 1
                self.writer.display_terminal(partial_epoch, epoch, "train")

        self.writer.display_terminal(self.len_epoch, epoch, mode="train", summary=True)

    def validate_epoch(self, epoch):
        """
        Perform one validation epoch.

        Args:
            epoch (int): Current epoch number.
        """
        self.model.eval()

        for batch_idx, input_subject in enumerate(self.valid_data_loader):  # type: ignore
            with torch.no_grad():
                input_tensor, target = prepare_input(
                    input_subject=input_subject, config=self.config
                )
                input_tensor.requires_grad = False

                output = self.model(input_tensor)
                loss, per_ch_score = self.criterion(output, target)

                self.writer.update_scores(
                    batch_idx,
                    loss.item(),
                    per_ch_score,
                    "val",
                    epoch * self.len_epoch + batch_idx,
                )

        self.writer.display_terminal(
            len(self.valid_data_loader), epoch, mode="val", summary=True  # type: ignore
        )
