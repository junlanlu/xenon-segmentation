"""Tester class for evaluating model."""

import logging
import pdb
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from torch.utils.data.dataloader import DataLoader

from config import base_config
from utils import img_utils, io_utils, metrics, plot
from utils.io_utils import prepare_input


class Tester:
    """Tester class for testing a model."""

    def __init__(
        self,
        config: base_config.Config,
        model: torch.nn.Module,
        test_data_loader: DataLoader,
    ):
        """
        Initialize the Tester class.

        Args:
            config (base_config.Config): Configuration object.
            model (torch.nn.Module): Model to be tested.
            test_data_loader (torch.utils.data.DataLoader): Test data loader.
        """
        self.config = config
        self.model = model
        self.test_data_loader = test_data_loader
        self.len_test = len(self.test_data_loader)
        self.total_f1 = 0
        self.total_mcc = 0
        self.total_sensitivity = 0

    def testing(self):
        self.model.eval()

        for batch_idx, input_subject in enumerate(self.test_data_loader):
            with torch.no_grad():
                input_tensor, target = prepare_input(
                    input_subject=input_subject, config=self.config
                )
                input_tensor.requires_grad = False
                plot.plot_subject(input_subject)
                output = self.model(input_tensor)
                x = img_utils.one_hot_to_label(output)

                self.generate_figures(input_tensor, target, x)
                self.update_metrics(target, x)
                # Here you can add any evaluation metrics you need
                # For example computing the dice score between output and target
                io_utils.save_tensor_to_nii(target, "tmp/target.nii")
                io_utils.save_tensor_to_nii(input_tensor, "tmp/input.nii")
                io_utils.save_tensor_to_nii(x, "tmp/output.nii")
                logging.info("processed subject idx: {}".format(input_subject["name"]))
        # Compute average metrics
        avg_f1 = self.total_f1 / self.len_test
        avg_mcc = self.total_mcc / self.len_test
        avg_sensitivity = self.total_sensitivity / self.len_test

        logging.info(f"Average Dice score: {avg_f1:.4f}")
        logging.info(f"Average MCC: {avg_mcc:.4f}")
        logging.info(f"Average Sensitivity: {avg_sensitivity:.4f}")
        logging.info("Finished testing")

    def generate_figures(
        self, input: torch.Tensor, target: torch.Tensor, output: torch.Tensor
    ):
        """Generate figures for the report."""
        input = input.squeeze().cpu().numpy()
        target = target.squeeze().cpu().numpy()
        output = output.squeeze().cpu().numpy()

        index_start, index_skip = plot.get_plot_indices(target)
        index_skip = max(index_skip, 1)
        plot.plot_montage_color(
            image=plot.overlay_mask_on_image(np.abs(input), target.astype("uint8")),
            path="tmp/target_overlay.png",
            index_start=index_start,
            index_skip=index_skip,
        )

        plot.plot_montage_color(
            image=plot.overlay_mask_on_image(np.abs(input), output.astype("uint8")),
            path="tmp/output_overlay.png",
            index_start=index_start,
            index_skip=index_skip,
        )

    def update_metrics(self, target: torch.Tensor, output: torch.Tensor):
        output_onehot = F.one_hot(output.to(torch.int64).flatten(), num_classes=2)
        dice = metrics.dice_score(output_onehot, target.flatten())
        # output metrics, but convert to numpy first
        output = torch.squeeze(output).detach().cpu().numpy()
        target = torch.squeeze(target).detach().cpu().numpy()
        f1 = metrics.f1_score(output, target)
        mcc = metrics.mcc(output, target)
        sensitivity = metrics.sensitivity(output, target)
        self.total_f1 += f1
        self.total_mcc += mcc
        self.total_sensitivity += sensitivity
        logging.info("Dice score: {}".format(f1))
        logging.info("MCC: {}".format(mcc))
        logging.info("Sensitivity: {}".format(sensitivity))

    def reset_metrics(self):
        self.total_f1 = 0
        self.total_mcc = 0
        self.total_sensitivity = 0
