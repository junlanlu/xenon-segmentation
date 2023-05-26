"""Tester class for evaluating model."""

import logging
import pdb
from typing import Optional

import numpy as np
import torch
import torchio as tio
from torch.utils.data.dataloader import DataLoader

from config import base_config
from utils import img_utils, io_utils, plot
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
                # Here you can add any evaluation metrics you need
                # For example computing the dice score between output and target
                io_utils.save_tensor_to_nii(target, "tmp/target.nii")
                io_utils.save_tensor_to_nii(input_tensor, "tmp/input.nii")
                io_utils.save_tensor_to_nii(x, "tmp/output.nii")
                logging.info("processed subject idx: {}".format(batch_idx))
        logging.info("Finished testing")

    def generate_figures(
        self, input: torch.Tensor, target: torch.Tensor, output: torch.Tensor
    ):
        """Generate figures for the report."""
        input = input.squeeze().cpu().numpy()
        target = target.squeeze().cpu().numpy()
        output = output.squeeze().cpu().numpy()

        index_start, index_skip = plot.get_plot_indices(target)

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
