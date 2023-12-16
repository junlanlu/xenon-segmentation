import os
import pdb

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config import base_config
from utils import constants, general, io_utils


class TensorboardWriter:
    def __init__(self, config: base_config.Config):
        name_model = (
            config.log_dir
            + config.model.name.value
            + "_"
            + config.data.name.value
            + "_"
            + general.datestr()
        )
        self.writer = SummaryWriter(
            log_dir=config.log_dir + name_model, comment=name_model
        )
        io_utils.make_dirs(config.save_dir)
        self.csv_train, self.csv_val = self.create_stats_files(config.save_dir)
        self.dataset_name = config.data.name.value
        self.classes = config.data.n_classes

        if config.data.name == constants.DatasetName.ISEG2019:
            self.label_names = constants.LabelName.ISEG2019
        elif config.data.name == constants.DatasetName.XENONSIMPLE:
            self.label_names = constants.LabelName.XENONSIMPLE
        elif config.data.name == constants.DatasetName.XENONTRACHEA:
            self.label_names = constants.LabelName.XENONTRACHEA
        else:
            raise NotImplementedError

        self.data = self.create_data_structure()

    def create_data_structure(
        self,
    ):
        data = {
            "train": dict((label, 0.0) for label in self.label_names),
            "val": dict((label, 0.0) for label in self.label_names),
        }
        data["train"]["loss"] = 0.0
        data["val"]["loss"] = 0.0
        data["train"]["count"] = 1.0
        data["val"]["count"] = 1.0
        data["train"]["dsc"] = 0.0
        data["val"]["dsc"] = 0.0
        return data

    def display_terminal(self, iter, epoch, mode="train", summary=False):
        """

        :param iter: iteration or partial epoch
        :param epoch: epoch of training
        :param loss: any loss numpy
        :param mode: train or val ( for training and validation)
        :param summary: to print total statistics at the end of epoch
        """
        if summary:
            info_print = (
                "\nSummary {} Epoch {:2d}:  Loss:{:.4f} \t DSC:{:.4f}  ".format(
                    mode,
                    epoch,
                    self.data[mode]["loss"] / self.data[mode]["count"],
                    self.data[mode]["dsc"] / self.data[mode]["count"],
                )
            )

            for i in range(len(self.label_names)):
                info_print += "\t{} : {:.4f}".format(
                    self.label_names[i],
                    self.data[mode][self.label_names[i]] / self.data[mode]["count"],
                )

            print(info_print)
        else:
            info_print = "\nEpoch: {:.2f} Loss:{:.4f} \t DSC:{:.4f}".format(
                iter,
                self.data[mode]["loss"] / self.data[mode]["count"],
                self.data[mode]["dsc"] / self.data[mode]["count"],
            )

            for i in range(len(self.label_names)):
                info_print += "\t{}:{:.4f}".format(
                    self.label_names[i],
                    self.data[mode][self.label_names[i]] / self.data[mode]["count"],
                )
            print(info_print)

    def create_stats_files(self, path):
        train_f = open(os.path.join(path, "train.csv"), "w")
        val_f = open(os.path.join(path, "val.csv"), "w")
        return train_f, val_f

    def reset(self, mode):
        self.data[mode]["dsc"] = 0.0
        self.data[mode]["loss"] = 0.0
        self.data[mode]["count"] = 1
        for i in range(len(self.label_names)):
            self.data[mode][self.label_names[i]] = 0.0

    def update_scores(self, iter, loss, channel_score, mode, writer_step):
        """
        :param iter: iteration or partial epoch
        :param loss: any loss torch.tensor.item()
        :param channel_score: per channel score or dice coef
        :param mode: train or val ( for training and validation)
        :param writer_step: tensorboard writer step
        """
        # WARNING ASSUMING THAT CHANNELS IN SAME ORDER AS DICTIONARY

        dice_coeff = np.mean(channel_score) * 100

        num_channels = len(channel_score)
        self.data[mode]["dsc"] += dice_coeff
        self.data[mode]["loss"] += loss
        self.data[mode]["count"] = iter + 1

        for i in range(num_channels):
            self.data[mode][self.label_names[i]] += channel_score[i]
            if self.writer is not None:
                self.writer.add_scalar(
                    mode + "/" + self.label_names[i],
                    channel_score[i],
                    global_step=writer_step,
                )

    def write_end_of_epoch(self, epoch):
        self.writer.add_scalars(
            "DSC/",
            {
                "train": self.data["train"]["dsc"] / self.data["train"]["count"],
                "val": self.data["val"]["dsc"] / self.data["val"]["count"],
            },
            epoch,
        )
        self.writer.add_scalars(
            "Loss/",
            {
                "train": self.data["train"]["loss"] / self.data["train"]["count"],
                "val": self.data["val"]["loss"] / self.data["val"]["count"],
            },
            epoch,
        )
        for i in range(len(self.label_names)):
            self.writer.add_scalars(
                self.label_names[i],
                {
                    "train": self.data["train"][self.label_names[i]]
                    / self.data["train"]["count"],
                    "val": self.data["val"][self.label_names[i]]
                    / self.data["train"]["count"],
                },
                epoch,
            )

        train_csv_line = "Epoch:{:2d} Loss:{:.4f} DSC:{:.4f}".format(
            epoch,
            self.data["train"]["loss"] / self.data["train"]["count"],
            self.data["train"]["dsc"] / self.data["train"]["count"],
        )
        val_csv_line = "Epoch:{:2d} Loss:{:.4f} DSC:{:.4f}".format(
            epoch,
            self.data["val"]["loss"] / self.data["val"]["count"],
            self.data["val"]["dsc"] / self.data["val"]["count"],
        )
        self.csv_train.write(train_csv_line + "\n")
        self.csv_val.write(val_csv_line + "\n")
