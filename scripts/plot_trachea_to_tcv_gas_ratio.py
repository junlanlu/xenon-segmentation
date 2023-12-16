"""Plot boxplots and perform statistical test for trachea signal and segmentation."""
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags
from scipy.stats import mannwhitneyu  # Import Mann-Whitney U test

sys.path.append(".")
from utils import io_utils, plot

HEALTHY_DIR = "datasets/xenon/healthy"
ILD_DIR = "datasets/xenon/genentech-ipf"


def main(unused_argv):
    healthy_subjects = os.listdir(HEALTHY_DIR)
    ild_subjects = os.listdir(ILD_DIR)

    trachea_to_tcv_ratio_healthy_list = []
    trachea_to_tcv_ratio_ild_list = []
    for healthy_subject in healthy_subjects:
        image = io_utils.import_nii(
            path=os.path.join(HEALTHY_DIR, healthy_subject, "gas.nii")
        )
        mask_tcv = io_utils.import_nii(
            path=os.path.join(HEALTHY_DIR, healthy_subject, "output_tcv.nii")
        ).astype("bool")
        mask_trachea = io_utils.import_nii(
            path=os.path.join(HEALTHY_DIR, healthy_subject, "output_trachea.nii")
        ).astype("bool")

        if np.sum(mask_tcv) == 0:
            trachea_to_tcv_ratio_healthy = 0
        else:
            trachea_to_tcv_ratio_healthy = np.mean(
                np.abs(image[mask_trachea])
            ) / np.mean(np.abs(image[mask_tcv]))
        trachea_to_tcv_ratio_healthy_list.append(trachea_to_tcv_ratio_healthy)

    for ild_subject in ild_subjects:
        image = io_utils.import_nii(path=os.path.join(ILD_DIR, ild_subject, "gas.nii"))
        mask_tcv = io_utils.import_nii(
            path=os.path.join(ILD_DIR, ild_subject, "output_tcv.nii")
        ).astype("bool")
        mask_trachea = io_utils.import_nii(
            path=os.path.join(ILD_DIR, ild_subject, "output_trachea.nii")
        ).astype("bool")
        if np.sum(mask_tcv) == 0:
            trachea_to_tcv_ratio_ild = 0
        else:
            trachea_to_tcv_ratio_ild = np.mean(np.abs(image[mask_trachea])) / np.mean(
                np.abs(image[mask_tcv])
            )
        trachea_to_tcv_ratio_ild_list.append(trachea_to_tcv_ratio_ild)
    # Create a boxplot to compare the two sets of data
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 14
    plt.boxplot([trachea_to_tcv_ratio_healthy_list, trachea_to_tcv_ratio_ild_list])
    plt.xticks([1, 2], ["Healthy", "ILD"])
    plt.xlabel("Group")
    plt.ylabel("Trachea to TCV Ratio")
    plt.title("Comparison of Trachea to TCV Ratio between Healthy and ILD Groups")
    plt.savefig("tmp/trachea_to_tcv_ratio_boxplot.png")

    # Perform the Mann-Whitney U test to compare the two groups
    statistic, p_value = mannwhitneyu(
        trachea_to_tcv_ratio_healthy_list, trachea_to_tcv_ratio_ild_list
    )
    print(f"Mann-Whitney U Statistic: {statistic}")
    print(f"p-value: {p_value}")


if __name__ == "__main__":
    app.run(main)
