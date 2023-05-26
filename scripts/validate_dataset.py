"""Ensure that all subject subdirectories contain all falls."""
import glob
import logging
import os

from absl import app, flags

flags.DEFINE_string("directory_path", "datasets/xenon", "Path to the directory")
FLAGS = flags.FLAGS


def check_directory_files(directory_path):
    is_valid = True
    for root, dirs, files in os.walk(directory_path):
        nifti_files = [file for file in files if file.endswith(".nii")]
        if len(nifti_files) > 0:
            if (
                "gas.nii" not in nifti_files
                or "mask.nii" not in nifti_files
                or "proton.nii" not in nifti_files
            ):
                logging.info("Missing files in {}".format(root))
                is_valid = False
    logging.info("Dataset is valid: {}".format(is_valid))


def main(argv):
    check_directory_files(FLAGS.directory_path)


if __name__ == "__main__":
    app.run(main)
