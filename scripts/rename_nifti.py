"""Rename files from network drive to standardized names."""
import glob
import logging
import os
import pdb

from absl import app, flags

flags.DEFINE_string("directory_path", "datasets/xenon", "Path to the directory")
FLAGS = flags.FLAGS


def rename_nifti_files(directory_path: str):
    """Rename files from network drive to standardized names.

    Crawls through the directory and renames files to the following:
        gas.nii
        proton.nii
        mask.nii

    Args:
        directory_path: Path to the directory.
    """
    for root, _, files in os.walk(directory_path):
        for file in files:
            if (
                file.lower().find("gas") != -1 or file.lower().find("vent") != -1
            ) and file.lower().find("mask") == -1:
                new_name = "gas.nii"
            elif file.lower().find("ute") != -1:
                new_name = "proton.nii"
            elif file.lower().find("mask") != -1:
                new_name = "mask.nii"

            else:
                continue  # Skip files that don't match the conditions

            file_path = os.path.join(root, file)
            new_file_path = os.path.join(root, new_name)

            if file != new_name:
                logging.info("Renaming {} to {}".format(file, new_name))
                os.rename(file_path, new_file_path)


def main(argv):
    rename_nifti_files(FLAGS.directory_path)


if __name__ == "__main__":
    app.run(main)
