"""Plot example images of trachea signal and segmentation."""
import os
import sys

import numpy as np
from absl import app, flags

sys.path.append(".")
from utils import io_utils, plot

DATA_DIR = flags.DEFINE_string(
    "data_dir", "datasets/xenon/train/009-008/gx", "Path to nifti data."
)


def main(unused_argv):
    image = io_utils.import_nii(path=DATA_DIR.value + "/gas.nii")
    mask = io_utils.import_nii(path=DATA_DIR.value + "/output.nii")
    plot.plot_slice_color(
        image=plot.overlay_mask_on_image(image=image, mask=mask.astype("uint8")),
        path="tmp/png/{}_overlay.png".format(os.path.basename(DATA_DIR.value)),
        index=55,
    )
    plot.plot_slice_grey(
        image=np.abs(image),
        path="tmp/png/{}_grey.png".format(os.path.basename(DATA_DIR.value)),
        index=55,
    )


if __name__ == "__main__":
    app.run(main)
