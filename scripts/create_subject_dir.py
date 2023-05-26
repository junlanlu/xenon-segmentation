"""Create a subject folder in the Xenon dataset.

Helps me save time when creating a new subject folder.
"""
import os
import sys
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("subject_id", None, "Subject ID")


def create_folder(_):
    if FLAGS.subject_id is None:
        print("Please provide a subject ID using the --subject_id flag.")
        sys.exit(1)

    subject_id = FLAGS.subject_id
    # Create the subject folder
    subject_folder = os.path.join("datasets/xenon/train", subject_id, "gx")

    os.makedirs(subject_folder, exist_ok=True)
    subject_folder = os.path.join(
        "datasets/xenon/train", subject_id, "dedicated_ventilation"
    )

    os.makedirs(subject_folder, exist_ok=True)


def main(argv):
    app.run(create_folder)


if __name__ == "__main__":
    main(sys.argv)
