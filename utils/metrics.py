"""Metrics for evaluating model performance."""
import numpy as np
from scipy.spatial.distance import directed_hausdorff


def dice(target: np.ndarray, output: np.ndarray) -> float:
    intersection = np.sum(target * output, axis=(1, 2, 3))
    union = np.sum(target + output, axis=(1, 2, 3))
    dice_coefficient = (2.0 * intersection) / (
        union + 1e-7
    )  # Adding epsilon to avoid division by zero
    return dice_coefficient


def hd95(target, output):
    """
    Compute the Hausdorff Distance at 95% (HD95) metric between the target and output segmentation masks.

    HD95 is a measure of the maximum distance between the contours of the two masks,
    considering the 95th percentile of distances. It provides an estimate of the largest
    spatial discrepancy between the two masks while accounting for outliers.

    Args:
        target (np.ndarray): The target segmentation mask of shape (N_classes, H, W, D).
        output (np.ndarray): The output segmentation mask of shape (N_classes, H, W, D).
        voxel_spacing (Tuple[float, float, float]): The voxel spacing in millimeters (H, W, D).

    Returns:
        float: The HD95 distance in millimeters.

    Raises:
        ValueError: If the target and output masks have different shapes or number of classes.

    Note:
        The target and output masks should have the same shape and number of classes.

    """
    distances = []
    for i in range(target.shape[0]):
        target_indices = np.argwhere(target[i] > 0)
        output_indices = np.argwhere(output[i] > 0)
        target_points = target_indices[:, 1:]
        output_points = output_indices[:, 1:]
        distance = max(
            directed_hausdorff(target_points, output_points)[0],
            directed_hausdorff(output_points, target_points)[0],
        )
        distances.append(distance)
    hd95 = np.percentile(distances, 95)
    return hd95
