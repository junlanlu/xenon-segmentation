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
