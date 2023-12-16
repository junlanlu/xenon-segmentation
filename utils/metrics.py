"""Metrics for evaluating model performance."""
from typing import Literal, Tuple

import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from torch import Tensor


def _stat_scores(
    preds: Tensor,
    target: Tensor,
    class_index: int,
    argmax_dim: int = 1,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Calculates the number of true positive, false positive, true negative and false negative for a specific
    class.

    Args:
        preds: prediction tensor
        target: target tensor
        class_index: class to calculate over
        argmax_dim: if pred is a tensor of probabilities, this indicates the
            axis the argmax transformation will be applied over

    Return:
        True Positive, False Positive, True Negative, False Negative, Support

    Example:
        >>> x = torch.tensor([1, 2, 3])
        >>> y = torch.tensor([0, 2, 3])
        >>> tp, fp, tn, fn, sup = _stat_scores(x, y, class_index=1)
        >>> tp, fp, tn, fn, sup
        (tensor(0), tensor(1), tensor(2), tensor(0), tensor(0))
    """
    if preds.ndim == target.ndim + 1:
        preds = to_categorical(preds, argmax_dim=argmax_dim)

    tp = ((preds == class_index) * (target == class_index)).to(torch.long).sum()
    fp = ((preds == class_index) * (target != class_index)).to(torch.long).sum()
    tn = ((preds != class_index) * (target != class_index)).to(torch.long).sum()
    fn = ((preds != class_index) * (target == class_index)).to(torch.long).sum()
    sup = (target == class_index).to(torch.long).sum()

    return tp, fp, tn, fn, sup


def to_categorical(x: Tensor, argmax_dim: int = 1) -> Tensor:
    """Converts a tensor of probabilities to a dense label tensor.

    Args:
        x: probabilities to get the categorical label [N, d1, d2, ...]
        argmax_dim: dimension to apply

    Return:
        A tensor with categorical labels [N, d2, ...]

    Example:
        >>> x = torch.tensor([[0.2, 0.5], [0.9, 0.1]])
        >>> to_categorical(x)
        tensor([1, 0])
    """
    return torch.argmax(x, dim=argmax_dim)


def reduce(
    to_reduce: Tensor, reduction: Literal["elementwise_mean", "sum", "none", None]
) -> Tensor:
    """Reduces a given tensor by a given reduction method.

    Args:
        to_reduce: the tensor, which shall be reduced
        reduction:  a string specifying the reduction method ('elementwise_mean', 'none', 'sum')

    Return:
        reduced Tensor

    Raise:
        ValueError if an invalid reduction parameter was given
    """
    if reduction == "elementwise_mean":
        return torch.mean(to_reduce)
    if reduction == "none" or reduction is None:
        return to_reduce
    if reduction == "sum":
        return torch.sum(to_reduce)
    raise ValueError("Reduction parameter unknown.")


def dice(target: np.ndarray, output: np.ndarray):
    intersection = np.sum(target * output, axis=(1, 2, 3))
    union = np.sum(target + output, axis=(1, 2, 3))
    dice_coefficient = (2.0 * intersection) / (
        union + 1e-7
    )  # Adding epsilon to avoid division by zero
    return dice_coefficient


def dice_score(
    preds: Tensor,
    target: Tensor,
    bg: bool = False,
    nan_score: float = 0.0,
    no_fg_score: float = 0.0,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
) -> Tensor:
    """Compute dice score from prediction scores.

    Args:
        preds: estimated probabilities
        target: ground-truth labels
        bg: whether to also compute dice for the background
        nan_score: score to return, if a NaN occurs during computation
        no_fg_score: score to return, if no foreground pixel was found in target
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

    Return:
        Tensor containing dice score

    Example:
        >>> from torchmetrics.functional import dice_score
        >>> pred = torch.tensor([[0.85, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.85, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.85, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.85]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> dice_score(pred, target)
        tensor(0.3333)
    """
    num_classes = preds.shape[1]
    bg_inv = 1 - int(bg)
    scores = torch.zeros(num_classes - bg_inv, device=preds.device, dtype=torch.float32)
    for i in range(bg_inv, num_classes):
        if not (target == i).any():
            # no foreground class
            scores[i - bg_inv] += no_fg_score
            continue

        # TODO: rewrite to use general `stat_scores`
        tp, fp, _, fn, _ = _stat_scores(preds=preds, target=target, class_index=i)
        denom = (2 * tp + fp + fn).to(torch.float)
        # nan result
        score_cls = (
            (2 * tp).to(torch.float) / denom if torch.is_nonzero(denom) else nan_score
        )

        scores[i - bg_inv] += score_cls
    return reduce(scores, reduction=reduction)


def weighted_dice_score(image1, image2, weight=0.5):
    """Compute the weighted Dice Similarity Coefficient between two images.

    Parameters:
    - image1, image2: Input binary images (same dimensions) to be compared.
    - weight: Weight for the foreground class.
        Background class will have a weight of (1 - weight).

    Returns:
    - Weighted DSC value.
    """

    # Ensure the images are binary (0 or 1)
    image1 = np.clip(image1, 0, 1)
    image2 = np.clip(image2, 0, 1)

    # Calculate the intersection and the sum of the two images
    intersection = np.sum(image1 * image2)
    im1_sum = np.sum(image1)
    im2_sum = np.sum(image2)

    # Compute the weighted DSC
    w_dsc = (2.0 * intersection + 1e-6) / (
        weight * im1_sum + (1 - weight) * im2_sum + 1e-6
    )

    return w_dsc


def weighted_f1_score(image1, image2, weight=0.5):
    """Compute the weighted F1 score between two images.

    Parameters:
    - image1, image2: Input binary images (same dimensions) to be compared.
    - weight: Weight for the foreground class.
        Background class will have a weight of (1 - weight).

    Returns:
    - Weighted F1 score value.
    """

    # Ensure the images are binary (0 or 1)
    image1 = np.clip(image1, 0, 1)
    image2 = np.clip(image2, 0, 1)

    # Calculate the intersection, the sum of the two images, and the true negatives
    intersection = np.sum(image1 * image2)
    im1_sum = np.sum(image1)
    im2_sum = np.sum(image2)

    # Compute the weighted F1 score
    w_f1 = (2.0 * intersection + 1e-6) / (
        weight * im1_sum + (1 - weight) * im2_sum + 1e-6
    )

    return w_f1


def f1_score(image1, image2):
    """Compute the F1 score between two images.

    Parameters:
    - image1, image2: Input binary images (same dimensions) to be compared.

    Returns:
    - F1 score value.
    """

    # Ensure the images are binary (0 or 1)
    image1 = np.clip(image1, 0, 1)
    image2 = np.clip(image2, 0, 1)

    # Calculate the intersection and the sum of the two images
    intersection = np.sum(image1 * image2)
    im1_sum = np.sum(image1)
    im2_sum = np.sum(image2)

    # Compute the F1 score
    f1 = (2.0 * intersection + 1e-6) / (im1_sum + im2_sum + 1e-6)

    return f1


def sensitivity(image1, image2):
    """Compute the sensitivity (recall) between two images.

    Parameters:
    - image1, image2: Input binary images (same dimensions) to be compared.

    Returns:
    - Sensitivity value.
    """

    # Ensure the images are binary (0 or 1)
    image1 = np.clip(image1, 0, 1)
    image2 = np.clip(image2, 0, 1)

    # Calculate true positives and false negatives
    true_positives = np.sum(image1 * image2)
    false_negatives = np.sum(image1 * (1 - image2))

    # Compute the sensitivity
    sens = true_positives / (true_positives + false_negatives + 1e-6)

    return sens


def mcc(image1, image2):
    """Compute the Matthews Correlation Coefficient between two images.

    Parameters:
    - image1, image2: Input binary images (same dimensions) to be compared.

    Returns:
    - MCC value.
    """

    # Ensure the images are binary (0 or 1)
    image1 = np.clip(image1, 0, 1)
    image2 = np.clip(image2, 0, 1)

    # Calculate true positives, false positives, true negatives, and false negatives
    true_positives = np.sum(image1 * image2)
    false_positives = np.sum((1 - image1) * image2)
    true_negatives = np.sum((1 - image1) * (1 - image2))
    false_negatives = np.sum(image1 * (1 - image2))

    # Compute the MCC
    numerator = true_positives * true_negatives - false_positives * false_negatives
    denominator = np.sqrt(
        (true_positives + false_positives)
        * (true_positives + false_negatives)
        * (true_negatives + false_positives)
        * (true_negatives + false_negatives)
    )

    mcc_value = numerator / (denominator + 1e-6)

    return mcc_value


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
