from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .utils import timer


def read_image(image_path: Path, to_rgb: bool = True) -> NDArray[np.uint8]:
    image = cv2.imread(str(image_path))
    if to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def scale(image: NDArray[np.float32], normalize: bool = True) -> NDArray[np.float32]:
    image = image.astype(np.float32) / 255.0
    if normalize:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
    return image


def gaussian_weight_map(size: int, sigma_ratio: float = 0.25) -> NDArray[np.float32]:
    tmp = np.linspace(-1, 1, size)
    x, y = np.meshgrid(tmp, tmp)
    d = np.sqrt(x * x + y * y)
    sigma = 2 * sigma_ratio**2
    g_map = np.exp(-((d**2) / sigma))
    return g_map.astype(np.float32)


def sigmoid(image: NDArray[np.float32]) -> NDArray[np.float32]:
    return 1.0 / (1.0 + np.exp(-image))


def remove_small_objects(mask: NDArray[np.uint8], min_size: int) -> NDArray[np.uint8]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    output_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output_mask[labels == i] = 1
    return output_mask


def fill_small_holes(mask: NDArray[np.uint8], min_size: int) -> NDArray[np.uint8]:
    inverted_mask = 1 - mask
    cleaned_inverted_mask = remove_small_objects(inverted_mask, min_size)
    return 1 - cleaned_inverted_mask


@timer
def post_process(
    image: NDArray[np.float32],
    confidence: float,
    smoothing_sigma: float = 1.0,
    closing_kernel_size: int = 5,
    min_object_size: int = 128,
) -> NDArray[np.uint8]:
    assert image.dtype == np.float32
    if smoothing_sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), smoothing_sigma)
    binary_mask = (image > confidence).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size)
    )
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    binary_mask = remove_small_objects(binary_mask, min_object_size)
    binary_mask = fill_small_holes(binary_mask, min_object_size)

    return binary_mask
