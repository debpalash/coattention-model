"""Image normalization and preprocessing for DINO feature extraction.

Handles resizing, color space conversion, ImageNet normalization,
and padding to patch_size multiples so the ViT backbone can process
the image without misaligned patch boundaries.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch


# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImageNormalizer:
    """Prepares raw images for DINO feature extraction.

    Performs the full preprocessing pipeline:
    1. Resize to target dimensions
    2. BGR → RGB conversion (OpenCV default is BGR)
    3. Float32 conversion and [0, 1] scaling
    4. ImageNet mean/std normalization
    5. Pad to the nearest multiple of patch_size
    6. Return as [1, 3, H, W] tensor on the target device

    Parameters
    ----------
    patch_size : int
        Patch size used by the DINO backbone (14 for DINOv2, 16 for DINOv3).
    device : torch.device
        Target device for the output tensor.
    """

    def __init__(self, patch_size: int = 14, device: torch.device | None = None) -> None:
        self.patch_size = patch_size
        self.device = device or torch.device("cpu")

    def normalize(self, image: np.ndarray, target_size: tuple[int, int]) -> torch.Tensor:
        """Normalize an image for DINO feature extraction.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format (H, W, 3), uint8.
        target_size : tuple[int, int]
            Desired (width, height) *before* padding. The image is first
            resized to this size, then padded to the nearest patch_size
            multiple.

        Returns
        -------
        torch.Tensor
            Normalized image tensor of shape [1, 3, H_padded, W_padded],
            where H_padded and W_padded are multiples of patch_size.

        Raises
        ------
        ValueError
            If the input image is not a valid 3-channel image.
        """
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected a 3-channel BGR image (H, W, 3), "
                f"got shape {getattr(image, 'shape', None)}"
            )

        target_w, target_h = target_size

        # 1. Resize to target dimensions
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # 2. BGR → RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 3. Float32, scale to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # 4. ImageNet normalization  (per-channel)
        normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD

        # 5. Pad to nearest multiple of patch_size
        padded = self._pad_to_patch_size(normalized)

        # 6. HWC → CHW, add batch dimension, move to device
        tensor = torch.from_numpy(padded.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(self.device)

        return tensor

    def _pad_to_patch_size(self, image: np.ndarray) -> np.ndarray:
        """Pad image (H, W, 3) so H and W are multiples of patch_size.

        Uses reflect-padding to avoid introducing artificial edges that
        would distort patch-level features near the border.
        """
        h, w, c = image.shape
        new_h = self._ceil_to_multiple(h, self.patch_size)
        new_w = self._ceil_to_multiple(w, self.patch_size)

        if new_h == h and new_w == w:
            return image

        pad_bottom = new_h - h
        pad_right = new_w - w

        padded = cv2.copyMakeBorder(
            image,
            top=0,
            bottom=pad_bottom,
            left=0,
            right=pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )
        return padded

    @staticmethod
    def _ceil_to_multiple(value: int, multiple: int) -> int:
        """Round *value* up to the nearest multiple of *multiple*."""
        return int(math.ceil(value / multiple)) * multiple

    def get_padded_size(self, target_size: tuple[int, int]) -> tuple[int, int]:
        """Return the (width, height) after padding a given target_size.

        Useful for pre-computing grid sizes without running the full
        normalization pipeline.
        """
        w, h = target_size
        return (
            self._ceil_to_multiple(w, self.patch_size),
            self._ceil_to_multiple(h, self.patch_size),
        )
