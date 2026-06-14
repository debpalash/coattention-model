"""Peak localization with sub-pixel refinement.

Takes a low-resolution patch-level heatmap, upsamples it to the original
canvas resolution, and finds the precise click target using Gaussian
smoothing + weighted centroid refinement around the peak.
"""

from __future__ import annotations

import cv2
import numpy as np


class PeakLocalizer:
    """Localize the best match position from a heatmap.

    Pipeline:
    1. Bicubic upsample heatmap → canvas resolution
    2. Gaussian blur to suppress noise / quantization artifacts
    3. Find global peak via ``cv2.minMaxLoc``
    4. Refine with weighted centroid in an 11×11 window around the peak

    Parameters
    ----------
    blur_sigma : float
        Sigma for Gaussian smoothing of the upsampled heatmap.
        Larger = smoother but less precise; smaller = noisier.
    refine_window : int
        Side length of the square window used for centroid refinement.
        Must be odd.
    """

    def __init__(self, blur_sigma: float = 3.0, refine_window: int = 11) -> None:
        if refine_window % 2 == 0:
            raise ValueError(f"refine_window must be odd, got {refine_window}")
        self.blur_sigma = blur_sigma
        self.refine_window = refine_window

    def localize(
        self,
        heatmap: np.ndarray,
        canvas_size: tuple[int, int],
    ) -> tuple[float, float]:
        """Find the best-match (x, y) position in canvas pixel coordinates.

        Parameters
        ----------
        heatmap : np.ndarray
            Heatmap of shape (H_patches, W_patches), float32 in [0, 1].
        canvas_size : tuple[int, int]
            Original canvas (width, height) in pixels.

        Returns
        -------
        tuple[float, float]
            Predicted (x, y) click position in canvas pixel space.
        """
        canvas_w, canvas_h = canvas_size

        # 1. Upsample heatmap to canvas resolution (bicubic)
        heatmap_up = cv2.resize(
            heatmap,
            (canvas_w, canvas_h),
            interpolation=cv2.INTER_CUBIC,
        )

        # Clamp to [0, 1] — bicubic can overshoot
        heatmap_up = np.clip(heatmap_up, 0.0, 1.0)

        # 2. Gaussian blur for smoothing
        ksize = self._auto_kernel_size(self.blur_sigma)
        heatmap_smooth = cv2.GaussianBlur(
            heatmap_up, (ksize, ksize), self.blur_sigma
        )

        # 3. Find global peak
        _, max_val, _, max_loc = cv2.minMaxLoc(heatmap_smooth)
        peak_x, peak_y = max_loc  # (x, y) — OpenCV convention

        # 4. Refine with weighted centroid
        refined_x, refined_y = self._refine_peak(
            heatmap_smooth, peak_x, peak_y
        )

        return (refined_x, refined_y)

    def _refine_peak(
        self,
        heatmap: np.ndarray,
        peak_x: int,
        peak_y: int,
    ) -> tuple[float, float]:
        """Refine the peak location with a weighted centroid in a local window.

        Extracts a window around the coarse peak and computes the
        intensity-weighted centroid for sub-pixel precision.
        """
        h, w = heatmap.shape
        half = self.refine_window // 2

        # Clamp window bounds to image
        y_min = max(0, peak_y - half)
        y_max = min(h, peak_y + half + 1)
        x_min = max(0, peak_x - half)
        x_max = min(w, peak_x + half + 1)

        window = heatmap[y_min:y_max, x_min:x_max]

        # Raise to a power to sharpen the peak for centroid computation
        weights = np.power(window, 2)
        total_weight = weights.sum()

        if total_weight < 1e-12:
            # Degenerate — return the raw peak
            return (float(peak_x), float(peak_y))

        # Build coordinate grids relative to the window origin
        ys = np.arange(y_min, y_max, dtype=np.float64)
        xs = np.arange(x_min, x_max, dtype=np.float64)
        xs_grid, ys_grid = np.meshgrid(xs, ys)

        centroid_x = float(np.sum(xs_grid * weights) / total_weight)
        centroid_y = float(np.sum(ys_grid * weights) / total_weight)

        return (centroid_x, centroid_y)

    @staticmethod
    def _auto_kernel_size(sigma: float) -> int:
        """Compute an odd kernel size that covers ±3σ."""
        ksize = int(2 * round(3 * sigma) + 1)
        return max(ksize, 3)

    def localize_with_heatmap(
        self,
        heatmap: np.ndarray,
        canvas_size: tuple[int, int],
    ) -> tuple[float, float, np.ndarray]:
        """Like ``localize`` but also returns the upsampled/smoothed heatmap.

        Useful for visualization and debugging.
        """
        canvas_w, canvas_h = canvas_size

        heatmap_up = cv2.resize(
            heatmap,
            (canvas_w, canvas_h),
            interpolation=cv2.INTER_CUBIC,
        )
        heatmap_up = np.clip(heatmap_up, 0.0, 1.0)

        ksize = self._auto_kernel_size(self.blur_sigma)
        heatmap_smooth = cv2.GaussianBlur(
            heatmap_up, (ksize, ksize), self.blur_sigma
        )

        _, _, _, max_loc = cv2.minMaxLoc(heatmap_smooth)
        peak_x, peak_y = max_loc

        refined_x, refined_y = self._refine_peak(
            heatmap_smooth, peak_x, peak_y
        )

        return (refined_x, refined_y, heatmap_smooth)
