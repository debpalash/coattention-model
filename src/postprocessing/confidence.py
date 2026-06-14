"""Confidence scoring for match quality assessment.

Produces a combined confidence score from three complementary signals
that together measure how "clean" and unambiguous the heatmap peak is.

Signals
-------
1. **peak_similarity** — raw cosine similarity at the best-match location.
   High = the template genuinely resembles that canvas region.

2. **peak_to_second_ratio** — ratio of the best peak to the second-best
   spatially-separated peak.  High = the match is unambiguous.

3. **peak_sharpness** — ratio of peak value to mean, measuring how
   concentrated the response is.  High = the peak stands out from noise.
"""

from __future__ import annotations

import cv2
import numpy as np


class ConfidenceScorer:
    """Score the quality/confidence of a heatmap match.

    Parameters
    ----------
    weight_sim : float
        Weight for peak_similarity in the combined score.
    weight_ratio : float
        Weight for peak_to_second_ratio in the combined score.
    weight_sharpness : float
        Weight for peak_sharpness in the combined score.
    min_peak_distance : int
        Minimum distance (in heatmap cells) between the primary peak and
        the second peak for the ratio computation.  Prevents the second
        peak from being a neighbor of the first.
    """

    def __init__(
        self,
        weight_sim: float = 0.40,
        weight_ratio: float = 0.35,
        weight_sharpness: float = 0.25,
        min_peak_distance: int = 3,
    ) -> None:
        total = weight_sim + weight_ratio + weight_sharpness
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.4f} "
                f"({weight_sim} + {weight_ratio} + {weight_sharpness})"
            )
        self.weight_sim = weight_sim
        self.weight_ratio = weight_ratio
        self.weight_sharpness = weight_sharpness
        self.min_peak_distance = min_peak_distance

    def score(self, heatmap: np.ndarray) -> float:
        """Compute the combined confidence score.

        Parameters
        ----------
        heatmap : np.ndarray
            Match heatmap, shape (H, W), float32, expected to be in [0, 1]
            (though it will still work if not — the individual signals
            handle arbitrary ranges).

        Returns
        -------
        float
            Combined confidence in [0, 1].
        """
        sim = self.peak_similarity(heatmap)
        ratio = self.peak_to_second_ratio(heatmap)
        sharpness = self.peak_sharpness(heatmap)

        combined = (
            self.weight_sim * sim
            + self.weight_ratio * ratio
            + self.weight_sharpness * sharpness
        )
        return float(np.clip(combined, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Individual signal methods
    # ------------------------------------------------------------------

    @staticmethod
    def peak_similarity(heatmap: np.ndarray) -> float:
        """Raw maximum value in the heatmap.

        When the heatmap is derived from cosine similarity (or normalized
        to [0, 1]), this directly reflects how similar the best canvas
        patch is to the query.
        """
        return float(heatmap.max())

    def peak_to_second_ratio(self, heatmap: np.ndarray) -> float:
        """Ratio of best peak to second-best spatially-separated peak.

        Returns a value in [0, 1] computed as
        ``1 - second_peak / first_peak`` (clamped).

        A value near 1 means the best peak is much stronger than any
        competitor — highly unambiguous.
        """
        if heatmap.size < 2:
            return 1.0

        # Find the primary peak
        _, max_val, _, max_loc = cv2.minMaxLoc(heatmap)
        if max_val < 1e-8:
            return 0.0

        # Suppress the region around the primary peak
        suppressed = heatmap.copy()
        h, w = suppressed.shape
        px, py = max_loc  # (x, y)
        d = self.min_peak_distance

        y_min = max(0, py - d)
        y_max = min(h, py + d + 1)
        x_min = max(0, px - d)
        x_max = min(w, px + d + 1)
        suppressed[y_min:y_max, x_min:x_max] = 0.0

        second_max = float(suppressed.max())

        # Ratio: how much stronger the primary peak is
        ratio = 1.0 - (second_max / max_val)
        return float(np.clip(ratio, 0.0, 1.0))

    @staticmethod
    def peak_sharpness(heatmap: np.ndarray) -> float:
        """Ratio of peak value to mean value (normalized to [0, 1]).

        A sharp, well-localized peak will have a high ratio; a diffuse
        response will have a ratio close to 1 (mapped to low confidence).
        """
        peak = float(heatmap.max())
        mean = float(heatmap.mean())

        if mean < 1e-8:
            return 1.0 if peak > 1e-8 else 0.0

        # Raw ratio — typically 2…20 for good matches
        raw_ratio = peak / mean

        # Sigmoid-like mapping to [0, 1]:
        # ratio=1 → 0.0 (flat, no peak),  ratio=5 → ~0.8,  ratio=10 → ~0.95
        sharpness = 1.0 - 1.0 / raw_ratio
        return float(np.clip(sharpness, 0.0, 1.0))

    def score_detailed(self, heatmap: np.ndarray) -> dict[str, float]:
        """Return all signals and the combined score as a dict."""
        sim = self.peak_similarity(heatmap)
        ratio = self.peak_to_second_ratio(heatmap)
        sharpness = self.peak_sharpness(heatmap)
        combined = float(np.clip(
            self.weight_sim * sim
            + self.weight_ratio * ratio
            + self.weight_sharpness * sharpness,
            0.0, 1.0,
        ))
        return {
            "peak_similarity": sim,
            "peak_to_second_ratio": ratio,
            "peak_sharpness": sharpness,
            "confidence": combined,
        }
