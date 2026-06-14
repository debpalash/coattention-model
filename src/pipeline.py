"""End-to-end CAPTCHA solver pipeline.

Orchestrates:  Preprocessing → DINO Feature Extraction → QATM Matching
→ Peak Localization → Confidence Scoring.

Usage
-----
>>> from src.pipeline import CaptchaSolver
>>> solver = CaptchaSolver()
>>> result = solver.solve(ref_image_bgr, canvas_image_bgr)
>>> print(f"Click at ({result.x:.1f}, {result.y:.1f})  conf={result.confidence:.3f}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from src.backbone.feature_extractor import DINOFeatureExtractor
from src.matching.qatm import QATMMatcher
from src.matching.correlation import CorrelationMatcher
from src.postprocessing.localizer import PeakLocalizer
from src.postprocessing.confidence import ConfidenceScorer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatchResult:
    """Result of a CAPTCHA solve attempt.

    Attributes
    ----------
    x : float
        Predicted click x-coordinate in canvas pixel space.
    y : float
        Predicted click y-coordinate in canvas pixel space.
    confidence : float
        Combined confidence score in [0, 1].
    peak_similarity : float
        Raw peak cosine similarity signal.
    peak_to_second_ratio : float
        Ratio of best peak to second-best (unambiguity).
    peak_sharpness : float
        Concentration of the heatmap peak.
    heatmap : np.ndarray | None
        Optional raw patch-level heatmap (H_patches, W_patches).
        Only populated when ``return_heatmap=True``.
    elapsed_ms : float
        Wall-clock time for the full solve, in milliseconds.
    """

    x: float
    y: float
    confidence: float
    peak_similarity: float
    peak_to_second_ratio: float
    peak_sharpness: float
    heatmap: np.ndarray | None = None
    elapsed_ms: float = 0.0


class CaptchaSolver:
    """Orchestrates the full CAPTCHA-solving pipeline.

    Parameters
    ----------
    ref_size : tuple[int, int]
        (width, height) to resize the reference icon to.
    matcher_type : Literal["qatm", "correlation"]
        Which matching algorithm to use.
    qatm_alpha : float
        Temperature for QATM softmax (only used if matcher_type="qatm").
    blur_sigma : float
        Gaussian sigma for peak localization smoothing.
    device : torch.device | None
        Inference device.  Auto-detected if None.
    force_dino_version : Literal["v2", "v3"] | None
        Force a specific DINO version (skip auto-detection).
    """

    def __init__(
        self,
        ref_size: tuple[int, int] = (224, 224),
        matcher_type: Literal["qatm", "correlation"] = "qatm",
        qatm_alpha: float = 1.0,
        blur_sigma: float = 3.0,
        device: torch.device | None = None,
        force_dino_version: Literal["v2", "v3"] | None = None,
    ) -> None:
        self.ref_size = ref_size
        self.matcher_type = matcher_type

        # ── Build components ──
        logger.info("Initializing CaptchaSolver (matcher=%s)…", matcher_type)

        self.feature_extractor = DINOFeatureExtractor(
            device=device, force_version=force_dino_version
        )

        if matcher_type == "qatm":
            self.matcher = QATMMatcher(alpha=qatm_alpha)
        elif matcher_type == "correlation":
            self.matcher = CorrelationMatcher()
        else:
            raise ValueError(f"Unknown matcher_type: {matcher_type!r}")

        self.localizer = PeakLocalizer(blur_sigma=blur_sigma)
        self.scorer = ConfidenceScorer()

        logger.info("CaptchaSolver ready.")

    def solve(
        self,
        ref_image: np.ndarray,
        canvas_image: np.ndarray,
        return_heatmap: bool = False,
    ) -> MatchResult:
        """Solve a CAPTCHA: find where *ref_image* appears in *canvas_image*.

        Parameters
        ----------
        ref_image : np.ndarray
            Reference icon (BGR, uint8), typically small (50–150 px).
        canvas_image : np.ndarray
            Full puzzle canvas (BGR, uint8).
        return_heatmap : bool
            If True, attach the raw patch-level heatmap to the result.

        Returns
        -------
        MatchResult
            Predicted click location + confidence metrics.
        """
        t0 = time.perf_counter()

        # ── Validate inputs ──
        self._validate_image(ref_image, "ref_image")
        self._validate_image(canvas_image, "canvas_image")

        canvas_h, canvas_w = canvas_image.shape[:2]

        # ── Compute canvas target size ──
        # Process at near-native resolution; the normalizer will pad to
        # the nearest patch_size multiple automatically.
        canvas_target = (canvas_w, canvas_h)

        # ── Extract features ──
        ref_feat = self.feature_extractor.extract(ref_image, self.ref_size)
        canvas_feat = self.feature_extractor.extract(canvas_image, canvas_target)

        # Patch grid for the canvas (after padding)
        canvas_grid = self.feature_extractor.get_padded_grid_size(canvas_target)

        logger.debug(
            "Features — ref: %s  canvas: %s  grid: %s",
            ref_feat.shape,
            canvas_feat.shape,
            canvas_grid,
        )

        # ── Match ──
        heatmap = self.matcher.match(ref_feat, canvas_feat, canvas_grid)

        # ── Localize ──
        x, y = self.localizer.localize(heatmap, (canvas_w, canvas_h))

        # ── Score confidence ──
        scores = self.scorer.score_detailed(heatmap)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = MatchResult(
            x=x,
            y=y,
            confidence=scores["confidence"],
            peak_similarity=scores["peak_similarity"],
            peak_to_second_ratio=scores["peak_to_second_ratio"],
            peak_sharpness=scores["peak_sharpness"],
            heatmap=heatmap if return_heatmap else None,
            elapsed_ms=elapsed_ms,
        )

        logger.info(
            "Solve complete: (%.1f, %.1f)  conf=%.3f  elapsed=%.0f ms",
            result.x,
            result.y,
            result.confidence,
            result.elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_image(image: np.ndarray, name: str) -> None:
        """Raise ValueError if *image* is not a valid BGR uint8 image."""
        if image is None:
            raise ValueError(f"{name} is None")
        if not isinstance(image, np.ndarray):
            raise ValueError(f"{name} must be a numpy array, got {type(image)}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"{name} must be a 3-channel image (H, W, 3), "
                f"got shape {image.shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(
                f"{name} must be uint8, got dtype {image.dtype}"
            )
