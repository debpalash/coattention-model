"""Quality-Aware Template Matching (QATM) via bidirectional softmax.

Implements the QATM scoring mechanism from *QATM: Quality-Aware Template
Matching For Deep Learning* (Cheng et al., CVPR 2019), adapted to work on
pre-extracted DINO patch tokens rather than convolutional feature maps.

The core idea: a good match is one where
  • the query patch chooses the canvas patch (forward direction), AND
  • the canvas patch chooses the query patch (backward direction).
The product of these two soft-assignment probabilities gives a robust,
bidirectional match score that suppresses spurious peaks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class QATMMatcher:
    """Bidirectional softmax template matcher operating on patch tokens.

    Parameters
    ----------
    alpha : float
        Temperature (softmax sharpness).  Lower = sharper distributions.
        Default 1.0 works well with L2-normalized DINO features whose
        cosine similarities typically lie in [−0.3, 0.8].
    """

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError(f"Temperature alpha must be positive, got {alpha}")
        self.alpha = alpha

    @torch.no_grad()
    def match(
        self,
        query_feat: torch.Tensor,
        canvas_feat: torch.Tensor,
        canvas_grid: tuple[int, int],
    ) -> np.ndarray:
        """Compute a QATM heatmap over the canvas.

        Parameters
        ----------
        query_feat : torch.Tensor
            Query (reference icon) patch tokens, shape [M, D].
            Should be L2-normalized.
        canvas_feat : torch.Tensor
            Canvas patch tokens, shape [N, D].
            Should be L2-normalized.
        canvas_grid : tuple[int, int]
            (H_patches, W_patches) — spatial layout of canvas tokens.

        Returns
        -------
        np.ndarray
            Heatmap of shape (H_patches, W_patches), float32 in [0, 1].
        """
        gh, gw = canvas_grid
        M = query_feat.shape[0]
        N = canvas_feat.shape[0]

        if N != gh * gw:
            raise ValueError(
                f"canvas_feat has {N} tokens but canvas_grid "
                f"implies {gh}×{gw}={gh * gw}"
            )

        # ── Step 1: pairwise cosine similarity  [M, N] ──
        # Features are already L2-normalized, so dot product = cosine sim.
        A = query_feat @ canvas_feat.T  # [M, N]

        # ── Step 2: bidirectional softmax ──
        # Forward:  each query token picks among canvas tokens
        L_canvas_given_query = F.softmax(A / self.alpha, dim=1)  # [M, N]

        # Backward: each canvas token picks among query tokens
        L_query_given_canvas = F.softmax(A / self.alpha, dim=0)  # [M, N]

        # ── Step 3: QATM score = element-wise product ──
        qatm_scores = L_canvas_given_query * L_query_given_canvas  # [M, N]

        # ── Step 4: aggregate over query dimension ──
        # Per-canvas score = max over all query patches
        per_canvas, _ = qatm_scores.max(dim=0)  # [N]

        # ── Step 5: reshape to spatial heatmap ──
        heatmap = per_canvas.reshape(gh, gw).cpu().numpy()

        # Normalize to [0, 1]
        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax - hmin > 1e-8:
            heatmap = (heatmap - hmin) / (hmax - hmin)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap.astype(np.float32)
