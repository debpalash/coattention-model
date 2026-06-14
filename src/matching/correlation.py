"""Simple mean-pool correlation matcher — the fast baseline.

Mean-pools the query tokens into a single prototype vector, then computes
cosine similarity with each canvas patch token.  Much faster than QATM
(O(N) vs O(M×N)) but less discriminative for fine-grained matching.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class CorrelationMatcher:
    """Correlation-based template matcher using mean-pooled query features.

    This is intentionally simple — it serves as the fast baseline against
    which the more expensive QATM matcher can be compared.
    """

    @torch.no_grad()
    def match(
        self,
        query_feat: torch.Tensor,
        canvas_feat: torch.Tensor,
        canvas_grid: tuple[int, int],
    ) -> np.ndarray:
        """Compute a correlation heatmap over the canvas.

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
        N = canvas_feat.shape[0]

        if N != gh * gw:
            raise ValueError(
                f"canvas_feat has {N} tokens but canvas_grid "
                f"implies {gh}×{gw}={gh * gw}"
            )

        # Mean-pool query tokens → single prototype vector [1, D]
        prototype = query_feat.mean(dim=0, keepdim=True)
        prototype = F.normalize(prototype, p=2, dim=-1)

        # Cosine similarity with each canvas token  [N]
        similarities = (canvas_feat @ prototype.T).squeeze(-1)

        # Reshape to spatial heatmap
        heatmap = similarities.reshape(gh, gw).cpu().numpy()

        # Normalize to [0, 1]
        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax - hmin > 1e-8:
            heatmap = (heatmap - hmin) / (hmax - hmin)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap.astype(np.float32)
