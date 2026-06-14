"""DINO feature extractor — loads DINOv3 (preferred) or DINOv2 fallback.

Extracts L2-normalized patch-level tokens from a Vision Transformer
backbone, suitable for downstream template matching (QATM, correlation).
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.preprocessing.normalizer import ImageNormalizer

logger = logging.getLogger(__name__)


class DINOFeatureExtractor:
    """Wrapper around DINO ViT backbones for dense patch feature extraction.

    Attempts to load DINOv3 first; on any failure falls back to DINOv2.
    Automatically detects the embedding dimension and patch size from the
    loaded model so downstream code can be model-agnostic.

    Parameters
    ----------
    device : torch.device | None
        Device to run inference on.  Defaults to CUDA if available, else CPU.
    force_version : Literal["v2", "v3"] | None
        If set, skip auto-detection and load only the specified version.
    """

    # Model configs: (hub_repo, hub_model, patch_size)
    _V3_CONFIG = ("facebookresearch/dinov3", "dinov3_vitb16", 16)
    _V2_CONFIG = ("facebookresearch/dinov2", "dinov2_vitb14", 14)

    def __init__(
        self,
        device: torch.device | None = None,
        force_version: Literal["v2", "v3"] | None = None,
    ) -> None:
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.model: torch.nn.Module
        self.version: str
        self.patch_size: int
        self.embed_dim: int

        self._load_model(force_version)

        # Build a normalizer that matches this backbone's patch size
        self.normalizer = ImageNormalizer(patch_size=self.patch_size, device=self.device)

        logger.info(
            "DINOFeatureExtractor ready — version=%s  patch=%d  dim=%d  device=%s",
            self.version,
            self.patch_size,
            self.embed_dim,
            self.device,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, force_version: Literal["v2", "v3"] | None) -> None:
        """Load the DINO backbone, trying v3 first then v2."""
        if force_version == "v3":
            self._try_load_v3()
            return
        if force_version == "v2":
            self._try_load_v2()
            return

        # Auto: try v3 first, fall back to v2
        try:
            self._try_load_v3()
        except Exception as exc:
            logger.warning("DINOv3 unavailable (%s), falling back to DINOv2.", exc)
            self._try_load_v2()

    def _try_load_v3(self) -> None:
        repo, model_name, patch_size = self._V3_CONFIG
        logger.info("Attempting to load DINOv3 (%s/%s)…", repo, model_name)
        model = torch.hub.load(repo, model_name, pretrained=True)
        model.eval()
        model.to(self.device)
        self.model = model
        self.version = "v3"
        self.patch_size = patch_size
        self.embed_dim = self._detect_embed_dim(model)

    def _try_load_v2(self) -> None:
        repo, model_name, patch_size = self._V2_CONFIG
        logger.info("Loading DINOv2 (%s/%s)…", repo, model_name)
        model = torch.hub.load(repo, model_name, pretrained=True)
        model.eval()
        model.to(self.device)
        self.model = model
        self.version = "v2"
        self.patch_size = patch_size
        self.embed_dim = self._detect_embed_dim(model)

    @staticmethod
    def _detect_embed_dim(model: torch.nn.Module) -> int:
        """Auto-detect the embedding dimension from the loaded model.

        Looks for common attribute names used by DINOv2/v3 ViT
        implementations.  Falls back to probing with a dummy forward pass.
        """
        # DINOv2 (and most ViTs) expose `embed_dim` directly
        for attr in ("embed_dim", "num_features", "hidden_dim"):
            if hasattr(model, attr):
                dim = getattr(model, attr)
                if isinstance(dim, int) and dim > 0:
                    return dim

        # Last-resort: probe with a 224×224 dummy image
        logger.debug("Probing embed_dim with a dummy forward pass.")
        dummy = torch.zeros(1, 3, 224, 224)
        dummy = dummy.to(next(model.parameters()).device)
        with torch.no_grad():
            out = model.forward_features(dummy)
            if isinstance(out, dict):
                tokens = out.get("x_norm_patchtokens", out.get("x_patchtokens"))
                if tokens is not None:
                    return tokens.shape[-1]
            elif isinstance(out, torch.Tensor):
                return out.shape[-1]

        raise RuntimeError("Could not determine DINO embedding dimension.")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract(self, image: np.ndarray, target_size: tuple[int, int]) -> torch.Tensor:
        """Extract L2-normalized patch tokens from an image.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format (H, W, 3), uint8.
        target_size : tuple[int, int]
            (width, height) to resize the image to before extraction.
            The image will also be padded to the nearest patch_size multiple.

        Returns
        -------
        torch.Tensor
            Patch token features of shape [N, D] where
            N = (H_padded // patch_size) × (W_padded // patch_size)
            and D = embed_dim.  Tokens are L2-normalized along dim=-1.
        """
        # Preprocess: resize, normalize, pad
        tensor = self.normalizer.normalize(image, target_size)  # [1, 3, H, W]

        # Forward pass — DINOv2 returns a dict from forward_features
        features = self._forward(tensor)

        # L2 normalize along the feature dimension
        features = F.normalize(features, p=2, dim=-1)

        return features  # [N, D]

    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run the backbone and return patch tokens as [N, D].

        Handles the different output formats of DINOv2 / DINOv3.
        """
        output = self.model.forward_features(tensor)

        if isinstance(output, dict):
            # DINOv2 returns {"x_norm_patchtokens": ..., "x_norm_clstoken": ..., ...}
            tokens = output.get("x_norm_patchtokens")
            if tokens is None:
                # Older builds or custom wrappers might use a different key
                tokens = output.get("x_patchtokens")
            if tokens is None:
                raise RuntimeError(
                    f"Unexpected output keys from forward_features: {list(output.keys())}"
                )
        elif isinstance(output, torch.Tensor):
            # Some models return [B, 1+N, D] (CLS + patches) or [B, N, D]
            if output.shape[1] == 1 + self._expected_num_patches(tensor):
                tokens = output[:, 1:]  # drop CLS
            else:
                tokens = output
        else:
            raise RuntimeError(f"Unexpected forward_features return type: {type(output)}")

        # Remove batch dimension → [N, D]
        return tokens.squeeze(0)

    def _expected_num_patches(self, tensor: torch.Tensor) -> int:
        """Compute how many patches the ViT should produce for *tensor*."""
        _, _, h, w = tensor.shape
        return (h // self.patch_size) * (w // self.patch_size)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_grid_size(self, h: int, w: int) -> tuple[int, int]:
        """Return the (rows, cols) patch grid for an image of size (h, w).

        The image dimensions should already be padded to patch_size multiples
        (i.e. the values returned by ``normalizer.get_padded_size``).
        """
        return (h // self.patch_size, w // self.patch_size)

    def get_padded_grid_size(self, target_size: tuple[int, int]) -> tuple[int, int]:
        """Convenience: return the patch grid for a given target_size (w, h).

        Accounts for padding to the nearest patch_size multiple.
        """
        padded_w, padded_h = self.normalizer.get_padded_size(target_size)
        return self.get_grid_size(padded_h, padded_w)
