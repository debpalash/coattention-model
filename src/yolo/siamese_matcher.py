"""Siamese network-based crop matcher for the YOLO solver pipeline.

Uses a trained SiameseMatchNet to compare the reference icon crop
against each YOLO-detected shape crop on the canvas.
"""
from __future__ import annotations

import json
import logging
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from scripts.train_siamese import SiameseMatchNet

logger = logging.getLogger(__name__)


class SiameseCropMatcher:
    """Match ref icon against detected shape crops using a trained Siamese network.
    
    Parameters
    ----------
    model_path : str
        Path to trained Siamese model weights (.pt).
    info_path : str or None
        Path to model info JSON (for embed_dim, img_size). If None, uses defaults.
    device : str
        Torch device.
    """
    
    def __init__(
        self,
        model_path: str,
        info_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        # Load model config
        self.img_size = 64
        embed_dim = 128
        
        if info_path and os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            self.img_size = info.get("img_size", 64)
            embed_dim = info.get("embed_dim", 128)
        
        # Load model
        self.device = device
        self.model = SiameseMatchNet(embed_dim=embed_dim)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"SiameseCropMatcher loaded: {model_path} (embed={embed_dim})")
    
    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess a BGR crop for the Siamese network."""
        crop = cv2.resize(crop_bgr, (self.img_size, self.img_size))
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(crop_rgb).float().permute(2, 0, 1) / 127.5 - 1.0
        return tensor.unsqueeze(0).to(self.device)
    
    def _get_rotated_crops(self, crop: np.ndarray) -> list[np.ndarray]:
        """Generate 8 rotated versions of a crop (45-degree increments)."""
        crops = []
        h, w = crop.shape[:2]
        center = (w // 2, h // 2)
        
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                crop, M, (w, h), 
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            crops.append(rotated)
        return crops

    def _get_shape_contour(self, crop: np.ndarray):
        """Extract the main contour from a crop."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Use adaptive thresholding to handle plasma backgrounds
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def match_all(
        self,
        ref_crop: np.ndarray,
        candidate_crops: list[np.ndarray],
    ) -> list[tuple[int, float]]:
        """Compare ref against all candidates using Siamese + Contour matching.
        
        Returns
        -------
        list[tuple[int, float]]
            (candidate_index, similarity) sorted by similarity descending.
        """
        # 1. Evaluate Siamese Network (with TTA rotations)
        ref_rotations = self._get_rotated_crops(ref_crop)
        ref_tensors = torch.cat([self._preprocess(r) for r in ref_rotations], dim=0)
        
        siamese_scores = []
        with torch.no_grad():
            ref_embs = self.model.encode(ref_tensors)
            for crop in candidate_crops:
                cand_tensor = self._preprocess(crop)
                cand_emb = self.model.encode(cand_tensor)
                sims = F.cosine_similarity(ref_embs, cand_emb, dim=-1)
                
                penalty = torch.tensor([1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], device=self.device)
                penalized_sims = sims * penalty
                siamese_scores.append(penalized_sims.max().item())
                
        # 2. Evaluate Contour Matching
        ref_contour = self._get_shape_contour(ref_crop)
        contour_scores = []
        
        for crop in candidate_crops:
            if ref_contour is None:
                contour_scores.append(0.0)
                continue
                
            cand_contour = self._get_shape_contour(crop)
            if cand_contour is None:
                contour_scores.append(0.0)
                continue
                
            # matchShapes returns distance (0 = perfect match)
            # Map distance to similarity [0, 1]
            dist = cv2.matchShapes(ref_contour, cand_contour, cv2.CONTOURS_MATCH_I1, 0)
            
            # Distance is usually 0.1 to 5.0. 
            # We want dist 0.0 -> 1.0 sim, dist 1.0 -> 0.0 sim
            sim = max(0.0, 1.0 - dist)
            contour_scores.append(sim)
            
        # 3. Combine Scores (Contour matching is much more reliable for pure geometry)
        results = []
        for i in range(len(candidate_crops)):
            s_score = siamese_scores[i]
            c_score = contour_scores[i]
            
            # If contour extraction worked well, trust it completely.
            # Siamese network was not trained on rotations and often gives 
            # false positives on similar textures but wrong shapes.
            if c_score > 0.1:
                final_score = c_score
            else:
                # Fallback to Siamese if contour failed
                final_score = s_score
                
            results.append((i, final_score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
