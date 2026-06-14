"""Crop-to-crop matching for identifying which detected shape matches the reference icon."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CropMatch:
    """Result of matching a reference crop against a candidate crop."""
    score: float
    method: str


class CropMatcher:
    """Compare reference icon crop against detected shape crops.
    
    Uses multiple lightweight similarity metrics since YOLO already
    provides tight bounding boxes — we just need to compare two
    small crops, not search a full canvas.
    
    Strategies:
    1. Structural histogram comparison (shape of intensity distribution)
    2. Template matching on edges (structural similarity)  
    3. Hu moments (geometric shape similarity)
    """
    
    def __init__(self, target_size: int = 64) -> None:
        self.target_size = target_size
    
    def match(
        self,
        ref_crop: np.ndarray,
        candidate_crop: np.ndarray,
    ) -> CropMatch:
        """Compare two crops and return a similarity score.
        
        Parameters
        ----------
        ref_crop : np.ndarray
            The reference icon crop (BGR).
        candidate_crop : np.ndarray
            A detected shape crop from the canvas (BGR).
            
        Returns
        -------
        CropMatch
            Combined similarity score (0-1, higher = more similar).
        """
        # Resize both to same size for fair comparison
        sz = self.target_size
        ref = cv2.resize(ref_crop, (sz, sz))
        cand = cv2.resize(candidate_crop, (sz, sz))
        
        scores = {}
        
        # 1. Edge template matching (structural similarity)
        scores['edge_tm'] = self._edge_template_score(ref, cand)
        
        # 2. Histogram comparison on gradient magnitudes
        scores['hist'] = self._histogram_score(ref, cand)
        
        # 3. Hu moments shape similarity
        scores['hu'] = self._hu_moments_score(ref, cand)
        
        # 4. Contour matching
        scores['contour'] = self._contour_score(ref, cand)
        
        # Weighted combination
        weights = {
            'edge_tm': 0.35,
            'hist': 0.15,
            'hu': 0.25,
            'contour': 0.25,
        }
        
        combined = sum(scores[k] * weights[k] for k in scores)
        
        detail = ' '.join(f'{k}={v:.2f}' for k, v in scores.items())
        
        return CropMatch(score=combined, method=detail)
    
    def match_all(
        self,
        ref_crop: np.ndarray,
        candidates: list[np.ndarray],
    ) -> list[tuple[int, CropMatch]]:
        """Match reference against all candidates, return sorted results.
        
        Returns
        -------
        list[tuple[int, CropMatch]]
            (candidate_index, match_result) sorted by score descending.
        """
        results = []
        for i, cand in enumerate(candidates):
            m = self.match(ref_crop, cand)
            results.append((i, m))
        
        results.sort(key=lambda x: x[1].score, reverse=True)
        return results
    
    def _edge_template_score(self, ref: np.ndarray, cand: np.ndarray) -> float:
        """Template match on Canny edges."""
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        cand_gray = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY)
        
        ref_edges = cv2.Canny(ref_gray, 50, 150)
        cand_edges = cv2.Canny(cand_gray, 30, 100)
        
        # Dilate for thicker edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ref_edges = cv2.dilate(ref_edges, kernel)
        cand_edges = cv2.dilate(cand_edges, kernel)
        
        # Since they're same size, use direct NCC
        result = cv2.matchTemplate(cand_edges, ref_edges, cv2.TM_CCOEFF_NORMED)
        return float(max(0, result[0, 0]))
    
    def _histogram_score(self, ref: np.ndarray, cand: np.ndarray) -> float:
        """Compare gradient magnitude histograms."""
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        cand_gray = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY)
        
        # Compute gradient magnitudes
        ref_gx = cv2.Sobel(ref_gray, cv2.CV_64F, 1, 0)
        ref_gy = cv2.Sobel(ref_gray, cv2.CV_64F, 0, 1)
        ref_mag = np.sqrt(ref_gx**2 + ref_gy**2).astype(np.float32)
        
        cand_gx = cv2.Sobel(cand_gray, cv2.CV_64F, 1, 0)
        cand_gy = cv2.Sobel(cand_gray, cv2.CV_64F, 0, 1)
        cand_mag = np.sqrt(cand_gx**2 + cand_gy**2).astype(np.float32)
        
        # Compute histograms
        ref_hist = cv2.calcHist([ref_mag], [0], None, [32], [0, 256])
        cand_hist = cv2.calcHist([cand_mag], [0], None, [32], [0, 256])
        
        cv2.normalize(ref_hist, ref_hist)
        cv2.normalize(cand_hist, cand_hist)
        
        # Compare using correlation
        score = cv2.compareHist(ref_hist, cand_hist, cv2.HISTCMP_CORREL)
        return float(max(0, score))
    
    def _hu_moments_score(self, ref: np.ndarray, cand: np.ndarray) -> float:
        """Hu moments shape distance (converted to 0-1 similarity)."""
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        cand_gray = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY)
        
        _, ref_bin = cv2.threshold(ref_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, cand_bin = cv2.threshold(cand_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        dist = cv2.matchShapes(ref_bin, cand_bin, cv2.CONTOURS_MATCH_I2, 0)
        
        # Convert distance to similarity (exponential decay)
        similarity = np.exp(-dist)
        return float(similarity)
    
    def _contour_score(self, ref: np.ndarray, cand: np.ndarray) -> float:
        """Compare dominant contours between ref and candidate."""
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        cand_gray = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY)
        
        _, ref_bin = cv2.threshold(ref_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, cand_bin = cv2.threshold(cand_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        ref_cnts, _ = cv2.findContours(ref_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cand_cnts, _ = cv2.findContours(cand_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not ref_cnts or not cand_cnts:
            return 0.0
        
        # Get largest contour from each
        ref_cnt = max(ref_cnts, key=cv2.contourArea)
        cand_cnt = max(cand_cnts, key=cv2.contourArea)
        
        dist = cv2.matchShapes(ref_cnt, cand_cnt, cv2.CONTOURS_MATCH_I2, 0)
        similarity = np.exp(-dist * 2)
        return float(similarity)
