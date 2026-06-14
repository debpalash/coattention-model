"""Multi-strategy CAPTCHA shape matcher.

Combines multiple matching approaches into an ensemble for robust
shape localization on plasma CAPTCHA backgrounds:

1. Multi-scale Template Matching (grayscale + edge representations)
2. DINO Feature QATM Matching (edge-preprocessed)  
3. Hu Moments contour shape matching
4. Multi-channel gradient matching

Each strategy votes on the location; the ensemble picks the consensus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ShapeCandidate:
    """A candidate match location with confidence from multiple strategies."""
    x: float
    y: float
    votes: int
    total_score: float
    strategies: dict[str, float]  # strategy_name -> score


class MultiStrategyMatcher:
    """Ensemble matcher that combines multiple approaches.
    
    Parameters
    ----------
    merge_radius : int
        Pixel radius within which two candidate peaks are considered the same.
    """
    
    def __init__(self, merge_radius: int = 40) -> None:
        self.merge_radius = merge_radius
    
    def match(
        self,
        ref_icon: np.ndarray,
        canvas: np.ndarray,
        exclude_box: tuple[int, int, int, int] | None = None,
    ) -> list[ShapeCandidate]:
        """Find the reference icon shape on the canvas.
        
        Parameters
        ----------
        ref_icon : np.ndarray
            Reference icon (BGR uint8).
        canvas : np.ndarray
            Canvas image (BGR uint8).
        exclude_box : tuple
            (x, y, w, h) region to exclude from search (e.g., floating ref icon area).
            
        Returns
        -------
        list[ShapeCandidate]
            Sorted by total_score (descending).
        """
        ref_gray = cv2.cvtColor(ref_icon, cv2.COLOR_BGR2GRAY)
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        
        all_peaks: list[tuple[float, float, float, str]] = []  # (x, y, score, strategy)
        
        # Strategy 1: Multi-scale Template Matching
        peaks_tm = self._template_matching(ref_gray, canvas_gray, exclude_box)
        all_peaks.extend(peaks_tm)
        
        # Strategy 2: Edge Template Matching
        peaks_edge = self._edge_template_matching(ref_gray, canvas_gray, exclude_box)
        all_peaks.extend(peaks_edge)
        
        # Strategy 3: Gradient Magnitude Matching
        peaks_grad = self._gradient_matching(ref_gray, canvas_gray, exclude_box)
        all_peaks.extend(peaks_grad)
        
        # Strategy 4: Hu Moments Shape Matching
        peaks_hu = self._hu_moments_matching(ref_icon, canvas, exclude_box)
        all_peaks.extend(peaks_hu)
        
        # Merge nearby peaks into consensus candidates
        candidates = self._merge_peaks(all_peaks)
        
        # Sort by total score
        candidates.sort(key=lambda c: c.total_score, reverse=True)
        
        return candidates
    
    def _apply_exclusion(
        self, result: np.ndarray, 
        exclude_box: tuple | None, 
        ref_w: int, ref_h: int
    ) -> np.ndarray:
        """Zero out exclusion zone in a matchTemplate result array."""
        if exclude_box is None:
            return result
        bx, by, bw, bh = exclude_box
        y1 = max(0, by - ref_h)
        y2 = min(result.shape[0], by + bh)
        x1 = max(0, bx - ref_w)
        x2 = min(result.shape[1], bx + bw)
        result[y1:y2, x1:x2] = -1.0
        return result
    
    def _get_top_peaks(
        self, result: np.ndarray, 
        n: int, suppress_radius: int,
        ref_w: int, ref_h: int,
        strategy_name: str
    ) -> list[tuple[float, float, float, str]]:
        """Extract top-N peaks from a matchTemplate result with NMS."""
        peaks = []
        heatmap = result.copy()
        for _ in range(n):
            _, max_val, _, max_loc = cv2.minMaxLoc(heatmap)
            if max_val < 0:
                break
            cx = max_loc[0] + ref_w // 2
            cy = max_loc[1] + ref_h // 2
            peaks.append((float(cx), float(cy), float(max_val), strategy_name))
            cv2.circle(heatmap, max_loc, suppress_radius, -1, -1)
        return peaks
    
    def _template_matching(
        self, ref_gray: np.ndarray, canvas_gray: np.ndarray,
        exclude_box: tuple | None
    ) -> list[tuple[float, float, float, str]]:
        """Multi-scale normalized cross-correlation template matching."""
        all_peaks = []
        rh, rw = ref_gray.shape[:2]
        
        for scale in np.arange(0.7, 2.2, 0.15):
            sw = int(rw * scale)
            sh = int(rh * scale)
            if sw > canvas_gray.shape[1] or sh > canvas_gray.shape[0] or sw < 20:
                continue
            ref_scaled = cv2.resize(ref_gray, (sw, sh))
            result = cv2.matchTemplate(canvas_gray, ref_scaled, cv2.TM_CCOEFF_NORMED)
            result = self._apply_exclusion(result, exclude_box, sw, sh)
            
            peaks = self._get_top_peaks(result, 3, 40, sw, sh, f"TM_s{scale:.1f}")
            all_peaks.extend(peaks)
        
        return all_peaks
    
    def _edge_template_matching(
        self, ref_gray: np.ndarray, canvas_gray: np.ndarray,
        exclude_box: tuple | None
    ) -> list[tuple[float, float, float, str]]:
        """Template matching on Canny edge images."""
        ref_edges = cv2.Canny(ref_gray, 50, 150)
        canvas_edges = cv2.Canny(canvas_gray, 30, 100)
        
        # Dilate edges for better matching
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ref_edges = cv2.dilate(ref_edges, kernel)
        canvas_edges = cv2.dilate(canvas_edges, kernel)
        
        all_peaks = []
        rh, rw = ref_edges.shape[:2]
        
        for scale in np.arange(0.8, 2.0, 0.2):
            sw = int(rw * scale)
            sh = int(rh * scale)
            if sw > canvas_edges.shape[1] or sh > canvas_edges.shape[0] or sw < 20:
                continue
            ref_scaled = cv2.resize(ref_edges, (sw, sh))
            result = cv2.matchTemplate(canvas_edges, ref_scaled, cv2.TM_CCOEFF_NORMED)
            result = self._apply_exclusion(result, exclude_box, sw, sh)
            
            peaks = self._get_top_peaks(result, 3, 40, sw, sh, f"EdgeTM_s{scale:.1f}")
            all_peaks.extend(peaks)
        
        return all_peaks
    
    def _gradient_matching(
        self, ref_gray: np.ndarray, canvas_gray: np.ndarray,
        exclude_box: tuple | None
    ) -> list[tuple[float, float, float, str]]:
        """Template matching on gradient magnitude images."""
        gx_r = cv2.Sobel(ref_gray, cv2.CV_64F, 1, 0, ksize=3)
        gy_r = cv2.Sobel(ref_gray, cv2.CV_64F, 0, 1, ksize=3)
        ref_mag = np.sqrt(gx_r**2 + gy_r**2)
        ref_mag = (ref_mag / (ref_mag.max() + 1e-8) * 255).astype(np.uint8)
        
        gx_c = cv2.Sobel(canvas_gray, cv2.CV_64F, 1, 0, ksize=3)
        gy_c = cv2.Sobel(canvas_gray, cv2.CV_64F, 0, 1, ksize=3)
        canvas_mag = np.sqrt(gx_c**2 + gy_c**2)
        canvas_mag = (canvas_mag / (canvas_mag.max() + 1e-8) * 255).astype(np.uint8)
        
        all_peaks = []
        rh, rw = ref_mag.shape[:2]
        
        for scale in [0.8, 1.0, 1.2, 1.5, 1.8]:
            sw = int(rw * scale)
            sh = int(rh * scale)
            if sw > canvas_mag.shape[1] or sh > canvas_mag.shape[0] or sw < 20:
                continue
            ref_scaled = cv2.resize(ref_mag, (sw, sh))
            result = cv2.matchTemplate(canvas_mag, ref_scaled, cv2.TM_CCOEFF_NORMED)
            result = self._apply_exclusion(result, exclude_box, sw, sh)
            
            peaks = self._get_top_peaks(result, 3, 40, sw, sh, f"GradTM_s{scale:.1f}")
            all_peaks.extend(peaks)
        
        return all_peaks
    
    def _hu_moments_matching(
        self, ref_icon: np.ndarray, canvas: np.ndarray,
        exclude_box: tuple | None
    ) -> list[tuple[float, float, float, str]]:
        """Find shapes on canvas via saturation thresholding + Hu moment comparison."""
        ref_gray = cv2.cvtColor(ref_icon, cv2.COLOR_BGR2GRAY)
        _, ref_binary = cv2.threshold(ref_gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        canvas_hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
        sat = canvas_hsv[:, :, 1]
        
        # Extract high-saturation regions (the glowing shapes)
        _, sat_mask = cv2.threshold(sat, 80, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(sat_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        peaks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300 or area > 80000:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            
            # Skip if in exclusion zone
            if exclude_box is not None:
                bx, by, bw, bh = exclude_box
                if bx < cx < bx + bw and by < cy < by + bh:
                    continue
            
            # Hu moments distance
            cnt_mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
            hu_dist = cv2.matchShapes(ref_binary, cnt_mask, cv2.CONTOURS_MATCH_I2, 0)
            
            # Convert distance to score (lower distance = higher score)
            score = max(0, 1.0 - hu_dist / 5.0)
            if score > 0.1:
                peaks.append((float(cx), float(cy), score, "HuMoments"))
        
        return peaks
    
    def _merge_peaks(
        self, peaks: list[tuple[float, float, float, str]]
    ) -> list[ShapeCandidate]:
        """Merge nearby peaks into consensus candidates."""
        if not peaks:
            return []
        
        # Sort by score descending
        peaks.sort(key=lambda p: p[2], reverse=True)
        
        merged: list[ShapeCandidate] = []
        used = [False] * len(peaks)
        
        for i, (x, y, score, strategy) in enumerate(peaks):
            if used[i]:
                continue
            
            # Start a new cluster
            cluster_x = [x]
            cluster_y = [y]
            cluster_scores = [score]
            strategies = {strategy: score}
            used[i] = True
            
            # Find all peaks within merge_radius
            for j, (x2, y2, score2, strat2) in enumerate(peaks):
                if used[j]:
                    continue
                dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                if dist <= self.merge_radius:
                    cluster_x.append(x2)
                    cluster_y.append(y2)
                    cluster_scores.append(score2)
                    if strat2 not in strategies or score2 > strategies[strat2]:
                        strategies[strat2] = score2
                    used[j] = True
            
            # Weighted centroid
            weights = np.array(cluster_scores)
            weights = weights / weights.sum()
            cx = float(np.dot(weights, cluster_x))
            cy = float(np.dot(weights, cluster_y))
            
            # Count unique strategy families (TM, EdgeTM, GradTM, HuMoments)
            families = set()
            for s in strategies:
                if s.startswith("TM"):
                    families.add("TM")
                elif s.startswith("Edge"):
                    families.add("Edge")
                elif s.startswith("Grad"):
                    families.add("Grad")
                elif s.startswith("Hu"):
                    families.add("Hu")
            
            merged.append(ShapeCandidate(
                x=cx, y=cy,
                votes=len(cluster_scores),
                total_score=sum(cluster_scores),
                strategies=strategies,
            ))
        
        return merged
