"""YOLO-based end-to-end CAPTCHA solver."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

from src.yolo.detector import ShapeDetector, Detection
from src.yolo.matcher import CropMatcher

logger = logging.getLogger(__name__)


@dataclass
class SolveResult:
    """Result of solving a CAPTCHA."""
    x: float
    y: float
    confidence: float
    match_score: float
    match_detail: str
    ref_detection: Detection
    target_detection: Detection
    all_detections: list[Detection]
    elapsed_ms: float


class YOLOCaptchaSolver:
    """End-to-end CAPTCHA solver using YOLO detection + crop matching.
    
    Pipeline:
    1. YOLO detects all shape bounding boxes
    2. Identify which detection is the floating reference icon
    3. Compare the reference icon crop against all other detected shapes
    4. Return the center of the best-matching shape
    
    Parameters
    ----------
    model_path : str
        Path to trained YOLO weights (.pt).
    conf_threshold : float
        YOLO detection confidence threshold.
    """
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
    ) -> None:
        self.detector = ShapeDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
        )
        self.matcher = CropMatcher(target_size=64)
        logger.info("YOLOCaptchaSolver ready")
    
    def solve(self, image: np.ndarray) -> SolveResult | None:
        """Solve a CAPTCHA from a screenshot or canvas image.
        
        Parameters
        ----------
        image : np.ndarray
            BGR image containing the CAPTCHA (can be full screenshot or canvas crop).
            
        Returns
        -------
        SolveResult or None
            The click coordinates and metadata, or None if solving failed.
        """
        t0 = time.perf_counter()
        
        # Step 1: Detect all shapes
        detections = self.detector.detect(image)
        if len(detections) < 2:
            logger.warning(f"Need at least 2 detections, got {len(detections)}")
            return None
        
        # Step 2: Identify reference icon
        ref_det, canvas_dets = self.detector.identify_reference(detections, image)
        if ref_det is None or not canvas_dets:
            logger.warning("Could not identify reference icon")
            return None
        
        logger.info(
            f"Reference: ({ref_det.cx:.0f},{ref_det.cy:.0f}) "
            f"{ref_det.width}x{ref_det.height} conf={ref_det.confidence:.2f}"
        )
        
        # Step 3: Match reference crop against canvas shape crops
        candidate_crops = [d.crop for d in canvas_dets]
        match_results = self.matcher.match_all(ref_det.crop, candidate_crops)
        
        # Log all matches
        for rank, (idx, m) in enumerate(match_results):
            det = canvas_dets[idx]
            logger.info(
                f"  #{rank+1}: ({det.cx:.0f},{det.cy:.0f}) "
                f"score={m.score:.3f} [{m.method}]"
            )
        
        # Step 4: Best match = target
        best_idx, best_match = match_results[0]
        target = canvas_dets[best_idx]
        
        elapsed = (time.perf_counter() - t0) * 1000
        
        result = SolveResult(
            x=target.cx,
            y=target.cy,
            confidence=target.confidence,
            match_score=best_match.score,
            match_detail=best_match.method,
            ref_detection=ref_det,
            target_detection=target,
            all_detections=detections,
            elapsed_ms=elapsed,
        )
        
        logger.info(
            f"SOLVED: click ({result.x:.0f},{result.y:.0f}) "
            f"match={result.match_score:.3f} elapsed={elapsed:.0f}ms"
        )
        
        return result
    
    def solve_and_visualize(
        self,
        image: np.ndarray,
        output_path: str | None = None,
    ) -> tuple[SolveResult | None, np.ndarray]:
        """Solve and create a visualization image.
        
        Returns
        -------
        tuple
            (SolveResult, visualization_image)
        """
        result = self.solve(image)
        vis = image.copy()
        
        if result is None:
            return None, vis
        
        # Draw all detections in blue
        for det in result.all_detections:
            cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), (255, 150, 0), 1)
        
        # Draw reference icon in yellow
        ref = result.ref_detection
        cv2.rectangle(vis, (ref.x1, ref.y1), (ref.x2, ref.y2), (0, 255, 255), 2)
        cv2.putText(vis, "REF", (ref.x1, ref.y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw target in green with crosshair
        tgt = result.target_detection
        cv2.rectangle(vis, (tgt.x1, tgt.y1), (tgt.x2, tgt.y2), (0, 255, 0), 3)
        ix, iy = int(result.x), int(result.y)
        cv2.circle(vis, (ix, iy), 25, (0, 255, 0), 3)
        cv2.line(vis, (ix - 35, iy), (ix + 35, iy), (0, 255, 0), 2)
        cv2.line(vis, (ix, iy - 35), (ix, iy + 35), (0, 255, 0), 2)
        
        # Label
        label = f"TARGET score={result.match_score:.2f}"
        cv2.putText(vis, label, (tgt.x1, tgt.y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw other candidates in orange with rank
        match_results = self.matcher.match_all(ref.crop, [d.crop for d in result.all_detections if d is not ref])
        
        if output_path:
            cv2.imwrite(output_path, vis)
        
        return result, vis
