"""YOLO-based shape detector for CAPTCHA solving."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A detected shape bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    crop: np.ndarray = field(repr=False)
    
    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2
    
    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height


class ShapeDetector:
    """YOLO-based detector for shapes on CAPTCHA canvases.
    
    Parameters
    ----------
    model_path : str
        Path to YOLO .pt weights file.
    conf_threshold : float
        Minimum confidence for a detection.
    iou_threshold : float
        NMS IoU threshold.
    """
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        from ultralytics import YOLO
        
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        logger.info(f"ShapeDetector loaded: {model_path}")
    
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect all shape bounding boxes in an image.
        
        Parameters
        ----------
        image : np.ndarray
            BGR image (screenshot or canvas crop).
            
        Returns
        -------
        list[Detection]
            Detected shapes sorted by confidence (descending).
        """
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                
                # Clip to image bounds
                h, w = image.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Extract crop
                crop = image[y1:y2, x1:x2].copy()
                
                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf,
                    crop=crop,
                ))
        
        detections.sort(key=lambda d: d.confidence, reverse=True)
        logger.info(f"Detected {len(detections)} shapes")
        return detections
    
    def identify_reference(
        self, 
        detections: list[Detection],
        image: np.ndarray,
    ) -> tuple[Detection | None, list[Detection]]:
        if not detections:
            return None, []
        
        # The prompt always says "drag the icon on the top" or "on the right".
        # So the reference icon is ALWAYS either the top-most or right-most shape.
        top_most = min(detections, key=lambda d: d.cy)
        right_most = max(detections, key=lambda d: d.cx)
        
        # Compare saturation between these two candidates only
        candidates = [top_most]
        if right_most != top_most:
            candidates.append(right_most)
        
        scored = []
        for det in candidates:
            sat = 255.0
            if det.crop.size > 0:
                hsv = cv2.cvtColor(det.crop, cv2.COLOR_BGR2HSV)
                sat = hsv[:, :, 1].mean()
            scored.append((sat, det))
            
        scored.sort(key=lambda x: x[0])
        ref_det = scored[0][1]
        
        remaining = [d for d in detections if d != ref_det]
        
        logger.info(
            f"Reference icon: ({ref_det.cx:.0f},{ref_det.cy:.0f}) "
            f"sat={scored[0][0]:.1f} conf={ref_det.confidence:.2f}"
        )
        
        return ref_det, remaining
