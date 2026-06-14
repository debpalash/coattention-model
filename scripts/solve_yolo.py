#!/usr/bin/env python3
"""Run the end-to-end YOLO + Siamese CAPTCHA solver on an image."""
import argparse
import logging
import os

import cv2

from src.yolo.detector import ShapeDetector
from src.yolo.siamese_matcher import SiameseCropMatcher
from src.yolo.solver import YOLOCaptchaSolver, SolveResult

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s — %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Solve CAPTCHA using YOLO + Siamese")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--yolo-model", default="runs/detect/runs/detect/captcha_shapes/weights/best.pt", help="Path to YOLO model")
    parser.add_argument("--siamese-model", default="runs/siamese/best.pt", help="Path to Siamese model")
    parser.add_argument("--info-path", default="runs/siamese/info.json", help="Path to Siamese model info")
    parser.add_argument("--output", help="Optional path to save visualization")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        logger.error(f"Image not found: {args.image}")
        return

    logger.info("Initializing solver...")
    detector = ShapeDetector(args.yolo_model, conf_threshold=0.05)
    matcher = SiameseCropMatcher(args.siamese_model, info_path=args.info_path)
    
    class DummyMatchResult:
        def __init__(self, score):
            self.score = score
            self.method = "siamese"

    class WrappedSiameseMatcher:
        def __init__(self, matcher):
            self.matcher = matcher
        def match_all(self, ref_crop, candidate_crops):
            results = self.matcher.match_all(ref_crop, candidate_crops)
            return [(idx, DummyMatchResult(sim)) for idx, sim in results]

    class SiameseYOLOSolver(YOLOCaptchaSolver):
        def __init__(self, detector, matcher):
            self.detector = detector
            self.matcher = WrappedSiameseMatcher(matcher)
            
    solver = SiameseYOLOSolver(detector, matcher)
    
    img = cv2.imread(args.image)
    if img is None:
        logger.error("Could not read image")
        return

    logger.info(f"Solving {args.image}...")
    if args.output:
        result, vis = solver.solve_and_visualize(img, args.output)
        if result:
            logger.info(f"Target found at ({result.x:.0f}, {result.y:.0f}) with score {result.match_score:.3f}")
            logger.info(f"Saved visualization to {args.output}")
        else:
            logger.error("Failed to solve")
    else:
        result = solver.solve(img)
        if result:
            logger.info(f"Target found at ({result.x:.0f}, {result.y:.0f}) with score {result.match_score:.3f}")
        else:
            logger.error("Failed to solve")

if __name__ == "__main__":
    main()
