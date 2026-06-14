#!/usr/bin/env python3
"""CLI script for solving CAPTCHA puzzles.

Usage
-----
    python -m scripts.solve_puzzle \\
        --ref test_puzzles/puzzle_0_ref.png \\
        --canvas test_puzzles/puzzle_0_canvas.png \\
        --output result.png

    # With correlation matcher instead of QATM:
    python -m scripts.solve_puzzle --ref ref.png --canvas canvas.png --matcher correlation

    # Force DINOv2:
    python -m scripts.solve_puzzle --ref ref.png --canvas canvas.png --dino-version v2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from src.pipeline import CaptchaSolver


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Solve a CAPTCHA puzzle by locating a reference icon in a canvas image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ref",
        type=str,
        required=True,
        help="Path to the reference icon image.",
    )
    parser.add_argument(
        "--canvas",
        type=str,
        required=True,
        help="Path to the canvas (puzzle) image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the visualization image.  If not set, shows on screen.",
    )
    parser.add_argument(
        "--matcher",
        type=str,
        choices=["qatm", "correlation"],
        default="qatm",
        help="Matching algorithm (default: qatm).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="QATM temperature parameter (default: 1.0).",
    )
    parser.add_argument(
        "--dino-version",
        type=str,
        choices=["v2", "v3"],
        default=None,
        help="Force a specific DINO version.  Auto-detect if not set.",
    )
    parser.add_argument(
        "--ref-size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("W", "H"),
        help="Reference icon resize dimensions (default: 224 224).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    return parser.parse_args(argv)


def draw_crosshair(
    image: np.ndarray,
    x: float,
    y: float,
    size: int = 20,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a crosshair marker at (x, y) on the image."""
    ix, iy = int(round(x)), int(round(y))
    out = image.copy()

    # Horizontal line
    cv2.line(out, (ix - size, iy), (ix + size, iy), color, thickness, cv2.LINE_AA)
    # Vertical line
    cv2.line(out, (ix, iy - size), (ix, iy + size), color, thickness, cv2.LINE_AA)
    # Circle at center
    cv2.circle(out, (ix, iy), size // 2, color, thickness, cv2.LINE_AA)

    return out


def overlay_heatmap(
    canvas: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a semi-transparent heatmap on the canvas.

    Parameters
    ----------
    canvas : np.ndarray
        Original canvas image (BGR, uint8).
    heatmap : np.ndarray
        Heatmap array, any resolution (will be resized to canvas).
    alpha : float
        Opacity of the heatmap overlay (0 = invisible, 1 = opaque).
    colormap : int
        OpenCV colormap ID.

    Returns
    -------
    np.ndarray
        Blended visualization image.
    """
    h, w = canvas.shape[:2]

    # Resize heatmap to canvas resolution
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

    # Convert to uint8 and apply colormap
    heatmap_u8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_u8, colormap)

    # Blend
    blended = cv2.addWeighted(canvas, 1.0 - alpha, heatmap_colored, alpha, 0)
    return blended


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    # ── Logging ──
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Load images ──
    ref_path = Path(args.ref)
    canvas_path = Path(args.canvas)

    if not ref_path.is_file():
        print(f"Error: Reference image not found: {ref_path}", file=sys.stderr)
        sys.exit(1)
    if not canvas_path.is_file():
        print(f"Error: Canvas image not found: {canvas_path}", file=sys.stderr)
        sys.exit(1)

    ref_image = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    canvas_image = cv2.imread(str(canvas_path), cv2.IMREAD_COLOR)

    if ref_image is None:
        print(f"Error: Failed to decode reference image: {ref_path}", file=sys.stderr)
        sys.exit(1)
    if canvas_image is None:
        print(f"Error: Failed to decode canvas image: {canvas_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reference image: {ref_path}  ({ref_image.shape[1]}×{ref_image.shape[0]})")
    print(f"Canvas image:    {canvas_path}  ({canvas_image.shape[1]}×{canvas_image.shape[0]})")

    # ── Solve ──
    solver = CaptchaSolver(
        ref_size=tuple(args.ref_size),
        matcher_type=args.matcher,
        qatm_alpha=args.alpha,
        force_dino_version=args.dino_version,
    )

    result = solver.solve(ref_image, canvas_image, return_heatmap=True)

    # ── Print results ──
    print()
    print("=" * 50)
    print(f"  Predicted location:   ({result.x:.1f}, {result.y:.1f})")
    print(f"  Confidence:           {result.confidence:.4f}")
    print(f"  Peak similarity:      {result.peak_similarity:.4f}")
    print(f"  Peak-to-second ratio: {result.peak_to_second_ratio:.4f}")
    print(f"  Peak sharpness:       {result.peak_sharpness:.4f}")
    print(f"  Elapsed:              {result.elapsed_ms:.0f} ms")
    print("=" * 50)

    # ── Visualize ──
    viz = canvas_image.copy()

    # Overlay heatmap if available
    if result.heatmap is not None:
        viz = overlay_heatmap(viz, result.heatmap, alpha=0.35)

    # Draw crosshair at predicted position
    viz = draw_crosshair(viz, result.x, result.y, size=25)

    # Add text overlay
    label = f"({result.x:.0f}, {result.y:.0f})  conf={result.confidence:.3f}"
    cv2.putText(
        viz,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        viz,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 200, 0),
        1,
        cv2.LINE_AA,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), viz)
        print(f"\nVisualization saved to: {output_path}")
    else:
        print("\nNo --output specified; displaying result window.")
        print("Press any key to close.")
        cv2.imshow("CAPTCHA Solver Result", viz)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
