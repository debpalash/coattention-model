"""End-to-end test of the CAPTCHA solver on real screenshots.

The CAPTCHA layout (from visual inspection):
- Screenshot 3 (media__1779543415160.jpg): 694×500
  - This is a cropped view of just the puzzle canvas with the floating ref icon
  - The reference icon (⊘ symbol) is floating near top-center inside a small grey box
  - The canvas fills the entire image with plasma background + glowing shapes
  
- Screenshot 4 (media__1779543415161.jpg): 710×756  
  - Full CAPTCHA widget including header text "Please drag the icon..."
  - Canvas is embedded in the middle section
  
For this test, we manually identify the reference icon crop coordinates from the screenshots.
"""

from __future__ import annotations

import os
import sys
import logging
import time

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from src.pipeline import CaptchaSolver

ARTIFACT_DIR = os.path.expanduser(
    "~/.gemini/antigravity/brain/fac872ba-4da8-448e-9e7c-8eda3b1ab3b6"
)
OUTPUT_DIR = "test_output"


def extract_ref_icon(screenshot: np.ndarray) -> tuple[np.ndarray | None, tuple | None]:
    """Auto-detect and extract the reference icon from the screenshot.
    
    Returns (ref_crop, ref_bbox) where ref_bbox = (x, y, w, h) in screenshot coords.
    """
    h, w = screenshot.shape[:2]
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    best_box = None
    best_score = 0
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = min(cw, ch) / max(cw, ch) if max(cw, ch) > 0 else 0
        
        if (50 < cw < 130 and 50 < ch < 130 and 
            aspect > 0.65 and y < h * 0.4):
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) >= 4:
                score = aspect * (cw * ch)
                if score > best_score:
                    best_score = score
                    best_box = (x, y, cw, ch)
    
    if best_box is None:
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        low_sat = cv2.threshold(sat, 30, 255, cv2.THRESH_BINARY_INV)[1]
        bright = cv2.threshold(val, 100, 255, cv2.THRESH_BINARY)[1]
        combined = cv2.bitwise_and(low_sat, bright)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        contours2, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours2:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = min(cw, ch) / max(cw, ch) if max(cw, ch) > 0 else 0
            if 40 < cw < 150 and 40 < ch < 150 and aspect > 0.6 and y < h * 0.5:
                score = aspect * (cw * ch)
                if score > best_score:
                    best_score = score
                    best_box = (x, y, cw, ch)
    
    if best_box is None:
        return None, None
    
    bx, by, bcw, bch = best_box
    pad = 3
    x1 = max(0, bx + pad)
    y1 = max(0, by + pad)
    x2 = min(w, bx + bcw - pad)
    y2 = min(h, by + bch - pad)
    
    ref_crop = screenshot[y1:y2, x1:x2]
    print(f"  Auto-detected ref icon at ({bx},{by}) size {bcw}×{bch}")
    return ref_crop, best_box


def extract_canvas(screenshot: np.ndarray, ref_box: tuple | None = None) -> np.ndarray:
    """Extract the canvas, masking out the floating ref icon to prevent self-matching.
    
    Uses cv2.inpaint to fill the ref icon area with surrounding background texture,
    so DINO features in that region won't create a false peak.
    
    Also crops out the decorative border frame around the canvas (grey/blue edges)
    and detects canvas boundaries in full-page screenshots.
    """
    canvas = screenshot.copy()
    h, w = canvas.shape[:2]
    
    # Step 1: Detect and crop the canvas boundary (remove decorative border)
    # The plasma canvas is colorful (high saturation). Border/UI elements are grey.
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    
    # Find rows/cols with high saturation (plasma content)
    row_sat = sat.mean(axis=1)
    col_sat = sat.mean(axis=0)
    
    # Threshold: plasma rows have saturation > 50 
    canvas_rows = np.where(row_sat > 50)[0]
    canvas_cols = np.where(col_sat > 50)[0]
    
    if len(canvas_rows) > 10 and len(canvas_cols) > 10:
        # Find the contiguous block of high-saturation rows (skip isolated header pixels)
        # Group consecutive rows and find the largest group
        row_groups = np.split(canvas_rows, np.where(np.diff(canvas_rows) > 5)[0] + 1)
        largest_row_group = max(row_groups, key=len)
        
        top = largest_row_group[0]
        bottom = largest_row_group[-1]
        left = canvas_cols[0]
        right = canvas_cols[-1]
        
        # Ensure minimum canvas size
        if (bottom - top) > 100 and (right - left) > 100:
            # Add small margin to avoid cutting into shapes at edges
            margin = 5
            top = min(top + margin, bottom)
            bottom = max(bottom - margin, top)
            left = min(left + margin, right)
            right = max(right - margin, left)
            
            canvas = canvas[top:bottom+1, left:right+1]
            print(f"  Cropped canvas to ({left},{top})-({right},{bottom}): {canvas.shape[1]}×{canvas.shape[0]}")
            
            # Adjust ref_box coordinates relative to cropped canvas
            if ref_box is not None:
                bx, by, bw, bh = ref_box
                ref_box = (bx - left, by - top, bw, bh)
    
    # Step 2: Mask out the ref icon area via inpainting
    if ref_box is not None:
        bx, by, bw, bh = ref_box
        # Only mask if the ref box is inside the canvas
        ch, cw = canvas.shape[:2]
        if 0 <= bx < cw and 0 <= by < ch:
            pad = 12
            x1 = max(0, bx - pad)
            y1 = max(0, by - pad)
            x2 = min(cw, bx + bw + pad)
            y2 = min(ch, by + bh + pad)
            
            mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            canvas = cv2.inpaint(canvas, mask, inpaintRadius=15, flags=cv2.INPAINT_NS)
            print(f"  Masked ref icon region ({x1},{y1})-({x2},{y2}) via inpainting")
    
    return canvas


def draw_result(canvas: np.ndarray, x: float, y: float,
                heatmap: np.ndarray | None = None) -> np.ndarray:
    """Draw crosshair and heatmap overlay on canvas."""
    vis = canvas.copy()
    
    if heatmap is not None:
        h, w = canvas.shape[:2]
        hm_up = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        hm_norm = ((hm_up - hm_up.min()) / (hm_up.max() - hm_up.min() + 1e-8) * 255)
        hm_color = cv2.applyColorMap(hm_norm.astype(np.uint8), cv2.COLORMAP_JET)
        vis = cv2.addWeighted(vis, 0.6, hm_color, 0.4, 0)
    
    ix, iy = int(round(x)), int(round(y))
    cv2.circle(vis, (ix, iy), 22, (0, 255, 0), 3)
    cv2.line(vis, (ix - 30, iy), (ix + 30, iy), (0, 255, 0), 2)
    cv2.line(vis, (ix, iy - 30), (ix, iy + 30), (0, 255, 0), 2)
    cv2.putText(vis, f"({ix}, {iy}) conf={conf:.2f}" if 'conf' in dir() else f"({ix}, {iy})",
                (ix + 28, iy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return vis


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("CAPTCHA Solver — End-to-End Test on Real Screenshots")
    print("=" * 60)
    
    # Initialize solver
    print("\nLoading DINO model...")
    t0 = time.time()
    solver = CaptchaSolver(
        ref_size=(224, 224),
        matcher_type="qatm",
        qatm_alpha=1.0,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"  Version: DINO{solver.feature_extractor.version}")
    print(f"  Patch size: {solver.feature_extractor.patch_size}")
    print(f"  Embed dim: {solver.feature_extractor.embed_dim}")
    print(f"  Device: {solver.feature_extractor.device}")
    print()

    # Test files
    samples = [
        (os.path.join(ARTIFACT_DIR, "media__1779543415160.jpg"), "sample_3"),
        (os.path.join(ARTIFACT_DIR, "media__1779541802429.jpg"), "sample_1"),
        (os.path.join(ARTIFACT_DIR, "media__1779541802713.jpg"), "sample_2"),
    ]

    for path, name in samples:
        if not os.path.exists(path):
            print(f"⚠️  {name}: file not found")
            continue
        
        print(f"── {name}: {os.path.basename(path)} ──")
        
        screenshot = cv2.imread(path)
        if screenshot is None:
            print("  ❌ Could not load")
            continue
        
        h, w = screenshot.shape[:2]
        print(f"  Size: {w}×{h}")
        
        # Extract reference icon + its bounding box
        ref_icon, ref_bbox = extract_ref_icon(screenshot)
        if ref_icon is None:
            print("  ❌ Could not detect reference icon")
            continue
        
        rh, rw = ref_icon.shape[:2]
        print(f"  Reference icon: {rw}×{rh}")
        
        # Remove grey background from ref icon — replace with random noise so
        # DINO doesn't match the grey bg against edge/transition zones in the canvas.
        # The symbol is dark on a light grey background.
        ref_gray = cv2.cvtColor(ref_icon, cv2.COLOR_BGR2GRAY)
        # Dark pixels (< 120) are the symbol; light pixels are background
        _, symbol_mask = cv2.threshold(ref_gray, 140, 255, cv2.THRESH_BINARY_INV)
        # Dilate to capture anti-aliased edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        symbol_mask = cv2.dilate(symbol_mask, kernel, iterations=1)
        # Fill background with Gaussian noise (simulates textured background)
        noise = np.random.randint(100, 180, ref_icon.shape, dtype=np.uint8)
        ref_clean = ref_icon.copy()
        ref_clean[symbol_mask == 0] = noise[symbol_mask == 0]
        
        # Save ref icon for inspection
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_ref_icon.png"), ref_icon)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_ref_clean.png"), ref_clean)
        
        # Canvas: cropped + ref icon masked out via inpainting
        canvas = extract_canvas(screenshot, ref_bbox)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_canvas_masked.jpg"), canvas)
        
        ch, cw = canvas.shape[:2]
        
        # Run solver with BOTH original and cleaned ref icons
        print("  Running solver (cleaned ref icon)...")
        result = solver.solve(ref_clean, canvas, return_heatmap=True)
        
        print(f"  ✅ Match: ({result.x:.1f}, {result.y:.1f})")
        print(f"     Confidence: {result.confidence:.3f}")
        print(f"     Peak similarity: {result.peak_similarity:.3f}")
        print(f"     Peak-to-second ratio: {result.peak_to_second_ratio:.3f}")
        print(f"     Peak sharpness: {result.peak_sharpness:.3f}")
        print(f"     Elapsed: {result.elapsed_ms:.0f} ms")
        
        # Also try with original ref icon for comparison
        print("  Running solver (original ref icon)...")
        result_orig = solver.solve(ref_icon, canvas, return_heatmap=True)
        print(f"  📊 Original: ({result_orig.x:.1f}, {result_orig.y:.1f}) conf={result_orig.confidence:.3f}")
        
        # Save visualization (on cropped canvas)
        vis = canvas.copy()
        if result.heatmap is not None:
            hm_up = cv2.resize(result.heatmap, (cw, ch), interpolation=cv2.INTER_CUBIC)
            hm_norm = ((hm_up - hm_up.min()) / (hm_up.max() - hm_up.min() + 1e-8) * 255)
            hm_color = cv2.applyColorMap(hm_norm.astype(np.uint8), cv2.COLORMAP_JET)
            vis = cv2.addWeighted(vis, 0.6, hm_color, 0.4, 0)
        
        ix, iy = int(round(result.x)), int(round(result.y))
        cv2.circle(vis, (ix, iy), 22, (0, 255, 0), 3)
        cv2.line(vis, (ix - 30, iy), (ix + 30, iy), (0, 255, 0), 2)
        cv2.line(vis, (ix, iy - 30), (ix, iy + 30), (0, 255, 0), 2)
        label = f"({ix},{iy}) conf={result.confidence:.2f}"
        cv2.putText(vis, label, (ix + 28, iy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        out_path = os.path.join(OUTPUT_DIR, f"{name}_result.jpg")
        cv2.imwrite(out_path, vis)
        print(f"  Saved: {out_path}")
        
        # Save heatmap only
        if result.heatmap is not None:
            hm_only = cv2.resize(result.heatmap, (cw, ch), interpolation=cv2.INTER_CUBIC)
            hm_only = ((hm_only - hm_only.min()) / (hm_only.max() - hm_only.min() + 1e-8) * 255)
            hm_only = cv2.applyColorMap(hm_only.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_heatmap.jpg"), hm_only)

        
        print()

    print("=" * 60)
    print(f"All results saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
