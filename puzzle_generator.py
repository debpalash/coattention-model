import os
import random
import numpy as np
import cv2

def generate_wavy_background(width, height):
    """
    Generates a colorful background with concentric circular ripples / plasma patterns.
    """
    x = np.arange(width)
    y = np.arange(height)
    xv, yv = np.meshgrid(x, y)
    
    # Define a few ripple centers to create overlapping concentric waves
    num_centers = 3
    val = np.zeros((height, width), dtype=np.float32)
    for _ in range(num_centers):
        cx = random.uniform(0, width)
        cy = random.uniform(0, height)
        freq = random.uniform(0.05, 0.15)
        dist = np.sqrt((xv - cx)**2 + (yv - cy)**2)
        val += np.sin(dist * freq)
    
    # Normalize val to 0-255
    val = (val - val.min()) / (val.max() - val.min()) * 255
    val = val.astype(np.uint8)
    
    # Apply a color map to make it vibrant and plasma-like
    background = cv2.applyColorMap(val, cv2.COLORMAP_PLASMA)
    
    # Add some large slow gradients
    gradient = np.zeros((height, width, 3), dtype=np.float32)
    for c in range(3):
        # random linear gradient direction
        gx, gy = random.uniform(-1, 1), random.uniform(-1, 1)
        norm = np.sqrt(gx**2 + gy**2)
        gx, gy = gx / norm, gy / norm
        proj = xv * gx + yv * gy
        proj = (proj - proj.min()) / (proj.max() - proj.min())
        gradient[:, :, c] = proj
        
    background = (background.astype(np.float32) * 0.7 + gradient * 255 * 0.3).astype(np.uint8)
    return background

def draw_shape_taurus(img, center, size, color, thickness=3):
    # Circle with two ears
    cx, cy = center
    r = int(size * 0.35)
    cv2.circle(img, (cx, cy + int(size * 0.1)), r, color, thickness, lineType=cv2.LINE_AA)
    # Draw ears
    ear_w = int(size * 0.15)
    ear_h = int(size * 0.25)
    # Left ear
    cv2.ellipse(img, (cx - int(size * 0.2), cy - int(size * 0.2)), (ear_w, ear_h), -30, 0, 360, color, thickness, lineType=cv2.LINE_AA)
    # Right ear
    cv2.ellipse(img, (cx + int(size * 0.2), cy - int(size * 0.2)), (ear_w, ear_h), 30, 0, 360, color, thickness, lineType=cv2.LINE_AA)

def draw_shape_butterfly(img, center, size, color, thickness=3):
    # Two intersecting curved arches (cross/butterfly)
    cx, cy = center
    hs = size // 2
    # Draw curved shapes using Bezier curves or simple arcs
    # We will draw a curvy X
    pts_left = np.array([
        [cx - hs, cy - hs],
        [cx - int(size * 0.1), cy],
        [cx - hs, cy + hs]
    ], dtype=np.int32)
    pts_right = np.array([
        [cx + hs, cy - hs],
        [cx + int(size * 0.1), cy],
        [cx + hs, cy + hs]
    ], dtype=np.int32)
    cv2.polylines(img, [pts_left], False, color, thickness, lineType=cv2.LINE_AA)
    cv2.polylines(img, [pts_right], False, color, thickness, lineType=cv2.LINE_AA)
    # Connect with a horizontal bar
    cv2.line(img, (cx - int(size * 0.15), cy), (cx + int(size * 0.15), cy), color, thickness, lineType=cv2.LINE_AA)

def draw_shape_triangle_circle(img, center, size, color, thickness=3):
    # Triangle with a circle on top
    cx, cy = center
    hs = size // 2
    # Triangle vertices
    pt1 = (cx, cy - int(hs * 0.2))
    pt2 = (cx - int(hs * 0.8), cy + hs)
    pt3 = (cx + int(hs * 0.8), cy + hs)
    cv2.line(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, pt2, pt3, color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, pt3, pt1, color, thickness, lineType=cv2.LINE_AA)
    # Circle at the peak
    cv2.circle(img, (cx, cy - int(hs * 0.7)), int(size * 0.2), color, thickness, lineType=cv2.LINE_AA)

def draw_shape_lens_circle(img, center, size, color, thickness=3):
    # Circle containing a vertical lens/ellipse inside
    cx, cy = center
    r = int(size * 0.45)
    cv2.circle(img, (cx, cy), r, color, thickness, lineType=cv2.LINE_AA)
    # Vertical lens/football shape (intersection of two arcs)
    # Draw left arc
    cv2.ellipse(img, (cx - int(size * 0.25), cy), (int(size * 0.3), int(size * 0.45)), 0, -60, 60, color, thickness, lineType=cv2.LINE_AA)
    # Draw right arc
    cv2.ellipse(img, (cx + int(size * 0.25), cy), (int(size * 0.3), int(size * 0.45)), 0, 120, 240, color, thickness, lineType=cv2.LINE_AA)

def draw_shape_circle_slash(img, center, size, color, thickness=3):
    # Circle with a diagonal slash inside
    cx, cy = center
    r = int(size * 0.45)
    cv2.circle(img, (cx, cy), r, color, thickness, lineType=cv2.LINE_AA)
    # Diagonal line
    offset = int(r * 0.7)
    cv2.line(img, (cx - offset, cy + offset), (cx + offset, cy - offset), color, thickness, lineType=cv2.LINE_AA)

SHAPE_DRAW_FUNCS = [
    draw_shape_taurus,
    draw_shape_butterfly,
    draw_shape_triangle_circle,
    draw_shape_lens_circle,
    draw_shape_circle_slash
]

def generate_puzzle(width=400, height=300, num_shapes=4):
    canvas = generate_wavy_background(width, height)
    
    # Define shape positions that don't overlap
    grid_positions = [
        (int(width * 0.2), int(height * 0.35)),
        (int(width * 0.4), int(height * 0.7)),
        (int(width * 0.6), int(height * 0.35)),
        (int(width * 0.8), int(height * 0.7)),
    ]
    random.shuffle(grid_positions)
    positions = grid_positions[:num_shapes]
    
    # Choose unique shape index for each position
    shape_indices = list(range(len(SHAPE_DRAW_FUNCS)))
    random.shuffle(shape_indices)
    shape_indices = shape_indices[:num_shapes]
    
    # Select one of the positions as the target (matching shape)
    target_idx = random.randint(0, num_shapes - 1)
    target_shape_idx = shape_indices[target_idx]
    target_center = positions[target_idx]
    
    shape_size = 60
    
    # Draw shapes on the canvas
    for i, (center, s_idx) in enumerate(zip(positions, shape_indices)):
        # Draw background glow (thick red/orange blurred shape)
        glow_mask = np.zeros((height, width), dtype=np.uint8)
        SHAPE_DRAW_FUNCS[s_idx](glow_mask, center, shape_size, 255, thickness=12)
        glow_mask_blurred = cv2.GaussianBlur(glow_mask, (15, 15), 0)
        
        # Glow color is bright red/orange
        glow_color = np.array([20, 50, 255], dtype=np.float32) # BGR
        glow_img = np.zeros_like(canvas, dtype=np.float32)
        for c in range(3):
            glow_img[:, :, c] = (glow_mask_blurred / 255.0) * glow_color[c]
            
        canvas_f = canvas.astype(np.float32)
        # Add glow (clamped to 255)
        canvas_f = np.clip(canvas_f + glow_img, 0, 255)
        canvas = canvas_f.astype(np.uint8)
        
        # Draw main outline (thick dark-red or black outline on top)
        SHAPE_DRAW_FUNCS[s_idx](canvas, center, shape_size, (10, 10, 80), thickness=4)
        # Inner glow / highlights (thin light orange/yellow)
        SHAPE_DRAW_FUNCS[s_idx](canvas, center, shape_size, (80, 200, 255), thickness=1)

    # Now create the reference/query icon
    # It should represent the target shape drawn cleanly on a neutral background
    # Let's draw it in dark grey/black on a light grey-blue background (like the CAPTCHA reference icon)
    ref_size = 100
    ref_img = np.ones((ref_size, ref_size, 3), dtype=np.uint8) * 220 # Light grey background
    # Add subtle background pattern or gradient to simulate the screenshot
    for y in range(ref_size):
        for x in range(ref_size):
            ref_img[y, x, 0] += int(10 * np.sin(x * 0.1) + 10 * np.cos(y * 0.1)) # B
            ref_img[y, x, 1] += int(5 * np.cos(x * 0.05)) # G
            ref_img[y, x, 2] += int(15 * np.sin(y * 0.08)) # R
            
    # Apply a subtle blur to background pattern
    ref_img = cv2.GaussianBlur(ref_img, (3, 3), 0)
    
    # Draw the target shape in the center of the reference icon
    ref_center = (ref_size // 2, ref_size // 2)
    # The symbol itself is black with smooth edges
    SHAPE_DRAW_FUNCS[target_shape_idx](ref_img, ref_center, 65, (30, 30, 30), thickness=5)
    # Optional smoothing
    ref_img = cv2.GaussianBlur(ref_img, (3, 3), 0.5).astype(np.uint8)
    
    # Bounding box of the target shape in the canvas
    bx = target_center[0] - shape_size // 2
    by = target_center[1] - shape_size // 2
    bw = shape_size
    bh = shape_size
    
    return canvas, ref_img, target_center, (bx, by, bw, bh)

if __name__ == "__main__":
    # Generate and save a few examples
    os.makedirs("test_puzzles", exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    
    for i in range(10):
        canvas, ref, center, bbox = generate_puzzle()
        cv2.imwrite(f"test_puzzles/puzzle_{i}_canvas.png", canvas)
        cv2.imwrite(f"test_puzzles/puzzle_{i}_ref.png", ref)
        with open(f"test_puzzles/puzzle_{i}_gt.txt", "w") as f:
            f.write(f"{center[0]},{center[1]},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")
        print(f"Generated puzzle {i}: target center = {center}, bbox = {bbox}")
