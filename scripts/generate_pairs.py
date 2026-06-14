"""Generate training data for the Siamese shape matcher.

Creates pairs of (ref_icon_crop, canvas_shape_crop, is_match) 
using the puzzle generator. This data trains the matching model
to recognize that a clean ref icon and a plasma-embedded shape
are the same shape.
"""
import os
import random
import json

import cv2
import numpy as np

# Fix the syntax bug in puzzle_generator
import puzzle_generator as pg


def generate_training_pairs(
    num_puzzles: int = 2000,
    output_dir: str = "data/siamese",
    canvas_size: tuple[int, int] = (400, 300),
    num_shapes: int = 4,
    seed: int = 42,
):
    """Generate Siamese training pairs.
    
    For each puzzle:
    - Extract ref icon crop
    - Extract each shape crop from canvas (using known positions)
    - Label: 1 if same shape type as ref, 0 otherwise
    
    Creates balanced positive/negative pairs.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    pairs_dir = os.path.join(output_dir, "pairs")
    os.makedirs(pairs_dir, exist_ok=True)
    
    manifest = []
    pair_id = 0
    
    for puzzle_idx in range(num_puzzles):
        width, height = canvas_size
        
        # ---- Replicate puzzle generation to get shape positions & indices ----
        canvas = pg.generate_wavy_background(width, height)
        
        grid_positions = [
            (int(width * 0.2), int(height * 0.35)),
            (int(width * 0.4), int(height * 0.7)),
            (int(width * 0.6), int(height * 0.35)),
            (int(width * 0.8), int(height * 0.7)),
        ]
        random.shuffle(grid_positions)
        positions = grid_positions[:num_shapes]
        
        shape_indices = list(range(len(pg.SHAPE_DRAW_FUNCS)))
        random.shuffle(shape_indices)
        shape_indices = shape_indices[:num_shapes]
        
        target_idx = random.randint(0, num_shapes - 1)
        target_shape_idx = shape_indices[target_idx]
        
        shape_size = 60
        
        # Draw shapes on canvas (same as puzzle_generator)
        for i, (center, s_idx) in enumerate(zip(positions, shape_indices)):
            glow_mask = np.zeros((height, width), dtype=np.uint8)
            pg.SHAPE_DRAW_FUNCS[s_idx](glow_mask, center, shape_size, 255, thickness=12)
            glow_mask_blurred = cv2.GaussianBlur(glow_mask, (15, 15), 0)
            glow_color = np.array([20, 50, 255], dtype=np.float32)
            glow_img = np.zeros_like(canvas, dtype=np.float32)
            for c in range(3):
                glow_img[:, :, c] = (glow_mask_blurred / 255.0) * glow_color[c]
            canvas_f = canvas.astype(np.float32)
            canvas_f = np.clip(canvas_f + glow_img, 0, 255)
            canvas = canvas_f.astype(np.uint8)
            pg.SHAPE_DRAW_FUNCS[s_idx](canvas, center, shape_size, (10, 10, 80), thickness=4)
            pg.SHAPE_DRAW_FUNCS[s_idx](canvas, center, shape_size, (80, 200, 255), thickness=1)
        
        # Create reference icon
        ref_size = 100
        ref_img = np.ones((ref_size, ref_size, 3), dtype=np.float32) * 220
        for y in range(ref_size):
            for x in range(ref_size):
                ref_img[y, x, 0] += 10 * np.sin(x * 0.1) + 10 * np.cos(y * 0.1)
                ref_img[y, x, 1] += 5 * np.cos(x * 0.05)
                ref_img[y, x, 2] += 15 * np.sin(y * 0.08)
        ref_img = np.clip(ref_img, 0, 255).astype(np.uint8)
        ref_center = (ref_size // 2, ref_size // 2)
        pg.SHAPE_DRAW_FUNCS[target_shape_idx](ref_img, ref_center, 65, (30, 30, 30), thickness=5)
        ref_img = cv2.GaussianBlur(ref_img, (3, 3), 0.5).astype(np.uint8)
        
        # Save ref icon (one per puzzle)
        ref_filename = f"ref_p{puzzle_idx:05d}.png"
        ref_path = os.path.join(pairs_dir, ref_filename)
        cv2.imwrite(ref_path, ref_img)
        
        # Extract shape crops from canvas and create pairs
        pad = 15  # extra padding around shape center
        crop_size = shape_size + 2 * pad
        
        for i, (center, s_idx) in enumerate(zip(positions, shape_indices)):
            cx, cy = center
            x1 = max(0, cx - crop_size // 2)
            y1 = max(0, cy - crop_size // 2)
            x2 = min(width, cx + crop_size // 2)
            y2 = min(height, cy + crop_size // 2)
            
            crop = canvas[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            is_match = 1 if s_idx == target_shape_idx else 0
            
            cand_filename = f"cand_{pair_id:06d}.png"
            crop_path = os.path.join(pairs_dir, cand_filename)
            cv2.imwrite(crop_path, crop)
            
            manifest.append({
                "pair_id": pair_id,
                "ref": ref_filename,
                "candidate": cand_filename,
                "label": is_match,
                "ref_shape": target_shape_idx,
                "cand_shape": s_idx,
                "puzzle_idx": puzzle_idx,
            })
            pair_id += 1
        
        if puzzle_idx % 200 == 0:
            n_pos = sum(1 for m in manifest if m["label"] == 1)
            n_neg = sum(1 for m in manifest if m["label"] == 0)
            print(f"  Puzzle {puzzle_idx}/{num_puzzles}: {pair_id} pairs ({n_pos} pos, {n_neg} neg)")
    
    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    n_pos = sum(1 for m in manifest if m["label"] == 1)
    n_neg = sum(1 for m in manifest if m["label"] == 0)
    print(f"\nDone! Generated {pair_id} pairs ({n_pos} positive, {n_neg} negative)")
    print(f"Manifest: {manifest_path}")
    print(f"Pairs dir: {pairs_dir}")


if __name__ == "__main__":
    generate_training_pairs(num_puzzles=2000)
