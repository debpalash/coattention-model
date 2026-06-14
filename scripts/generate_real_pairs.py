"""Generate realistic training pairs from real CAPTCHA crops.

Strategy: Take each real ref icon, apply plasma-like augmentations to simulate
how it would look on the canvas, then create positive pairs (ref → augmented_ref)
and negative pairs (ref → different_shape augmented).

This bridges the domain gap between clean ref icons and plasma-embedded shapes.
"""
import os
import json
import random

import cv2
import numpy as np


def apply_plasma_augmentation(crop: np.ndarray, seed: int = 0) -> np.ndarray:
    """Transform a clean ref icon crop to look like a plasma-embedded shape."""
    rng = np.random.RandomState(seed)
    h, w = crop.shape[:2]
    
    # 1. Extract the shape mask (dark regions on light background)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Generate a plasma-like background
    x = np.arange(w)
    y = np.arange(h)
    xv, yv = np.meshgrid(x, y)
    
    val = np.zeros((h, w), dtype=np.float32)
    for _ in range(3):
        cx = rng.uniform(0, w)
        cy = rng.uniform(0, h)
        freq = rng.uniform(0.05, 0.2)
        val += np.sin(np.sqrt((xv - cx)**2 + (yv - cy)**2) * freq + rng.uniform(0, 2*np.pi))
    
    val = ((val - val.min()) / (val.max() - val.min()) * 255).astype(np.uint8)
    bg = cv2.applyColorMap(val, rng.choice([
        cv2.COLORMAP_PLASMA, cv2.COLORMAP_INFERNO, cv2.COLORMAP_MAGMA,
        cv2.COLORMAP_HOT, cv2.COLORMAP_OCEAN, cv2.COLORMAP_JET,
    ]))
    
    # 3. Create glowing shape outline
    # Dilate mask for glow
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    glow_mask = cv2.dilate(mask, kernel, iterations=2)
    glow_mask = cv2.GaussianBlur(glow_mask, (11, 11), 3)
    
    # Glow color (random warm color)
    glow_hue = rng.randint(0, 30)  # Red-orange range
    glow_color = np.array([glow_hue, 200, 255], dtype=np.uint8).reshape(1, 1, 3)
    glow_bgr = cv2.cvtColor(glow_color, cv2.COLOR_HSV2BGR).astype(np.float32).flatten()
    
    glow_img = np.zeros_like(bg, dtype=np.float32)
    for c in range(3):
        glow_img[:, :, c] = (glow_mask / 255.0) * glow_bgr[c]
    
    result = np.clip(bg.astype(np.float32) + glow_img, 0, 255).astype(np.uint8)
    
    # 4. Draw dark shape outline on top
    edge_mask = cv2.Canny(mask, 50, 150)
    edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)
    
    result[edge_mask > 0] = result[edge_mask > 0] * 0.3  # Darken edges
    
    # 5. Add noise
    noise = rng.normal(0, 8, result.shape).astype(np.float32)
    result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # 6. Random brightness/contrast
    alpha = rng.uniform(0.7, 1.3)
    beta = rng.uniform(-20, 20)
    result = np.clip(result * alpha + beta, 0, 255).astype(np.uint8)
    
    return result


def generate_real_pairs(
    crop_dir: str = "data/real_crops",
    output_dir: str = "data/siamese_real",
    num_augmentations: int = 10,
    seed: int = 42,
):
    """Generate training pairs from real ref icon crops."""
    random.seed(seed)
    np.random.seed(seed)
    
    pairs_dir = os.path.join(output_dir, "pairs")
    os.makedirs(pairs_dir, exist_ok=True)
    
    # Load all ref icon crops
    all_crops = sorted(os.listdir(crop_dir))
    ref_crops = []
    
    for name in all_crops:
        crop = cv2.imread(os.path.join(crop_dir, name))
        if crop is None:
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_sat = hsv[:, :, 1].mean()
        if mean_sat < 50:  # Low saturation = ref icon
            ref_crops.append((name, crop))
    
    print(f"Found {len(ref_crops)} ref icons")
    
    manifest = []
    pair_id = 0
    
    for ref_idx, (ref_name, ref_crop) in enumerate(ref_crops):
        # Save original ref
        ref_filename = f"ref_{ref_idx:04d}.png"
        cv2.imwrite(os.path.join(pairs_dir, ref_filename), ref_crop)
        
        for aug_idx in range(num_augmentations):
            seed_val = ref_idx * 1000 + aug_idx
            
            # POSITIVE: augment THIS ref to look like plasma shape
            pos_aug = apply_plasma_augmentation(ref_crop, seed=seed_val)
            pos_filename = f"pos_{pair_id:06d}.png"
            cv2.imwrite(os.path.join(pairs_dir, pos_filename), pos_aug)
            
            manifest.append({
                "pair_id": pair_id,
                "ref": ref_filename,
                "candidate": pos_filename,
                "label": 1,
            })
            pair_id += 1
            
            # NEGATIVE: augment a DIFFERENT ref to look like plasma shape
            other_idx = random.choice([i for i in range(len(ref_crops)) if i != ref_idx])
            neg_crop = ref_crops[other_idx][1]
            neg_aug = apply_plasma_augmentation(neg_crop, seed=seed_val + 500)
            neg_filename = f"neg_{pair_id:06d}.png"
            cv2.imwrite(os.path.join(pairs_dir, neg_filename), neg_aug)
            
            manifest.append({
                "pair_id": pair_id,
                "ref": ref_filename,
                "candidate": neg_filename,
                "label": 0,
            })
            pair_id += 1
        
        if ref_idx % 20 == 0:
            print(f"  Ref {ref_idx}/{len(ref_crops)}: {pair_id} pairs")
    
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    n_pos = sum(1 for m in manifest if m["label"] == 1)
    n_neg = sum(1 for m in manifest if m["label"] == 0)
    print(f"\nDone! {pair_id} pairs ({n_pos} pos, {n_neg} neg)")


if __name__ == "__main__":
    generate_real_pairs(num_augmentations=15)
