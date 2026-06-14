"""Split YOLO dataset into train/val sets (80/20)."""
import os
import random
import shutil
from pathlib import Path


def split_dataset(
    data_dir: str = "data",
    train_ratio: float = 0.8,
    seed: int = 42,
):
    """Split images and labels into train/ and val/ subdirs."""
    data_path = Path(data_dir)
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"
    
    # Get all image files
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ])
    
    print(f"Found {len(image_files)} images")
    
    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(image_files)
    
    # Split
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Create output dirs
    for split_name, files in [("train", train_files), ("val", val_files)]:
        img_out = data_path / split_name / "images"
        lbl_out = data_path / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        
        for img_file in files:
            # Copy image
            dst_img = img_out / img_file.name
            if not dst_img.exists():
                shutil.copy2(img_file, dst_img)
            
            # Copy label
            lbl_file = labels_dir / (img_file.stem + ".txt")
            if lbl_file.exists():
                dst_lbl = lbl_out / lbl_file.name
                if not dst_lbl.exists():
                    shutil.copy2(lbl_file, dst_lbl)
            else:
                print(f"  Warning: no label for {img_file.name}")
        
        print(f"  {split_name}: {len(files)} images → {img_out}")
    
    print("Done!")


if __name__ == "__main__":
    split_dataset()
