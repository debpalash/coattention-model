"""Train YOLO11n on the CAPTCHA shape detection dataset."""
from ultralytics import YOLO


def train():
    # Load pretrained YOLO11n (nano — fastest, best for real-time inference)
    model = YOLO("yolo11n.pt")
    
    # Fine-tune on our dataset
    results = model.train(
        data="data/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,          # Early stopping
        lr0=0.01,             # Initial learning rate
        lrf=0.01,             # Final learning rate factor
        mosaic=1.0,           # Mosaic augmentation
        mixup=0.1,            # Mixup augmentation
        copy_paste=0.1,       # Copy-paste augmentation
        degrees=15,           # Rotation augmentation
        scale=0.5,            # Scale augmentation
        flipud=0.5,           # Vertical flip
        fliplr=0.5,           # Horizontal flip
        hsv_h=0.015,          # HSV-Hue augmentation
        hsv_s=0.7,            # HSV-Saturation augmentation
        hsv_v=0.4,            # HSV-Value augmentation
        project="runs/detect",
        name="captcha_shapes",
        exist_ok=True,
        device="cpu",         # Mac CPU (use "mps" if Metal works)
        workers=4,
        verbose=True,
    )
    
    print(f"\nTraining complete!")
    print(f"Best model: runs/detect/captcha_shapes/weights/best.pt")
    
    # Validate
    metrics = model.val()
    print(f"\nValidation metrics:")
    print(f"  mAP@50:    {metrics.box.map50:.3f}")
    print(f"  mAP@50-95: {metrics.box.map:.3f}")
    print(f"  Precision:  {metrics.box.mp:.3f}")
    print(f"  Recall:     {metrics.box.mr:.3f}")


if __name__ == "__main__":
    train()
