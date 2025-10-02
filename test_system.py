"""Test if everything is set up correctly."""
import sys
from pathlib import Path

print("=" * 60)
print("SYSTEM CHECK")
print("=" * 60)

# Check imports
try:
    from digit_detector import DigitDetector
    print("✓ digit_detector.py - OK")
except Exception as e:
    print(f"✗ digit_detector.py - ERROR: {e}")
    sys.exit(1)

try:
    from trainer import YOLOTrainer
    print("✓ trainer.py - OK")
except Exception as e:
    print(f"✗ trainer.py - ERROR: {e}")
    sys.exit(1)

try:
    from config import config
    print("✓ config.py - OK")
    print(f"  Device: {config.model.device} ({'GPU' if config.model.device >= 0 else 'CPU'})")
except Exception as e:
    print(f"✗ config.py - ERROR: {e}")
    sys.exit(1)

# Check data.yaml
data_yaml = Path("data/data.yaml")
if data_yaml.exists():
    print("✓ data/data.yaml - EXISTS")
    import yaml
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
        print(f"  Classes: {data['nc']}")
        print(f"  Names: {data['names']}")
else:
    print("✗ data/data.yaml - NOT FOUND")
    sys.exit(1)

# Check model
model_path = Path("models/yolov8n.pt")
if model_path.exists():
    print(f"✓ Pretrained model - EXISTS ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
else:
    print("✗ Pretrained model - NOT FOUND")
    print("  Download: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt")

# Check dataset
train_images = Path("data/train/images")
val_images = Path("data/val/images")

if train_images.exists():
    train_count = len(list(train_images.glob("*.jpg"))) + len(list(train_images.glob("*.png")))
    print(f"✓ Training images: {train_count}")
else:
    print("✗ Training images - DIRECTORY NOT FOUND")

if val_images.exists():
    val_count = len(list(val_images.glob("*.jpg"))) + len(list(val_images.glob("*.png")))
    print(f"✓ Validation images: {val_count}")
else:
    print("✗ Validation images - DIRECTORY NOT FOUND")

print("=" * 60)
print("✓ SYSTEM READY!")
print("\nNext step: python main.py train --data data/data.yaml --epochs 100 --device cpu")
print("=" * 60)