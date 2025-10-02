# verify_dataset.py
from pathlib import Path

def verify_dataset():
    data_root = Path("data")
    
    # Check images
    train_images = list((data_root / "train/images").glob("*.jpg"))
    val_images = list((data_root / "val/images").glob("*.jpg"))
    
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    
    # Check labels
    train_labels = list((data_root / "train/labels").glob("*.txt"))
    val_labels = list((data_root / "val/labels").glob("*.txt"))
    
    print(f"Training labels: {len(train_labels)}")
    print(f"Validation labels: {len(val_labels)}")
    
    # Verify matching
    if len(train_images) != len(train_labels):
        print("⚠️  Mismatch in training data!")
    else:
        print("✓ Training data OK")
    
    if len(val_images) != len(val_labels):
        print("⚠️  Mismatch in validation data!")
    else:
        print("✓ Validation data OK")

if __name__ == "__main__":
    verify_dataset()