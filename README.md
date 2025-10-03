```markdown
# 🔢 Digital Number OCR System

A production-ready OCR system for detecting and recognizing digital numbers from displays using YOLOv8. Detects 12 classes: digits **0-9**, **colon (:)**, and **period (.)** from digital watches, thermometers, meters, and other digital displays.

---

## 📋 Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Detection](#detection)
- [Output Format](#output-format)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Performance Tips](#performance-tips)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

- **12-Class Detection**: Recognizes digits 0-9, colon (:), and period (.)
- **High Accuracy**: YOLOv8-based detection with confidence scoring
- **Smart Filtering**: Automatically removes temperature symbols (°, C, F)
- **Production Ready**: Complete error handling, logging, and validation
- **Batch Processing**: Process multiple images efficiently
- **Visual Feedback**: Generates annotated images with bounding boxes
- **JSONL Output**: Structured output format for easy integration
- **GPU/CPU Support**: Auto-detects and uses available hardware
- **Easy Training**: Simple CLI for model training and validation

---

## 💻 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended)
- **Storage**: 2GB free space
- **OS**: Windows 10/11, Linux, macOS

### Recommended for Training
- **GPU**: NVIDIA GPU with 4GB+ VRAM (RTX 3060 or better)
- **CUDA**: 11.8 or 12.1+
- **RAM**: 16GB
- **Storage**: 5GB free space

### For CPU-Only Users
- **RAM**: 8GB minimum
- Training will be slower but functional
- Use smaller batch sizes (4-8)

---

## 🚀 Installation

### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd digital-number-ocr
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

#### For CPU:
```bash
pip install -r requirements.txt
```

#### For GPU (CUDA 12.1):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python numpy pyyaml
```

#### For GPU (CUDA 11.8):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python numpy pyyaml
```

### Step 4: Download Pretrained Model

```bash
# Create models directory
mkdir models

# Download YOLOv8n pretrained weights
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt

# Or using curl
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -o models/yolov8n.pt
```

### Step 5: Verify Installation

```bash
python test_system.py
```

Expected output:
```
✓ config.py - OK
✓ digit_detector.py - OK
✓ trainer.py - OK
✓ data/data.yaml - EXISTS
✓ Pretrained model - EXISTS
✓ SYSTEM READY!
```

---

## 📁 Project Structure

```
digital-number-ocr/
│
├── config.py                   # Configuration management
├── logger.py                   # Logging system
├── exceptions.py               # Custom exceptions
├── digit_detector.py           # Main detection module
├── trainer.py                  # Training module
├── main.py                     # CLI interface
│
├── data/
│   ├── data.yaml              # Dataset configuration
│   ├── train/
│   │   ├── images/            # Training images
│   │   └── labels/            # Training labels (YOLO format)
│   └── val/
│       ├── images/            # Validation images
│       └── labels/            # Validation labels
│
├── models/
│   ├── yolov8n.pt            # Pretrained model
│   └── yolov8n-finetuned.pt  # Your trained model (generated)
│
├── logs/                      # Log files
│   ├── digital_ocr_YYYYMMDD.log
│   └── digital_ocr_error_YYYYMMDD.log
│
├── results/                   # Detection results
│   ├── results.jsonl         # Detection output
│   └── vis_*.jpg             # Visualizations
│
├── runs/                      # Training outputs
│   └── train/
│       └── ocr_yolov8/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           └── results.png
│
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── test_system.py            # System verification script
```

---

## 🎯 Quick Start

### 1. Prepare Your Dataset

```bash
# Place your images and labels
data/
  train/
    images/  ← Your training images (.jpg, .png)
    labels/  ← YOLO format labels (.txt)
  val/
    images/  ← Validation images
    labels/  ← Validation labels
```

### 2. Train the Model

```bash
# CPU Training
python main.py train --data data/data.yaml --epochs 100 --batch 8 --device cpu

# GPU Training
python main.py train --data data/data.yaml --epochs 300 --batch 16 --device 0
```

### 3. Detect Digits

```bash
# Single image
python main.py detect --image clock.jpg

# Multiple images
python main.py detect --image-dir test_images/

# Custom output file
python main.py detect --image-dir test_images/ --output my_results.jsonl
```

---

## 📊 Dataset Preparation

### Required Format: YOLO

Each image needs a corresponding `.txt` file with the same name.

**Example:**
- Image: `train/images/clock_001.jpg`
- Label: `train/labels/clock_001.txt`

### Label Format

Each line in the label file:
```
<class_id> <x_center> <y_center> <width> <height>
```

All values normalized to [0, 1].

**Example (`clock_001.txt`):**
```
1 0.234 0.456 0.045 0.089    # Digit '1'
2 0.289 0.456 0.045 0.089    # Digit '2'
10 0.344 0.456 0.020 0.089   # Colon ':'
3 0.389 0.456 0.045 0.089    # Digit '3'
5 0.444 0.456 0.045 0.089    # Digit '5'
```

### Class IDs

```
0  → '0'
1  → '1'
2  → '2'
3  → '3'
4  → '4'
5  → '5'
6  → '6'
7  → '7'
8  → '8'
9  → '9'
10 → ':'
11 → '.'
```

### data.yaml Configuration

Create `data/data.yaml`:

```yaml
# Dataset configuration
path: ./data
train: train/images
val: val/images

# 12 classes
nc: 12
names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '.']
```

### Labeling Tools

Recommended tools for creating labels:

1. **LabelImg** (Free)
   ```bash
   pip install labelImg
   labelImg
   ```

2. **Roboflow** (Web-based, Free tier)
   - Visit: https://roboflow.com
   - Upload images
   - Label and export in YOLO format

3. **CVAT** (Free, Open-source)
   - Visit: https://www.cvat.ai

### Dataset Size Recommendations

| Dataset Size | Training Epochs | Batch Size | Expected Time (GPU) |
|--------------|----------------|------------|---------------------|
| Small (<500) | 300-500        | 8          | 1-2 hours          |
| Medium (500-2000) | 200-300   | 16         | 2-4 hours          |
| Large (>2000) | 100-200       | 32         | 2-3 hours          |

---

## 🏋️ Training

### Basic Training

```bash
python main.py train --data data/data.yaml --epochs 300
```

### Training with Custom Parameters

```bash
python main.py train \
  --data data/data.yaml \
  --epochs 300 \
  --batch 16 \
  --device 0
```

### Training Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--data` | Path to data.yaml | data/data.yaml | `--data data/data.yaml` |
| `--epochs` | Number of epochs | 300 | `--epochs 500` |
| `--batch` | Batch size | 16 | `--batch 32` |
| `--device` | GPU device or cpu | auto | `--device 0` or `--device cpu` |
| `--resume` | Resume training | False | `--resume` |
| `--model` | Pretrained model | models/yolov8n.pt | `--model custom.pt` |

### Training for Different Scenarios

#### Small Dataset (< 500 images)
```bash
python main.py train \
  --epochs 500 \
  --batch 8 \
  --device cpu
```

#### Large Dataset (> 2000 images)
```bash
python main.py train \
  --epochs 200 \
  --batch 32 \
  --device 0
```

#### Resume Interrupted Training
```bash
python main.py train --resume
```

### Monitor Training

Training progress is saved to:
```
runs/train/ocr_yolov8/
├── weights/
│   ├── best.pt     ← Best model
│   └── last.pt     ← Last checkpoint
├── results.png     ← Training curves
├── confusion_matrix.png
└── val_batch0_pred.jpg
```

**View logs:**
```bash
# Real-time
tail -f logs/digital_ocr_*.log

# All logs
cat logs/digital_ocr_*.log
```

### Understanding Training Metrics

- **box_loss**: Bounding box localization error (lower is better)
- **cls_loss**: Classification error (lower is better)
- **dfl_loss**: Distribution focal loss (lower is better)
- **mAP50**: Mean Average Precision at IoU=0.5 (higher is better, >0.85 is good)
- **mAP50-95**: mAP at IoU 0.5-0.95 (higher is better, >0.6 is good)
- **Precision**: Correct detections / Total detections (>0.9 is good)
- **Recall**: Detected objects / Total objects (>0.8 is good)

### Training Complete

After training:
```
✓ TRAINING COMPLETED SUCCESSFULLY!
Model saved to: models/yolov8n-finetuned.pt
```

---

## 🔍 Detection

### Single Image Detection

```bash
python main.py detect --image clock.jpg
```

**Output:**
```
DETECTION RESULT
================================================================================
Image:            clock.jpg
Detected Number:  12:35
Confidence:       0.953
Number of Digits: 5
================================================================================
```

### Batch Detection

```bash
python main.py detect --image-dir test_images/ --output results.jsonl
```

**Output:**
```
PROCESSING SUMMARY
================================================================================
Total images:          10
Successful detections: 9
Failed detections:     1
Average confidence:    0.912
Average digits:        4.2
================================================================================
```

### Detection Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--image` | Single image path | - | `--image test.jpg` |
| `--image-dir` | Directory of images | - | `--image-dir images/` |
| `--output` | Output JSONL file | results.jsonl | `--output my_results.jsonl` |
| `--conf` | Confidence threshold | 0.25 | `--conf 0.5` |
| `--no-visualization` | Disable visual output | False | `--no-visualization` |
| `--use-pretrained` | Use pretrained (not finetuned) | False | `--use-pretrained` |

### Adjust Confidence Threshold

```bash
# More detections (may include false positives)
python main.py detect --image test.jpg --conf 0.15

# Fewer, more confident detections
python main.py detect --image test.jpg --conf 0.5
```

---

## 📤 Output Format

### results.jsonl

Each line is a JSON object:

```json
{
  "image_path": "clock.jpg",
  "detected_number": "12:35",
  "confidence": 0.953,
  "num_digits": 5,
  "detections": [
    {
      "digit": "1",
      "confidence": 0.960,
      "bbox": [45, 120, 78, 195],
      "x_center": 61.5,
      "y_center": 157.5,
      "class_id": 1
    },
    {
      "digit": "2",
      "confidence": 0.955,
      "bbox": [82, 120, 115, 195],
      "x_center": 98.5,
      "y_center": 157.5,
      "class_id": 2
    },
    {
      "digit": ":",
      "confidence": 0.940,
      "bbox": [119, 135, 135, 180],
      "x_center": 127.0,
      "y_center": 157.5,
      "class_id": 10
    },
    {
      "digit": "3",
      "confidence": 0.958,
      "bbox": [139, 120, 172, 195],
      "x_center": 155.5,
      "y_center": 157.5,
      "class_id": 3
    },
    {
      "digit": "5",
      "confidence": 0.952,
      "bbox": [176, 120, 209, 195],
      "x_center": 192.5,
      "y_center": 157.5,
      "class_id": 5
    }
  ],
  "timestamp": "2025-01-15T10:30:45.123456"
}
```

### Visualization Images

Saved to `results/vis_*.jpg`:
- Green boxes: Individual digit detections
- Labels: Digit + confidence score
- Top banner: Final result and average confidence

---

## ⚙️ Configuration

### Edit config.py

```python
# Model settings
config.model.conf_threshold = 0.25  # Confidence threshold
config.model.iou_threshold = 0.45   # IoU threshold
config.model.img_size = 640         # Input image size

# Training settings
config.training.epochs = 300
config.training.batch_size = 16
config.training.learning_rate = 0.01

# Validation settings
config.validation.allowed_chars = {'0','1','2','3','4','5','6','7','8','9',':','.'}
config.validation.ignore_chars = {'°', 'C', 'F'}
```

### Environment Variables

```bash
# Set GPU device
export OCR_DEVICE=0

# Set confidence threshold
export OCR_CONF_THRESHOLD=0.3

# Set base directory
export OCR_BASE_DIR=/path/to/project
```

---

## 🐛 Troubleshooting

### Issue 1: CUDA Not Available

**Error:** `ValueError: Invalid CUDA 'device=0' requested`

**Solution:**
```bash
# Train on CPU
python main.py train --device cpu

# Or install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Issue 2: Low Detection Accuracy

**Solutions:**

1. **Lower confidence threshold:**
   ```bash
   python main.py detect --image test.jpg --conf 0.15
   ```

2. **More training data:**
   - Add more varied examples
   - Balance class distribution

3. **Train longer:**
   ```bash
   python main.py train --epochs 500
   ```

### Issue 3: Out of Memory

**Solutions:**

1. **Reduce batch size:**
   ```bash
   python main.py train --batch 4
   ```

2. **Use smaller model:**
   - Already using YOLOv8n (smallest)

3. **Use CPU:**
   ```bash
   python main.py train --device cpu
   ```

### Issue 4: Import Errors

**Solution:**
```bash
# Reinstall dependencies
pip uninstall -y ultralytics opencv-python
pip install ultralytics opencv-python
```

### Issue 5: No Digits Detected

**Solutions:**

1. **Check image quality:**
   - Ensure good contrast
   - Sufficient resolution
   - Clear digits

2. **Lower confidence:**
   ```bash
   python main.py detect --image test.jpg --conf 0.1
   ```

3. **Retrain model:**
   - Add similar images to training set

### Check Logs

```bash
# View recent errors
tail -n 50 logs/digital_ocr_error_*.log

# View all activity
cat logs/digital_ocr_*.log
```

---

## 💡 Examples

### Example 1: Digital Clock (12:35)

```bash
python main.py detect --image examples/clock.jpg
```

**Output:**
```
Detected Number: 12:35
Confidence: 0.953
```

### Example 2: Thermometer (35.8°F)

```bash
python main.py detect --image examples/thermometer.jpg
```

**Output:**
```
Detected Number: 35.8
Confidence: 0.887
```
*(°F automatically removed)*

### Example 3: Digital Meter (123.45)

```bash
python main.py detect --image examples/meter.jpg
```

**Output:**
```
Detected Number: 123.45
Confidence: 0.921
```

### Example 4: Batch Processing

```bash
python main.py detect --image-dir examples/ --output batch_results.jsonl
```

**Output:**
```
PROCESSING SUMMARY
Total images:          3
Successful detections: 3
Average confidence:    0.920
```

---

## 📚 API Reference

### Using as Python Module

```python
from digit_detector import DigitDetector
from config import config

# Initialize detector
detector = DigitDetector(use_finetuned=True)

# Detect from single image
result = detector.detect_digits("test.jpg", conf_threshold=0.25)

print(f"Detected: {result['detected_number']}")
print(f"Confidence: {result['confidence']:.3f}")

# Process multiple images
results = detector.process_batch("test_images/", "output.jsonl")

# Access individual detections
for det in result['detections']:
    print(f"Digit: {det['digit']}, Confidence: {det['confidence']:.3f}")
```

### Training Programmatically

```python
from trainer import YOLOTrainer

# Initialize trainer
trainer = YOLOTrainer("data/data.yaml")

# Train model
results = trainer.train(
    epochs=300,
    batch_size=16,
    device=0
)

# Validate model
val_results = trainer.validate()

print(f"mAP50: {val_results.box.map50:.4f}")
```

---

## ⚡ Performance Tips

### For Faster Training

1. **Use GPU:**
   ```bash
   python main.py train --device 0
   ```

2. **Increase batch size:**
   ```bash
   python main.py train --batch 32
   ```

3. **Use smaller image size:**
   - Edit `config.py`: `img_size = 480`

4. **Fewer epochs for testing:**
   ```bash
   python main.py train --epochs 50
   ```

### For Better Accuracy

1. **More training data:**
   - At least 500 images
   - Balanced classes

2. **Data augmentation:**
   - Already enabled in training

3. **Train longer:**
   ```bash
   python main.py train --epochs 500
   ```

4. **Higher resolution:**
   - Edit `config.py`: `img_size = 1280`

### For Faster Detection

1. **Higher confidence threshold:**
   ```bash
   python main.py detect --image test.jpg --conf 0.5
   ```

2. **Disable visualization:**
   ```bash
   python main.py detect --image test.jpg --no-visualization
   ```

3. **Use GPU:**
   - Automatic if available

---

## 🤝 Contributing

Contributions welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .

# Lint
flake8 .
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [PyTorch](https://pytorch.org/) - Deep learning framework

---

## 📧 Support

For issues, questions, or suggestions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review [closed issues](https://github.com/your-repo/issues?q=is%3Aissue+is%3Aclosed)
3. Open a [new issue](https://github.com/your-repo/issues/new)

---

## 📈 Roadmap

- [ ] Web interface for easy upload and detection
- [ ] Support for more symbol classes (-, +, /, etc.)
- [ ] Real-time video detection
- [ ] Mobile app (iOS/Android)
- [ ] REST API for integration
- [ ] Docker containerization
- [ ] Pre-trained models for download



**Made with ❤️ for accurate digit detection**

**Star ⭐ this repo if you find it helpful!**
```
