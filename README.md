<<<<<<< HEAD
# ocr_digit_detect
=======
# Digital Number OCR - Production System

A production-ready OCR system for detecting and recognizing digital numbers.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download pretrained model:
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt
```

3. Prepare your dataset in YOLO format

4. Train the model:
```bash
python main.py train --data data/data.yaml --epochs 300
```

5. Run inference:
```bash
python main.py inference --image test.jpg
```

## Project Structure

```
digital-number-ocr/
├── config.py              # Configuration
├── logger.py              # Logging
├── exceptions.py          # Custom exceptions
├── detector.py            # YOLO detector
├── recognizer.py          # OCR recognizer
├── pipeline.py            # Main pipeline
├── trainer.py             # Training module
├── main.py                # CLI
├── preprocess/            # Preprocessing modules
├── data/                  # Dataset
├── models/                # Model weights
├── logs/                  # Log files
└── results/               # Results
```

## Training

See TRAINING_GUIDE.md for detailed instructions.

## Testing

```bash
python main.py inference --image test.jpg --output results/
```
>>>>>>> 458328e (Initial commit: OCR digit detection project)
