"""
Complete Project Setup Script
Run this script to generate all production-ready OCR files.

Usage: python setup_project.py
"""

import os
from pathlib import Path

FILES = {
    "config.py": '''"""
Configuration management for Digital Number OCR system.
"""
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """YOLO model configuration."""
    pretrained_path: str = "models/yolov8n.pt"
    finetuned_path: str = "models/yolov8n-finetuned.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    img_size: int = 640
    device: int = 0


@dataclass
class OCRConfig:
    """PaddleOCR configuration."""
    use_angle_cls: bool = True
    lang: str = 'en'
    use_gpu: bool = False
    use_textline_orientation: bool = True
    min_confidence: float = 0.5


@dataclass
class PreprocessConfig:
    """Preprocessing configuration."""
    stages: list = field(default_factory=lambda: ['s0', 's1', 's2', 's3'])
    scale_factors: list = field(default_factory=lambda: [1.0, 1.5, 2.0])
    roi_min_size: int = 100
    roi_scale_factor: float = 2.0
    clahe_clip_limit: float = 2.0
    clahe_tile_size: tuple = (8, 8)
    morph_kernel_size: tuple = (3, 3)


@dataclass
class ValidationConfig:
    """Text validation configuration."""
    min_digit_count: int = 2
    numeric_pattern: str = r"^(?!.*[IO]/[IO])(?![IO]+$)(?![A-Z]+$)[0-9OIl:.,+\\-_/\\\\()\\s°C°F%℃]+$"
    exclude_patterns: list = field(default_factory=lambda: [
        r'^\\.', r'^0:\\d{2}$', r'^\\d{1,2}\\.\\d{3,}$',
        r'^\\d+\\.\\s+\\d+$', r'^0{3,}$', r'^\\d{1}[°℃°F%]$',
        r'^[°℃°FC%]+$', r'^\\([IO]/[IO]\\)$'
    ])
    non_numeric_keywords: list = field(default_factory=lambda: [
        'I/O', 'O/I', 'ON', 'OFF', 'IO', 'OI'
    ])


@dataclass
class PathConfig:
    """Path configuration."""
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    data_yaml: str = "data/data.yaml"
    output_dir: str = "runs/train/ocr_yolov8"
    logs_dir: str = "logs"
    results_dir: str = "results"


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    epochs: int = 300
    learning_rate: float = 0.001
    weight_decay: float = 5e-4
    momentum: float = 0.937
    optimizer: str = "AdamW"
    cos_lr: bool = True
    amp: bool = True


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.ocr = OCRConfig()
        self.preprocess = PreprocessConfig()
        self.validation = ValidationConfig()
        self.paths = PathConfig()
        self.training = TrainingConfig()
        self._load_from_env()
        self._create_directories()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        if env_device := os.getenv('OCR_DEVICE'):
            self.model.device = int(env_device)
        if env_conf := os.getenv('OCR_CONF_THRESHOLD'):
            self.model.conf_threshold = float(env_conf)
        if env_gpu := os.getenv('OCR_USE_GPU'):
            self.ocr.use_gpu = env_gpu.lower() == 'true'
        if env_base := os.getenv('OCR_BASE_DIR'):
            self.paths.base_dir = Path(env_base)
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        dirs = [
            self.paths.base_dir / self.paths.logs_dir,
            self.paths.base_dir / self.paths.results_dir,
            self.paths.base_dir / self.paths.output_dir,
            self.paths.base_dir / "models"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, finetuned: bool = True) -> Path:
        """Get model path."""
        path = self.model.finetuned_path if finetuned else self.model.pretrained_path
        return self.paths.base_dir / path
    
    def get_output_path(self, filename: str) -> Path:
        """Get output file path."""
        return self.paths.base_dir / self.paths.results_dir / filename
    
    def get_log_path(self, filename: str) -> Path:
        """Get log file path."""
        return self.paths.base_dir / self.paths.logs_dir / filename


config = Config()
''',

    "exceptions.py": '''"""
Custom exceptions for Digital Number OCR system.
"""


class OCRException(Exception):
    """Base exception for OCR system."""
    pass


class ModelLoadError(OCRException):
    """Exception raised when model fails to load."""
    pass


class ImageLoadError(OCRException):
    """Exception raised when image fails to load."""
    pass


class DetectionError(OCRException):
    """Exception raised when detection fails."""
    pass


class RecognitionError(OCRException):
    """Exception raised when OCR recognition fails."""
    pass


class PreprocessingError(OCRException):
    """Exception raised during preprocessing."""
    pass


class ValidationError(OCRException):
    """Exception raised during result validation."""
    pass


class ConfigurationError(OCRException):
    """Exception raised for configuration issues."""
    pass


class InsufficientDataError(OCRException):
    """Exception raised when insufficient data is available."""
    pass
''',

    "logger.py": '''"""
Production-grade logging system for Digital Number OCR.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional
import json


class OCRLogger:
    """Centralized logger for OCR system."""
    
    def __init__(self, name: str = "digital_ocr", log_dir: Optional[Path] = None):
        self.name = name
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        log_file = self.log_dir / f"{self.name}_{datetime.now():%Y%m%d}.log"
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        error_file = self.log_dir / f"{self.name}_error_{datetime.now():%Y%m%d}.log"
        error_handler = RotatingFileHandler(
            error_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
    
    def debug(self, msg: str, **kwargs):
        self.logger.debug(msg, extra=kwargs)
    
    def info(self, msg: str, **kwargs):
        self.logger.info(msg, extra=kwargs)
    
    def warning(self, msg: str, **kwargs):
        self.logger.warning(msg, extra=kwargs)
    
    def error(self, msg: str, exc_info: bool = False, **kwargs):
        self.logger.error(msg, exc_info=exc_info, extra=kwargs)
    
    def critical(self, msg: str, exc_info: bool = True, **kwargs):
        self.logger.critical(msg, exc_info=exc_info, extra=kwargs)


class ProcessingLogger:
    """Logger for tracking preprocessing attempts."""
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.attempts = []
        self.summary = {
            "success": 0,
            "detection_failed": 0,
            "no_numeric_content": 0,
            "regex_too_strict": 0,
            "total_attempts": 0
        }
    
    def log_attempt(self, stage: str, result: dict):
        """Log a preprocessing attempt."""
        attempt = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "result": result
        }
        self.attempts.append(attempt)
        self.summary["total_attempts"] += 1
        
        if result.get("success"):
            self.summary["success"] += 1
        elif not result.get("bbox"):
            self.summary["detection_failed"] += 1
        elif not result.get("has_numeric"):
            self.summary["no_numeric_content"] += 1
        else:
            self.summary["regex_too_strict"] += 1
    
    def save_log(self, filename: str):
        """Save processing log to file."""
        log_path = self.log_dir / filename
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": self.summary,
                "attempts": self.attempts
            }, f, indent=2, ensure_ascii=False)
    
    def print_summary(self):
        """Print processing summary."""
        print("\\n" + "=" * 50)
        print("Processing Summary")
        print("=" * 50)
        for key, value in self.summary.items():
            print(f"{key:25s}: {value}")
        
        if self.summary["total_attempts"] > 0:
            success_rate = (self.summary["success"] / self.summary["total_attempts"]) * 100
            print(f"{'Success Rate':25s}: {success_rate:.2f}%")
        print("=" * 50 + "\\n")


ocr_logger = OCRLogger()
''',

    "requirements.txt": '''# Core dependencies
numpy>=1.21.0,<2.0.0
opencv-python>=4.6.0
Pillow>=9.0.0

# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0

# OCR
paddleocr>=2.6.0
paddlepaddle>=2.4.0

# Development
pytest>=7.0.0
pytest-cov>=4.0.0

# Utilities
tqdm>=4.64.0
pyyaml>=6.0
''',

    "README.md": '''# Digital Number OCR - Production System

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
'''
}


def create_file(filepath, content):
    """Create a file with content."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Created: {filepath}")


def main():
    print("=" * 60)
    print("Digital Number OCR - Project Setup")
    print("=" * 60)
    print()
    
    # Create directory structure
    dirs = [
        "preprocess", "tests", "models", "data", 
        "logs", "results", "runs", "data/train/images",
        "data/train/labels", "data/val/images", "data/val/labels"
    ]
    
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Create files
    for filename, content in FILES.items():
        create_file(filename, content)
    
    # Create __init__ files
    create_file("preprocess/__init__.py", "")
    create_file("tests/__init__.py", "")
    create_file("__init__.py", "")
    
    # Create .gitkeep files
    for d in ["models", "data", "logs", "results"]:
        create_file(f"{d}/.gitkeep", "")
    
    print()
    print("=" * 60)
    print("✓ PROJECT SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download model: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt")
    print("3. Prepare your dataset in data/ folder")
    print("4. Create data/data.yaml configuration file")
    print("5. Run training: python main.py train --data data/data.yaml")
    print()
    print("NOTE: You still need to copy the remaining modules:")
    print("- detector.py, recognizer.py, pipeline.py")
    print("- trainer.py, main.py")
    print("- preprocess/operations.py, preprocess/engine.py")
    print()
    print("Download the complete ZIP from the next message!")
    print("=" * 60)


if __name__ == "__main__":
    main()
