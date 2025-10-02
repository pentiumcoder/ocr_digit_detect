"""
Configuration management for Digital Number OCR system.
"""
import os
import torch
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
    device: int = 0 if torch.cuda.is_available() else -1  # Auto-detect GPU/CPU
    
    # 12 classes: digits 0-9, colon, period
    nc: int = 12
    class_names: list = field(default_factory=lambda: [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '.'
    ])


@dataclass
class ValidationConfig:
    """Text validation configuration."""
    min_digit_count: int = 1
    
    # Allowed characters (our 12 classes)
    allowed_chars: set = field(default_factory=lambda: {
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '.'
    })
    
    # Characters to ignore/remove (temperature symbols, etc.)
    ignore_chars: set = field(default_factory=lambda: {
        '°', 'C', 'F', 'c', 'f', 'degree', '℃', '℉', ' '
    })
    
    numeric_only: bool = True


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
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    momentum: float = 0.937
    optimizer: str = "AdamW"
    cos_lr: bool = True
    amp: bool = True


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.validation = ValidationConfig()
        self.paths = PathConfig()
        self.training = TrainingConfig()
        self._load_from_env()
        self._create_directories()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        if env_device := os.getenv('OCR_DEVICE'):
            if env_device.lower() == 'cpu':
                self.model.device = -1
            else:
                self.model.device = int(env_device)
        
        if env_conf := os.getenv('OCR_CONF_THRESHOLD'):
            self.model.conf_threshold = float(env_conf)
        
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