"""
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
        print("\n" + "=" * 50)
        print("Processing Summary")
        print("=" * 50)
        for key, value in self.summary.items():
            print(f"{key:25s}: {value}")
        
        if self.summary["total_attempts"] > 0:
            success_rate = (self.summary["success"] / self.summary["total_attempts"]) * 100
            print(f"{'Success Rate':25s}: {success_rate:.2f}%")
        print("=" * 50 + "\n")


ocr_logger = OCRLogger()
