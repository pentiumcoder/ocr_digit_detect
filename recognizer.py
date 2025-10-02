"""
PaddleOCR-based text recognition module.
"""
import numpy as np
import re
from typing import List, Dict, Optional
from paddleocr import PaddleOCR

from config import config
from logger import ocr_logger
from exceptions import RecognitionError


class TextRecognizer:
    """PaddleOCR-based text recognizer with normalization."""
    
    def __init__(self):
        """Initialize PaddleOCR recognizer."""
        try:
            ocr_logger.info("Initializing PaddleOCR...")
            self.ocr = PaddleOCR(
                use_angle_cls=config.ocr.use_angle_cls,
                lang=config.ocr.lang,
                use_gpu=config.ocr.use_gpu,
                show_log=False
            )
            ocr_logger.info("✓ PaddleOCR initialized successfully")
        except Exception as e:
            raise RecognitionError(f"Failed to initialize PaddleOCR: {e}")
    
    def recognize(self, image: np.ndarray, return_confidence: bool = True) -> List[Dict]:
        """Recognize text in image."""
        if image is None or image.size == 0:
            raise RecognitionError("Invalid input image")
        
        try:
            min_dim = min(image.shape[:2])
            if min_dim < config.preprocess.roi_min_size:
                scale_factor = config.preprocess.roi_scale_factor
                import cv2
                image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor,
                                 interpolation=cv2.INTER_CUBIC)
                ocr_logger.debug(f"Upscaled image by {scale_factor}x")
            
            result = self.ocr.ocr(image, cls=True)
            
            if not result or not result[0]:
                return []
            
            texts = []
            for line in result[0]:
                try:
                    text = line[1][0]
                    conf = line[1][1] if len(line[1]) > 1 else None
                    
                    if return_confidence and conf is not None:
                        texts.append({'text': text, 'conf': float(conf)})
                    else:
                        texts.append({'text': text})
                except (IndexError, TypeError):
                    continue
            
            return texts
            
        except Exception as e:
            ocr_logger.debug(f"OCR recognition failed: {e}")
            return []
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize OCR output text."""
        if not text:
            return ""
        
        if text.upper() in config.validation.non_numeric_keywords:
            return text
        
        cleaned = text.strip()
        cleaned = re.sub(r'^[.\-\s]+', '', cleaned)
        cleaned = re.sub(r'[.\-\s]+$', '', cleaned)
        cleaned = re.sub(r'(\d)\s+(\d)', r'\1\2', cleaned)
        
        if re.search(r'\d', cleaned):
            cleaned = cleaned.replace('O', '0')
            cleaned = cleaned.replace('o', '0')
            cleaned = cleaned.replace('I', '1')
            cleaned = cleaned.replace('l', '1')
            cleaned = cleaned.replace('|', '1')
        
        return cleaned
    
    @staticmethod
    def validate_numeric(text: str) -> bool:
        """Validate if text is valid numeric content."""
        if not text or len(text.strip()) == 0:
            return False
        
        if text.upper() in config.validation.non_numeric_keywords:
            return False
        
        for pattern in config.validation.exclude_patterns:
            if re.match(pattern, text):
                return False
        
        digit_count = sum(1 for c in text if c.isdigit())
        if digit_count < config.validation.min_digit_count:
            return False
        
        if not re.match(config.validation.numeric_pattern, text):
            return False
        
        return True
    
    def recognize_and_filter(self, image: np.ndarray, normalize: bool = True, 
                           validate: bool = True) -> List[Dict]:
        """Recognize text with normalization and validation."""
        results = self.recognize(image)
        
        if not results:
            return []
        
        filtered_results = []
        for result in results:
            text = result['text']
            
            if normalize:
                text = self.normalize_text(text)
                result['normalized_text'] = text
            
            if validate:
                if self.validate_numeric(text):
                    filtered_results.append(result)
            else:
                filtered_results.append(result)
        
        return filtered_results