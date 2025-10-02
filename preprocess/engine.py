"""
Main preprocessing engine for Digital Number OCR.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple

from preprocess.operations import PreprocessingOperations
from config import config
from logger import ocr_logger, ProcessingLogger
from exceptions import PreprocessingError


class PreprocessingEngine:
    """Multi-stage preprocessing engine with early termination."""
    
    def __init__(self):
        self.ops = PreprocessingOperations()
        self.config = config.preprocess
        self.proc_logger = ProcessingLogger()
        
        self.s1_presets = [
            ('as-is', []),
            ('invert', [self.ops.auto_invert]),
            ('clahe', [self.ops.clahe]),
            ('lcd_strong', [self.ops.lcd_strong]),
            ('decimal_enhance', [self.ops.decimal_enhance]),
            ('invert+clahe', [self.ops.auto_invert, self.ops.clahe])
        ]
        
        self.s2_presets = [
            (1.5, 'closing', [self.ops.morphological_closing]),
            (2.0, 'closing', [self.ops.morphological_closing]),
            (1.5, 'as-is', []),
            (2.0, 'as-is', [])
        ]
    
    def process_image(self, image: np.ndarray, ocr_callback, 
                     early_terminate: bool = True) -> Tuple[List[Dict], str]:
        """Execute multi-stage preprocessing pipeline."""
        if image is None or image.size == 0:
            raise PreprocessingError("Invalid input image")
        
        ocr_logger.info("Starting multi-stage preprocessing")
        
        # Stage 0
        results, success = self._try_stage_0(image, ocr_callback)
        if success and early_terminate:
            ocr_logger.info("✓ Success at Stage 0")
            return results, "S0"
        
        # Stage 1
        results, success = self._try_stage_1(image, ocr_callback)
        if success and early_terminate:
            ocr_logger.info("✓ Success at Stage 1")
            return results, "S1"
        
        # Stage 2
        results, success = self._try_stage_2(image, ocr_callback)
        if success and early_terminate:
            ocr_logger.info("✓ Success at Stage 2")
            return results, "S2"
        
        # Stage 3
        results, success = self._try_stage_3(image, ocr_callback)
        if success:
            ocr_logger.info("✓ Success at Stage 3")
            return results, "S3"
        
        ocr_logger.warning("All preprocessing stages failed")
        return [], "FAILED"
    
    def _try_stage_0(self, image: np.ndarray, ocr_callback) -> Tuple[List[Dict], bool]:
        """Stage 0: Pass-through."""
        ocr_logger.debug("Trying Stage 0: Pass-through")
        results = self._try_ocr(image, ocr_callback, stage="S0")
        return results, len(results) > 0
    
    def _try_stage_1(self, image: np.ndarray, ocr_callback) -> Tuple[List[Dict], bool]:
        """Stage 1: Basic presets."""
        ocr_logger.debug("Trying Stage 1: Basic presets")
        
        for preset_name, operations in self.s1_presets:
            try:
                processed = image.copy()
                for op in operations:
                    processed = op(processed)
                
                results = self._try_ocr(processed, ocr_callback, stage=f"S1-{preset_name}")
                if len(results) > 0:
                    return results, True
            except Exception as e:
                ocr_logger.debug(f"S1 preset '{preset_name}' failed: {e}")
                continue
        
        return [], False
    
    def _try_stage_2(self, image: np.ndarray, ocr_callback) -> Tuple[List[Dict], bool]:
        """Stage 2: Scaling + presets."""
        ocr_logger.debug("Trying Stage 2: Scaling + presets")
        
        for scale, preset_name, operations in self.s2_presets:
            try:
                scaled = self.ops.scale_image(image, scale)
                
                processed = scaled
                for op in operations:
                    processed = op(processed)
                
                results = self._try_ocr(processed, ocr_callback, stage=f"S2-{scale}x-{preset_name}")
                if len(results) > 0:
                    return results, True
            except Exception as e:
                ocr_logger.debug(f"S2 preset '{scale}x-{preset_name}' failed: {e}")
                continue
        
        return [], False
    
    def _try_stage_3(self, image: np.ndarray, ocr_callback) -> Tuple[List[Dict], bool]:
        """Stage 3: ROI extraction."""
        ocr_logger.debug("Trying Stage 3: ROI extraction")
        
        try:
            rois = self.ops.extract_horizontal_rois(image)
            if not rois:
                return [], False
            
            ocr_logger.debug(f"Extracted {len(rois)} ROIs")
            all_results = []
            
            for idx, roi in enumerate(rois):
                roi_img = self.ops.crop_roi(image, roi)
                if roi_img is None:
                    continue
                
                for preset_name, operations in self.s1_presets[:3]:
                    try:
                        processed = roi_img.copy()
                        for op in operations:
                            processed = op(processed)
                        
                        results = self._try_ocr(processed, ocr_callback, stage=f"S3-ROI{idx}-{preset_name}")
                        if results:
                            for r in results:
                                r['roi'] = roi.xyxy
                            all_results.extend(results)
                            break
                    except:
                        continue
            
            return all_results, len(all_results) > 0
        except Exception as e:
            ocr_logger.error(f"Stage 3 failed: {e}")
            return [], False
    
    def _try_ocr(self, image: np.ndarray, ocr_callback, stage: str) -> List[Dict]:
        """Attempt OCR on processed image."""
        try:
            raw_results = ocr_callback(image)
            
            if not raw_results:
                self.proc_logger.log_attempt(stage, {"success": False, "bbox": False, "has_numeric": False})
                return []
            
            valid_results = []
            has_numeric = False
            
            for result in raw_results:
                text = result.get('text', '')
                if any(c.isdigit() for c in text):
                    has_numeric = True
                    if self._is_valid_numeric(text):
                        result['stage'] = stage
                        valid_results.append(result)
            
            self.proc_logger.log_attempt(stage, {
                "success": len(valid_results) > 0,
                "bbox": True,
                "has_numeric": has_numeric
            })
            
            return valid_results
            
        except Exception as e:
            self.proc_logger.log_attempt(stage, {"success": False, "bbox": False, "has_numeric": False})
            return []
    
    def _is_valid_numeric(self, text: str) -> bool:
        """Check if text is valid numeric content."""
        import re
        
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
    
    def get_processing_summary(self):
        """Get processing summary."""
        return self.proc_logger.summary
    
    def print_summary(self):
        """Print processing summary."""
        self.proc_logger.print_summary()