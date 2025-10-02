"""
Main OCR pipeline orchestrating detection, recognition, and preprocessing.
"""
import cv2
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

from digit_detector import YOLODetector
from recognizer import TextRecognizer
from preprocess.engine import PreprocessingEngine
from config import config
from logger import ocr_logger
from exceptions import ImageLoadError, OCRException


@dataclass
class OCRResult:
    """OCR result data structure."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    stage: str
    normalized_text: Optional[str] = None
    processing_time: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class DigitalOCRPipeline:
    """Complete OCR pipeline for digital number recognition."""
    
    def __init__(self, use_finetuned_model: bool = True):
        """Initialize OCR pipeline."""
        ocr_logger.info("Initializing Digital OCR Pipeline...")
        
        try:
            self.digit_detector = YOLODetector(use_finetuned=use_finetuned_model)
            self.recognizer = TextRecognizer()
            self.preprocess_engine = PreprocessingEngine()
            
            ocr_logger.info("✓ Pipeline initialized successfully")
        except Exception as e:
            ocr_logger.critical(f"Failed to initialize pipeline: {e}")
            raise
    
    def process_image(self, image_path: str,
                     early_terminate: bool = True,
                     save_visualization: bool = True,
                     output_dir: Optional[Path] = None) -> List[OCRResult]:
        """Process image through complete OCR pipeline."""
        start_time = time.time()
        ocr_logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = self._load_image(image_path)
        
        # Setup output directory
        if output_dir is None:
            output_dir = config.paths.base_dir / config.paths.results_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect regions
        ocr_logger.info("Step 1: Detecting text regions...")
        boxes = self.detector.detect(image)
        
        if not boxes:
            ocr_logger.warning("No text regions detected")
            return []
        
        ocr_logger.info(f"Detected {len(boxes)} regions")
        
        # Process each detected region
        all_results = []
        
        for idx, box in enumerate(boxes):
            ocr_logger.info(f"Processing region {idx + 1}/{len(boxes)}")
            
            # Crop region
            x1, y1, x2, y2 = box
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
            
            # Multi-stage preprocessing and recognition
            def ocr_callback(processed_image):
                return self.recognizer.recognize_and_filter(
                    processed_image,
                    normalize=True,
                    validate=True
                )
            
            region_results, success_stage = self.preprocess_engine.process_image(
                roi,
                ocr_callback,
                early_terminate=early_terminate
            )
            
            # Convert to OCRResult objects
            for result in region_results:
                ocr_result = OCRResult(
                    text=result.get('normalized_text', result['text']),
                    confidence=result.get('conf', 0.0),
                    bbox=box,
                    stage=success_stage,
                    normalized_text=result.get('normalized_text'),
                    processing_time=time.time() - start_time
                )
                all_results.append(ocr_result)
        
        total_time = time.time() - start_time
        ocr_logger.info(f"Processing complete: {len(all_results)} results in {total_time:.2f}s")
        
        # Save visualization
        if save_visualization and all_results:
            self._save_visualization(image, all_results, image_path, output_dir)
        
        # Save results
        self._save_results(all_results, image_path, output_dir)
        
        return all_results
    
    def process_batch(self, image_paths: List[str],
                     output_dir: Optional[Path] = None) -> Dict[str, List[OCRResult]]:
        """Process multiple images."""
        ocr_logger.info(f"Processing batch of {len(image_paths)} images")
        
        results = {}
        for path in image_paths:
            try:
                results[path] = self.process_image(path, output_dir=output_dir)
            except Exception as e:
                ocr_logger.error(f"Failed to process {path}: {e}")
                results[path] = []
        
        return results
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and validate image."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ImageLoadError(f"Failed to load image: {image_path}")
            
            ocr_logger.debug(f"Loaded image: {image.shape}")
            return image
        except Exception as e:
            raise ImageLoadError(f"Error loading image: {e}")
    
    def _save_visualization(self, image: np.ndarray,
                          results: List[OCRResult],
                          image_path: str,
                          output_dir: Path):
        """Save visualization with bounding boxes and text."""
        try:
            vis_image = image.copy()
            
            for result in results:
                x1, y1, x2, y2 = result.bbox
                
                # Draw bounding box (green for valid numeric)
                color = (0, 255, 0)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # Add text label
                label = f"{result.text} ({result.confidence:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                # Draw label background
                cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(vis_image, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Save visualization
            vis_filename = f"vis_{Path(image_path).name}"
            vis_path = output_dir / vis_filename
            cv2.imwrite(str(vis_path), vis_image)
            ocr_logger.info(f"Saved visualization: {vis_path}")
            
        except Exception as e:
            ocr_logger.warning(f"Failed to save visualization: {e}")
    
    def _save_results(self, results: List[OCRResult],
                     image_path: str,
                     output_dir: Path):
        """Save results to JSON file."""
        try:
            import json
            
            result_filename = f"results_{Path(image_path).stem}.json"
            result_path = output_dir / result_filename
            
            # Convert results to dictionaries
            results_dict = {
                "image_path": image_path,
                "num_results": len(results),
                "results": [r.to_dict() for r in results]
            }
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
            
            ocr_logger.info(f"Saved results: {result_path}")
            
        except Exception as e:
            ocr_logger.warning(f"Failed to save results: {e}")
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics."""
        return self.preprocess_engine.get_processing_summary()
    
    def print_summary(self):
        """Print processing summary."""
        self.preprocess_engine.print_summary()