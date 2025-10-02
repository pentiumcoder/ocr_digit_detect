"""
Simple digit detection module for digital displays.
Detects individual digits and symbols, then combines them.
"""
import cv2
import numpy as np
import json
from typing import List, Dict, Tuple
from pathlib import Path
from ultralytics import YOLO

from config import config
from logger import ocr_logger
from exceptions import ModelLoadError, DetectionError


class DigitDetector:
    """
    Detect individual digits (0-9, :, .) from digital displays.
    """
    
    # Class names from your dataset
    CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '.']
    
    def __init__(self, model_path: str = None, use_finetuned: bool = True):
        """Initialize digit detector."""
        if model_path is None:
            model_path = config.get_model_path(finetuned=use_finetuned)
        
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model."""
        try:
            if not self.model_path.exists():
                raise ModelLoadError(f"Model not found at: {self.model_path}")
            
            ocr_logger.info(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(str(self.model_path))
            ocr_logger.info("✓ Model loaded successfully")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")
    
    def detect_digits(self, image_path: str, 
                     conf_threshold: float = None,
                     save_visualization: bool = True) -> Dict:
        """
        Detect digits from image and return combined result.
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold
            save_visualization: Save visualization with bounding boxes
            
        Returns:
            Dictionary with detected number and metadata
        """
        conf = conf_threshold or config.model.conf_threshold
        
        ocr_logger.info(f"Processing: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise DetectionError(f"Failed to load image: {image_path}")
        
        # Run detection
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=config.model.iou_threshold,
            device=config.model.device,
            imgsz=config.model.img_size,
            verbose=False
        )
        
        # Parse detections
        detections = []
        for r in results:
            if not hasattr(r, 'boxes') or len(r.boxes) == 0:
                continue
            
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            confidences = r.boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box
                digit = self.CLASS_NAMES[cls]
                
                detections.append({
                    'digit': digit,
                    'confidence': float(conf),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'x_center': (x1 + x2) / 2,
                    'y_center': (y1 + y2) / 2
                })
        
        if not detections:
            ocr_logger.warning("No digits detected")
            return {
                'image_path': image_path,
                'detected_number': '',
                'confidence': 0.0,
                'num_digits': 0,
                'detections': []
            }
        
        # Sort detections left to right
        detections.sort(key=lambda d: d['x_center'])
        
        # Combine digits into number
        detected_number = ''.join([d['digit'] for d in detections])
        avg_confidence = np.mean([d['confidence'] for d in detections])
        
        # Filter unwanted characters
        detected_number = self._clean_number(detected_number)
        
        result = {
            'image_path': image_path,
            'detected_number': detected_number,
            'confidence': float(avg_confidence),
            'num_digits': len(detections),
            'detections': detections
        }
        
        ocr_logger.info(f"Detected: {detected_number} (confidence: {avg_confidence:.3f})")
        
        # Save visualization
        if save_visualization:
            self._save_visualization(image, detections, detected_number, image_path)
        
        return result
    
    def _clean_number(self, text: str) -> str:
        """Remove unwanted characters like °, C, F."""
        # Remove ignored characters
        for char in config.validation.ignore_chars:
            text = text.replace(char, '')
        
        # Keep only allowed characters
        cleaned = ''.join([c for c in text if c in config.validation.allowed_chars])
        
        return cleaned.strip()
    
    def _save_visualization(self, image: np.ndarray, detections: List[Dict], 
                          detected_number: str, image_path: str):
        """Save visualization with bounding boxes."""
        vis_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            digit = detection['digit']
            conf = detection['confidence']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{digit} {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Background for label
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            
            # Label text
            cv2.putText(vis_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add detected number at top
        result_text = f"Result: {detected_number}"
        cv2.putText(vis_image, result_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Save
        output_dir = config.paths.base_dir / config.paths.results_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        vis_filename = f"vis_{Path(image_path).name}"
        vis_path = output_dir / vis_filename
        cv2.imwrite(str(vis_path), vis_image)
        
        ocr_logger.info(f"Saved visualization: {vis_path}")
    
    def process_batch(self, image_dir: str, output_file: str = "results.jsonl") -> List[Dict]:
        """
        Process multiple images and save to JSONL.
        
        Args:
            image_dir: Directory containing images
            output_file: Output JSONL filename
            
        Returns:
            List of all results
        """
        image_dir = Path(image_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        image_files = [
            f for f in image_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            ocr_logger.error(f"No images found in: {image_dir}")
            return []
        
        ocr_logger.info(f"Processing {len(image_files)} images...")
        
        all_results = []
        for image_file in image_files:
            try:
                result = self.detect_digits(str(image_file))
                all_results.append(result)
            except Exception as e:
                ocr_logger.error(f"Failed to process {image_file}: {e}")
        
        # Save to JSONL
        output_path = config.get_output_path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in all_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        ocr_logger.info(f"Results saved to: {output_path}")
        
        # Print summary
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: List[Dict]):
        """Print processing summary."""
        total = len(results)
        successful = sum(1 for r in results if r['detected_number'])
        
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total images: {total}")
        print(f"Successful detections: {successful}")
        print(f"Failed detections: {total - successful}")
        
        if successful > 0:
            avg_conf = np.mean([r['confidence'] for r in results if r['detected_number']])
            print(f"Average confidence: {avg_conf:.3f}")
        
        print("\nDetected Numbers:")
        print("-" * 60)
        for r in results:
            if r['detected_number']:
                filename = Path(r['image_path']).name
                print(f"  {filename:30s} -> {r['detected_number']:15s} ({r['confidence']:.3f})")
        print("=" * 60 + "\n")