"""
Image preprocessing operations for Digital Number OCR.
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

from exceptions import PreprocessingError
from logger import ocr_logger


@dataclass
class ROI:
    """Region of Interest data structure."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        """Get coordinates in xyxy format."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    @property
    def area(self) -> int:
        """Get ROI area."""
        return self.width * self.height


class PreprocessingOperations:
    """Image preprocessing operations."""
    
    @staticmethod
    def invert(image: np.ndarray) -> np.ndarray:
        """Invert image colors."""
        try:
            return cv2.bitwise_not(image)
        except Exception as e:
            raise PreprocessingError(f"Failed to invert image: {e}")
    
    @staticmethod
    def auto_invert(image: np.ndarray, threshold: int = 128) -> np.ndarray:
        """Automatically invert image if mean brightness is high."""
        try:
            mean_val = np.mean(image)
            if mean_val > threshold:
                ocr_logger.debug(f"Auto-inverting image (mean={mean_val:.1f})")
                return cv2.bitwise_not(image)
            return image
        except Exception as e:
            raise PreprocessingError(f"Failed in auto_invert: {e}")
    
    @staticmethod
    def clahe(image: np.ndarray, clip_limit: float = 2.0, 
              tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        try:
            clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            
            if len(image.shape) == 3 and image.shape[2] == 3:
                enhanced = image.copy()
                for i in range(3):
                    enhanced[:, :, i] = clahe_obj.apply(image[:, :, i])
                return enhanced
            elif len(image.shape) == 2:
                enhanced = clahe_obj.apply(image)
                return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            else:
                return image
        except Exception as e:
            raise PreprocessingError(f"Failed to apply CLAHE: {e}")
    
    @staticmethod
    def morphological_closing(image: np.ndarray, 
                            kernel_size: Tuple[int, int] = (3, 3)) -> np.ndarray:
        """Apply morphological closing operation."""
        try:
            kernel = np.ones(kernel_size, np.uint8)
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        except Exception as e:
            raise PreprocessingError(f"Failed to apply closing: {e}")
    
    @staticmethod
    def lcd_strong(image: np.ndarray) -> np.ndarray:
        """Strong preprocessing for LCD displays."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            
            thresh = cv2.adaptiveThreshold(
                bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            kernel = np.ones((2, 2), np.uint8)
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            return cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            raise PreprocessingError(f"Failed to apply LCD strong: {e}")
    
    @staticmethod
    def decimal_enhance(image: np.ndarray) -> np.ndarray:
        """Enhance decimal point detection."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(blurred, -1, kernel)
            
            return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            raise PreprocessingError(f"Failed to apply decimal enhance: {e}")
    
    @staticmethod
    def scale_image(image: np.ndarray, scale_factor: float = 1.5,
                   interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """Scale image by given factor."""
        try:
            if scale_factor == 1.0:
                return image
            return cv2.resize(image, None, fx=scale_factor, fy=scale_factor,
                            interpolation=interpolation)
        except Exception as e:
            raise PreprocessingError(f"Failed to scale image: {e}")
    
    @staticmethod
    def extract_horizontal_rois(image: np.ndarray, min_height: int = 20,
                               min_width: int = 30) -> List[ROI]:
        """Extract horizontal regions of interest automatically."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            rois = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w >= min_width and h >= min_height:
                    rois.append(ROI(x, y, w, h))
            
            rois.sort(key=lambda r: r.y)
            return rois
        except Exception as e:
            ocr_logger.warning(f"Failed to extract ROIs: {e}")
            return []
    
    @staticmethod
    def crop_roi(image: np.ndarray, roi: ROI, padding: int = 5) -> Optional[np.ndarray]:
        """Crop region of interest from image with padding."""
        try:
            h, w = image.shape[:2]
            x1 = max(0, roi.x - padding)
            y1 = max(0, roi.y - padding)
            x2 = min(w, roi.x + roi.width + padding)
            y2 = min(h, roi.y + roi.height + padding)
            
            return image[y1:y2, x1:x2]
        except Exception as e:
            ocr_logger.warning(f"Failed to crop ROI: {e}")
            return None