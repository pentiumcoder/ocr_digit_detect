"""
YOLO model training module for custom digital number detection.
Includes comprehensive hyperparameter configuration.
"""
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from ultralytics import YOLO

from config import config
from logger import ocr_logger
from exceptions import ModelLoadError, ConfigurationError


class YOLOTrainer:
    """
    Trainer for YOLO model on custom digital number dataset.
    
    Supports full hyperparameter customization for fine-tuning.
    """
    
    def __init__(self, data_yaml: Optional[str] = None):
        """
        Initialize YOLO trainer.
        
        Args:
            data_yaml: Path to dataset YAML configuration
        """
        self.data_yaml = data_yaml or config.paths.data_yaml
        self.training_config = config.training
        self.model_config = config.model
        
        # Validate data YAML exists
        if not Path(self.data_yaml).exists():
            raise ConfigurationError(f"Data YAML not found: {self.data_yaml}")
        
        ocr_logger.info(f"Trainer initialized with data: {self.data_yaml}")
    
    def train(self, 
              pretrained_model: Optional[str] = None,
              epochs: Optional[int] = None,
              batch_size: Optional[int] = None,
              device: Optional[int] = None,
              resume: bool = False,
              save_dir: Optional[str] = None,
              # Image settings
              imgsz: Optional[int] = None,
              # Optimization hyperparameters
              lr0: Optional[float] = None,
              lrf: Optional[float] = None,
              momentum: Optional[float] = None,
              weight_decay: Optional[float] = None,
              warmup_epochs: Optional[float] = None,
              warmup_momentum: Optional[float] = None,
              warmup_bias_lr: Optional[float] = None,
              # Augmentation hyperparameters
              hsv_h: Optional[float] = None,
              hsv_s: Optional[float] = None,
              hsv_v: Optional[float] = None,
              degrees: Optional[float] = None,
              translate: Optional[float] = None,
              scale: Optional[float] = None,
              shear: Optional[float] = None,
              perspective: Optional[float] = None,
              flipud: Optional[float] = None,
              fliplr: Optional[float] = None,
              mosaic: Optional[float] = None,
              mixup: Optional[float] = None,
              copy_paste: Optional[float] = None,
              # Loss hyperparameters
              box: Optional[float] = None,
              cls: Optional[float] = None,
              dfl: Optional[float] = None,
              # Advanced settings
              optimizer: Optional[str] = None,
              cos_lr: Optional[bool] = None,
              close_mosaic: Optional[int] = None,
              amp: Optional[bool] = None,
              fraction: Optional[float] = None,
              profile: bool = False,
              patience: Optional[int] = None,
              # Custom kwargs
              **kwargs) -> Dict:
        """
        Train YOLO model with comprehensive hyperparameter support.
        
        Args:
            pretrained_model: Path to pretrained model weights
            epochs: Number of training epochs (default: 300)
            batch_size: Batch size (default: 16)
            device: GPU device ID or 'cpu' (default: 0)
            resume: Resume from last checkpoint
            save_dir: Directory to save training outputs
            
            # Image settings
            imgsz: Input image size (default: 640)
            
            # Optimization hyperparameters
            lr0: Initial learning rate (default: 0.01)
            lrf: Final learning rate factor (default: 0.01)
            momentum: SGD momentum/Adam beta1 (default: 0.937)
            weight_decay: Optimizer weight decay (default: 0.0005)
            warmup_epochs: Warmup epochs (default: 3.0)
            warmup_momentum: Warmup initial momentum (default: 0.8)
            warmup_bias_lr: Warmup initial bias learning rate (default: 0.1)
            
            # Augmentation hyperparameters
            hsv_h: HSV-Hue augmentation (default: 0.015)
            hsv_s: HSV-Saturation augmentation (default: 0.7)
            hsv_v: HSV-Value augmentation (default: 0.4)
            degrees: Rotation degrees (default: 0.0)
            translate: Translation (default: 0.1)
            scale: Scaling (default: 0.5)
            shear: Shear (default: 0.0)
            perspective: Perspective (default: 0.0)
            flipud: Vertical flip probability (default: 0.0)
            fliplr: Horizontal flip probability (default: 0.5)
            mosaic: Mosaic augmentation probability (default: 1.0)
            mixup: Mixup augmentation probability (default: 0.0)
            copy_paste: Copy-paste augmentation probability (default: 0.0)
            
            # Loss hyperparameters
            box: Box loss gain (default: 7.5)
            cls: Class loss gain (default: 0.5)
            dfl: DFL loss gain (default: 1.5)
            
            # Advanced settings
            optimizer: Optimizer choice ('SGD', 'Adam', 'AdamW', 'RMSProp')
            cos_lr: Use cosine learning rate scheduler
            close_mosaic: Disable mosaic last N epochs (default: 10)
            amp: Automatic Mixed Precision training
            fraction: Dataset fraction to train on (default: 1.0)
            profile: Profile ONNX and TensorRT speeds
            patience: Early stopping patience epochs (default: 50)
            
        Returns:
            Training results dictionary
        """
        
        # Set default parameters
        model_path = pretrained_model or config.model.pretrained_path
        epochs = epochs or self.training_config.epochs
        batch = batch_size or self.training_config.batch_size
        device = device if device is not None else self.model_config.device
        save_dir = save_dir or config.paths.output_dir
        imgsz = imgsz or self.model_config.img_size
        
        # Optimization defaults
        lr0 = lr0 or self.training_config.learning_rate
        momentum = momentum or self.training_config.momentum
        weight_decay = weight_decay or self.training_config.weight_decay
        optimizer = optimizer or self.training_config.optimizer
        cos_lr = cos_lr if cos_lr is not None else self.training_config.cos_lr
        amp = amp if amp is not None else self.training_config.amp
        
        # Build training arguments
        train_args = {
            'data': self.data_yaml,
            'epochs': epochs,
            'batch': batch,
            'imgsz': imgsz,
            'device': device,
            'lr0': lr0,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'optimizer': optimizer,
            'cos_lr': cos_lr,
            'amp': amp,
            'project': save_dir,
            'name': 'train',
            'exist_ok': True,
            'resume': resume,
            'verbose': True,
            'plots': True,
            'save': True,
            'save_period': -1,  # Save checkpoint every N epochs (-1 = disabled)
        }
        
        # Add optional hyperparameters if specified
        if lrf is not None:
            train_args['lrf'] = lrf
        if warmup_epochs is not None:
            train_args['warmup_epochs'] = warmup_epochs
        if warmup_momentum is not None:
            train_args['warmup_momentum'] = warmup_momentum
        if warmup_bias_lr is not None:
            train_args['warmup_bias_lr'] = warmup_bias_lr
            
        # Augmentation parameters
        if hsv_h is not None:
            train_args['hsv_h'] = hsv_h
        if hsv_s is not None:
            train_args['hsv_s'] = hsv_s
        if hsv_v is not None:
            train_args['hsv_v'] = hsv_v
        if degrees is not None:
            train_args['degrees'] = degrees
        if translate is not None:
            train_args['translate'] = translate
        if scale is not None:
            train_args['scale'] = scale
        if shear is not None:
            train_args['shear'] = shear
        if perspective is not None:
            train_args['perspective'] = perspective
        if flipud is not None:
            train_args['flipud'] = flipud
        if fliplr is not None:
            train_args['fliplr'] = fliplr
        if mosaic is not None:
            train_args['mosaic'] = mosaic
        if mixup is not None:
            train_args['mixup'] = mixup
        if copy_paste is not None:
            train_args['copy_paste'] = copy_paste
            
        # Loss parameters
        if box is not None:
            train_args['box'] = box
        if cls is not None:
            train_args['cls'] = cls
        if dfl is not None:
            train_args['dfl'] = dfl
            
        # Advanced parameters
        if close_mosaic is not None:
            train_args['close_mosaic'] = close_mosaic
        if fraction is not None:
            train_args['fraction'] = fraction
        if profile:
            train_args['profile'] = profile
        if patience is not None:
            train_args['patience'] = patience
        
        # Add any additional kwargs
        train_args.update(kwargs)
        
        # Log training configuration
        ocr_logger.info("=" * 80)
        ocr_logger.info("YOLO TRAINING CONFIGURATION")
        ocr_logger.info("=" * 80)
        ocr_logger.info(f"Model: {model_path}")
        ocr_logger.info(f"Dataset: {self.data_yaml}")
        ocr_logger.info("-" * 80)
        ocr_logger.info("Training Parameters:")
        ocr_logger.info(f"  Epochs: {epochs}")
        ocr_logger.info(f"  Batch size: {batch}")
        ocr_logger.info(f"  Image size: {imgsz}")
        ocr_logger.info(f"  Device: {device}")
        ocr_logger.info("-" * 80)
        ocr_logger.info("Optimization:")
        ocr_logger.info(f"  Optimizer: {optimizer}")
        ocr_logger.info(f"  Learning rate (lr0): {lr0}")
        ocr_logger.info(f"  Momentum: {momentum}")
        ocr_logger.info(f"  Weight decay: {weight_decay}")
        ocr_logger.info(f"  Cosine LR: {cos_lr}")
        ocr_logger.info(f"  AMP: {amp}")
        
        if any([hsv_h, hsv_s, hsv_v, degrees, translate, scale]):
            ocr_logger.info("-" * 80)
            ocr_logger.info("Augmentation:")
            if hsv_h is not None:
                ocr_logger.info(f"  HSV-H: {hsv_h}")
            if hsv_s is not None:
                ocr_logger.info(f"  HSV-S: {hsv_s}")
            if hsv_v is not None:
                ocr_logger.info(f"  HSV-V: {hsv_v}")
            if degrees is not None:
                ocr_logger.info(f"  Rotation: {degrees}°")
            if translate is not None:
                ocr_logger.info(f"  Translate: {translate}")
            if scale is not None:
                ocr_logger.info(f"  Scale: {scale}")
            if mosaic is not None:
                ocr_logger.info(f"  Mosaic: {mosaic}")
            if mixup is not None:
                ocr_logger.info(f"  Mixup: {mixup}")
        
        ocr_logger.info("=" * 80)
        
        try:
            # Load model
            ocr_logger.info(f"Loading model from: {model_path}")
            model = YOLO(model_path)
            
            # Start training
            ocr_logger.info("Starting training...")
            ocr_logger.info("=" * 80)
            
            results = model.train(**train_args)
            
            ocr_logger.info("=" * 80)
            ocr_logger.info("✓ Training completed successfully!")
            
            # Save best model
            self._save_best_model(save_dir)
            
            return results
            
        except Exception as e:
            ocr_logger.error(f"Training failed: {e}", exc_info=True)
            raise
    
    def _save_best_model(self, save_dir: str):
        """Copy best model to models directory."""
        try:
            best_model_path = Path(save_dir) / "train" / "weights" / "best.pt"
            
            if not best_model_path.exists():
                ocr_logger.warning(f"Best model not found at: {best_model_path}")
                return
            
            target_path = config.get_model_path(finetuned=True)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy(best_model_path, target_path)
            ocr_logger.info(f"✓ Best model saved to: {target_path}")
            
        except Exception as e:
            ocr_logger.warning(f"Failed to save best model: {e}")
    
    def train_with_preset(self, preset: str = 'default', **kwargs) -> Dict:
        """
        Train with predefined hyperparameter presets.
        
        Args:
            preset: Preset name ('default', 'fast', 'accurate', 'small_dataset', 'large_dataset')
            **kwargs: Override any preset parameters
            
        Returns:
            Training results
        """
        presets = {
            'default': {
                'epochs': 300,
                'batch_size': 16,
                'lr0': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'mosaic': 1.0,
                'mixup': 0.0,
            },
            'fast': {
                'epochs': 100,
                'batch_size': 32,
                'lr0': 0.02,
                'close_mosaic': 5,
                'amp': True,
            },
            'accurate': {
                'epochs': 500,
                'batch_size': 8,
                'lr0': 0.005,
                'weight_decay': 0.001,
                'mosaic': 1.0,
                'mixup': 0.1,
                'copy_paste': 0.1,
                'patience': 100,
            },
            'small_dataset': {
                'epochs': 500,
                'batch_size': 8,
                'lr0': 0.001,
                'mosaic': 0.5,
                'mixup': 0.2,
                'copy_paste': 0.2,
                'hsv_h': 0.02,
                'hsv_s': 0.8,
                'hsv_v': 0.5,
                'degrees': 10,
                'translate': 0.2,
                'scale': 0.7,
                'patience': 100,
            },
            'large_dataset': {
                'epochs': 100,
                'batch_size': 32,
                'lr0': 0.02,
                'mosaic': 1.0,
                'close_mosaic': 20,
            }
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Choose from: {list(presets.keys())}")
        
        # Get preset parameters
        params = presets[preset].copy()
        
        # Override with user parameters
        params.update(kwargs)
        
        ocr_logger.info(f"Training with preset: {preset}")
        return self.train(**params)
    
    def validate(self, model_path: Optional[str] = None,
                device: Optional[int] = None,
                imgsz: Optional[int] = None,
                batch: Optional[int] = None,
                conf: Optional[float] = None,
                iou: Optional[float] = None) -> Dict:
        """
        Validate trained model.
        
        Args:
            model_path: Path to model weights
            device: GPU device ID
            imgsz: Image size
            batch: Batch size
            conf: Confidence threshold
            iou: IoU threshold
            
        Returns:
            Validation metrics
        """
        model_path = model_path or config.get_model_path(finetuned=True)
        device = device if device is not None else self.model_config.device
        imgsz = imgsz or self.model_config.img_size
        batch = batch or 16
        conf = conf or self.model_config.conf_threshold
        iou = iou or self.model_config.iou_threshold
        
        ocr_logger.info("=" * 80)
        ocr_logger.info(f"Validating model: {model_path}")
        ocr_logger.info("=" * 80)
        
        try:
            model = YOLO(str(model_path))
            
            results = model.val(
                data=self.data_yaml,
                device=device,
                imgsz=imgsz,
                batch=batch,
                conf=conf,
                iou=iou,
                verbose=True
            )
            
            ocr_logger.info("=" * 80)
            ocr_logger.info("✓ Validation completed")
            
            # Log key metrics
            if hasattr(results, 'box'):
                ocr_logger.info(f"mAP50: {results.box.map50:.4f}")
                ocr_logger.info(f"mAP50-95: {results.box.map:.4f}")
                ocr_logger.info(f"Precision: {results.box.mp:.4f}")
                ocr_logger.info(f"Recall: {results.box.mr:.4f}")
            
            ocr_logger.info("=" * 80)
            
            return results
            
        except Exception as e:
            ocr_logger.error(f"Validation failed: {e}", exc_info=True)
            raise
    
    def export_model(self, model_path: Optional[str] = None,
                    format: str = 'onnx',
                    output_path: Optional[str] = None,
                    imgsz: Optional[int] = None,
                    half: bool = False,
                    simplify: bool = True) -> str:
        """
        Export model to different format.
        
        Args:
            model_path: Path to model weights
            format: Export format ('onnx', 'torchscript', 'tflite', 'edgetpu', 'coreml')
            output_path: Output path for exported model
            imgsz: Image size for export
            half: FP16 quantization
            simplify: Simplify ONNX model
            
        Returns:
            Path to exported model
        """
        model_path = model_path or config.get_model_path(finetuned=True)
        imgsz = imgsz or self.model_config.img_size
        
        ocr_logger.info(f"Exporting model to {format} format")
        
        try:
            model = YOLO(str(model_path))
            
            export_path = model.export(
                format=format,
                imgsz=imgsz,
                half=half,
                simplify=simplify
            )
            
            if output_path:
                shutil.move(export_path, output_path)
                export_path = output_path
            
            ocr_logger.info(f"✓ Model exported to: {export_path}")
            return export_path
            
        except Exception as e:
            ocr_logger.error(f"Export failed: {e}", exc_info=True)
            raise