"""
Main CLI application for 12-class Digit Detection.
"""
import argparse
import sys
from pathlib import Path
import json 
from detector import DigitDetector
from trainer import YOLOTrainer
from config import config
from logger import ocr_logger
from exceptions import OCRException


def setup_argparse() -> argparse.ArgumentParser:
    """Setup command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="12-Class Digit Detection System (0-9, :, .)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model on 12 classes
  python main.py train --data data/data.yaml --epochs 300 --device cpu
  
  # Detect digits from single image
  python main.py detect --image clock.jpg
  
  # Process directory and save to results.jsonl
  python main.py detect --image-dir test_images/
  
  # Validate model
  python main.py validate
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect digits from images')
    detect_parser.add_argument('--image', type=str, help='Single image path')
    detect_parser.add_argument('--image-dir', type=str, help='Directory of images')
    detect_parser.add_argument('--output', type=str, default='results.jsonl',
                              help='Output JSONL file (default: results.jsonl)')
    detect_parser.add_argument('--conf', type=float, default=0.25,
                              help='Confidence threshold (default: 0.25)')
    detect_parser.add_argument('--no-visualization', action='store_true',
                              help='Disable visualization')
    detect_parser.add_argument('--use-pretrained', action='store_true',
                              help='Use pretrained YOLOv8 (not recommended)')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train 12-class YOLO model')
    train_parser.add_argument('--data', type=str, default='data/data.yaml',
                             help='Path to data.yaml')
    train_parser.add_argument('--epochs', type=int, default=300,
                             help='Number of epochs')
    train_parser.add_argument('--batch', type=int, default=16,
                             help='Batch size')
    train_parser.add_argument('--device', type=str, default=None,
                             help='Device: 0 for GPU, cpu for CPU')
    train_parser.add_argument('--resume', action='store_true',
                             help='Resume training')
    train_parser.add_argument('--model', type=str,
                             help='Pretrained model path')
    
    # Validation command
    validate_parser = subparsers.add_parser('validate', help='Validate model')
    validate_parser.add_argument('--model', type=str,
                                help='Model path to validate')
    validate_parser.add_argument('--data', type=str, default='data/data.yaml',
                                help='Path to data.yaml')
    
    return parser


def run_detection(args):
    """Run digit detection."""
    try:
        use_finetuned = not args.use_pretrained
        
        ocr_logger.info("="*60)
        ocr_logger.info("DIGIT DETECTION - 12 Classes (0-9, :, .)")
        ocr_logger.info("="*60)
        
        detector = DigitDetector(use_finetuned=use_finetuned)
        
        # Single image
        if args.image:
            if not Path(args.image).exists():
                ocr_logger.error(f"Image not found: {args.image}")
                return 1
            
            result = detector.detect_digits(
                args.image,
                conf_threshold=args.conf,
                save_visualization=not args.no_visualization
            )
            
            # Display result
            print("\n" + "=" * 80)
            print("DETECTION RESULT")
            print("=" * 80)
            print(f"Image:            {args.image}")
            print(f"Detected Number:  {result['detected_number']}")
            print(f"Confidence:       {result['confidence']:.3f}")
            print(f"Number of Digits: {result['num_digits']}")
            print(f"Timestamp:        {result['timestamp']}")
            print("=" * 80)
            
            # Show individual detections
            if result['detections']:
                print("\nIndividual Detections:")
                print("-" * 80)
                for i, det in enumerate(result['detections'], 1):
                    print(f"  {i}. Digit: '{det['digit']}'  Confidence: {det['confidence']:.3f}  BBox: {det['bbox']}")
                print("=" * 80 + "\n")
            
            # Save to JSONL
            output_path = config.get_output_path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            print(f"✓ Result saved to: {output_path}\n")
            
        # Directory
        elif args.image_dir:
            if not Path(args.image_dir).exists():
                ocr_logger.error(f"Directory not found: {args.image_dir}")
                return 1
            
            results = detector.process_batch(args.image_dir, args.output)
            
        else:
            ocr_logger.error("Must provide --image or --image-dir")
            return 1
        
        return 0
        
    except OCRException as e:
        ocr_logger.error(f"Detection error: {e}")
        return 1
    except Exception as e:
        ocr_logger.critical(f"Unexpected error: {e}", exc_info=True)
        return 1


def run_training(args):
    """Run model training."""
    try:
        ocr_logger.info("="*60)
        ocr_logger.info("TRAINING 12-CLASS MODEL (0-9, :, .)")
        ocr_logger.info("="*60)
        
        trainer = YOLOTrainer(data_yaml=args.data)
        
        # Parse device
        device = args.device
        if device is None:
            device = config.model.device
        elif device.lower() == 'cpu':
            device = -1
        else:
            device = int(device)
        
        if args.resume:
            results = trainer.train(resume=True, device=device)
        else:
            results = trainer.train(
                pretrained_model=args.model,
                epochs=args.epochs,
                batch_size=args.batch,
                device=device
            )
        
        ocr_logger.info("\n" + "="*60)
        ocr_logger.info("✓ TRAINING COMPLETED SUCCESSFULLY!")
        ocr_logger.info("="*60)
        ocr_logger.info(f"Model saved to: {config.get_model_path(finetuned=True)}")
        ocr_logger.info("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        ocr_logger.critical(f"Training failed: {e}", exc_info=True)
        return 1


def run_validation(args):
    """Run model validation."""
    try:
        trainer = YOLOTrainer(data_yaml=args.data)
        
        results = trainer.validate(model_path=args.model)
        
        ocr_logger.info("\n✓ Validation completed successfully!")
        return 0
        
    except Exception as e:
        ocr_logger.critical(f"Validation failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command
    if args.command == 'detect':
        return run_detection(args)
    elif args.command == 'train':
        return run_training(args)
    elif args.command == 'validate':
        return run_validation(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())