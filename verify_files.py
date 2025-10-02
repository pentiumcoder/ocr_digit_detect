# verify_files.py
import os
from pathlib import Path

required_files = {
    'config.py': ['Config', 'config'],
    'logger.py': ['OCRLogger', 'ocr_logger'],
    'exceptions.py': ['OCRException', 'ModelLoadError'],
    'detector.py': ['YOLODetector'],
    'recognizer.py': ['TextRecognizer'],
    'pipeline.py': ['DigitalOCRPipeline', 'OCRResult'],
    'trainer.py': ['YOLOTrainer'],
    'main.py': ['main'],
    'preprocess/operations.py': ['PreprocessingOperations', 'ROI'],
    'preprocess/engine.py': ['PreprocessingEngine']
}

print("=" * 60)
print("FILE VERIFICATION")
print("=" * 60)

all_ok = True

for filepath, expected_content in required_files.items():
    file_path = Path(filepath)
    
    if not file_path.exists():
        print(f"✗ {filepath} - FILE NOT FOUND")
        all_ok = False
        continue
    
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if file is empty
    if len(content.strip()) == 0:
        print(f"✗ {filepath} - FILE IS EMPTY")
        all_ok = False
        continue
    
    # Check for expected content
    missing = []
    for item in expected_content:
        if item not in content:
            missing.append(item)
    
    if missing:
        print(f"✗ {filepath} - MISSING: {', '.join(missing)}")
        all_ok = False
    else:
        print(f"✓ {filepath} - OK ({len(content)} bytes)")

print("=" * 60)

if all_ok:
    print("✓ ALL FILES VERIFIED!")
    print("\nNow run: python main.py train --data data/data.yaml --epochs 50")
else:
    print("✗ SOME FILES HAVE ISSUES - Please fix them")
    
print("=" * 60)