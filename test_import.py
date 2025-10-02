# test_import.py
try:
    from digit_detector import YOLODetector
    print("✓ detector.py - OK")
except Exception as e:
    print(f"✗ detector.py - ERROR: {e}")

try:
    from recognizer import TextRecognizer
    print("✓ recognizer.py - OK")
except Exception as e:
    print(f"✗ recognizer.py - ERROR: {e}")

try:
    from pipeline import DigitalOCRPipeline
    print("✓ pipeline.py - OK")
except Exception as e:
    print(f"✗ pipeline.py - ERROR: {e}")

print("\nAll imports successful!")