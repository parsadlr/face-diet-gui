"""Verify GPU usage in face detection."""
import sys
import os

# Set UTF-8 encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("=" * 60)
print("Verifying GPU Usage in Face Detection")
print("=" * 60)

# Test 1: Check current initialization
print("\n1. Testing current initialization (without providers)...")
try:
    from insightface.app import FaceAnalysis
    
    # This is how it's currently initialized in face_detection.py
    app1 = FaceAnalysis(name='buffalo_l')
    app1.prepare(ctx_id=0, det_size=(640, 640))
    
    # Check which providers are actually being used
    # InsightFace doesn't expose this directly, but we can check the models
    print("   [INFO] Initialized with ctx_id=0 (GPU)")
    print("   [WARNING] Providers not explicitly set - may fall back to CPU")
    
except Exception as e:
    print(f"   [ERROR] Initialization failed: {e}")

# Test 2: Check with explicit CUDA providers
print("\n2. Testing with explicit CUDA providers...")
try:
    app2 = FaceAnalysis(
        name='buffalo_l',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app2.prepare(ctx_id=0, det_size=(640, 640))
    print("   [OK] Initialized with CUDA providers explicitly set")
    
    # Try to detect on a test image
    import numpy as np
    import cv2
    
    # Create a dummy test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = app2.get(test_img)
    print(f"   [OK] Detection test successful (found {len(faces)} faces)")
    
except Exception as e:
    print(f"   [ERROR] Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check ONNX Runtime providers
print("\n3. Checking ONNX Runtime providers...")
try:
    import onnxruntime as ort
    
    providers = ort.get_available_providers()
    print(f"   Available providers: {providers}")
    
    if 'CUDAExecutionProvider' in providers:
        print("   [OK] CUDA provider is available")
    else:
        print("   [WARNING] CUDA provider not available")
        
except Exception as e:
    print(f"   [ERROR] Failed to check providers: {e}")

print("\n" + "=" * 60)
print("Recommendation")
print("=" * 60)
print("To ensure GPU is used, initialize FaceAnalysis with:")
print("  app = FaceAnalysis(")
print("      name='buffalo_l',")
print("      providers=['CUDAExecutionProvider', 'CPUExecutionProvider']")
print("  )")
print("  app.prepare(ctx_id=0, det_size=(640, 640))")
print("=" * 60)

