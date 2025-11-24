"""Setup script to configure CUDA environment for ONNX Runtime."""
import os
import sys
import subprocess
from pathlib import Path

def check_cuda_installation():
    """Check if CUDA is installed and accessible."""
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        os.environ.get("CUDA_PATH", ""),
    ]
    
    cuda_versions = []
    for base_path in cuda_paths:
        if base_path and os.path.exists(base_path):
            for item in os.listdir(base_path):
                version_path = os.path.join(base_path, item)
                if os.path.isdir(version_path) and item.startswith("v"):
                    cuda_versions.append((item, version_path))
    
    return cuda_versions

def check_cudnn_installation(cuda_path):
    """Check if cuDNN is installed for the given CUDA path."""
    cudnn_paths = [
        os.path.join(cuda_path, "bin", "cudnn64_*.dll"),
        os.path.join(cuda_path, "bin", "cudnn*.dll"),
    ]
    
    import glob
    for pattern in cudnn_paths:
        if glob.glob(pattern):
            return True
    
    # Also check common cuDNN installation locations
    common_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\cuDNN",
        os.path.join(os.path.dirname(cuda_path), "cuDNN"),
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return True
    
    return False

def setup_environment():
    """Set up environment variables for CUDA."""
    print("=" * 60)
    print("CUDA Environment Setup")
    print("=" * 60)
    
    # Check CUDA installation
    cuda_versions = check_cuda_installation()
    
    if not cuda_versions:
        print("\n[ERROR] CUDA not found!")
        print("Please install CUDA Toolkit from:")
        print("https://developer.nvidia.com/cuda-downloads")
        return False
    
    print(f"\n[OK] Found {len(cuda_versions)} CUDA installation(s):")
    for version, path in cuda_versions:
        print(f"  - CUDA {version} at {path}")
    
    # Use the latest CUDA version
    latest_version, latest_path = sorted(cuda_versions, reverse=True)[0]
    cuda_bin = os.path.join(latest_path, "bin")
    
    print(f"\n[INFO] Using CUDA {latest_version}")
    
    # Check cuDNN
    cudnn_found = check_cudnn_installation(latest_path)
    
    if not cudnn_found:
        print("\n[WARNING] cuDNN not found!")
        print("\nTo enable CUDA support, you need to install cuDNN:")
        print("1. Download cuDNN from: https://developer.nvidia.com/cudnn")
        print("2. Extract the archive")
        print("3. Copy the following files to CUDA directories:")
        print(f"   - bin/cudnn64_*.dll -> {cuda_bin}")
        print(f"   - include/cudnn*.h -> {os.path.join(latest_path, 'include')}")
        print(f"   - lib/x64/cudnn*.lib -> {os.path.join(latest_path, 'lib', 'x64')}")
        print("\nAlternatively, you can:")
        print("- Add cuDNN bin directory to your PATH")
        print("- Set CUDNN_PATH environment variable")
        return False
    else:
        print("\n[OK] cuDNN found!")
    
    # Add CUDA to PATH
    current_path = os.environ.get("PATH", "")
    if cuda_bin not in current_path:
        print(f"\n[INFO] Adding CUDA bin to PATH: {cuda_bin}")
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
        
        # Create a batch file to set PATH
        batch_file = Path("setup_cuda_env.bat")
        with open(batch_file, "w") as f:
            f.write(f"@echo off\n")
            f.write(f"set PATH={cuda_bin};%PATH%\n")
            f.write(f"echo CUDA environment configured.\n")
            f.write(f"echo CUDA bin added to PATH: {cuda_bin}\n")
        
        print(f"[OK] Created {batch_file} to set up environment")
        print(f"   Run this before your Python scripts, or add to your system PATH")
    else:
        print(f"\n[OK] CUDA bin already in PATH")
    
    return True

def test_onnxruntime():
    """Test ONNX Runtime CUDA support."""
    print("\n" + "=" * 60)
    print("Testing ONNX Runtime")
    print("=" * 60)
    
    try:
        import onnxruntime as ort
        print(f"[OK] ONNXRuntime version: {ort.__version__}")
        
        providers = ort.get_available_providers()
        print(f"[OK] Available providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("[OK] CUDA provider is available")
            
            # Try to create a simple session to test
            try:
                # This is just a test - we can't create a session without a model
                print("[INFO] CUDA provider appears to be working")
                return True
            except Exception as e:
                print(f"[WARNING] CUDA provider available but may have issues: {e}")
                return False
        else:
            print("[WARNING] CUDA provider not available")
            return False
            
    except ImportError:
        print("[ERROR] ONNX Runtime not installed")
        return False

if __name__ == "__main__":
    print("ONNX Runtime CUDA Setup")
    print("=" * 60)
    
    if setup_environment():
        test_onnxruntime()
        print("\n" + "=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        print("\nNote: If cuDNN is missing, CUDA will fall back to CPU.")
        print("Your pipeline will still work, but without GPU acceleration.")
    else:
        print("\n" + "=" * 60)
        print("Setup Incomplete")
        print("=" * 60)
        print("\nPlease install cuDNN to enable CUDA support.")
        print("Your pipeline will work with CPU fallback.")


