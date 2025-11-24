"""Check CUDA and cuDNN installation status."""
import os
import sys
import glob
from pathlib import Path

def check_cudnn_files():
    """Check if cuDNN files are installed."""
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
    
    checks = {
        "bin": {
            "path": os.path.join(cuda_path, "bin"),
            "patterns": ["cudnn64_*.dll", "cudnn*.dll"],
            "found": False,
            "files": []
        },
        "include": {
            "path": os.path.join(cuda_path, "include"),
            "patterns": ["cudnn*.h", "cudnn.h"],
            "found": False,
            "files": []
        },
        "lib": {
            "path": os.path.join(cuda_path, "lib", "x64"),
            "patterns": ["cudnn*.lib", "cudnn64_*.lib"],
            "found": False,
            "files": []
        }
    }
    
    for key, check in checks.items():
        if os.path.exists(check["path"]):
            for pattern in check["patterns"]:
                full_pattern = os.path.join(check["path"], pattern)
                matches = glob.glob(full_pattern)
                if matches:
                    check["found"] = True
                    check["files"].extend([os.path.basename(f) for f in matches])
    
    return checks

def check_path():
    """Check if CUDA is in PATH."""
    path = os.environ.get("PATH", "")
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
    cuda_lib = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64"
    
    bin_in_path = cuda_bin.lower() in path.lower()
    lib_in_path = cuda_lib.lower() in path.lower()
    
    return {
        "cuda_bin_in_path": bin_in_path,
        "cuda_lib_in_path": lib_in_path,
        "cuda_bin": cuda_bin,
        "cuda_lib": cuda_lib
    }

def main():
    print("=" * 60)
    print("CUDA Installation Check")
    print("=" * 60)
    
    # Check cuDNN files
    print("\n1. Checking cuDNN files...")
    cudnn_checks = check_cudnn_files()
    
    all_found = True
    for key, check in cudnn_checks.items():
        status = "✓ FOUND" if check["found"] else "✗ NOT FOUND"
        print(f"   {key.capitalize()}: {status}")
        if check["found"]:
            print(f"      Files: {', '.join(check['files'][:3])}")
            if len(check["files"]) > 3:
                print(f"      ... and {len(check['files']) - 3} more")
        else:
            print(f"      Path: {check['path']}")
            all_found = False
    
    # Check PATH
    print("\n2. Checking PATH environment variable...")
    path_check = check_path()
    print(f"   CUDA bin in PATH: {'✓ YES' if path_check['cuda_bin_in_path'] else '✗ NO'}")
    print(f"   CUDA lib in PATH: {'✓ YES' if path_check['cuda_lib_in_path'] else '✗ NO'}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if all_found:
        print("✓ cuDNN files are installed correctly")
    else:
        print("✗ cuDNN files are missing or incomplete")
        print("\n  Please verify step 3 was completed:")
        print("  - Copy bin/cudnn*.dll to CUDA bin folder")
        print("  - Copy include/cudnn*.h to CUDA include folder")
        print("  - Copy lib/x64/cudnn*.lib to CUDA lib/x64 folder")
    
    if path_check['cuda_bin_in_path']:
        print("✓ CUDA bin is already in PATH")
        print("\n  Step 4 (PATH setup) is NOT needed - it's already configured!")
    else:
        print("✗ CUDA bin is NOT in PATH")
        print("\n  Step 4 (PATH setup) IS needed")
        print(f"  Add this to your PATH: {path_check['cuda_bin']}")
    
    if path_check['cuda_lib_in_path']:
        print("✓ CUDA lib is already in PATH")
    else:
        print("⚠ CUDA lib is NOT in PATH (usually not required)")
    
    print("\n" + "=" * 60)
    
    # Final recommendation
    if all_found and path_check['cuda_bin_in_path']:
        print("\n✓ Installation looks complete!")
        print("  Try running: python test_cuda.py")
    elif not all_found:
        print("\n⚠ cuDNN files are missing - complete step 3 first")
    elif not path_check['cuda_bin_in_path']:
        print("\n⚠ PATH needs to be updated - complete step 4")

if __name__ == "__main__":
    main()

