"""Monitor GPU usage during processing."""
import subprocess
import time
import sys

def get_gpu_usage():
    """Get current GPU utilization and memory usage."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 3:
                return {
                    'utilization': int(parts[0]),
                    'memory_used': int(parts[1]),
                    'memory_total': int(parts[2]),
                    'memory_percent': int(parts[1]) * 100 // int(parts[2]) if int(parts[2]) > 0 else 0
                }
    except:
        pass
    return None

def monitor_gpu_continuous(duration=60, interval=1):
    """Monitor GPU continuously for specified duration."""
    print("=" * 60)
    print("GPU Monitoring")
    print("=" * 60)
    print(f"Monitoring for {duration} seconds (checking every {interval}s)")
    print("Press Ctrl+C to stop early")
    print("=" * 60)
    print()
    
    start_time = time.time()
    samples = []
    
    try:
        while time.time() - start_time < duration:
            gpu_info = get_gpu_usage()
            if gpu_info:
                samples.append(gpu_info)
                print(f"GPU: {gpu_info['utilization']:3d}% | "
                      f"Memory: {gpu_info['memory_used']:5d}MB / {gpu_info['memory_total']:5d}MB "
                      f"({gpu_info['memory_percent']:3d}%)")
            else:
                print("GPU: Unable to read (nvidia-smi not available)")
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    
    if samples:
        avg_util = sum(s['utilization'] for s in samples) / len(samples)
        max_util = max(s['utilization'] for s in samples)
        avg_mem = sum(s['memory_percent'] for s in samples) / len(samples)
        max_mem = max(s['memory_percent'] for s in samples)
        
        print()
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Average GPU utilization: {avg_util:.1f}%")
        print(f"Maximum GPU utilization: {max_util}%")
        print(f"Average memory usage: {avg_mem:.1f}%")
        print(f"Maximum memory usage: {max_mem}%")
        print("=" * 60)
        
        if avg_util < 10:
            print("\n⚠ WARNING: Low GPU utilization detected!")
            print("   This suggests GPU may not be fully utilized.")
            print("   Possible reasons:")
            print("   - Processing is I/O bound (reading video)")
            print("   - Models are too small to benefit from GPU")
            print("   - Other operations (attribute extraction) use CPU")
        elif avg_util > 50:
            print("\n✓ GPU is being actively used")
    else:
        print("\nNo GPU samples collected")

if __name__ == "__main__":
    duration = 60
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except:
            pass
    
    monitor_gpu_continuous(duration=duration)

