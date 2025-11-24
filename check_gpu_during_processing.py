"""Check if GPU is being used during actual processing."""
import sys
import subprocess
import threading
import time

def monitor_gpu_background(stop_event, results):
    """Monitor GPU in background thread."""
    samples = []
    while not stop_event.is_set():
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                if len(parts) >= 3:
                    samples.append({
                        'utilization': int(parts[0]),
                        'memory_used': int(parts[1]),
                        'memory_total': int(parts[2]),
                        'time': time.time()
                    })
        except:
            pass
        time.sleep(0.5)  # Check every 0.5 seconds
    
    results['samples'] = samples

if __name__ == "__main__":
    print("=" * 60)
    print("GPU Monitoring During Processing")
    print("=" * 60)
    print("\nThis script will monitor GPU while you run your processing.")
    print("Run your command in another terminal, then press Enter here when done.")
    print("=" * 60)
    print()
    
    input("Press Enter to start monitoring (then run your processing command)...")
    
    stop_event = threading.Event()
    results = {'samples': []}
    
    monitor_thread = threading.Thread(target=monitor_gpu_background, args=(stop_event, results))
    monitor_thread.start()
    
    print("\nMonitoring GPU... (Press Enter again when processing is complete)")
    input()
    
    stop_event.set()
    monitor_thread.join(timeout=2)
    
    samples = results.get('samples', [])
    
    if samples:
        print("\n" + "=" * 60)
        print("GPU Usage Statistics")
        print("=" * 60)
        
        utilizations = [s['utilization'] for s in samples]
        memories = [s['memory_used'] for s in samples]
        
        print(f"Total samples: {len(samples)}")
        print(f"Average GPU utilization: {sum(utilizations) / len(utilizations):.1f}%")
        print(f"Maximum GPU utilization: {max(utilizations)}%")
        print(f"Minimum GPU utilization: {min(utilizations)}%")
        print(f"Average memory used: {sum(memories) / len(memories):.0f}MB")
        print(f"Maximum memory used: {max(memories)}MB")
        
        # Count how many samples had significant GPU usage
        high_usage = sum(1 for u in utilizations if u > 30)
        print(f"\nSamples with >30% GPU usage: {high_usage}/{len(samples)} ({100*high_usage/len(samples):.1f}%)")
        
        if sum(utilizations) / len(utilizations) < 20:
            print("\n⚠ WARNING: Low average GPU utilization!")
            print("   GPU may not be fully utilized during processing.")
        elif sum(utilizations) / len(utilizations) > 50:
            print("\n✓ GPU is being actively used")
        
        print("=" * 60)
    else:
        print("\nNo GPU samples collected. Make sure nvidia-smi is available.")

