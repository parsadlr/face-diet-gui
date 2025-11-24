"""
Performance profiling module for face-diet.

Tracks timing for different operations to identify bottlenecks.
"""

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List


class PerformanceProfiler:
    """Profiler to track timing for different operations."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.counts: Dict[str, int] = defaultdict(int)
        self.current_timers: Dict[str, float] = {}
    
    @contextmanager
    def time_block(self, operation_name: str):
        """Context manager to time a code block."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.timings[operation_name].append(elapsed)
            self.counts[operation_name] += 1
    
    def start_timer(self, operation_name: str):
        """Start a timer for an operation."""
        self.current_timers[operation_name] = time.perf_counter()
    
    def end_timer(self, operation_name: str):
        """End a timer and record the elapsed time."""
        if operation_name in self.current_timers:
            elapsed = time.perf_counter() - self.current_timers[operation_name]
            self.timings[operation_name].append(elapsed)
            self.counts[operation_name] += 1
            del self.current_timers[operation_name]
            return elapsed
        return 0.0
    
    def get_stats(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        if operation_name not in self.timings or len(self.timings[operation_name]) == 0:
            return {
                'count': 0,
                'total': 0.0,
                'mean': 0.0,
                'min': 0.0,
                'max': 0.0,
            }
        
        times = self.timings[operation_name]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
        }
    
    def print_summary(self):
        """Print a summary of all timings."""
        print("\n" + "=" * 80)
        print("PERFORMANCE PROFILING SUMMARY")
        print("=" * 80)
        
        # Calculate totals
        total_time = sum(sum(times) for times in self.timings.values())
        
        # Sort by total time
        sorted_ops = sorted(
            self.timings.items(),
            key=lambda x: sum(x[1]),
            reverse=True
        )
        
        print(f"\n{'Operation':<40} {'Count':<10} {'Total (s)':<12} {'Mean (s)':<12} {'Min (s)':<12} {'Max (s)':<12} {'% of Total':<12}")
        print("-" * 80)
        
        for op_name, times in sorted_ops:
            stats = self.get_stats(op_name)
            percentage = (stats['total'] / total_time * 100) if total_time > 0 else 0.0
            print(
                f"{op_name:<40} "
                f"{stats['count']:<10} "
                f"{stats['total']:<12.4f} "
                f"{stats['mean']:<12.4f} "
                f"{stats['min']:<12.4f} "
                f"{stats['max']:<12.4f} "
                f"{percentage:<12.2f}%"
            )
        
        print("-" * 80)
        print(f"{'TOTAL':<40} {'':<10} {total_time:<12.4f}")
        print("=" * 80)
        
        # Identify bottlenecks
        if sorted_ops:
            top_bottleneck = sorted_ops[0]
            bottleneck_pct = (sum(top_bottleneck[1]) / total_time * 100) if total_time > 0 else 0.0
            print(f"\n🔴 Top Bottleneck: {top_bottleneck[0]} ({bottleneck_pct:.1f}% of total time)")
            
            if len(sorted_ops) > 1:
                second_bottleneck = sorted_ops[1]
                second_pct = (sum(second_bottleneck[1]) / total_time * 100) if total_time > 0 else 0.0
                print(f"🟡 Second Bottleneck: {second_bottleneck[0]} ({second_pct:.1f}% of total time)")
    
    def reset(self):
        """Reset all timings."""
        self.timings.clear()
        self.counts.clear()
        self.current_timers.clear()


# Global profiler instance
_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _profiler


def reset_profiler():
    """Reset the global profiler."""
    _profiler.reset()

