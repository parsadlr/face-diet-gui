"""
Face-Diet: Video Face Analysis and Tracking

Main entry point for processing egocentric videos with face detection,
identity tracking, and attribute extraction.
"""

import argparse
import os
import sys
from typing import Dict

from profiler import get_profiler
from video_processor import process_video


def parse_time_string(time_str: str) -> float:
    """
    Parse time string in format MM:SS or HH:MM:SS to seconds.
    
    Parameters
    ----------
    time_str : str
        Time string like "2:30" or "0:02:30"
    
    Returns
    -------
    float
        Time in seconds
    """
    parts = time_str.split(':')
    
    if len(parts) == 2:
        # MM:SS format
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        # HH:MM:SS format
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        raise ValueError(f"Invalid time format: {time_str}. Use MM:SS or HH:MM:SS")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Face-Diet: Analyze faces in egocentric videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to input video file'
    )
    
    # Output arguments
    parser.add_argument(
        '-o', '--output-csv',
        type=str,
        default=None,
        help='Path to output CSV file (default: <video_name>_faces.csv)'
    )
    
    parser.add_argument(
        '-v', '--output-video',
        type=str,
        default=None,
        help='Path to output annotated video (default: no video output)'
    )
    
    # Processing arguments
    parser.add_argument(
        '-s', '--sampling-rate',
        type=int,
        default=30,
        help='Process every N frames (1 = every frame, 30 = 1 per second for 30fps video)'
    )
    
    parser.add_argument(
        '--start-time',
        type=str,
        default=None,
        help='Start time in format MM:SS or HH:MM:SS (e.g., "2:00" or "0:02:00")'
    )
    
    parser.add_argument(
        '--end-time',
        type=str,
        default=None,
        help='End time in format MM:SS or HH:MM:SS (e.g., "2:30" or "0:02:30")'
    )
    
    # Clustering arguments
    parser.add_argument(
        '-c', '--clustering-method',
        type=str,
        choices=['threshold', 'dbscan'],
        default='threshold',
        help='Clustering method for face identity assignment'
    )
    
    parser.add_argument(
        '-t', '--similarity-threshold',
        type=float,
        default=0.6,
        help='Cosine similarity threshold for threshold-based clustering'
    )
    
    parser.add_argument(
        '--dbscan-eps',
        type=float,
        default=0.4,
        help='DBSCAN epsilon parameter (distance threshold)'
    )
    
    parser.add_argument(
        '--dbscan-min-samples',
        type=int,
        default=2,
        help='DBSCAN minimum samples parameter'
    )
    
    # Hardware arguments
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for processing (if available)'
    )
    
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable performance profiling and print timing summary'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers for attribute extraction (default: 4)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for DeepFace processing (default: 8, set to 1 to disable batching)'
    )
    
    parser.add_argument(
        '--no-batch',
        action='store_true',
        help='Disable DeepFace batch processing (use parallel processing instead)'
    )
    
    return parser.parse_args()


def validate_config(args) -> Dict:
    """
    Validate and prepare configuration from arguments.
    
    Parameters
    ----------
    args
        Parsed command line arguments
    
    Returns
    -------
    Dict
        Configuration dictionary
    """
    # Check video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}", file=sys.stderr)
        sys.exit(1)
    
    # Set default output CSV path
    if args.output_csv is None:
        video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
        args.output_csv = f"{video_basename}_faces.csv"
    
    # Check output directory exists
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        print(f"Error: Output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)
    
    if args.output_video:
        output_video_dir = os.path.dirname(args.output_video)
        if output_video_dir and not os.path.exists(output_video_dir):
            print(f"Error: Output video directory not found: {output_video_dir}", file=sys.stderr)
            sys.exit(1)
    
    # Validate sampling rate
    if args.sampling_rate < 1:
        print(f"Error: Sampling rate must be >= 1", file=sys.stderr)
        sys.exit(1)
    
    # Validate threshold
    if args.similarity_threshold < 0.0 or args.similarity_threshold > 1.0:
        print(f"Error: Similarity threshold must be in [0.0, 1.0]", file=sys.stderr)
        sys.exit(1)
    
    # Parse time range if provided
    start_time_sec = None
    end_time_sec = None
    
    if args.start_time:
        try:
            start_time_sec = parse_time_string(args.start_time)
        except ValueError as e:
            print(f"Error: Invalid start time format: {e}", file=sys.stderr)
            sys.exit(1)
    
    if args.end_time:
        try:
            end_time_sec = parse_time_string(args.end_time)
        except ValueError as e:
            print(f"Error: Invalid end time format: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Validate time range
    if start_time_sec is not None and end_time_sec is not None:
        if start_time_sec >= end_time_sec:
            print(f"Error: Start time must be before end time", file=sys.stderr)
            sys.exit(1)
    
    config = {
        'video_path': args.video_path,
        'output_csv': args.output_csv,
        'output_video': args.output_video,
        'sampling_rate': args.sampling_rate,
        'start_time': start_time_sec,
        'end_time': end_time_sec,
        'clustering_method': args.clustering_method,
        'similarity_threshold': args.similarity_threshold,
        'dbscan_eps': args.dbscan_eps,
        'dbscan_min_samples': args.dbscan_min_samples,
        'use_gpu': args.gpu,
        'max_workers': args.max_workers,
        'batch_size': args.batch_size,
        'use_batch': not args.no_batch,
    }
    
    return config


def print_config(config: Dict) -> None:
    """Print configuration summary."""
    print("=" * 60)
    print("Face-Diet Video Processing")
    print("=" * 60)
    print(f"Input video:        {config['video_path']}")
    print(f"Output CSV:         {config['output_csv']}")
    if config['output_video']:
        print(f"Output video:       {config['output_video']}")
    else:
        print(f"Output video:       (not generating)")
    print(f"Sampling rate:      Every {config['sampling_rate']} frame(s)")
    
    # Display time range if specified
    if config['start_time'] is not None or config['end_time'] is not None:
        start_str = f"{config['start_time']:.1f}s" if config['start_time'] is not None else "beginning"
        end_str = f"{config['end_time']:.1f}s" if config['end_time'] is not None else "end"
        print(f"Time range:         {start_str} to {end_str}")
    
    print(f"Clustering method:  {config['clustering_method']}")
    if config['clustering_method'] == 'threshold':
        print(f"Similarity thresh:  {config['similarity_threshold']:.2f}")
    else:
        print(f"DBSCAN eps:         {config['dbscan_eps']:.2f}")
        print(f"DBSCAN min samples: {config['dbscan_min_samples']}")
    print(f"GPU acceleration:   {'Yes' if config['use_gpu'] else 'No'}")
    print(f"Parallel workers:   {config.get('max_workers', 4)}")
    if config.get('use_batch', True):
        print(f"Batch processing:   Enabled (batch size: {config.get('batch_size', 8)})")
    else:
        print(f"Batch processing:   Disabled")
    print("=" * 60)
    print()


def print_summary(results: Dict) -> None:
    """Print processing summary."""
    print()
    print("=" * 60)
    print("Processing Summary")
    print("=" * 60)
    print(f"Total face instances detected: {results['total_detections']}")
    print(f"Unique face identities found:  {results['unique_faces']}")
    print()
    
    if results['representatives']:
        print("Representative instances (closest to cluster centroid):")
        for face_id, rep in sorted(results['representatives'].items()):
            frame = rep['frame_number']
            time = rep['time_seconds']
            print(f"  {face_id}: Frame {frame} (t={time:.2f}s)")
    
    print("=" * 60)


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Validate and prepare config
    config = validate_config(args)
    
    # Print configuration
    print_config(config)
    
    try:
        # Process video
        results = process_video(**config)
        
        # Print summary
        print_summary(results)
        
        print(f"\n✓ Success! Results saved to: {config['output_csv']}")
        if config['output_video']:
            print(f"✓ Annotated video saved to: {config['output_video']}")
        
        # Print profiling summary if enabled
        if args.profile:
            profiler = get_profiler()
            profiler.print_summary()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

