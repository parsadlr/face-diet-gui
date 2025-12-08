"""
Convenience script to process all sessions in a participant directory at once.

This runs:
1. process-session for each session folder
2. combine-sessions to assign global IDs

Usage:
    python process_all_sessions.py <participant_dir> [options]
"""

import os
import sys
from pathlib import Path
from process_participant_v2 import process_single_session, combine_sessions_with_global_ids


def process_all_sessions(
    participant_dir: str,
    sampling_rate: int = 30,
    test_duration: float = None,
    global_threshold: float = 0.6,
    use_gpu: bool = False,
    batch_size: int = 32,
    clustering_method: str = 'threshold',
):
    """
    Process all sessions in a participant directory.
    
    Detects faces in each session without assigning IDs, then assigns
    global face IDs across all sessions at once.
    
    Parameters
    ----------
    participant_dir : str
        Path to participant directory containing session subdirectories
    sampling_rate : int
        Process every N frames
    test_duration : float, optional
        If specified, only process first N seconds of each video
    global_threshold : float
        Similarity threshold for global face ID assignment across sessions
    use_gpu : bool
        Whether to use GPU
    batch_size : int
        Batch size for DeepFace processing
    clustering_method : str
        'threshold' or 'dbscan' for global clustering
    """
    participant_path = Path(participant_dir)
    
    # Find all session directories
    session_dirs = []
    for subdir in sorted(participant_path.iterdir()):
        if not subdir.is_dir():
            continue
        
        # Check if it has a scenevideo file
        video_files = list(subdir.glob("scenevideo.*"))
        if video_files:
            session_dirs.append(subdir)
    
    if not session_dirs:
        print(f"Error: No session directories found in {participant_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print(f"PROCESSING PARTICIPANT: {participant_path.name}")
    print("=" * 80)
    print(f"Found {len(session_dirs)} session(s):")
    for session_dir in session_dirs:
        print(f"  - {session_dir.name}")
    print()
    
    if test_duration:
        print(f"[TEST MODE] Processing only first {test_duration} seconds of each video\n")
    
    # Step 1: Process each session
    print("=" * 80)
    print("STEP 1: PROCESSING EACH SESSION")
    print("=" * 80)
    print()
    
    for i, session_dir in enumerate(session_dirs, 1):
        print(f"\n{'='*80}")
        print(f"Session {i}/{len(session_dirs)}: {session_dir.name}")
        print(f"{'='*80}\n")
        
        try:
            result = process_single_session(
                session_dir=str(session_dir),
                output_csv=None,  # Use default: <session_dir>/faces_detections.csv
                sampling_rate=sampling_rate,
                start_time=None,  # Will be randomly selected if test_duration set
                end_time=test_duration,  # Used as duration if start_time is None
                use_gpu=use_gpu,
                batch_size=batch_size,
            )
            print(f"✓ Session {session_dir.name} completed successfully")
        
        except Exception as e:
            print(f"✗ Error processing session {session_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nContinuing with remaining sessions...")
    
    # Step 2: Combine sessions with global IDs
    print("\n\n" + "=" * 80)
    print("STEP 2: COMBINING SESSIONS WITH GLOBAL IDs")
    print("=" * 80)
    print()
    
    try:
        result = combine_sessions_with_global_ids(
            participant_dir=participant_dir,
            output_csv=None,  # Use default: <participant_dir>/faces_combined.csv
            similarity_threshold=global_threshold,
            clustering_method=clustering_method,
        )
        
        print("\n" + "=" * 80)
        print("✓ ALL PROCESSING COMPLETE!")
        print("=" * 80)
        print(f"Total faces: {result['total_faces']}")
        print(f"Unique global IDs: {result['unique_global_ids']}")
        print(f"Sessions processed: {result['num_sessions']}")
        print(f"\nFinal output: {result['output_csv']}")
        print("=" * 80)
    
    except Exception as e:
        print(f"\n✗ Error combining sessions: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process all sessions in a participant directory"
    )
    
    parser.add_argument(
        'participant_dir',
        help='Path to participant directory containing session folders'
    )
    parser.add_argument(
        '-s', '--sampling-rate',
        type=int,
        default=30,
        help='Process every N frames (default: 30)'
    )
    parser.add_argument(
        '--test-duration',
        type=float,
        default=None,
        help='For testing: process only first N seconds of each video'
    )
    parser.add_argument(
        '-t', '--global-threshold',
        type=float,
        default=0.6,
        help='Similarity threshold for global face IDs across all sessions (default: 0.6)'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for processing'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for DeepFace processing (default: 32)'
    )
    parser.add_argument(
        '-c', '--clustering',
        choices=['threshold', 'dbscan'],
        default='threshold',
        help='Clustering method for global IDs (default: threshold)'
    )
    
    args = parser.parse_args()
    
    process_all_sessions(
        participant_dir=args.participant_dir,
        sampling_rate=args.sampling_rate,
        test_duration=args.test_duration,
        global_threshold=args.global_threshold,
        use_gpu=args.gpu,
        batch_size=args.batch_size,
        clustering_method=args.clustering,
    )

