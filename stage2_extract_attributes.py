"""
Stage 2: Extract Face Attributes with DeepFace

Reads Stage 1 detections and adds demographic attributes:
- Age
- Gender
- Race
- Emotion

Input: <session_dir>/stage1_detections.csv
Output: <session_dir>/stage2_attributes.csv

Can be run in parallel for multiple sessions!
"""

import argparse
import sys
from pathlib import Path

from video_processor import process_video_stage2


def stage2_extract_attributes(
    session_dir: str,
    batch_size: int = 32,
    limit: int = None,
):
    """
    Stage 2: Extract attributes for faces in a single session.
    
    Parameters
    ----------
    session_dir : str
        Path to session directory containing stage1_detections.csv
    batch_size : int
        Batch size for DeepFace processing (larger = more memory, faster)
    """
    session_path = Path(session_dir).resolve()
    
    # Find video file
    video_files = list(session_path.glob("scenevideo.*"))
    if not video_files:
        raise FileNotFoundError(f"No scenevideo file found in {session_dir}")
    
    video_path = str(video_files[0])
    
    # Input/Output paths
    input_csv = session_path / "stage1_detections.csv"
    output_csv = session_path / "stage2_attributes.csv"
    
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Stage 1 output not found: {input_csv}\n"
            f"Run stage1_detect_faces.py first!"
        )
    
    print("=" * 80)
    print("STAGE 2: ATTRIBUTE EXTRACTION")
    print("=" * 80)
    print(f"Session: {session_path.name}")
    print(f"Video: {video_path}")
    print(f"Input: {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Batch size: {batch_size}")
    if limit:
        print(f"Limit: Processing only first {limit} faces (TEST MODE)")
    print()
    
    # If limit specified, create a temporary CSV with only first N rows
    stage1_csv_to_use = str(input_csv)
    temp_csv_created = False
    
    if limit:
        import pandas as pd
        df_full = pd.read_csv(input_csv)
        df_limited = df_full.head(limit)
        
        temp_csv = session_path / "stage1_detections_limited.csv"
        df_limited.to_csv(temp_csv, index=False)
        
        stage1_csv_to_use = str(temp_csv)
        temp_csv_created = True
        
        print(f"Created temporary CSV with {len(df_limited)} faces for testing")
    
    # Process with DeepFace
    print("Processing with DeepFace (chunked for memory efficiency)...")
    result = process_video_stage2(
        video_path=video_path,
        stage1_csv=stage1_csv_to_use,
        output_csv=str(output_csv),
        batch_size=batch_size,
        progress_callback=None,
    )
    
    # Cleanup temp file
    if temp_csv_created:
        import os
        os.remove(stage1_csv_to_use)
    
    print("\n" + "=" * 80)
    print("STAGE 2 COMPLETE")
    print("=" * 80)
    print(f"Output: {output_csv}")
    print(f"Processed faces: {result['processed_faces']}")
    print("\nNext: Run stage3_combine_sessions.py to assign global face IDs")
    print("=" * 80)
    
    return {
        'session_dir': str(session_path),
        'output_csv': str(output_csv),
        'processed_faces': result['processed_faces'],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2: Extract attributes for faces in a single session"
    )
    
    parser.add_argument(
        'session_dir',
        help='Path to session directory (contains stage1_detections.csv)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for DeepFace processing (default: 32)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='For testing: only process first N faces'
    )
    
    args = parser.parse_args()
    
    try:
        stage2_extract_attributes(
            session_dir=args.session_dir,
            batch_size=args.batch_size,
            limit=args.limit,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

