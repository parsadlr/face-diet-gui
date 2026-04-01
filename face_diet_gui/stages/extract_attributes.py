"""
Extract Face Attributes with DeepFace

Reads face detections and adds demographic attributes:
- Age
- Gender
- Race
- Emotion

Input:  <session_dir>/face_detections.csv  (or --input-csv for BIDS path)
Output: Same CSV updated in-place with attribute columns

Can be run in parallel for multiple sessions!
"""

import argparse
import sys
from pathlib import Path

from face_diet_gui.processing.video_processor import process_video_stage2


def extract_attributes(
    session_dir: str,
    batch_size: int = 32,
    input_csv: str = None,
    video_path: str = None,
):
    """
    Extract attributes for faces in a single session.

    Parameters
    ----------
    session_dir : str
        Path to the DATA session directory (contains scenevideo.*).
        Used to find the video file when --video-path is not specified.
    batch_size : int
        Batch size for DeepFace processing (larger = more memory, faster)
    input_csv : str, optional
        Full path to the face-detections CSV (in derivatives_dir).
        When provided, this is used instead of <session_dir>/face_detections.csv.
        The CSV is updated in-place.
    video_path : str, optional
        Full path to the scene video file. When provided, used directly
        instead of searching session_dir for scenevideo.*.
    """
    session_path = Path(session_dir).resolve()

    # Resolve video path
    if video_path:
        resolved_video = str(Path(video_path).resolve())
    else:
        video_files = list(session_path.glob("scenevideo.*"))
        if not video_files:
            raise FileNotFoundError(f"No scenevideo file found in {session_dir}")
        resolved_video = str(video_files[0])

    # Resolve CSV paths
    if input_csv:
        csv_path = Path(input_csv).resolve()
    else:
        csv_path = session_path / "face_detections.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Stage 1 output not found: {csv_path}\n"
            f"Run detect_faces.py first!"
        )

    print("=" * 80)
    print("ATTRIBUTE EXTRACTION")
    print("=" * 80)
    print(f"Session: {session_path.name}")
    print(f"Video: {resolved_video}")
    print(f"Input CSV: {csv_path}")
    print(f"Output CSV: {csv_path} (in-place)")
    print(f"Batch size: {batch_size}")
    print()

    # Process with DeepFace
    print("Processing with DeepFace (chunked for memory efficiency)...")
    result = process_video_stage2(
        video_path=resolved_video,
        stage1_csv=str(csv_path),
        output_csv=str(csv_path),
        batch_size=batch_size,
        progress_callback=None,
    )

    print("\n" + "=" * 80)
    print("ATTRIBUTE EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Output: {csv_path}")
    print(f"Processed faces: {result['processed_faces']}")
    print("\nNext: Run cluster_face_ids.py to assign global face IDs")
    print("=" * 80)

    return {
        'session_dir': str(session_path),
        'output_csv': str(csv_path),
        'processed_faces': result['processed_faces'],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract demographic attributes for faces in a single session"
    )

    parser.add_argument(
        'session_dir',
        help='Path to data session directory (contains scenevideo.*)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for DeepFace processing (default: 32)'
    )
    parser.add_argument(
        '--input-csv',
        type=str,
        default=None,
        help='Full path to face-detections CSV (BIDS derivatives path). '
             'Overrides default <session_dir>/face_detections.csv. Updated in-place.'
    )
    parser.add_argument(
        '--video-path',
        type=str,
        default=None,
        help='Full path to scene video file. Overrides searching session_dir for scenevideo.*.'
    )

    args = parser.parse_args()

    try:
        extract_attributes(
            session_dir=args.session_dir,
            batch_size=args.batch_size,
            input_csv=args.input_csv,
            video_path=args.video_path,
        )
    except Exception as e:
        print(f"\n[ERROR] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
