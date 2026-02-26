"""
Face Detection with InsightFace

Detects faces in a single session and saves:
- Bounding boxes
- Face embeddings
- Pose angles
- Attended flag (if eye tracking available)

Output: <session_dir>/face_detections.csv

Can be run in parallel for multiple sessions!
"""

import argparse
import sys
from pathlib import Path

from face_diet_gui.processing.video_processor import collect_detections_insightface_only, write_csv_stage1
from face_diet_gui.processing.face_detection import initialize_detector


def detect_faces(
    session_dir: str,
    sampling_rate: int = 30,
    start_time: float = None,
    end_time: float = None,
    use_gpu: bool = False,
    min_confidence: float = 0.0,
):
    """
    Detect faces in a single session.
    
    Parameters
    ----------
    session_dir : str
        Path to session directory containing scenevideo.*
    sampling_rate : int
        Process every N frames
    start_time : float, optional
        Start time in seconds
    end_time : float, optional
        End time in seconds (or duration if start_time is None)
    use_gpu : bool
        Whether to use GPU
    min_confidence : float
        Minimum detection confidence (0.0-1.0), filters out low-confidence detections
    """
    session_path = Path(session_dir).resolve()
    
    # Find video file
    video_files = list(session_path.glob("scenevideo.*"))
    if not video_files:
        raise FileNotFoundError(f"No scenevideo file found in {session_dir}")
    
    video_path = str(video_files[0])
    
    # Find eye tracking file
    eye_tracking_path = session_path / "eye_tracking.tsv"
    if not eye_tracking_path.exists():
        eye_tracking_path = None
        print("WARNING: No eye_tracking.tsv found. 'attended' flag will be False for all faces.")
    else:
        eye_tracking_path = str(eye_tracking_path)
    
    # Handle random test window selection
    if end_time is not None and start_time is None:
        import cv2
        import random
        
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps
            cap.release()
            
            test_duration = end_time
            max_start = max(0, video_duration - test_duration)
            if max_start > 0:
                start_time = random.uniform(0, max_start)
                end_time = start_time + test_duration
                print(f"[RANDOM] Randomly selected test window: {start_time:.1f}s - {end_time:.1f}s")
            else:
                start_time = 0.0
                end_time = video_duration
    
    # Output path
    output_csv = str(session_path / "face_detections.csv")
    
    print("=" * 80)
    print("FACE DETECTION")
    print("=" * 80)
    print(f"Session: {session_path.name}")
    print(f"Video: {video_path}")
    print(f"Eye tracking: {eye_tracking_path if eye_tracking_path else 'Not found'}")
    print(f"Output: {output_csv}")
    print(f"Sampling rate: Every {sampling_rate} frame(s)")
    if start_time is not None and end_time is not None:
        print(f"Time range: {start_time:.1f}s - {end_time:.1f}s")
    print(f"GPU: {'Enabled' if use_gpu else 'Disabled'}")
    print(f"Min confidence: {min_confidence}")
    print()
    
    # Initialize detector
    print("Initializing detector...")
    detector = initialize_detector(use_gpu=use_gpu)
    
    # Detect faces
    print("\nDetecting faces with InsightFace...")
    detections = collect_detections_insightface_only(
        video_path=video_path,
        detector=detector,
        sampling_rate=sampling_rate,
        start_time=start_time,
        end_time=end_time,
        progress_callback=None,
        eye_tracking_path=eye_tracking_path,
    )
    
    print(f"\n[OK] Detected {len(detections)} face instances")
    
    # Filter by confidence if threshold > 0
    if min_confidence > 0.0:
        detections_before = len(detections)
        detections = [d for d in detections if d.get('confidence', 0.0) >= min_confidence]
        print(f"[OK] Filtered by confidence >= {min_confidence}: {detections_before} -> {len(detections)} faces")
    
    # Write CSV
    print(f"Writing to {output_csv}...")
    write_csv_stage1(output_csv, detections)
    
    print("\n" + "=" * 80)
    print("FACE DETECTION COMPLETE")
    print("=" * 80)
    print(f"Output: {output_csv}")
    print(f"Total faces: {len(detections)}")
    
    if eye_tracking_path:
        attended_count = sum(1 for d in detections if d.get('attended', False))
        if len(detections) > 0:
            print(f"Attended faces: {attended_count} ({100*attended_count/len(detections):.1f}%)")
        else:
            print(f"Attended faces: {attended_count} (0.0%)")
    
    print("\nNext: Run extract_attributes.py on this session")
    print("=" * 80)
    
    return {
        'session_dir': str(session_path),
        'output_csv': output_csv,
        'total_faces': len(detections),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect faces in a single session"
    )
    
    parser.add_argument(
        'session_dir',
        help='Path to session directory (contains scenevideo.*)'
    )
    parser.add_argument(
        '-s', '--sampling-rate',
        type=int,
        default=30,
        help='Process every N frames (default: 30)'
    )
    parser.add_argument(
        '--start-time',
        type=float,
        help='Start time in seconds'
    )
    parser.add_argument(
        '--end-time',
        type=float,
        help='End time in seconds (or duration if start-time not specified)'
    )
    parser.add_argument(
        '--test-duration',
        type=float,
        help='Randomly select N seconds to process (alternative to start/end)'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for processing'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.0,
        help='Minimum detection confidence (0.0-1.0, default: 0.0)'
    )
    
    args = parser.parse_args()
    
    # Handle test-duration as end-time
    end_time = args.end_time
    if args.test_duration:
        end_time = args.test_duration
    
    try:
        detect_faces(
            session_dir=args.session_dir,
            sampling_rate=args.sampling_rate,
            start_time=args.start_time,
            end_time=end_time,
            use_gpu=args.gpu,
            min_confidence=args.min_confidence,
        )
    except Exception as e:
        print(f"\n[ERROR] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

