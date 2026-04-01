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
import random
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from face_diet_gui.processing.video_processor import collect_detections_insightface_only, write_csv_stage1
from face_diet_gui.processing.face_detection import initialize_detector


def _video_info(video_path: str):
    """Return (fps, total_frames, duration_seconds) for a video file."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = total_frames / fps if fps > 0 else 0.0
    return fps, total_frames, duration


def _intervals_overlap(a_start: float, a_end: float, accepted: List[Tuple[float, float]]) -> bool:
    """Return True if [a_start, a_end) overlaps any interval in *accepted*."""
    for s, e in accepted:
        if a_start < e and a_end > s:
            return True
    return False


# After each accepted interval (and before the first), stop if this many attempts in a row
# fail to yield another qualifying interval (overlap skip or low face fraction).
MAX_INTERVAL_SEARCH_ATTEMPTS_PER_SUCCESS = 50


def _count_processed_frames(start_time: float, end_time: float, fps: float, sampling_rate: int) -> int:
    """
    Compute how many frames will be processed by collect_detections_insightface_only.

    Uses the same start/end frame truncation behavior as the collector:
      start_frame = int(start_time * fps)
      end_frame = int(end_time * fps)   (exclusive)
      process when (frame_number - start_frame) % sampling_rate == 0
    """
    start_frame = max(0, int(start_time * fps))
    end_frame = max(start_frame, int(end_time * fps))
    span = end_frame - start_frame
    if span <= 0:
        return 0
    return ((span - 1) // sampling_rate) + 1


def detect_faces_with_intervals(
    video_path: str,
    detector,
    eye_tracking_path: Optional[str],
    sampling_rate: int,
    min_confidence: float,
    interval_length: float,
    num_intervals: int,
    min_face_fraction: float,
) -> List[Dict]:
    """
    Collect face detections from randomly sampled, non-overlapping video intervals.

    For each candidate interval:
      1. Pre-scan at ~1 fps to estimate what fraction of frames contain at least one face.
      2. Accept the interval if face_fraction >= min_face_fraction.
      3. Repeat until *num_intervals* qualifying intervals have been found.

    Full-resolution detection (using *sampling_rate*) is then run on each accepted
    interval. Results from all intervals are merged and sorted by time.

    Parameters
    ----------
    video_path : str
        Path to video file.
    detector
        Initialised InsightFace detector.
    eye_tracking_path : str or None
        Path to eye_tracking.tsv (for attended flag).
    sampling_rate : int
        Downsampling factor for full detection (1 = every frame).
    min_confidence : float
        Minimum detection confidence threshold.
    interval_length : float
        Length of each interval in seconds.
    num_intervals : int
        Number of qualifying intervals to collect.
    min_face_fraction : float
        Minimum fraction of pre-scanned frames that must contain at least one face.
    """
    fps, total_frames, duration = _video_info(video_path)

    if interval_length <= 0:
        raise ValueError("interval_length must be positive")
    if interval_length > duration:
        raise ValueError(
            f"Interval length {interval_length:.1f}s exceeds video duration {duration:.1f}s"
        )

    # Sampling rate for the quick face-check pre-scan (~1 fps)
    pre_scan_rate = max(1, int(fps))

    accepted: List[Tuple[float, float]] = []
    attempt = 0
    attempts_since_success = 0

    print(
        f"[INTERVAL MODE] Searching for {num_intervals} intervals x {interval_length:.1f}s, "
        f"min {min_face_fraction * 100:.0f}% face frames "
        f"(max {MAX_INTERVAL_SEARCH_ATTEMPTS_PER_SUCCESS} attempts per interval without success)"
    )
    print(f"  Video duration: {duration:.1f}s | pre-scan rate: every {pre_scan_rate} frames (~1 fps)")

    while len(accepted) < num_intervals:
        if attempts_since_success >= MAX_INTERVAL_SEARCH_ATTEMPTS_PER_SUCCESS:
            raise RuntimeError(
                f"Interval sampling: could not find qualifying interval "
                f"{len(accepted) + 1}/{num_intervals} after "
                f"{MAX_INTERVAL_SEARCH_ATTEMPTS_PER_SUCCESS} consecutive attempts "
                f"(non-overlapping candidates that pass the min face fraction "
                f"{min_face_fraction * 100:.0f}%). "
                f"Try lowering the min face fraction, shortening the interval length, "
                f"or disabling interval sampling."
            )

        attempt += 1

        # Random non-overlapping candidate
        max_start = duration - interval_length
        if max_start <= 0:
            candidate_start = 0.0
        else:
            candidate_start = random.uniform(0.0, max_start)
        candidate_end = candidate_start + interval_length

        if _intervals_overlap(candidate_start, candidate_end, accepted):
            attempts_since_success += 1
            continue

        # Pre-scan: quick face detection at ~1 fps
        pre_detections = collect_detections_insightface_only(
            video_path=video_path,
            detector=detector,
            sampling_rate=pre_scan_rate,
            start_time=candidate_start,
            end_time=candidate_end,
            progress_callback=None,
            eye_tracking_path=None,  # No gaze needed for the pre-scan
        )

        frames_with_face = len(set(d["frame_number"] for d in pre_detections))
        # Total frames sampled during pre-scan
        pre_scan_frames = max(
            1,
            _count_processed_frames(
                start_time=candidate_start,
                end_time=candidate_end,
                fps=fps,
                sampling_rate=pre_scan_rate,
            ),
        )
        face_fraction = frames_with_face / pre_scan_frames

        status = "ACCEPTED" if face_fraction >= min_face_fraction else "rejected"
        print(
            f"  Attempt {attempt}: [{candidate_start:.1f}s-{candidate_end:.1f}s] "
            f"face fraction {face_fraction * 100:.0f}% ({frames_with_face}/{pre_scan_frames}) -> {status}"
        )

        if face_fraction >= min_face_fraction:
            accepted.append((candidate_start, candidate_end))
            attempts_since_success = 0
            print(f"  -> Collected {len(accepted)}/{num_intervals} intervals")
        else:
            attempts_since_success += 1

    # Sort accepted intervals chronologically
    accepted.sort(key=lambda x: x[0])

    # Compute total frames for global progress display
    global_total = sum(
        max(
            1,
            _count_processed_frames(
                start_time=s,
                end_time=e,
                fps=fps,
                sampling_rate=sampling_rate,
            ),
        )
        for s, e in accepted
    )

    print(
        f"\n[INTERVAL MODE] Running full detection on {len(accepted)} interval(s) "
        f"(total ~{global_total} frames at rate={sampling_rate})"
    )

    all_detections: List[Dict] = []
    global_offset = 0

    for i, (start_time, end_time) in enumerate(accepted):
        interval_frames = max(
            1,
            _count_processed_frames(
                start_time=start_time,
                end_time=end_time,
                fps=fps,
                sampling_rate=sampling_rate,
            ),
        )
        print(
            f"\n  Interval {i + 1}/{len(accepted)}: "
            f"[{start_time:.1f}s-{end_time:.1f}s] (~{interval_frames} frames)"
        )

        interval_detections = collect_detections_insightface_only(
            video_path=video_path,
            detector=detector,
            sampling_rate=sampling_rate,
            start_time=start_time,
            end_time=end_time,
            progress_callback=None,
            eye_tracking_path=eye_tracking_path,
            global_offset=global_offset,
            global_total=global_total,
        )
        all_detections.extend(interval_detections)
        global_offset += interval_frames

    # Final global progress line so the GUI progress bar reaches 100 %
    print(f"  [{100:3d}%] {global_total}/{global_total} frames, {len(all_detections)} faces")

    # Sort all detections by time
    all_detections.sort(key=lambda d: d["time_seconds"])

    return all_detections


def detect_faces(
    session_dir: str,
    sampling_rate: int = 30,
    start_time: float = None,
    end_time: float = None,
    use_gpu: bool = False,
    min_confidence: float = 0.0,
    use_interval_sampling: bool = False,
    interval_length: float = 30.0,
    num_intervals: int = 5,
    min_face_fraction: float = 0.1,
    output_csv_path: Optional[str] = None,
):
    """
    Detect faces in a single session.

    Parameters
    ----------
    session_dir : str
        Path to session directory containing scenevideo.*
    sampling_rate : int
        Process every N frames (downsampling factor).
    start_time : float, optional
        Start time in seconds (normal mode only).
    end_time : float, optional
        End time in seconds (normal mode only).
    use_gpu : bool
        Whether to use GPU.
    min_confidence : float
        Minimum detection confidence (0.0-1.0).
    use_interval_sampling : bool
        When True, randomly sample *num_intervals* qualifying intervals instead of
        processing the whole video sequentially.
    interval_length : float
        Length of each interval in seconds (interval sampling mode).
    num_intervals : int
        Number of qualifying intervals to collect (interval sampling mode).
    min_face_fraction : float
        Minimum fraction of pre-scanned frames with at least one face (interval sampling mode).
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

    # For normal mode: handle a random test window when only end_time is given
    if not use_interval_sampling and end_time is not None and start_time is None:
        import cv2
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

    if output_csv_path:
        output_csv = str(Path(output_csv_path).resolve())
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    else:
        output_csv = str(session_path / "face_detections.csv")

    print("=" * 80)
    print("FACE DETECTION")
    print("=" * 80)
    print(f"Session: {session_path.name}")
    print(f"Video: {video_path}")
    print(f"Eye tracking: {eye_tracking_path if eye_tracking_path else 'Not found'}")
    print(f"Output: {output_csv}")
    print(f"Sampling rate: Every {sampling_rate} frame(s)")
    if use_interval_sampling:
        print(f"Mode: Interval sampling ({num_intervals} x {interval_length:.1f}s, "
              f"min face fraction {min_face_fraction * 100:.0f}%)")
    elif start_time is not None and end_time is not None:
        print(f"Time range: {start_time:.1f}s - {end_time:.1f}s")
    print(f"GPU: {'Enabled' if use_gpu else 'Disabled'}")
    print(f"Min confidence: {min_confidence}")
    print()

    # Initialise detector
    print("Initializing detector...")
    detector = initialize_detector(use_gpu=use_gpu)

    # Detect faces
    if use_interval_sampling:
        print("\nDetecting faces with InsightFace (interval sampling mode)...")
        detections = detect_faces_with_intervals(
            video_path=video_path,
            detector=detector,
            eye_tracking_path=eye_tracking_path,
            sampling_rate=sampling_rate,
            min_confidence=min_confidence,
            interval_length=interval_length,
            num_intervals=num_intervals,
            min_face_fraction=min_face_fraction,
        )
    else:
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

    # Filter by confidence
    if min_confidence > 0.0:
        detections_before = len(detections)
        detections = [d for d in detections if d.get("confidence", 0.0) >= min_confidence]
        print(
            f"[OK] Filtered by confidence >= {min_confidence}: "
            f"{detections_before} -> {len(detections)} faces"
        )

    # Write CSV
    print(f"Writing to {output_csv}...")
    write_csv_stage1(output_csv, detections)

    print("\n" + "=" * 80)
    print("FACE DETECTION COMPLETE")
    print("=" * 80)
    print(f"Output: {output_csv}")
    print(f"Total faces: {len(detections)}")

    if eye_tracking_path:
        attended_count = sum(1 for d in detections if d.get("attended", False))
        if len(detections) > 0:
            print(f"Attended faces: {attended_count} ({100 * attended_count / len(detections):.1f}%)")
        else:
            print(f"Attended faces: {attended_count} (0.0%)")

    print("\nNext: Run extract_attributes.py on this session")
    print("=" * 80)

    return {
        "session_dir": str(session_path),
        "output_csv": output_csv,
        "total_faces": len(detections),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect faces in a single session")

    parser.add_argument(
        "session_dir",
        help="Path to session directory (contains scenevideo.*)"
    )
    parser.add_argument(
        "-s", "--sampling-rate",
        type=int,
        default=1,
        help="Process every N frames / downsampling factor (default: 1)"
    )
    parser.add_argument(
        "--start-time",
        type=float,
        help="Start time in seconds (normal mode)"
    )
    parser.add_argument(
        "--end-time",
        type=float,
        help="End time in seconds (normal mode, or duration if start-time not specified)"
    )
    parser.add_argument(
        "--test-duration",
        type=float,
        help="Randomly select N seconds to process (alternative to start/end)"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for processing"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum detection confidence (0.0-1.0, default: 0.0)"
    )
    # Interval sampling
    parser.add_argument(
        "--use-interval-sampling",
        action="store_true",
        help="Randomly sample qualifying intervals instead of processing the full video"
    )
    parser.add_argument(
        "--interval-length",
        type=float,
        default=30.0,
        help="Length of each interval in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--num-intervals",
        type=int,
        default=5,
        help="Number of qualifying intervals to collect (default: 5)"
    )
    parser.add_argument(
        "--min-face-fraction",
        type=float,
        default=0.1,
        help="Minimum fraction of pre-scanned frames with at least one face (default: 0.1)"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Full path for the output face-detections CSV (overrides default <session_dir>/face_detections.csv)"
    )

    args = parser.parse_args()

    # --test-duration as alias for --end-time
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
            use_interval_sampling=args.use_interval_sampling,
            interval_length=args.interval_length,
            num_intervals=args.num_intervals,
            min_face_fraction=args.min_face_fraction,
            output_csv_path=args.output_csv,
        )
    except Exception as e:
        print(f"\n[ERROR] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
