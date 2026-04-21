"""
overlay_gaze_on_video.py

Loads a session video and an eye-tracking TSV file, then renders the gaze
location as a dot overlay on every frame and saves the result to a new video.

Gaze loading and frame–timestamp mapping are intentionally identical to the
logic used by the Tab 1 (Face Detection) pipeline in face_diet_gui, so this
script can be used to visually verify that gaze data is being read and
synchronised correctly.

Usage
-----
    python overlay_gaze_on_video.py
        --video   path/to/scenevideo.mp4
        --gaze    path/to/eye_tracking.tsv
        --output  path/to/output.mp4
        [--start  <seconds>]
        [--end    <seconds>]
        [--max-diff-ms  <ms>]   # max gaze–frame time gap to accept (default 50)
        [--dot-radius   <px>]   # radius of the drawn gaze dot (default 20)
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running from the repo root without installing the package.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

import cv2

# Import the exact same helpers used by the Tab 1 pipeline.
from face_diet_gui.processing.video_processor import (
    load_gaze_data_for_video,
    find_closest_gaze,
)


# ---------------------------------------------------------------------------
# Overlay logic
# ---------------------------------------------------------------------------

def overlay_gaze(
    video_path: str,
    eye_tracking_path: str,
    output_path: str,
    start_time: float | None = None,
    end_time: float | None = None,
    max_diff_ms: float = 50.0,
    dot_radius: int = 20,
) -> None:
    """
    Read every frame inside [start_time, end_time], look up the closest gaze
    sample using the same `find_closest_gaze` call as the pipeline, draw a dot
    at that position, and write the result to *output_path*.

    Parameters
    ----------
    video_path : str
        Path to the input scene video.
    eye_tracking_path : str
        Path to the Tobii Pro Lab ``eye_tracking.tsv`` export.
    output_path : str
        Destination path for the rendered video (mp4).
    start_time : float, optional
        Start of the clip in seconds (defaults to beginning of video).
    end_time : float, optional
        End of the clip in seconds (defaults to end of video).
    max_diff_ms : float
        Maximum acceptable time gap between a video frame and its closest gaze
        sample. Frames with no gaze sample within this window are written
        without a dot.  Mirrors the default used by `find_closest_gaze`.
    dot_radius : int
        Radius in pixels of the drawn gaze dot.
    """

    # ------------------------------------------------------------------
    # 1.  Load gaze data – identical call to the pipeline.
    # ------------------------------------------------------------------
    print(f"Loading gaze data from: {eye_tracking_path}")
    gaze_data = load_gaze_data_for_video(eye_tracking_path)
    if gaze_data:
        print(f"  Loaded {len(gaze_data)} gaze samples")
        ts_values = list(gaze_data.keys())
        print(f"  Timestamp range: {min(ts_values):.0f} ms – {max(ts_values):.0f} ms")
    else:
        print("  WARNING: No gaze data loaded – output video will have no dots")

    # ------------------------------------------------------------------
    # 2.  Open video.
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo: {total_frames} frames, {fps:.2f} fps, {width}x{height}")

    # ------------------------------------------------------------------
    # 3.  Determine frame range – identical arithmetic to the pipeline.
    # ------------------------------------------------------------------
    start_frame = 0
    end_frame = total_frames

    if start_time is not None:
        start_frame = int(start_time * fps)
    if end_time is not None:
        end_frame = int(end_time * fps)

    start_frame = max(0, start_frame)
    end_frame = min(total_frames, end_frame)

    print(f"Processing frames {start_frame} – {end_frame}  "
          f"({start_frame / fps:.2f}s – {end_frame / fps:.2f}s)")

    # ------------------------------------------------------------------
    # 4.  Set up output writer.
    # ------------------------------------------------------------------
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video writer for: {output_path}")

    # ------------------------------------------------------------------
    # 5.  Seek to start frame (same as pipeline).
    # ------------------------------------------------------------------
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # ------------------------------------------------------------------
    # 6.  Frame loop – identical timestamp formula to the pipeline:
    #       timestamp_ms = (frame_number / fps) * 1000
    # ------------------------------------------------------------------
    frame_number = start_frame
    frames_written = 0
    gaze_hit = 0
    gaze_miss = 0

    while frame_number < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Compute frame timestamp in ms – same formula as the pipeline.
        timestamp_ms = (frame_number / fps) * 1000

        # Look up the closest gaze sample – same call as the pipeline.
        gaze = find_closest_gaze(gaze_data, timestamp_ms, max_diff_ms=max_diff_ms)

        if gaze is not None:
            gaze_x, gaze_y = gaze
            cx, cy = int(round(gaze_x)), int(round(gaze_y))

            # Outer white ring for contrast.
            cv2.circle(frame, (cx, cy), dot_radius + 3, (255, 255, 255), 3, cv2.LINE_AA)
            # Inner filled red dot.
            cv2.circle(frame, (cx, cy), dot_radius, (0, 0, 220), -1, cv2.LINE_AA)
            # Tiny centre black dot so fine position is visible.
            cv2.circle(frame, (cx, cy), max(2, dot_radius // 5), (0, 0, 0), -1, cv2.LINE_AA)
            gaze_hit += 1
        else:
            gaze_miss += 1

        # Timestamp text (frame number + seconds)
        label = f"frame {frame_number}  t={timestamp_ms / 1000:.3f}s"
        cv2.putText(
            frame, label, (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, label, (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )

        writer.write(frame)
        frames_written += 1
        frame_number += 1

        if frames_written % 100 == 0:
            pct = (frame_number - start_frame) / max(1, end_frame - start_frame) * 100
            print(f"  {pct:5.1f}%  frame {frame_number}/{end_frame}  "
                  f"gaze hits so far: {gaze_hit}")

    cap.release()
    writer.release()

    total = frames_written or 1
    print(f"\nDone.")
    print(f"  Frames written : {frames_written}")
    print(f"  Gaze hits      : {gaze_hit}  ({gaze_hit / total * 100:.1f}%)")
    print(f"  Gaze misses    : {gaze_miss}  ({gaze_miss / total * 100:.1f}%)")
    print(f"  Output         : {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overlay gaze location on a scene video for pipeline verification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, help="Path to input scene video")
    p.add_argument("--gaze", required=True, help="Path to eye_tracking.tsv")
    p.add_argument("--output", required=True, help="Path for the output video")
    p.add_argument(
        "--start", type=float, default=None,
        help="Start time in seconds (default: beginning of video)",
    )
    p.add_argument(
        "--end", type=float, default=None,
        help="End time in seconds (default: end of video)",
    )
    p.add_argument(
        "--max-diff-ms", type=float, default=50.0,
        help="Max gaze–frame time gap in ms to accept a gaze sample",
    )
    p.add_argument(
        "--dot-radius", type=int, default=20,
        help="Radius in pixels of the drawn gaze dot",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    overlay_gaze(
        video_path=args.video,
        eye_tracking_path=args.gaze,
        output_path=args.output,
        start_time=args.start,
        end_time=args.end,
        max_diff_ms=args.max_diff_ms,
        dot_radius=args.dot_radius,
    )
