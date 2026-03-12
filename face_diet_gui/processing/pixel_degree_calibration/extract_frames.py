import argparse
from pathlib import Path
from typing import Optional

import cv2


def extract_frames(
    video_path: str,
    output_dir: str,
    stride: int = 5,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> None:
    """
    Extract frames from a video and save them as images.

    Parameters
    ----------
    video_path : str
        Path to input calibration video.
    output_dir : str
        Directory where extracted frames will be saved.
    stride : int
        Save every N-th frame (default: 5).
    start_time : float, optional
        Start time in seconds (default: from beginning).
    end_time : float, optional
        End time in seconds (default: until end).
    """
    video_path = str(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = 0
    end_frame = total_frames

    if start_time is not None:
        start_frame = int(start_time * fps)
    if end_time is not None:
        end_frame = int(end_time * fps)

    start_frame = max(0, start_frame)
    end_frame = min(total_frames, end_frame)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    saved_count = 0

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}, total frames: {total_frames}")
    print(f"Extracting frames {start_frame} to {end_frame} (stride={stride})")
    print(f"Output dir: {out_dir}")

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % stride == 0:
            rel_idx = (frame_idx - start_frame) // stride
            frame_name = f"frame_{rel_idx:06d}.png"
            out_path = out_dir / frame_name
            cv2.imwrite(str(out_path), frame)
            saved_count += 1

            if saved_count % 50 == 0:
                print(f"Saved {saved_count} frames (up to frame {frame_idx})")

        frame_idx += 1

    cap.release()
    print(f"Done. Saved {saved_count} frames to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frames from a calibration video for "
            "pixel-to-degree mapping."
        )
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input calibration video (e.g., scenevideo.mp4).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory to save extracted frames.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Save every N-th frame (default: 5).",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Optional start time in seconds.",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="Optional end time in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_frames(
        video_path=args.video,
        output_dir=args.out_dir,
        stride=args.stride,
        start_time=args.start,
        end_time=args.end,
    )


if __name__ == "__main__":
    main()

