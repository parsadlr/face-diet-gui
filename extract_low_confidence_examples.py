"""
Extract frames with lowest confidence face detections or closest to a target confidence.

Usage:
    python extract_low_confidence_examples.py <data_dir> [--num_examples 5] [--target_confidence 0.5]

Examples:
    # Extract 5 lowest confidence examples
    python extract_low_confidence_examples.py "C:\\Users\\parsadlr\\Documents\\Parsa\\Projects\\fps_experiment\\data\\pilot-1\\20251127T195837Z" --num_examples 5
    
    # Extract 10 examples closest to confidence 0.5
    python extract_low_confidence_examples.py "C:\\Users\\parsadlr\\Documents\\Parsa\\Projects\\fps_experiment\\data\\pilot-1\\20251127T195837Z" --target_confidence 0.5 --num_examples 10
"""

import argparse
import csv
from pathlib import Path
import cv2
import pandas as pd


def extract_low_confidence_examples(data_dir, num_examples=5, target_confidence=None):
    """
    Extract and save frames with lowest confidence detections or closest to target confidence.
    
    Parameters
    ----------
    data_dir : str
        Path to directory containing stage1_detections.csv and scenevideo.mp4
    num_examples : int
        Number of examples to extract
    target_confidence : float, optional
        If specified, extract examples closest to this confidence value.
        If None, extract lowest confidence examples.
    """
    data_path = Path(data_dir)
    
    # Find CSV file
    csv_path = data_path / "stage1_detections.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Find video file
    video_files = list(data_path.glob("scenevideo.*"))
    if not video_files:
        raise FileNotFoundError(f"No scenevideo file found in {data_dir}")
    video_path = video_files[0]
    
    print(f"Reading CSV: {csv_path}")
    print(f"Video: {video_path}")
    
    # Read CSV with pandas for easier manipulation
    df = pd.read_csv(csv_path)
    
    # Display CSV structure
    print(f"\nCSV columns: {list(df.columns)}")
    print(f"Total detections: {len(df)}")
    print(f"\nFirst row:")
    print(df.head(1).to_string())
    print(f"\nConfidence stats:")
    print(df['confidence'].describe())
    
    # Select examples based on mode
    if target_confidence is not None:
        # Find examples closest to target confidence
        df['conf_distance'] = abs(df['confidence'] - target_confidence)
        df_sorted = df.sort_values('conf_distance', ascending=True)
        selected = df_sorted.head(num_examples)
        mode_desc = f"{num_examples} examples closest to confidence {target_confidence:.3f}"
    else:
        # Get lowest confidence examples
        df_sorted = df.sort_values('confidence', ascending=True)
        selected = df_sorted.head(num_examples)
        mode_desc = f"{num_examples} lowest confidence examples"
    
    print(f"\n{'='*80}")
    print(f"Extracting {mode_desc}:")
    print(f"{'='*80}")
    print(selected[['frame_number', 'time_seconds', 'confidence', 'x', 'y', 'w', 'h']].to_string(index=False))
    
    # Create output directory
    if target_confidence is not None:
        output_dir = data_path / f"confidence_examples_{target_confidence:.3f}"
    else:
        output_dir = data_path / "low_confidence_examples"
    output_dir.mkdir(exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo info: {fps:.2f} fps, {total_frames} frames")
    print(f"\nExtracting frames to: {output_dir}")
    
    # Extract each frame
    for idx, row in selected.iterrows():
        frame_num = int(row['frame_number'])
        confidence = row['confidence']
        x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
        time_sec = row['time_seconds']
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"  WARNING: Could not read frame {frame_num}")
            continue
        
        # Draw bounding box (red for low confidence)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Add confidence text
        label = f"conf: {confidence:.3f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), (0, 0, 255), -1)
        
        # Draw text
        cv2.putText(frame, label, (x, y - 5), font, font_scale, (255, 255, 255), thickness)
        
        # Save frame
        output_filename = f"frame_{frame_num:06d}_conf_{confidence:.4f}.jpg"
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), frame)
        
        print(f"  ✓ Saved: {output_filename} (time: {time_sec:.2f}s)")
    
    cap.release()
    
    print(f"\n{'='*80}")
    print(f"Done! Extracted {num_examples} examples to: {output_dir}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames with lowest confidence or closest to target confidence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract 5 lowest confidence examples
    python extract_low_confidence_examples.py "C:\\path\\to\\data\\dir"
    
    # Extract 10 lowest confidence examples
    python extract_low_confidence_examples.py "C:\\path\\to\\data\\dir" --num_examples 10
    
    # Extract 10 examples closest to confidence 0.5
    python extract_low_confidence_examples.py "C:\\path\\to\\data\\dir" --target_confidence 0.5 --num_examples 10
        """
    )
    
    parser.add_argument(
        'data_dir',
        type=str,
        help='Path to directory containing stage1_detections.csv and scenevideo.mp4'
    )
    
    parser.add_argument(
        '--num_examples',
        type=int,
        default=5,
        help='Number of examples to extract (default: 5)'
    )
    
    parser.add_argument(
        '--target_confidence',
        type=float,
        default=None,
        help='Target confidence value. If specified, extracts examples closest to this value. '
             'If not specified, extracts lowest confidence examples.'
    )
    
    args = parser.parse_args()
    
    extract_low_confidence_examples(args.data_dir, args.num_examples, args.target_confidence)


if __name__ == "__main__":
    main()
