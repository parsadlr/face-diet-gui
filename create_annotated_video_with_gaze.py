"""
Create annotated video with face bounding boxes and gaze overlay.

Shows:
- Face bounding boxes with global IDs
- Gaze point (red dot)
- Face attributes (age, gender, emotion)
"""

import csv
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import sys


def load_gaze_data(eye_tracking_path):
    """Load gaze data from eye tracking TSV."""
    gaze_points = {}  # timestamp_ms -> (x, y)
    
    with open(eye_tracking_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        
        # Find column indices
        timestamp_idx = header.index('Recording timestamp [ms]')
        sensor_idx = header.index('Sensor')
        gaze_x_idx = header.index('Gaze point X [MCS px]')
        gaze_y_idx = header.index('Gaze point Y [MCS px]')
        
        for row in reader:
            if len(row) <= max(timestamp_idx, sensor_idx):
                continue
            
            try:
                if row[sensor_idx] == 'Eye Tracker':
                    ts_ms = float(row[timestamp_idx])
                    x = float(row[gaze_x_idx])
                    y = float(row[gaze_y_idx])
                    gaze_points[ts_ms] = (x, y)
            except (ValueError, IndexError):
                continue
    
    return gaze_points


def find_closest_gaze(gaze_points, timestamp_ms, max_diff_ms=50):
    """Find closest gaze point to a timestamp."""
    if not gaze_points:
        return None
    
    closest_ts = min(gaze_points.keys(), key=lambda t: abs(t - timestamp_ms))
    
    if abs(closest_ts - timestamp_ms) <= max_diff_ms:
        return gaze_points[closest_ts]
    
    return None


def draw_face_bbox(frame, detection, color=(0, 255, 0)):
    """Draw face bounding box and label."""
    x, y, w, h = detection['bbox']
    face_id = detection['face_id']
    confidence = detection.get('confidence', 0.0)
    
    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Prepare label
    label_lines = [f"{face_id} ({confidence:.2f})"]  # Face ID with confidence
    
    # Add attributes if available
    if detection.get('age') and pd.notna(detection['age']):
        age = int(detection['age'])
        gender = detection.get('gender', '')
        if gender:
            label_lines.append(f"{gender}, {age}")
    
    if detection.get('emotion') and pd.notna(detection['emotion']):
        label_lines.append(detection['emotion'])
    
    # Draw labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    y_offset = y - 10
    for line in label_lines:
        (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        
        # Background rectangle
        cv2.rectangle(
            frame,
            (x, y_offset - text_height - 5),
            (x + text_width + 5, y_offset),
            color,
            -1
        )
        
        # Text
        cv2.putText(
            frame,
            line,
            (x + 2, y_offset - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )
        
        y_offset -= (text_height + 8)
    
    return frame


def draw_gaze(frame, gaze_x, gaze_y, radius=15, color=(0, 0, 255)):
    """Draw gaze point."""
    gx, gy = int(gaze_x), int(gaze_y)
    
    # Draw outer circle
    cv2.circle(frame, (gx, gy), radius, color, 2)
    
    # Draw crosshair
    cv2.line(frame, (gx - radius, gy), (gx + radius, gy), color, 2)
    cv2.line(frame, (gx, gy - radius), (gx, gy + radius), color, 2)
    
    # Draw center dot
    cv2.circle(frame, (gx, gy), 3, color, -1)
    
    return frame


def create_annotated_video(
    video_path,
    faces_csv,
    eye_tracking_tsv,
    output_video,
    session_name=None,
):
    """
    Create annotated video with faces and gaze overlay.
    
    Parameters
    ----------
    video_path : str
        Path to input video
    faces_csv : str
        Path to CSV with face detections and global IDs
    eye_tracking_tsv : str
        Path to eye tracking data
    output_video : str
        Path to output annotated video
    session_name : str, optional
        If processing combined CSV, filter by session name
    """
    print("=" * 80)
    print("CREATING ANNOTATED VIDEO WITH GAZE")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"Faces: {faces_csv}")
    print(f"Gaze: {eye_tracking_tsv}")
    print(f"Output: {output_video}")
    if session_name:
        print(f"Session filter: {session_name}")
    print()
    
    # Load face detections
    print("Loading face detections...")
    df = pd.read_csv(faces_csv)
    
    # Filter by session if specified
    if session_name and 'session_name' in df.columns:
        df = df[df['session_name'] == session_name]
        print(f"Filtered to session {session_name}: {len(df)} detections")
    
    # Group by frame
    detections_by_frame = {}
    for _, row in df.iterrows():
        frame_num = int(row['frame_number'])
        if frame_num not in detections_by_frame:
            detections_by_frame[frame_num] = []
        
        detections_by_frame[frame_num].append({
            'bbox': (int(row['x']), int(row['y']), int(row['w']), int(row['h'])),
            'face_id': row['face_id'],
            'confidence': float(row['confidence']) if 'confidence' in row and pd.notna(row['confidence']) else 0.0,
            'age': row.get('age'),
            'gender': row.get('gender'),
            'emotion': row.get('emotion'),
            'timestamp_ms': float(row['time_seconds']) * 1000,
        })
    
    print(f"Loaded {len(df)} face detections across {len(detections_by_frame)} frames")
    
    # Load gaze data
    print("Loading gaze data...")
    gaze_points = load_gaze_data(eye_tracking_tsv)
    print(f"Loaded {len(gaze_points)} gaze points")
    
    # Open video
    print("\nProcessing video...")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate timestamp for this frame
        timestamp_ms = (frame_number / fps) * 1000
        
        # Draw face bounding boxes
        if frame_number in detections_by_frame:
            for detection in detections_by_frame[frame_number]:
                frame = draw_face_bbox(frame, detection)
        
        # Draw gaze point
        gaze = find_closest_gaze(gaze_points, timestamp_ms)
        if gaze:
            gx, gy = gaze
            # Check if gaze is within frame bounds
            if 0 <= gx < width and 0 <= gy < height:
                frame = draw_gaze(frame, gx, gy)
        
        # Draw timestamp and frame info
        info_text = f"Frame: {frame_number} | Time: {timestamp_ms/1000:.2f}s"
        cv2.putText(
            frame,
            info_text,
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        out.write(frame)
        
        # Progress update
        if frame_number % 100 == 0:
            percent = int(100 * frame_number / total_frames)
            print(f"  [{percent:3d}%] Processing frame {frame_number}/{total_frames}")
        
        frame_number += 1
    
    cap.release()
    out.release()
    
    print(f"\n✓ Annotated video created: {output_video}")
    print("=" * 80)


def create_annotated_videos_for_participant(participant_dir, output_in_session_folders=True):
    """Create annotated videos for all sessions in a participant directory."""
    participant_path = Path(participant_dir).resolve()
    
    print("=" * 80)
    print(f"CREATING ANNOTATED VIDEOS: {participant_path.name}")
    print("=" * 80)
    print(f"Participant directory: {participant_path}")
    if output_in_session_folders:
        print(f"Output: In each session folder (scenevideo_annotated.mp4)")
    else:
        print(f"Output: In annotated_videos/ directory")
    print()
    
    # Check if we have combined CSV
    combined_csv = participant_path / "faces_combined.csv"
    use_combined = combined_csv.exists()
    
    # Find all sessions
    session_dirs = []
    print(f"Scanning for sessions in: {participant_path}")
    
    try:
        for subdir in sorted(participant_path.iterdir()):
            if not subdir.is_dir():
                continue
            
            print(f"  Checking: {subdir.name}")
            video_files = list(subdir.glob("scenevideo.*"))
            eye_tracking_file = subdir / "eye_tracking.tsv"
            
            if video_files:
                print(f"    Found video: {video_files[0].name}")
            if eye_tracking_file.exists():
                print(f"    Found eye tracking: {eye_tracking_file.name}")
            
            if video_files and eye_tracking_file.exists():
                session_dirs.append(subdir)
                print(f"    ✓ Valid session")
    except Exception as e:
        print(f"Error scanning directory: {e}")
        return
    
    if not session_dirs:
        print("\n❌ No sessions with video and eye tracking found")
        return
    
    print(f"Found {len(session_dirs)} session(s)")
    if use_combined:
        print(f"Using combined CSV: {combined_csv}")
    print()
    
    # Process each session
    for i, session_dir in enumerate(session_dirs, 1):
        session_name = session_dir.name
        print(f"\n{'='*80}")
        print(f"Session {i}/{len(session_dirs)}: {session_name}")
        print(f"{'='*80}\n")
        
        video_path = list(session_dir.glob("scenevideo.*"))[0]
        eye_tracking_path = session_dir / "eye_tracking.tsv"
        
        # Save in session folder if requested
        if output_in_session_folders:
            output_video = session_dir / "scenevideo_annotated.mp4"
        else:
            output_dir = participant_path / "annotated_videos"
            output_dir.mkdir(exist_ok=True)
            output_video = output_dir / f"{session_name}_annotated.mp4"
        
        # Determine which CSV to use
        if use_combined:
            faces_csv = str(combined_csv)
            session_filter = session_name
        else:
            # Use individual session CSV
            faces_csv = str(session_dir / "stage2_attributes.csv")
            if not Path(faces_csv).exists():
                faces_csv = str(session_dir / "faces_detections.csv")
            if not Path(faces_csv).exists():
                faces_csv = str(session_dir / "faces.csv")
            session_filter = None
        
        try:
            create_annotated_video(
                video_path=str(video_path),
                faces_csv=faces_csv,
                eye_tracking_tsv=str(eye_tracking_path),
                output_video=str(output_video),
                session_name=session_filter,
            )
            print(f"✓ Session {session_name} completed")
        except Exception as e:
            print(f"✗ Error processing session {session_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("ALL ANNOTATED VIDEOS CREATED")
    print("=" * 80)
    if output_in_session_folders:
        print(f"Annotated videos saved in each session folder as: scenevideo_annotated.mp4")
    else:
        print(f"Output directory: {participant_path / 'annotated_videos'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create annotated videos with face bounding boxes and gaze overlay"
    )
    
    parser.add_argument(
        'participant_dir',
        help='Path to participant directory containing session folders'
    )
    parser.add_argument(
        '--separate-folder',
        action='store_true',
        help='Save in annotated_videos/ folder instead of session folders'
    )
    
    args = parser.parse_args()
    
    create_annotated_videos_for_participant(
        participant_dir=args.participant_dir,
        output_in_session_folders=not args.separate_folder,
    )

