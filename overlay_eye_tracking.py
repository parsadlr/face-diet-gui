"""
Overlay eye tracking data on annotated video.

Displays gaze position as a circle with a fading trail.
"""

import argparse
import csv
import sys
from typing import List, Tuple

import cv2
import numpy as np


class GazePoint:
    """Represents a gaze point with fading."""
    
    def __init__(self, x: float, y: float, timestamp: float):
        self.x = x
        self.y = y
        self.timestamp = timestamp


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
        description="Overlay eye tracking data on video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to input video file'
    )
    
    parser.add_argument(
        'eye_tracking_path',
        type=str,
        help='Path to eye tracking TSV file'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Path to output video file'
    )
    
    # Gaze visualization parameters
    parser.add_argument(
        '--gaze-color',
        type=str,
        default='255,255,255',
        help='Current gaze circle color in BGR format (B,G,R), e.g., "255,255,255" for white'
    )
    
    parser.add_argument(
        '--trail-color',
        type=str,
        default='0,0,255',
        help='Trail circle color in BGR format (B,G,R), e.g., "0,0,255" for red'
    )
    
    parser.add_argument(
        '--gaze-size',
        type=int,
        default=30,
        help='Radius of gaze circle in pixels'
    )
    
    parser.add_argument(
        '--gaze-opacity',
        type=float,
        default=0.7,
        help='Opacity of current gaze point (0.0 to 1.0)'
    )
    
    parser.add_argument(
        '--trail-length',
        type=int,
        default=10,
        help='Number of previous gaze points to show as trail'
    )
    
    parser.add_argument(
        '--fade-rate',
        type=float,
        default=0.1,
        help='Opacity decrease per trail point (0.0 to 1.0)'
    )
    
    parser.add_argument(
        '--initial-fade',
        type=float,
        default=0.3,
        help='Initial opacity drop from current to first trail point (0.0 to 1.0)'
    )
    
    parser.add_argument(
        '--thickness',
        type=int,
        default=3,
        help='Thickness of trail circle outline in pixels'
    )
    
    parser.add_argument(
        '--current-thickness',
        type=int,
        default=2,
        help='Thickness of current gaze circle outline in pixels'
    )
    
    parser.add_argument(
        '--time-offset',
        type=float,
        default=0.0,
        help='Time offset to sync eye tracking with video (in seconds)'
    )
    
    parser.add_argument(
        '--start-time',
        type=str,
        default=None,
        help='Start time in format MM:SS or HH:MM:SS (e.g., "2:00")'
    )
    
    parser.add_argument(
        '--end-time',
        type=str,
        default=None,
        help='End time in format MM:SS or HH:MM:SS (e.g., "2:30")'
    )
    
    return parser.parse_args()


def load_eye_tracking_data(tsv_path: str) -> Tuple[List[Tuple[float, float, float]], float]:
    """
    Load eye tracking data from TSV file.
    
    Parameters
    ----------
    tsv_path : str
        Path to TSV file
    
    Returns
    -------
    Tuple[List[Tuple[float, float, float]], float]
        List of (timestamp_ms, gaze_x, gaze_y) tuples and sync offset in ms
    """
    gaze_data = []
    sync_offset_ms = 0.0
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)  # Skip header
        
        # Try to determine column indices
        try:
            timestamp_idx = header.index('Recording timestamp [ms]')
            event_idx = header.index('Event') if 'Event' in header else None
        except ValueError:
            # Fallback for old format
            timestamp_idx = 0
            event_idx = None
        
        # Find gaze point X and Y columns
        gaze_x_idx = None
        gaze_y_idx = None
        for i, col in enumerate(header):
            if 'Gaze point X' in col:
                gaze_x_idx = i
            elif 'Gaze point Y' in col:
                gaze_y_idx = i
        
        # Fallback if not found
        if gaze_x_idx is None:
            gaze_x_idx = 3
        if gaze_y_idx is None:
            gaze_y_idx = 4
        
        for row in reader:
            try:
                # Check for sync event (use only the first one)
                if event_idx is not None and len(row) > event_idx and sync_offset_ms == 0.0:
                    event = row[event_idx].strip()
                    if event == 'SyncPortOutHigh':
                        sync_offset_ms = float(row[timestamp_idx])
                        print(f"Found sync event at {sync_offset_ms:.1f}ms - using this as video start")
                        continue
                
                timestamp_ms = float(row[timestamp_idx])
                gaze_x = float(row[gaze_x_idx])
                gaze_y = float(row[gaze_y_idx])
                
                # Only include valid gaze points
                if not (np.isnan(gaze_x) or np.isnan(gaze_y)):
                    gaze_data.append((timestamp_ms, gaze_x, gaze_y))
            except (ValueError, IndexError):
                continue
    
    print(f"Loaded {len(gaze_data)} gaze points")
    
    return gaze_data, sync_offset_ms


def find_gaze_by_index(
    gaze_data: List[Tuple[float, float, float]],
    index: int
) -> Tuple[float, float]:
    """
    Find gaze point by index.
    
    Parameters
    ----------
    gaze_data : List[Tuple[float, float, float]]
        List of (timestamp_ms, gaze_x, gaze_y)
    index : int
        Index into gaze data
    
    Returns
    -------
    Tuple[float, float]
        (gaze_x, gaze_y) or (None, None) if out of range
    """
    if not gaze_data or index < 0 or index >= len(gaze_data):
        return None, None
    
    return gaze_data[index][1], gaze_data[index][2]


def draw_gaze_with_trail(
    frame: np.ndarray,
    gaze_history: List[GazePoint],
    current_color: Tuple[int, int, int],
    trail_color: Tuple[int, int, int],
    size: int,
    opacity: float,
    fade_rate: float,
    trail_thickness: int = 3,
    current_thickness: int = 2,
    initial_fade: float = 0.3
) -> np.ndarray:
    """
    Draw gaze point with fading trail.
    
    Parameters
    ----------
    frame : np.ndarray
        Frame to draw on
    gaze_history : List[GazePoint]
        List of recent gaze points (newest first)
    current_color : Tuple[int, int, int]
        BGR color for current point
    trail_color : Tuple[int, int, int]
        BGR color for trail points
    size : int
        Circle radius
    opacity : float
        Opacity of current point
    fade_rate : float
        Opacity decrease per trail point (after initial fade)
    trail_thickness : int
        Trail circle outline thickness
    current_thickness : int
        Current gaze circle outline thickness
    initial_fade : float
        Initial opacity drop from current to first trail point
    
    Returns
    -------
    np.ndarray
        Frame with gaze overlay
    """
    overlay = frame.copy()
    
    # Draw trail (oldest to newest, so newer points cover older)
    for i in range(len(gaze_history) - 1, -1, -1):
        gaze = gaze_history[i]
        
        # Choose color, opacity, and thickness based on position
        if i == 0:
            # Current point: use current color with full opacity (0% fade)
            circle_color = current_color
            trail_opacity = 1.0  # Always 100% for current point
            circle_thickness = current_thickness
        else:
            # Trail points: use trail color with fading
            circle_color = trail_color
            # i=1 gets (opacity - initial_fade), then decreases by fade_rate per point
            trail_opacity = opacity - initial_fade - ((i - 1) * fade_rate)
            trail_opacity = max(0.0, trail_opacity)
            circle_thickness = trail_thickness
        
        if trail_opacity > 0.0:
            # Draw circle outline on overlay
            cv2.circle(
                overlay,
                (int(gaze.x), int(gaze.y)),
                size,
                circle_color,
                circle_thickness  # Outline, not filled
            )
            
            # Blend with original frame
            alpha = trail_opacity
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            overlay = frame.copy()
    
    return frame


def overlay_eye_tracking(
    video_path: str,
    eye_tracking_path: str,
    output_path: str,
    gaze_color: Tuple[int, int, int],
    trail_color: Tuple[int, int, int],
    gaze_size: int,
    gaze_opacity: float,
    trail_length: int,
    fade_rate: float,
    trail_thickness: int,
    current_thickness: int,
    initial_fade: float,
    time_offset: float,
    start_time_sec: float = None,
    end_time_sec: float = None
):
    """
    Overlay eye tracking data on video.
    
    Parameters
    ----------
    video_path : str
        Input video path
    eye_tracking_path : str
        Eye tracking TSV path
    output_path : str
        Output video path
    gaze_color : Tuple[int, int, int]
        BGR color for gaze circle
    gaze_size : int
        Radius of gaze circle
    gaze_opacity : float
        Opacity of current gaze point
    trail_length : int
        Number of trail points
    fade_rate : float
        Opacity decrease per trail point
    time_offset : float
        Time offset in seconds
    """
    # Load eye tracking data
    print(f"Loading eye tracking data from: {eye_tracking_path}")
    gaze_data, sync_offset_ms = load_eye_tracking_data(eye_tracking_path)
    
    if not gaze_data:
        print("Error: No valid gaze data found")
        return
    
    # Get timestamp range
    min_ts = gaze_data[0][0]
    max_ts = gaze_data[-1][0]
    print(f"Gaze data time range: {min_ts:.0f}ms to {max_ts:.0f}ms ({(max_ts-min_ts)/1000:.1f}s)")
    print(f"Video start sync offset: {sync_offset_ms:.1f}ms")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo: {width}x{height}, {fps:.2f} fps, {total_frames} frames")
    print(f"Time offset: {time_offset:.2f}s")
    
    # Calculate frame range
    start_frame = 0
    end_frame = total_frames
    
    if start_time_sec is not None:
        start_frame = int(start_time_sec * fps)
        print(f"Start time: {start_time_sec:.1f}s (frame {start_frame})")
    
    if end_time_sec is not None:
        end_frame = int(end_time_sec * fps)
        print(f"End time: {end_time_sec:.1f}s (frame {end_frame})")
    
    # Clamp to valid range
    start_frame = max(0, start_frame)
    end_frame = min(total_frames, end_frame)
    frames_to_process = end_frame - start_frame
    
    # Seek to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video
    frame_number = start_frame
    frames_written = 0
    gaze_history = []
    gaze_found_count = 0
    
    print(f"\nProcessing {frames_to_process} frames (frame {start_frame} to {end_frame})...")
    print("Overlaying gaze data...")
    
    # Create a lookup for faster access
    gaze_by_timestamp = {int(ts): (x, y) for ts, x, y in gaze_data}
    
    while frame_number < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate video time in milliseconds
        video_time_ms = (frame_number / fps) * 1000
        
        # Map to eye tracking timestamp using sync offset
        target_timestamp_ms = sync_offset_ms + video_time_ms + (time_offset * 1000)
        
        # Find closest gaze point (within 25ms tolerance - half the sampling rate)
        gaze_x, gaze_y = None, None
        min_diff = float('inf')
        
        for ts_offset in range(-25, 26, 5):  # Search within ±25ms
            check_ts = int(target_timestamp_ms + ts_offset)
            if check_ts in gaze_by_timestamp:
                diff = abs(ts_offset)
                if diff < min_diff:
                    min_diff = diff
                    gaze_x, gaze_y = gaze_by_timestamp[check_ts]
        
        # If not found in lookup, search linearly
        if gaze_x is None:
            for ts, x, y in gaze_data:
                diff = abs(ts - target_timestamp_ms)
                if diff < 25:  # Within 25ms
                    gaze_x, gaze_y = x, y
                    break
                elif ts > target_timestamp_ms:
                    break  # Past target, stop searching
        
        if gaze_x is not None:
            gaze_found_count += 1
            
            # Add to history
            gaze_history.insert(0, GazePoint(gaze_x, gaze_y, video_time_ms))
            
            # Keep only recent points
            gaze_history = gaze_history[:trail_length]
            
            # Draw gaze with trail
            frame = draw_gaze_with_trail(
                frame,
                gaze_history,
                gaze_color,
                trail_color,
                gaze_size,
                gaze_opacity,
                fade_rate,
                trail_thickness,
                current_thickness,
                initial_fade
            )
        
        out.write(frame)
        frames_written += 1
        
        # Progress update
        if frames_written % 300 == 0:
            percent = int(100 * frames_written / frames_to_process)
            print(f"  [{percent:3d}%] Frame {frame_number}/{end_frame} - {gaze_found_count} gaze points found")
        
        frame_number += 1
    
    cap.release()
    out.release()
    
    print(f"\n[OK] Gaze points matched: {gaze_found_count}/{frames_written} frames ({100*gaze_found_count/max(1, frames_written):.1f}%)")
    print(f"[OK] Output saved to: {output_path}")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Parse colors
    try:
        color_parts = args.gaze_color.split(',')
        gaze_color = (int(color_parts[0]), int(color_parts[1]), int(color_parts[2]))
    except:
        print(f"Error: Invalid gaze color format. Use B,G,R format, e.g., '255,255,255'")
        sys.exit(1)
    
    try:
        trail_parts = args.trail_color.split(',')
        trail_color = (int(trail_parts[0]), int(trail_parts[1]), int(trail_parts[2]))
    except:
        print(f"Error: Invalid trail color format. Use B,G,R format, e.g., '0,0,255'")
        sys.exit(1)
    
    # Parse time range
    start_time_sec = None
    end_time_sec = None
    
    if args.start_time:
        try:
            start_time_sec = parse_time_string(args.start_time)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    if args.end_time:
        try:
            end_time_sec = parse_time_string(args.end_time)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    if start_time_sec is not None and end_time_sec is not None:
        if start_time_sec >= end_time_sec:
            print("Error: Start time must be before end time")
            sys.exit(1)
    
    print("=" * 60)
    print("Eye Tracking Overlay")
    print("=" * 60)
    print(f"Input video:     {args.video_path}")
    print(f"Eye tracking:    {args.eye_tracking_path}")
    print(f"Output:          {args.output}")
    if start_time_sec or end_time_sec:
        start_str = f"{start_time_sec:.1f}s" if start_time_sec else "start"
        end_str = f"{end_time_sec:.1f}s" if end_time_sec else "end"
        print(f"Time range:      {start_str} to {end_str}")
    print(f"Current color:   {gaze_color} (BGR)")
    print(f"Trail color:     {trail_color} (BGR)")
    print(f"Gaze size:       {args.gaze_size}px")
    print(f"Gaze opacity:    {args.gaze_opacity}")
    print(f"Trail length:    {args.trail_length} points")
    print(f"Initial fade:    {args.initial_fade}")
    print(f"Fade rate:       {args.fade_rate}")
    print(f"Current thick:   {args.current_thickness}px")
    print(f"Trail thick:     {args.thickness}px")
    print("=" * 60)
    print()
    
    overlay_eye_tracking(
        video_path=args.video_path,
        eye_tracking_path=args.eye_tracking_path,
        output_path=args.output,
        gaze_color=gaze_color,
        trail_color=trail_color,
        gaze_size=args.gaze_size,
        gaze_opacity=args.gaze_opacity,
        trail_length=args.trail_length,
        fade_rate=args.fade_rate,
        trail_thickness=args.thickness,
        current_thickness=args.current_thickness,
        initial_fade=args.initial_fade,
        time_offset=args.time_offset,
        start_time_sec=start_time_sec,
        end_time_sec=end_time_sec
    )


if __name__ == '__main__':
    main()

