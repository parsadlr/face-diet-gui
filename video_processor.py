"""
Video processing module for face-diet.

Implements two-pass video processing:
1. Pass 1: Collect all face detections with embeddings and attributes
2. Pass 2: Cluster embeddings, assign IDs, and generate outputs
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import cv2
import numpy as np

from face_attributes import (
    extract_all_attributes,
    extract_all_attributes_batch,
    extract_pose_with_pnp,
    estimate_distance,
)
import deepface
from face_detection import (
    assign_face_ids,
    detect_faces_in_frame,
    find_representative_instances,
    initialize_detector,
)
from profiler import get_profiler
from utils import append_csv_row, frame_to_time, write_csv_header


def process_video(
    video_path: str,
    output_csv: str,
    output_video: Optional[str] = None,
    sampling_rate: int = 1,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    clustering_method: str = 'threshold',
    similarity_threshold: float = 0.6,
    dbscan_eps: float = 0.4,
    dbscan_min_samples: int = 2,
    use_gpu: bool = False,
    progress_callback: Optional[callable] = None,
    max_workers: Optional[int] = None,
    batch_size: int = 8,
    use_batch: bool = True,
) -> Dict:
    """
    Process video with two-pass face detection and identity assignment.
    
    Parameters
    ----------
    video_path : str
        Path to input video file
    output_csv : str
        Path to output CSV file
    output_video : str, optional
        Path to output annotated video (if None, skip video output)
    sampling_rate : int
        Process every N frames (1 = every frame)
    start_time : float, optional
        Start time in seconds (None = from beginning)
    end_time : float, optional
        End time in seconds (None = until end)
    clustering_method : str
        'threshold' or 'dbscan'
    similarity_threshold : float
        Threshold for threshold-based clustering
    dbscan_eps : float
        Epsilon for DBSCAN
    dbscan_min_samples : int
        Min samples for DBSCAN
    use_gpu : bool
        Whether to use GPU for detection
    progress_callback : callable, optional
        Function to call with progress updates (progress_callback(message, percent))
    
    Returns
    -------
    Dict
        Summary statistics
    """
    print(f"Starting video processing: {video_path}")
    print(f"Sampling rate: every {sampling_rate} frame(s)")
    print(f"Clustering method: {clustering_method}")
    
    # Set default max_workers if not provided
    if max_workers is None:
        max_workers = 4
    print(f"Parallel workers: {max_workers}")
    if use_batch:
        print(f"Batch processing: Enabled (batch size: {batch_size})")
    else:
        print(f"Batch processing: Disabled")
    
    # Initialize detector
    if progress_callback:
        progress_callback("Initializing detector...", 0)
    else:
        print("Initializing detector...")
    
    profiler = get_profiler()
    with profiler.time_block("detector_initialization"):
        detector = initialize_detector(use_gpu=use_gpu)
    
    # Pass 1: Collect all detections
    if progress_callback:
        progress_callback("Pass 1: Collecting detections...", 5)
    else:
        print("\nPass 1: Collecting detections...")
    
    detections = collect_all_detections(
        video_path=video_path,
        detector=detector,
        sampling_rate=sampling_rate,
        start_time=start_time,
        end_time=end_time,
        progress_callback=progress_callback,
        max_workers=max_workers,
        batch_size=batch_size,
        use_batch=use_batch,
    )
    
    print(f"\nCollected {len(detections)} face instances")
    
    # Pass 2: Assign IDs and generate outputs
    if progress_callback:
        progress_callback("Pass 2: Assigning face IDs...", 60)
    else:
        print("\nPass 2: Assigning face IDs...")
    
    profiler = get_profiler()
    with profiler.time_block("face_id_assignment"):
        assign_face_ids(
            detections=detections,
            clustering_method=clustering_method,
            similarity_threshold=similarity_threshold,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
        )
    
    # Find representative instances
    with profiler.time_block("find_representatives"):
        representatives = find_representative_instances(detections)
    
    # Count unique faces
    unique_ids = set(d['face_id'] for d in detections if d['face_id'] != 'UNKNOWN')
    print(f"Found {len(unique_ids)} unique face(s)")
    
    # Write CSV output
    if progress_callback:
        progress_callback("Writing CSV output...", 70)
    else:
        print("\nWriting CSV output...")
    
    profiler = get_profiler()
    with profiler.time_block("csv_writing"):
        write_csv_output(output_csv, detections)
    print(f"CSV written to: {output_csv}")
    
    # Write annotated video if requested
    if output_video:
        if progress_callback:
            progress_callback("Creating annotated video...", 80)
        else:
            print("\nCreating annotated video...")
        
        with profiler.time_block("video_writing"):
            write_annotated_video(
                video_path=video_path,
                output_video=output_video,
                detections=detections,
                sampling_rate=sampling_rate,
                progress_callback=progress_callback,
            )
        print(f"Annotated video written to: {output_video}")
    
    if progress_callback:
        progress_callback("Complete!", 100)
    
    print("\nProcessing complete!")
    
    return {
        'total_detections': len(detections),
        'unique_faces': len(unique_ids),
        'representatives': representatives,
    }


def collect_all_detections(
    video_path: str,
    detector,
    sampling_rate: int = 1,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    progress_callback: Optional[callable] = None,
    max_workers: Optional[int] = None,
    batch_size: int = 8,
    use_batch: bool = True,
) -> List[Dict]:
    """
    Pass 1: Collect all face detections from video.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    detector
        InsightFace detector
    sampling_rate : int
        Process every N frames
    start_time : float, optional
        Start time in seconds (None = from beginning)
    end_time : float, optional
        End time in seconds (None = until end)
    progress_callback : callable, optional
        Progress callback function
    
    Returns
    -------
    List[Dict]
        List of all detections with attributes
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame range from time range
    start_frame = 0
    end_frame = total_frames
    
    if start_time is not None:
        start_frame = int(start_time * fps)
    if end_time is not None:
        end_frame = int(end_time * fps)
    
    # Clamp to valid range
    start_frame = max(0, start_frame)
    end_frame = min(total_frames, end_frame)
    
    frames_to_process = (end_frame - start_frame) // sampling_rate
    
    print(f"Video: {total_frames} frames, {fps:.2f} fps")
    if start_time is not None or end_time is not None:
        print(f"Time range: {start_frame / fps:.1f}s to {end_frame / fps:.1f}s (frames {start_frame} to {end_frame})")
    print(f"Will process {frames_to_process} frames")
    
    # Seek to start frame if specified
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    all_detections = []
    frame_number = start_frame
    processed_count = 0
    
    # Set default max_workers if not provided
    if max_workers is None:
        max_workers = 4
    
    profiler = get_profiler()
    
    # Cross-frame batching: collect faces from multiple frames before processing
    pending_faces = []  # List of (frame_number, frame_image, detection) tuples
    
    def process_pending_faces_batch():
        """Process accumulated faces in a batch across frames."""
        if not pending_faces or not use_batch:
            return []
        
        # Extract all face objects, bboxes, and frame images
        all_face_objs = []
        all_bboxes = []
        all_frame_images = []
        all_frame_numbers = []
        all_detection_metadata = []
        
        for frame_num, frame_img, detection in pending_faces:
            all_face_objs.append(detection['face_obj'])
            all_bboxes.append(detection['bbox'])
            all_frame_images.append(frame_img)
            all_frame_numbers.append(frame_num)
            all_detection_metadata.append(detection)
        
        # Crop all faces first (needs original frame for each)
        face_crops = []
        valid_indices = []
        for i, (bbox, frame_img) in enumerate(zip(all_bboxes, all_frame_images)):
            x, y, w, h = bbox
            pad = int(max(w, h) * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame_img.shape[1], x + w + pad)
            y2 = min(frame_img.shape[0], y + h + pad)
            
            face_crop = frame_img[y1:y2, x1:x2]
            if face_crop.size > 0:
                face_crops.append(face_crop)
                valid_indices.append(i)
        
        if not face_crops:
            return []
        
        # Process in batches
        batch_results = []
        for batch_start in range(0, len(face_crops), batch_size):
            batch_end = min(batch_start + batch_size, len(face_crops))
            batch_crops = face_crops[batch_start:batch_end]
            batch_indices = valid_indices[batch_start:batch_end]
            
            # Extract poses individually (needs full frame)
            batch_poses = []
            for idx in batch_indices:
                with profiler.time_block("pose_extraction"):
                    pose = extract_pose_with_pnp(all_face_objs[idx], all_frame_images[idx])
                batch_poses.append(pose)
            
            # Batch process DeepFace with cropped faces
            with profiler.time_block("deepface_batch_analysis"):
                try:
                    from deepface import DeepFace
                    results = DeepFace.analyze(
                        img_path=batch_crops,  # List of numpy arrays
                        actions=['age', 'gender', 'emotion', 'race'],
                        enforce_detection=False,
                        detector_backend='skip',
                        silent=True,
                    )
                    
                    if not isinstance(results, list):
                        results = [results]
                    
                    # Map results back
                    for i, (idx, result, pose) in enumerate(zip(batch_indices, results, batch_poses)):
                        if isinstance(result, dict):
                            demographics = {
                                'age': result.get('age', None),
                                'gender': max(result.get('gender', {}).items(), key=lambda x: x[1])[0] if result.get('gender') else None,
                                'race': result.get('dominant_race', None),
                                'emotion': result.get('dominant_emotion', None),
                            }
                        else:
                            demographics = {'age': None, 'gender': None, 'race': None, 'emotion': None}
                        
                        metadata = all_detection_metadata[idx]
                        distance = estimate_distance(metadata['bbox'])
                        batch_results.append({
                            'frame_number': all_frame_numbers[idx],
                            'time_seconds': frame_to_time(all_frame_numbers[idx], fps),
                            'bbox': metadata['bbox'],
                            'embedding': metadata['embedding'],
                            'confidence': metadata['confidence'],
                            'pose': pose,
                            'age': demographics['age'],
                            'gender': demographics['gender'],
                            'race': demographics['race'],
                            'emotion': demographics['emotion'],
                            'distance': distance,
                            'face_id': None,
                        })
                except Exception as e:
                    # Fallback to individual processing if batch fails
                    print(f"Warning: Cross-frame batch failed, processing individually: {e}")
                    for idx in batch_indices:
                        metadata = all_detection_metadata[idx]
                        with profiler.time_block("attribute_extraction_per_face"):
                            attributes = extract_all_attributes(
                                face_obj=all_face_objs[idx],
                                image_bgr=all_frame_images[idx],
                                bbox=all_bboxes[idx],
                            )
                        batch_results.append({
                            'frame_number': all_frame_numbers[idx],
                            'time_seconds': frame_to_time(all_frame_numbers[idx], fps),
                            'bbox': metadata['bbox'],
                            'embedding': metadata['embedding'],
                            'confidence': metadata['confidence'],
                            'pose': attributes['pose'],
                            'age': attributes['age'],
                            'gender': attributes['gender'],
                            'race': attributes['race'],
                            'emotion': attributes['emotion'],
                            'distance': attributes['distance'],
                            'face_id': None,
                        })
        
        return batch_results
    
    while frame_number < end_frame:
        with profiler.time_block("video_frame_read"):
            ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames based on sampling rate
        if (frame_number - start_frame) % sampling_rate != 0:
            frame_number += 1
            continue
        
        # Detect faces in this frame
        with profiler.time_block("face_detection_per_frame"):
            frame_detections = detect_faces_in_frame(detector, frame)
        
        # Add detected faces to pending batch
        if frame_detections:
            for detection in frame_detections:
                pending_faces.append((frame_number, frame.copy(), detection))
        
        # Process batch when we have enough faces or at end of processing
        if use_batch and len(pending_faces) >= batch_size:
            with profiler.time_block("cross_frame_batch_processing"):
                batch_results = process_pending_faces_batch()
                all_detections.extend(batch_results)
                pending_faces = []  # Clear after processing
        elif not frame_detections and len(pending_faces) > 0:
            # Process remaining faces if we hit a frame with no faces
            with profiler.time_block("cross_frame_batch_processing"):
                batch_results = process_pending_faces_batch()
                all_detections.extend(batch_results)
                pending_faces = []
        
        processed_count += 1
        
        # Progress update - show every 10 frames or at 5% intervals
        show_progress = False
        if processed_count % 10 == 0:
            show_progress = True
        
        if show_progress:
            percent = int(100 * processed_count / max(1, frames_to_process))
            time_sec = frame_number / fps
            profiler = get_profiler()
            # Show recent performance stats
            det_stats = profiler.get_stats("face_detection_per_frame")
            attr_stats = profiler.get_stats("attribute_extraction_per_face")
            if progress_callback:
                progress_callback(f"Processing frame {frame_number}...", 5 + int(55 * processed_count / max(1, frames_to_process)))
            print(
                f"  [{percent:3d}%] Frame {frame_number} ({time_sec:.1f}s) - "
                f"{processed_count}/{frames_to_process} frames, {len(all_detections)} faces detected | "
                f"Detection: {det_stats['mean']:.3f}s/frame, Attributes: {attr_stats['mean']:.3f}s/face"
            )
        
        frame_number += 1
    
    # Process any remaining pending faces
    if pending_faces:
        with profiler.time_block("cross_frame_batch_processing"):
            batch_results = process_pending_faces_batch()
            all_detections.extend(batch_results)
    
    cap.release()
    
    return all_detections


def extract_attributes_parallel(
    frame_detections: List[Dict],
    frame: np.ndarray,
    frame_number: int,
    fps: float,
    max_workers: Optional[int] = None,
    use_batch: bool = True,
    batch_size: int = 8,
) -> List[Dict]:
    """
    Extract attributes for multiple faces using batch processing and/or parallel processing.
    
    Parameters
    ----------
    frame_detections : List[Dict]
        List of face detections from detect_faces_in_frame
    frame : np.ndarray
        Frame image in BGR format
    frame_number : int
        Current frame number
    fps : float
        Video FPS
    max_workers : int
        Maximum number of parallel workers (for non-batch fallback)
    use_batch : bool
        Whether to use DeepFace batch processing (faster for multiple faces)
    batch_size : int
        Maximum number of faces per DeepFace batch
    
    Returns
    -------
    List[Dict]
        List of detections with attributes
    """
    if not frame_detections:
        return []
    
    # Set default max_workers if not provided
    if max_workers is None:
        max_workers = 4
    
    profiler = get_profiler()
    all_detections = []
    
    # Use batch processing if multiple faces and enabled
    if len(frame_detections) > 1 and use_batch:
        with profiler.time_block("batch_attribute_extraction"):
            # Extract face objects and bboxes
            face_objs = [det['face_obj'] for det in frame_detections]
            bboxes = [det['bbox'] for det in frame_detections]
            
            # Use batch processing for DeepFace
            with profiler.time_block("attribute_extraction_per_face"):
                attributes_list = extract_all_attributes_batch(
                    face_objs=face_objs,
                    image_bgr=frame,
                    bboxes=bboxes,
                    batch_size=batch_size,
                )
            
            # Combine with detection data
            for detection, attributes in zip(frame_detections, attributes_list):
                detection_data = {
                    'frame_number': frame_number,
                    'time_seconds': frame_to_time(frame_number, fps),
                    'bbox': detection['bbox'],
                    'embedding': detection['embedding'],
                    'confidence': detection['confidence'],
                    'pose': attributes['pose'],
                    'age': attributes['age'],
                    'gender': attributes['gender'],
                    'race': attributes['race'],
                    'emotion': attributes['emotion'],
                    'distance': attributes['distance'],
                    'face_id': None,  # Will be assigned in Pass 2
                }
                all_detections.append(detection_data)
    
    # Fallback to parallel processing if batch disabled or single face
    elif len(frame_detections) > 1 and max_workers > 1:
        with profiler.time_block("parallel_attribute_extraction"):
            def extract_single_face(detection):
                """Extract attributes for a single face."""
                with profiler.time_block("attribute_extraction_per_face"):
                    attributes = extract_all_attributes(
                        face_obj=detection['face_obj'],
                        image_bgr=frame,
                        bbox=detection['bbox'],
                    )
                
                return {
                    'frame_number': frame_number,
                    'time_seconds': frame_to_time(frame_number, fps),
                    'bbox': detection['bbox'],
                    'embedding': detection['embedding'],
                    'confidence': detection['confidence'],
                    'pose': attributes['pose'],
                    'age': attributes['age'],
                    'gender': attributes['gender'],
                    'race': attributes['race'],
                    'emotion': attributes['emotion'],
                    'distance': attributes['distance'],
                    'face_id': None,  # Will be assigned in Pass 2
                }
            
            # Process faces in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(extract_single_face, det) for det in frame_detections]
                for future in as_completed(futures):
                    try:
                        all_detections.append(future.result())
                    except Exception as e:
                        print(f"Warning: Failed to extract attributes: {e}")
    else:
        # Sequential processing for single face or when max_workers=1
        for detection in frame_detections:
            with profiler.time_block("attribute_extraction_per_face"):
                attributes = extract_all_attributes(
                    face_obj=detection['face_obj'],
                    image_bgr=frame,
                    bbox=detection['bbox'],
                )
            
            detection_data = {
                'frame_number': frame_number,
                'time_seconds': frame_to_time(frame_number, fps),
                'bbox': detection['bbox'],
                'embedding': detection['embedding'],
                'confidence': detection['confidence'],
                'pose': attributes['pose'],
                'age': attributes['age'],
                'gender': attributes['gender'],
                'race': attributes['race'],
                'emotion': attributes['emotion'],
                'distance': attributes['distance'],
                'face_id': None,  # Will be assigned in Pass 2
            }
            all_detections.append(detection_data)
    
    return all_detections


def write_csv_output(csv_path: str, detections: List[Dict]) -> None:
    """
    Write all detections to CSV file.
    
    Parameters
    ----------
    csv_path : str
        Output CSV file path
    detections : List[Dict]
        List of all detections with assigned face IDs
    """
    # Write header
    write_csv_header(csv_path)
    
    # Write each detection as a row
    for detection in detections:
        append_csv_row(csv_path, detection)


def write_annotated_video(
    video_path: str,
    output_video: str,
    detections: List[Dict],
    sampling_rate: int = 1,
    progress_callback: Optional[callable] = None,
) -> None:
    """
    Create annotated video with face bounding boxes and IDs.
    
    Parameters
    ----------
    video_path : str
        Input video path
    output_video : str
        Output video path
    detections : List[Dict]
        All detections with assigned face IDs
    sampling_rate : int
        Sampling rate used during detection
    progress_callback : callable, optional
        Progress callback function
    """
    # Group detections by frame number for quick lookup
    detections_by_frame = {}
    for detection in detections:
        frame_num = detection['frame_number']
        if frame_num not in detections_by_frame:
            detections_by_frame[frame_num] = []
        detections_by_frame[frame_num].append(detection)
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw detections for this frame if available
        if frame_number in detections_by_frame:
            frame = draw_detections_on_frame(frame, detections_by_frame[frame_number])
        
        out.write(frame)
        
        # Progress update - show every 30 frames
        if frame_number % 30 == 0:
            percent = int(100 * frame_number / max(1, total_frames))
            if progress_callback:
                progress_callback(f"Writing frame {frame_number}...", 80 + int(20 * frame_number / total_frames))
            print(f"  [{percent:3d}%] Writing frame {frame_number}/{total_frames}")
        
        frame_number += 1
    
    cap.release()
    out.release()


def draw_detections_on_frame(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    Draw bounding boxes and labels on frame.
    
    Parameters
    ----------
    frame : np.ndarray
        Frame image
    detections : List[Dict]
        Detections for this frame
    
    Returns
    -------
    np.ndarray
        Annotated frame
    """
    annotated = frame.copy()
    
    for detection in detections:
        x, y, w, h = detection['bbox']
        face_id = detection.get('face_id', 'UNKNOWN')
        
        # Draw bounding box
        color = (0, 255, 0) if face_id != 'UNKNOWN' else (128, 128, 128)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        
        # Prepare label text
        label_lines = [face_id]
        
        # Add attributes
        age = detection.get('age')
        gender = detection.get('gender')
        
        if age is not None and gender is not None:
            label_lines.append(f"{gender}, {age}")
        
        # Draw label background and text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        y_offset = y - 10
        for line in label_lines:
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
            
            # Draw background rectangle
            cv2.rectangle(
                annotated,
                (x, y_offset - text_height - 5),
                (x + text_width + 5, y_offset),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                annotated,
                line,
                (x + 2, y_offset - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )
            
            y_offset -= (text_height + 8)
    
    return annotated

