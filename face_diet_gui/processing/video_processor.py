"""
Video processing module for face-diet.

Implements staged video processing:
Stage 1: InsightFace detection (fast) - Extract faces, bboxes, embeddings, assign IDs
Stage 2: DeepFace attributes (batched) - Extract age, gender, race, emotion for all faces

This staged approach enables:
- Better GPU utilization (batch all DeepFace calls together)
- Resumability (if Stage 2 fails, no need to re-run Stage 1)
- Flexibility (skip Stage 2 for fast face detection only)
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import json
import pandas as pd

import cv2
import numpy as np

from face_diet_gui.processing.face_attributes import (
    extract_all_attributes,
    extract_all_attributes_batch,
    extract_pose_with_pnp,
    estimate_distance,
    extract_age_gender_race_emotion_batch,
)
import deepface
from face_diet_gui.processing.face_detection import (
    assign_face_ids,
    detect_faces_in_frame,
    find_representative_instances,
    initialize_detector,
)
from face_diet_gui.profiler import get_profiler
from face_diet_gui.utils import append_csv_row, frame_to_time, write_csv_header


def process_video_stage1(
    video_path: str,
    output_csv: str,
    sampling_rate: int = 1,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    clustering_method: str = 'threshold',
    similarity_threshold: float = 0.6,
    dbscan_eps: float = 0.4,
    dbscan_min_samples: int = 2,
    use_gpu: bool = False,
    progress_callback: Optional[callable] = None,
    eye_tracking_path: Optional[str] = None,
) -> Dict:
    """
    STAGE 1: InsightFace detection only (no DeepFace attributes).
    
    Extracts faces, bounding boxes, embeddings, assigns IDs, and saves to CSV.
    This is fast and suitable for quick face detection.
    
    Parameters
    ----------
    video_path : str
        Path to input video file
    output_csv : str
        Path to output CSV file (will contain basic detection info)
    sampling_rate : int
        Process every N frames (1 = every frame)
    start_time : float, optional
        Start time in seconds
    end_time : float, optional
        End time in seconds
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
        Progress callback function
    eye_tracking_path : str, optional
        Path to eye_tracking.tsv file (for attended flag)
    
    Returns
    -------
    Dict
        Summary statistics
    """
    print(f"STAGE 1: InsightFace detection")
    print(f"Video: {video_path}")
    print(f"Sampling rate: every {sampling_rate} frame(s)")
    print(f"Clustering method: {clustering_method}")
    
    # Initialize detector
    if progress_callback:
        progress_callback("Initializing detector...", 0)
    
    profiler = get_profiler()
    with profiler.time_block("detector_initialization"):
        detector = initialize_detector(use_gpu=use_gpu)
    
    # Collect detections (InsightFace only)
    if progress_callback:
        progress_callback("Stage 1: Detecting faces...", 5)
    
    print("\nDetecting faces with InsightFace...")
    detections = collect_detections_insightface_only(
        video_path=video_path,
        detector=detector,
        sampling_rate=sampling_rate,
        start_time=start_time,
        end_time=end_time,
        progress_callback=progress_callback,
        eye_tracking_path=eye_tracking_path,
    )
    
    print(f"\nCollected {len(detections)} face instances")
    
    # Assign face IDs
    if progress_callback:
        progress_callback("Assigning face IDs...", 60)
    
    print("\nAssigning face IDs...")
    with profiler.time_block("face_id_assignment"):
        assign_face_ids(
            detections=detections,
            clustering_method=clustering_method,
            similarity_threshold=similarity_threshold,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
        )
    
    # Find representatives
    with profiler.time_block("find_representatives"):
        representatives = find_representative_instances(detections)
    
    unique_ids = set(d['face_id'] for d in detections if d['face_id'] != 'UNKNOWN')
    print(f"Found {len(unique_ids)} unique face(s)")
    
    # Write CSV output (without DeepFace attributes)
    if progress_callback:
        progress_callback("Writing CSV...", 80)
    
    print("\nWriting CSV output...")
    with profiler.time_block("csv_writing"):
        write_csv_stage1(output_csv, detections)
    print(f"Stage 1 CSV written to: {output_csv}")
    
    if progress_callback:
        progress_callback("Stage 1 complete!", 100)
    
    print("\nStage 1 complete!")
    print("Run process_video_stage2() to add DeepFace attributes.")
    
    return {
        'total_detections': len(detections),
        'unique_faces': len(unique_ids),
        'representatives': representatives,
        'csv_path': output_csv,
    }


def process_video_stage2(
    video_path: str,
    stage1_csv: str,
    output_csv: Optional[str] = None,
    batch_size: int = 32,
    progress_callback: Optional[callable] = None,
) -> Dict:
    """
    STAGE 2: DeepFace attribute extraction from Stage 1 CSV.
    
    Reads face detections from Stage 1 CSV, extracts face crops from video,
    batches all DeepFace processing, and updates CSV with attributes.
    
    Parameters
    ----------
    video_path : str
        Path to input video file
    stage1_csv : str
        Path to Stage 1 CSV output
    output_csv : str, optional
        Path to output CSV (if None, overwrites stage1_csv)
    batch_size : int
        Batch size for DeepFace processing (larger = more memory but faster)
    progress_callback : callable, optional
        Progress callback function
    
    Returns
    -------
    Dict
        Summary statistics
    """
    print(f"STAGE 2: DeepFace attribute extraction")
    print(f"Video: {video_path}")
    print(f"Stage 1 CSV: {stage1_csv}")
    print(f"Batch size: {batch_size}")
    
    if output_csv is None:
        output_csv = stage1_csv
    
    # Read Stage 1 CSV
    if progress_callback:
        progress_callback("Reading Stage 1 CSV...", 0)
    
    print("\nReading Stage 1 CSV...")
    df = pd.read_csv(stage1_csv)
    print(f"Loaded {len(df)} face detections")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # CHUNKED PROCESSING: Process faces in chunks to avoid loading all crops into RAM
    # This is critical for long videos with many faces
    profiler = get_profiler()
    
    if progress_callback:
        progress_callback("Processing faces in chunks...", 10)
    
    print(f"\nProcessing {len(df)} faces in memory-efficient chunks...")
    
    # Group by frame for efficient video reading
    df_sorted = df.sort_values('frame_number').reset_index(drop=True)
    
    # Process in chunks of faces (not just DeepFace batch size)
    CHUNK_SIZE = 2000  # Process 2000 faces at a time (~60 MB RAM vs 5+ GB for all)
    all_attributes = [None] * len(df)  # Pre-allocate results list
    
    for chunk_start in range(0, len(df_sorted), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(df_sorted))
        chunk_df = df_sorted.iloc[chunk_start:chunk_end]
        
        print(f"\n--- Processing chunk {chunk_start//CHUNK_SIZE + 1}/{(len(df_sorted)-1)//CHUNK_SIZE + 1} ---")
        print(f"    Faces {chunk_start} to {chunk_end} ({len(chunk_df)} faces)")
        
        # Extract face crops for this chunk only
        face_crops = []
        face_indices = []
        
        last_frame_number = -1
        current_frame = None
        
        for row_idx, row in chunk_df.iterrows():
            frame_number = int(row['frame_number'])
            
            # Read frame if different from last
            if frame_number != last_frame_number:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, current_frame = cap.read()
                if not ret:
                    print(f"    Warning: Could not read frame {frame_number}")
                    continue
                last_frame_number = frame_number
            
            # Extract face crop
            x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
            pad = int(max(w, h) * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(current_frame.shape[1], x + w + pad)
            y2 = min(current_frame.shape[0], y + h + pad)
            
            face_crop = current_frame[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                face_crops.append(face_crop)
                face_indices.append(row_idx)
        
        print(f"    Extracted {len(face_crops)} face crops from chunk")
        
        # Batch process this chunk with DeepFace
        chunk_attributes = []
        num_batches_in_chunk = (len(face_crops) + batch_size - 1) // batch_size
        
        for batch_idx, batch_start in enumerate(range(0, len(face_crops), batch_size)):
            batch_end = min(batch_start + batch_size, len(face_crops))
            batch_crops = face_crops[batch_start:batch_end]
            
            with profiler.time_block("deepface_batch_analysis"):
                try:
                    from deepface import DeepFace
                    results = DeepFace.analyze(
                        img_path=batch_crops,
                        actions=['age', 'gender', 'emotion', 'race'],
                        enforce_detection=False,
                        detector_backend='skip',
                        silent=True,
                    )
                    
                    # DeepFace can return nested lists when given multiple images
                    # Flatten if needed
                    if not isinstance(results, list):
                        results = [results]
                    
                    # Check if results are nested (list of lists)
                    if results and isinstance(results[0], list):
                        results = [item for sublist in results for item in (sublist if isinstance(sublist, list) else [sublist])]
                    
                    for idx, result in enumerate(results):
                        if isinstance(result, dict):
                            attrs = {
                                'age': result.get('age', None),
                                'gender': max(result.get('gender', {}).items(), key=lambda x: x[1])[0] if result.get('gender') else None,
                                'race': result.get('dominant_race', None),
                                'emotion': result.get('dominant_emotion', None),
                            }
                            chunk_attributes.append(attrs)
                        else:
                            print(f"      [WARNING] Result {idx} in batch is not a dict")
                            chunk_attributes.append({'age': None, 'gender': None, 'race': None, 'emotion': None})
                    
                    # Show progress only every 5 batches or first/last batch
                    if batch_idx == 0 or batch_idx == num_batches_in_chunk - 1 or batch_idx % 5 == 0:
                        print(f"      Batch {batch_idx+1}/{num_batches_in_chunk}: Processed {batch_end}/{len(face_crops)} faces in chunk")
                
                except Exception as e:
                    print(f"    [ERROR] Batch {batch_start}-{batch_end} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Add empty attributes for failed batch
                    for _ in range(len(batch_crops)):
                        chunk_attributes.append({'age': None, 'gender': None, 'race': None, 'emotion': None})
        
        # Store results in the pre-allocated list
        for idx, attributes in zip(face_indices, chunk_attributes):
            all_attributes[idx] = attributes
        
        # Progress update
        percent = int(100 * chunk_end / len(df))
        if progress_callback:
            progress_callback(f"Processing: {chunk_end}/{len(df)} faces", 10 + int(80 * chunk_end / len(df)))
        print(f"    [{percent:3d}%] Completed chunk. Total processed: {chunk_end}/{len(df)} faces")
        
        # Clear chunk data from memory
        del face_crops
        del chunk_df
    
    cap.release()
    print(f"\n[OK] All {len(df)} faces processed with DeepFace")
    
    # Update dataframe with attributes
    print("\nUpdating CSV with attributes...")
    updated_count = 0
    for idx in range(len(df_sorted)):
        if all_attributes[idx] is not None:
            attributes = all_attributes[idx]
            df_sorted.at[idx, 'age'] = attributes['age']
            df_sorted.at[idx, 'gender'] = attributes['gender']
            df_sorted.at[idx, 'race'] = attributes['race']
            df_sorted.at[idx, 'emotion'] = attributes['emotion']
            updated_count += 1
        else:
            # Face was skipped (invalid crop), keep attributes as None
            pass
    
    print(f"  Updated {updated_count}/{len(df_sorted)} faces with attributes")
    
    if updated_count < len(df_sorted):
        print(f"  [WARNING] {len(df_sorted) - updated_count} faces had invalid crops and were skipped")
    
    # Write output CSV (use df_sorted to maintain all rows)
    if progress_callback:
        progress_callback("Writing final CSV...", 95)
    
    df_sorted.to_csv(output_csv, index=False)
    print(f"Stage 2 CSV written to: {output_csv}")
    
    if progress_callback:
        progress_callback("Stage 2 complete!", 100)
    
    print("\nStage 2 complete!")
    
    return {
        'total_faces': len(df_sorted),
        'processed_faces': updated_count,
        'csv_path': output_csv,
    }


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
    use_staged: bool = False,
) -> Dict:
    """
    Process video with face detection and attribute extraction.
    
    Can operate in two modes:
    - Staged mode (use_staged=True): Run Stage 1 (InsightFace) then Stage 2 (DeepFace)
      Better for large batches and GPU utilization
    - Legacy mode (use_staged=False): Original interleaved processing
      May be faster for small videos or when batching isn't beneficial
    
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
    max_workers : int, optional
        Number of parallel workers (legacy mode only)
    batch_size : int
        Batch size for DeepFace processing
    use_batch : bool
        Whether to use batch processing (legacy mode only)
    use_staged : bool
        If True, use staged pipeline (Stage 1 + Stage 2). If False, use legacy mode.
    
    Returns
    -------
    Dict
        Summary statistics
    """
    # Use staged pipeline if requested
    if use_staged:
        print("=" * 80)
        print("USING STAGED PIPELINE")
        print("=" * 80)
        
        # Stage 1: InsightFace detection only
        stage1_result = process_video_stage1(
            video_path=video_path,
            output_csv=output_csv,
            sampling_rate=sampling_rate,
            start_time=start_time,
            end_time=end_time,
            clustering_method=clustering_method,
            similarity_threshold=similarity_threshold,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            use_gpu=use_gpu,
            progress_callback=progress_callback,
        )
        
        print("\n" + "=" * 80)
        print("Stage 1 complete. Starting Stage 2...")
        print("=" * 80 + "\n")
        
        # Stage 2: DeepFace attribute extraction
        stage2_result = process_video_stage2(
            video_path=video_path,
            stage1_csv=output_csv,
            output_csv=output_csv,
            batch_size=batch_size,
            progress_callback=progress_callback,
        )
        
        # Write annotated video if requested
        if output_video:
            if progress_callback:
                progress_callback("Creating annotated video...", 95)
            print("\nCreating annotated video...")
            
            # Need to read detections from CSV
            df = pd.read_csv(output_csv)
            detections = []
            for _, row in df.iterrows():
                detection = {
                    'frame_number': int(row['frame_number']),
                    'time_seconds': float(row['time_seconds']),
                    'bbox': (int(row['x']), int(row['y']), int(row['w']), int(row['h'])),
                    'face_id': row['face_id'],
                    'age': row['age'] if pd.notna(row['age']) else None,
                    'gender': row['gender'] if pd.notna(row['gender']) else None,
                }
                detections.append(detection)
            
            profiler = get_profiler()
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
        
        print("\n" + "=" * 80)
        print("STAGED PIPELINE COMPLETE")
        print("=" * 80)
        
        return {
            'total_detections': stage1_result['total_detections'],
            'unique_faces': stage1_result['unique_faces'],
            'representatives': stage1_result['representatives'],
            'processed_faces': stage2_result['processed_faces'],
        }
    
    # Legacy mode: original interleaved processing
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


def compute_face_sharpness(face_crop: np.ndarray) -> float:
    """
    Compute image sharpness/quality of a face crop using Laplacian variance.
    
    Higher values = sharper image
    Lower values = blurrier image
    
    Parameters
    ----------
    face_crop : np.ndarray
        Face image crop (BGR format)
    
    Returns
    -------
    float
        Sharpness score (typically 0-1000+, higher = sharper)
    """
    if face_crop.size == 0:
        return 0.0
    
    # Convert to grayscale
    if len(face_crop.shape) == 3:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_crop
    
    # Compute Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())
    
    return sharpness


def load_gaze_data_for_video(eye_tracking_path: str) -> Dict[float, tuple]:
    """
    Load gaze data from eye tracking TSV file.
    
    Parameters
    ----------
    eye_tracking_path : str
        Path to eye_tracking.tsv file
    
    Returns
    -------
    Dict[float, tuple]
        Mapping from timestamp_ms to (gaze_x, gaze_y)
    """
    import csv
    from pathlib import Path
    
    if not Path(eye_tracking_path).exists():
        return {}
    
    gaze_data = {}
    
    try:
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
                        gaze_x = float(row[gaze_x_idx])
                        gaze_y = float(row[gaze_y_idx])
                        gaze_data[ts_ms] = (gaze_x, gaze_y)
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"Warning: Could not load eye tracking data: {e}")
        return {}
    
    return gaze_data


def is_gaze_in_bbox(gaze_x: float, gaze_y: float, bbox: tuple) -> bool:
    """
    Check if gaze point is inside face bounding box.
    
    Parameters
    ----------
    gaze_x : float
        Gaze X coordinate
    gaze_y : float
        Gaze Y coordinate
    bbox : tuple
        (x, y, w, h) bounding box
    
    Returns
    -------
    bool
        True if gaze is inside bbox
    """
    x, y, w, h = bbox
    return x <= gaze_x <= (x + w) and y <= gaze_y <= (y + h)


def find_closest_gaze(gaze_data: Dict[float, tuple], timestamp_ms: float, max_diff_ms: float = 50) -> Optional[tuple]:
    """
    Find gaze point closest to a timestamp.
    
    Parameters
    ----------
    gaze_data : Dict[float, tuple]
        Mapping from timestamp to (gaze_x, gaze_y)
    timestamp_ms : float
        Target timestamp in milliseconds
    max_diff_ms : float
        Maximum time difference to consider valid (default: 50ms)
    
    Returns
    -------
    Optional[tuple]
        (gaze_x, gaze_y) or None if no valid gaze found
    """
    if not gaze_data:
        return None
    
    # Find closest timestamp
    closest_ts = min(gaze_data.keys(), key=lambda t: abs(t - timestamp_ms))
    
    if abs(closest_ts - timestamp_ms) <= max_diff_ms:
        return gaze_data[closest_ts]
    
    return None


def collect_detections_insightface_only(
    video_path: str,
    detector,
    sampling_rate: int = 1,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    progress_callback: Optional[callable] = None,
    eye_tracking_path: Optional[str] = None,
) -> List[Dict]:
    """
    Stage 1: Collect face detections using InsightFace only (no DeepFace).
    
    This is much faster than the full pipeline because it skips DeepFace analysis.
    Extracts: bbox, embedding, confidence, pose (from InsightFace), distance, attended flag.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    detector
        InsightFace detector
    sampling_rate : int
        Process every N frames
    start_time : float, optional
        Start time in seconds
    end_time : float, optional
        End time in seconds
    progress_callback : callable, optional
        Progress callback function
    eye_tracking_path : str, optional
        Path to eye_tracking.tsv file (for attended flag)
    
    Returns
    -------
    List[Dict]
        List of detections with basic info (no DeepFace attributes)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame range
    start_frame = 0
    end_frame = total_frames
    
    if start_time is not None:
        start_frame = int(start_time * fps)
    if end_time is not None:
        end_frame = int(end_time * fps)
    
    start_frame = max(0, start_frame)
    end_frame = min(total_frames, end_frame)
    
    frames_to_process = (end_frame - start_frame) // sampling_rate
    
    print(f"Video: {total_frames} frames, {fps:.2f} fps")
    if start_time is not None or end_time is not None:
        print(f"Time range: {start_frame / fps:.1f}s to {end_frame / fps:.1f}s")
    print(f"Will process {frames_to_process} frames")
    
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    all_detections = []
    frame_number = start_frame
    processed_count = 0
    
    # Load gaze data if provided
    gaze_data = {}
    if eye_tracking_path:
        print(f"Loading eye tracking data from: {eye_tracking_path}")
        gaze_data = load_gaze_data_for_video(eye_tracking_path)
        if gaze_data:
            print(f"  Loaded {len(gaze_data)} gaze points")
        else:
            print("  Warning: No gaze data loaded")
    
    profiler = get_profiler()
    
    while frame_number < end_frame:
        with profiler.time_block("video_frame_read"):
            ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames based on sampling rate
        if (frame_number - start_frame) % sampling_rate != 0:
            frame_number += 1
            continue
        
        # Detect faces (InsightFace only)
        with profiler.time_block("insightface_detection"):
            frame_detections = detect_faces_in_frame(detector, frame)
        
        # Extract basic info (no DeepFace)
        for detection in frame_detections:
            face_obj = detection['face_obj']
            bbox = detection['bbox']
            
            # Extract pose from InsightFace (fast)
            with profiler.time_block("pose_extraction"):
                pose = extract_pose_with_pnp(face_obj, frame)
            
            # Estimate distance (very fast)
            distance = estimate_distance(bbox)
            
            # Compute face sharpness/quality
            x, y, w, h = bbox
            face_crop = frame[y:y+h, x:x+w] if (y+h <= frame.shape[0] and x+w <= frame.shape[1]) else frame[y:min(y+h,frame.shape[0]), x:min(x+w,frame.shape[1])]
            sharpness = compute_face_sharpness(face_crop)
            
            # Check if face was attended (gaze in bbox)
            attended = False
            if gaze_data:
                timestamp_ms = (frame_number / fps) * 1000
                gaze = find_closest_gaze(gaze_data, timestamp_ms)
                if gaze:
                    gaze_x, gaze_y = gaze
                    attended = is_gaze_in_bbox(gaze_x, gaze_y, bbox)
            
            detection_data = {
                'frame_number': frame_number,
                'time_seconds': frame_to_time(frame_number, fps),
                'bbox': bbox,
                'embedding': detection['embedding'],
                'confidence': detection['confidence'],
                'sharpness': sharpness,  # Image quality measure
                'pose': pose,
                'distance': distance,
                'attended': attended,  # Binary flag
                # No DeepFace attributes yet
                'age': None,
                'gender': None,
                'race': None,
                'emotion': None,
            }
            all_detections.append(detection_data)
        
        processed_count += 1
        
        # Progress update - show every 100 frames
        if processed_count % 100 == 0:
            percent = int(100 * processed_count / max(1, frames_to_process))
            time_sec = frame_number / fps
            det_stats = profiler.get_stats("insightface_detection")
            if progress_callback:
                progress_callback(f"Detecting faces: frame {frame_number}...", 5 + int(55 * processed_count / max(1, frames_to_process)))
            print(
                f"  [{percent:3d}%] Frame {frame_number} ({time_sec:.1f}s) - "
                f"{processed_count}/{frames_to_process} frames, {len(all_detections)} faces | "
                f"Detection: {det_stats['mean']:.3f}s/frame"
            )
        
        frame_number += 1
    
    cap.release()
    
    return all_detections


def write_csv_stage1(csv_path: str, detections: List[Dict]) -> None:
    """
    Write Stage 1 CSV with basic detection info (no DeepFace attributes yet).
    
    Includes embeddings for later global ID assignment across sessions.
    Does NOT include face_id (assigned globally later).
    
    Parameters
    ----------
    csv_path : str
        Output CSV file path
    detections : List[Dict]
        List of detections from Stage 1
    """
    # Write CSV header (NO face_id, embedding comes last)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        import csv
        import json
        writer = csv.writer(f)
        writer.writerow([
            'frame_number', 'time_seconds',
            'x', 'y', 'w', 'h',
            'confidence', 'sharpness',  # Image quality measure
            'distance',
            'pitch', 'yaw', 'roll',
            'attended',  # Binary: was this face attended (gaze inside bbox)?
            'age', 'gender', 'race', 'emotion',
            'embedding',  # Embedding comes LAST
        ])
        
        # Write rows
        for detection in detections:
            bbox = detection['bbox']
            pose = detection['pose']
            embedding = detection.get('embedding', None)
            attended = detection.get('attended', False)
            
            # Pose values (may be None)
            pitch = pose['pitch'] if pose else None
            yaw = pose['yaw'] if pose else None
            roll = pose['roll'] if pose else None
            
            # Serialize embedding as JSON string (list of floats)
            if embedding is not None:
                if isinstance(embedding, np.ndarray):
                    embedding_str = json.dumps(embedding.tolist())
                else:
                    embedding_str = json.dumps(list(embedding))
            else:
                embedding_str = ''
            
            writer.writerow([
                detection['frame_number'],
                detection['time_seconds'],
                bbox[0], bbox[1], bbox[2], bbox[3],
                detection['confidence'],
                detection.get('sharpness', 0.0),  # Image quality
                detection['distance'],
                pitch, yaw, roll,
                1 if attended else 0,  # Binary: 1 if attended, 0 otherwise
                detection.get('age'),
                detection.get('gender'),
                detection.get('race'),
                detection.get('emotion'),
                embedding_str,  # Embedding as JSON (last column)
            ])


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
    # Store face info WITHOUT full frames to save memory
    pending_faces = []  # List of (frame_number, detection) tuples
    frame_cache = {}  # Cache recent frames: {frame_number: frame_image}
    MAX_CACHED_FRAMES = 10  # Keep only last 10 frames in cache
    
    def process_pending_faces_batch():
        """Process accumulated faces in a batch across frames."""
        if not pending_faces or not use_batch:
            return []
        
        # Re-read frames on demand (only if not in cache)
        # This uses less memory than storing all frames
        nonlocal frame_cache
        
        all_face_objs = []
        all_bboxes = []
        all_frame_images = []
        all_frame_numbers = []
        all_detection_metadata = []
        
        for frame_num, detection in pending_faces:
            # Get frame from cache or re-read
            if frame_num not in frame_cache:
                cap_temp = cv2.VideoCapture(video_path)
                cap_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame_img = cap_temp.read()
                cap_temp.release()
                if not ret:
                    continue
                # Only cache if we have room
                if len(frame_cache) < MAX_CACHED_FRAMES:
                    frame_cache[frame_num] = frame_img
            else:
                frame_img = frame_cache[frame_num]
            
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
        
        # Add detected faces to pending batch (without storing full frames)
        if frame_detections:
            # Cache current frame for this batch
            if frame_number not in frame_cache:
                frame_cache[frame_number] = frame.copy()
                # Limit cache size - remove oldest frame if needed
                if len(frame_cache) > MAX_CACHED_FRAMES:
                    oldest_frame = min(frame_cache.keys())
                    del frame_cache[oldest_frame]
            
            for detection in frame_detections:
                pending_faces.append((frame_number, detection))
        
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
    
    # Cleanup
    cap.release()
    frame_cache.clear()  # Free cached frames
    
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

