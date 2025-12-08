"""
Process multiple recording sessions for a participant with global face IDs (v2).

This improved version:
1. Processes each session independently and saves embeddings
2. Combines all sessions and assigns global face IDs using embeddings
3. No need to re-process videos - just re-cluster embeddings

Usage:
    # Step 1: Process each session (run once per session)
    python process_participant_v2.py process-session <session_dir> [options]
    
    # Step 2: Combine sessions with global IDs (run once per participant)
    python process_participant_v2.py combine-sessions <participant_dir> [options]
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from video_processor import process_video
from face_detection import assign_face_ids


def process_single_session(
    session_dir: str,
    output_csv: str = None,
    sampling_rate: int = 30,
    start_time: float = None,
    end_time: float = None,
    use_gpu: bool = False,
    batch_size: int = 32,
):
    """
    Process a single session with the staged pipeline.
    
    Detects faces and saves embeddings WITHOUT assigning face IDs.
    Face IDs will be assigned globally across all sessions later.
    
    Parameters
    ----------
    session_dir : str
        Path to session directory containing 'scenevideo.*'
    output_csv : str, optional
        Output CSV path (default: <session_dir>/faces_detections.csv)
    sampling_rate : int
        Process every N frames
    start_time : float, optional
        Start time in seconds (for testing)
    end_time : float, optional
        End time in seconds (for testing)
    use_gpu : bool
        Whether to use GPU
    batch_size : int
        Batch size for DeepFace Stage 2
    
    Returns
    -------
    Dict
        Processing results
    """
    session_path = Path(session_dir)
    
    # Find video file
    video_files = list(session_path.glob("scenevideo.*"))
    if not video_files:
        raise FileNotFoundError(f"No scenevideo file found in {session_dir}")
    
    video_path = str(video_files[0])
    
    # Find eye tracking file
    eye_tracking_path = session_path / "eye_tracking.tsv"
    if not eye_tracking_path.exists():
        eye_tracking_path = None
        print("Warning: No eye_tracking.tsv found. 'attended' flag will be False for all faces.")
    else:
        eye_tracking_path = str(eye_tracking_path)
    
    # If test duration specified, randomly select a window
    if end_time is not None and start_time is None:
        # end_time is being used as test_duration
        import cv2
        import random
        
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps
            cap.release()
            
            test_duration = end_time
            
            # Randomly choose start time that allows full test_duration
            max_start = max(0, video_duration - test_duration)
            if max_start > 0:
                start_time = random.uniform(0, max_start)
                end_time = start_time + test_duration
                print(f"Randomly selected test window: {start_time:.1f}s - {end_time:.1f}s (duration: {test_duration}s)")
            else:
                # Video shorter than test duration
                start_time = 0.0
                end_time = video_duration
                print(f"Video shorter than test duration. Using full video: 0.0s - {end_time:.1f}s")
    
    # Set default output path
    if output_csv is None:
        output_csv = str(session_path / "faces_detections.csv")
    
    print("=" * 80)
    print(f"PROCESSING SESSION: {session_path.name}")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"Eye tracking: {eye_tracking_path if eye_tracking_path else 'Not found'}")
    print(f"Output: {output_csv}")
    if start_time is not None and end_time is not None:
        print(f"Time range: {start_time}s - {end_time}s (testing mode)")
    print()
    
    # Process with Stage 1 only (no ID assignment, that happens globally later)
    from video_processor import process_video_stage1, process_video_stage2
    
    print("Stage 1: Detecting faces with InsightFace (no ID assignment)...")
    
    # Run Stage 1 with eye tracking
    stage1_result = process_video_stage1(
        video_path=video_path,
        output_csv=output_csv,
        sampling_rate=sampling_rate,
        start_time=start_time,
        end_time=end_time,
        clustering_method='threshold',
        similarity_threshold=0.6,  # Will be ignored since we skip ID assignment
        use_gpu=use_gpu,
        progress_callback=None,
        eye_tracking_path=eye_tracking_path,
    )
    
    print(f"\nStage 1 complete: {stage1_result['total_detections']} faces detected")
    print("Stage 2: Adding DeepFace attributes...")
    
    # Run Stage 2 for attributes
    stage2_result = process_video_stage2(
        video_path=video_path,
        stage1_csv=output_csv,
        output_csv=output_csv,
        batch_size=batch_size,
    )
    
    result = {
        'total_detections': stage1_result['total_detections'],
    }
    
    print("\n" + "=" * 80)
    print(f"SESSION COMPLETE: {session_path.name}")
    print("=" * 80)
    print(f"Total face detections: {result['total_detections']}")
    print(f"Output saved to: {output_csv}")
    print(f"Note: Face IDs will be assigned globally across all sessions")
    print()
    
    return {
        'total_detections': result['total_detections'],
    }


def load_embeddings_from_csv(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load face data and embeddings from CSV.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file with embeddings
    
    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        DataFrame with all face data, and array of embeddings
    """
    df = pd.read_csv(csv_path)
    
    # Parse embeddings from JSON strings
    embeddings = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        embedding_str = row.get('embedding', '')
        if embedding_str and not pd.isna(embedding_str):
            try:
                embedding = np.array(json.loads(embedding_str))
                embeddings.append(embedding)
                valid_indices.append(idx)
            except (json.JSONDecodeError, ValueError):
                print(f"Warning: Could not parse embedding at row {idx}")
    
    if not embeddings:
        raise ValueError(f"No valid embeddings found in {csv_path}")
    
    embeddings_array = np.array(embeddings)
    df_valid = df.loc[valid_indices].reset_index(drop=True)
    
    return df_valid, embeddings_array


def combine_sessions_with_global_ids(
    participant_dir: str,
    output_csv: str = None,
    similarity_threshold: float = 0.6,
    clustering_method: str = 'threshold',
):
    """
    Combine multiple session CSVs and assign global face IDs.
    
    Uses embeddings from each session to identify the same person across sessions.
    
    Parameters
    ----------
    participant_dir : str
        Path to participant directory containing session subdirectories
    output_csv : str, optional
        Output path for combined CSV (default: <participant_dir>/faces_combined.csv)
    similarity_threshold : float
        Cosine similarity threshold for matching faces across sessions (default: 0.6)
    clustering_method : str
        'threshold' or 'dbscan' for global clustering
    
    Returns
    -------
    Dict
        Summary statistics
    """
    participant_path = Path(participant_dir)
    
    if output_csv is None:
        output_csv = str(participant_path / "faces_combined.csv")
    
    print("=" * 80)
    print(f"COMBINING SESSIONS WITH GLOBAL IDs")
    print("=" * 80)
    print(f"Participant: {participant_path.name}")
    print(f"Similarity threshold: {similarity_threshold}")
    print()
    
    # Find all session CSV files
    session_csvs = []
    for session_dir in sorted(participant_path.iterdir()):
        if not session_dir.is_dir():
            continue
        
        # Try both filenames
        csv_path = session_dir / "faces_detections.csv"
        if not csv_path.exists():
            csv_path = session_dir / "faces.csv"
        
        if csv_path.exists():
            session_csvs.append({
                'session_name': session_dir.name,
                'csv_path': str(csv_path),
            })
    
    if not session_csvs:
        raise FileNotFoundError(f"No session CSV files (faces.csv) found in {participant_dir}")
    
    print(f"Found {len(session_csvs)} session(s):")
    for session in session_csvs:
        print(f"  - {session['session_name']}")
    print()
    
    # Load all sessions with embeddings
    all_data = []
    all_embeddings = []
    session_offsets = [0]  # Track where each session starts in the combined array
    
    for session in session_csvs:
        print(f"Loading {session['session_name']}...")
        try:
            df, embeddings = load_embeddings_from_csv(session['csv_path'])
            
            # Add session metadata
            df['session_name'] = session['session_name']
            # Note: session CSVs no longer have face_id (assigned globally only)
            
            all_data.append(df)
            all_embeddings.append(embeddings)
            session_offsets.append(session_offsets[-1] + len(embeddings))
            
            print(f"  Loaded {len(df)} faces with embeddings")
        except ValueError as e:
            print(f"  ⚠️  Skipping: {e}")
            continue
    
    # Check if we have any data
    if not all_data:
        raise ValueError("No sessions with valid face detections found. All sessions are empty.")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_embeddings = np.vstack(all_embeddings)
    
    print(f"\nTotal faces across all sessions: {len(combined_df)}")
    print(f"Combined embedding matrix shape: {combined_embeddings.shape}")
    
    # Assign global face IDs using embeddings
    print(f"\nAssigning global face IDs (method={clustering_method}, threshold={similarity_threshold})...")
    
    if clustering_method == 'threshold':
        global_ids = assign_global_ids_threshold(
            combined_embeddings,
            similarity_threshold=similarity_threshold
        )
    elif clustering_method == 'dbscan':
        from sklearn.cluster import DBSCAN
        # Convert similarity threshold to distance (1 - cosine similarity)
        eps = 1.0 - similarity_threshold
        clustering = DBSCAN(eps=eps, min_samples=1, metric='cosine')
        labels = clustering.fit_predict(combined_embeddings)
        global_ids = [f"FACE_{label:05d}" if label >= 0 else "UNKNOWN" for label in labels]
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}")
    
    # Update dataframe with global IDs
    combined_df['face_id'] = global_ids
    
    # Reorder columns: face_id, session_name first, embedding last
    cols = list(combined_df.columns)
    
    # Remove special columns that we'll reorder
    cols_to_reorder = []
    if 'embedding' in cols:
        cols.remove('embedding')
        cols_to_reorder.append('embedding')
    if 'face_id' in cols:
        cols.remove('face_id')
    if 'session_name' in cols:
        cols.remove('session_name')
    
    # New order: face_id, session_name, ...other columns..., embedding
    new_cols = ['face_id', 'session_name'] + cols
    if 'embedding' in cols_to_reorder:
        new_cols = new_cols + ['embedding']
    
    combined_df = combined_df[new_cols]
    
    unique_global_ids = len(set(global_ids) - {'UNKNOWN'})
    print(f"Assigned {unique_global_ids} unique global face IDs")
    
    # Print per-session breakdown
    print("\nPer-session breakdown:")
    for session in session_csvs:
        session_faces = combined_df[combined_df['session_name'] == session['session_name']]
        unique_in_session = session_faces['face_id'].nunique()
        print(f"  {session['session_name']}: {len(session_faces)} faces, {unique_in_session} global IDs")
    
    # Save combined CSV
    combined_df.to_csv(output_csv, index=False)
    print(f"\nCombined CSV saved to: {output_csv}")
    
    # Create ID mapping summary
    mapping_file = str(Path(output_csv).with_suffix('.mapping.txt'))
    with open(mapping_file, 'w') as f:
        f.write("Global Face ID Mapping\n")
        f.write("=" * 80 + "\n\n")
        f.write("Each face ID represents a unique person detected across all sessions.\n\n")
        
        for global_id in sorted(set(global_ids) - {'UNKNOWN'}):
            f.write(f"{global_id}:\n")
            faces_with_id = combined_df[combined_df['face_id'] == global_id]
            
            for session_name in faces_with_id['session_name'].unique():
                session_faces = faces_with_id[faces_with_id['session_name'] == session_name]
                num_attended = session_faces['attended'].sum() if 'attended' in session_faces.columns else 0
                f.write(f"  {session_name}: {len(session_faces)} instances")
                if 'attended' in session_faces.columns:
                    f.write(f" ({int(num_attended)} attended)")
                f.write("\n")
            f.write("\n")
    
    print(f"ID mapping saved to: {mapping_file}")
    
    print("\n" + "=" * 80)
    print("GLOBAL ID ASSIGNMENT COMPLETE")
    print("=" * 80)
    
    return {
        'total_faces': len(combined_df),
        'unique_global_ids': unique_global_ids,
        'num_sessions': len(session_csvs),
        'output_csv': output_csv,
    }


def assign_global_ids_threshold(
    embeddings: np.ndarray,
    similarity_threshold: float = 0.6
) -> List[str]:
    """
    Assign global IDs using cosine similarity threshold.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Array of face embeddings (N x D)
    similarity_threshold : float
        Minimum cosine similarity to consider two faces the same person
    
    Returns
    -------
    List[str]
        List of face IDs
    """
    n = len(embeddings)
    assigned_ids = [-1] * n
    next_id = 0
    
    for i in range(n):
        if assigned_ids[i] >= 0:
            continue  # Already assigned
        
        # Assign new ID
        assigned_ids[i] = next_id
        
        # Find all similar faces
        for j in range(i + 1, n):
            if assigned_ids[j] >= 0:
                continue  # Already assigned
            
            # Compute cosine similarity
            sim = cosine_similarity(
                embeddings[i:i+1],
                embeddings[j:j+1]
            )[0, 0]
            
            if sim >= similarity_threshold:
                assigned_ids[j] = next_id
        
        next_id += 1
    
    # Convert to face ID strings (5-digit format)
    face_ids = [f"FACE_{id:05d}" for id in assigned_ids]
    return face_ids


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process participant sessions with global face IDs (v2)"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Process single session command
    process_parser = subparsers.add_parser(
        'process-session',
        help='Process a single session'
    )
    process_parser.add_argument('session_dir', help='Path to session directory')
    process_parser.add_argument('-o', '--output', help='Output CSV path')
    process_parser.add_argument('-s', '--sampling-rate', type=int, default=30)
    process_parser.add_argument('--start-time', type=float, help='Start time (seconds)')
    process_parser.add_argument('--end-time', type=float, help='End time (seconds)')
    process_parser.add_argument('-t', '--threshold', type=float, default=0.4)
    process_parser.add_argument('--gpu', action='store_true')
    process_parser.add_argument('--batch-size', type=int, default=32)
    
    # Combine sessions command
    combine_parser = subparsers.add_parser(
        'combine-sessions',
        help='Combine sessions with global IDs'
    )
    combine_parser.add_argument('participant_dir', help='Path to participant directory')
    combine_parser.add_argument('-o', '--output', help='Output CSV path')
    combine_parser.add_argument('-t', '--threshold', type=float, default=0.6,
                                help='Similarity threshold for global IDs')
    combine_parser.add_argument('-c', '--clustering', choices=['threshold', 'dbscan'],
                                default='threshold', help='Clustering method')
    
    args = parser.parse_args()
    
    if args.command == 'process-session':
        process_single_session(
            session_dir=args.session_dir,
            output_csv=args.output,
            sampling_rate=args.sampling_rate,
            start_time=args.start_time,
            end_time=args.end_time,
            similarity_threshold=args.threshold,
            use_gpu=args.gpu,
            batch_size=args.batch_size,
        )
    
    elif args.command == 'combine-sessions':
        combine_sessions_with_global_ids(
            participant_dir=args.participant_dir,
            output_csv=args.output,
            similarity_threshold=args.threshold,
            clustering_method=args.clustering,
        )
    
    else:
        parser.print_help()

