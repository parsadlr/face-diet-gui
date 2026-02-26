"""
Utility functions for face-diet video processing.

Includes quality scoring, blur detection, CSV utilities, and helper functions.
"""

import csv
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def calculate_blur_score(image_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    """
    Calculate blur score for a face region using Laplacian variance.
    Higher score = sharper image.
    
    Parameters
    ----------
    image_bgr : np.ndarray
        Full image in BGR format
    bbox : tuple
        (x, y, w, h) bounding box of the face
    
    Returns
    -------
    float
        Blur score (higher = sharper)
    """
    x, y, w, h = bbox
    # Ensure coordinates are within image bounds
    h_img, w_img = image_bgr.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    face_crop = image_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return 0.0
    
    # Convert to grayscale
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    
    # Calculate Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    return float(variance)


def calculate_pose_frontality(pose: Optional[Dict[str, float]]) -> float:
    """
    Calculate how frontal a face pose is (0.0 = profile, 1.0 = perfect frontal).
    
    Parameters
    ----------
    pose : dict or None
        Dictionary with 'yaw', 'pitch', 'roll' angles in degrees
    
    Returns
    -------
    float
        Frontality score [0.0, 1.0]
    """
    if pose is None:
        return 0.0
    
    yaw = abs(pose.get('yaw', 0.0))
    pitch = abs(pose.get('pitch', 0.0))
    roll = abs(pose.get('roll', 0.0))
    
    # Weighted combination: yaw is most important for frontality
    # Perfect frontal: yaw=0, pitch=0, roll=0
    # Use exponential decay: score = exp(-weight * angle)
    yaw_score = np.exp(-0.03 * yaw)  # yaw weight higher
    pitch_score = np.exp(-0.02 * pitch)
    roll_score = np.exp(-0.01 * roll)  # roll weight lower
    
    # Weighted average
    frontality = 0.5 * yaw_score + 0.3 * pitch_score + 0.2 * roll_score
    
    return float(np.clip(frontality, 0.0, 1.0))


def calculate_face_quality(
    detection_confidence: float,
    bbox: Tuple[int, int, int, int],
    pose: Optional[Dict[str, float]],
    image_bgr: np.ndarray,
) -> float:
    """
    Calculate overall quality score for a detected face.
    Combines detection confidence, face size, pose frontality, and sharpness.
    
    Parameters
    ----------
    detection_confidence : float
        Confidence score from face detector [0.0, 1.0]
    bbox : tuple
        (x, y, w, h) bounding box of the face
    pose : dict or None
        Dictionary with 'yaw', 'pitch', 'roll' angles in degrees
    image_bgr : np.ndarray
        Full image in BGR format
    
    Returns
    -------
    float
        Overall quality score [0.0, 1.0]
    """
    # Face size score (normalized by image area)
    x, y, w, h = bbox
    face_area = w * h
    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    size_ratio = face_area / image_area
    # Normalize: assume good face size is 5-50% of image
    size_score = np.clip(size_ratio / 0.25, 0.0, 1.0)
    
    # Pose frontality score
    frontality_score = calculate_pose_frontality(pose)
    
    # Blur/sharpness score
    blur_raw = calculate_blur_score(image_bgr, bbox)
    # Normalize blur: typical sharp faces have variance > 100, blurry < 50
    sharpness_score = np.clip(blur_raw / 200.0, 0.0, 1.0)
    
    # Weighted combination
    quality = (
        0.3 * detection_confidence +
        0.2 * size_score +
        0.3 * frontality_score +
        0.2 * sharpness_score
    )
    
    return float(np.clip(quality, 0.0, 1.0))


def frame_to_time(frame_number: int, fps: float) -> float:
    """
    Convert frame number to timestamp in seconds.
    
    Parameters
    ----------
    frame_number : int
        Frame number (0-indexed)
    fps : float
        Frames per second
    
    Returns
    -------
    float
        Time in seconds
    """
    if fps <= 0:
        return 0.0
    return float(frame_number) / fps


def write_csv_header(csv_path: str) -> None:
    """
    Write CSV header for face detection results.
    
    Parameters
    ----------
    csv_path : str
        Path to output CSV file
    """
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame_number',
            'time_seconds',
            'face_id',
            'bbox_x',
            'bbox_y',
            'bbox_w',
            'bbox_h',
            'pose_yaw',
            'pose_pitch',
            'pose_roll',
            'age',
            'gender',
            'race',
            'emotion',
            'distance_estimate'
        ])


def append_csv_row(
    csv_path: str,
    detection: Dict,
) -> None:
    """
    Append a row to the CSV file for one face instance.
    
    Parameters
    ----------
    csv_path : str
        Path to output CSV file
    detection : dict
        Detection dictionary with all required fields
    """
    x, y, w, h = detection['bbox']
    pose = detection.get('pose')
    
    # Extract pose values
    yaw = pose.get('yaw', '') if pose else ''
    pitch = pose.get('pitch', '') if pose else ''
    roll = pose.get('roll', '') if pose else ''
    
    # Extract attributes with defaults
    age = detection.get('age', '')
    gender = detection.get('gender', '')
    race = detection.get('race', '')
    emotion = detection.get('emotion', '')
    distance = detection.get('distance', '')
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            detection['frame_number'],
            f"{detection['time_seconds']:.3f}",
            detection.get('face_id', ''),
            x, y, w, h,
            f"{yaw:.2f}" if isinstance(yaw, (int, float)) else yaw,
            f"{pitch:.2f}" if isinstance(pitch, (int, float)) else pitch,
            f"{roll:.2f}" if isinstance(roll, (int, float)) else roll,
            age,
            gender,
            race,
            emotion,
            f"{distance:.3f}" if isinstance(distance, (int, float)) else distance
        ])


def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Parameters
    ----------
    emb_a : np.ndarray
        First embedding vector
    emb_b : np.ndarray
        Second embedding vector
    
    Returns
    -------
    float
        Cosine similarity [-1.0, 1.0]
    """
    denom = (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
    if denom == 0:
        return 0.0
    return float(np.dot(emb_a, emb_b) / denom)
