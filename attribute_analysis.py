from typing import Dict, List, Optional
import warnings
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from deepface import DeepFace


def analyze_face_attributes(
    image_bgr: np.ndarray,
    face_box: tuple,
    enforce_detection: bool = False,
) -> Optional[Dict]:
    """
    Analyze facial attributes using DeepFace.
    
    Parameters
    ----------
    image_bgr : np.ndarray
        Full image in BGR format
    face_box : tuple
        (x, y, w, h) bounding box of the face
    enforce_detection : bool
        If False, will run analysis even if DeepFace doesn't detect a face
    
    Returns
    -------
    dict or None
        Dictionary with keys: age, gender, emotion, race (ethnicity)
        Returns None if analysis fails
    """
    try:
        x, y, w, h = face_box
        # Crop face with some padding
        pad = int(max(w, h) * 0.1)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image_bgr.shape[1], x + w + pad)
        y2 = min(image_bgr.shape[0], y + h + pad)
        
        face_crop = image_bgr[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return None
        
        # Run DeepFace analysis
        result = DeepFace.analyze(
            img_path=face_crop,
            actions=['age', 'gender', 'emotion', 'race'],
            enforce_detection=enforce_detection,
            detector_backend='skip',  # Skip detection since we already have the face
            silent=True,
        )
        
        # DeepFace returns a list, get first result
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        
        # Extract dominant values
        attributes = {
            'age': result.get('age', 0),
            'gender': max(result.get('gender', {}).items(), key=lambda x: x[1])[0] if result.get('gender') else 'Unknown',
            'gender_confidence': max(result.get('gender', {}).values()) if result.get('gender') else 0.0,
            'emotion': result.get('dominant_emotion', 'Unknown'),
            'emotion_scores': result.get('emotion', {}),
            'race': result.get('dominant_race', 'Unknown'),
            'race_scores': result.get('race', {}),
        }
        
        return attributes
        
    except Exception as e:
        print(f"Warning: Failed to analyze face attributes: {e}")
        return None


def analyze_all_faces(
    image_bgr: np.ndarray,
    face_boxes: List[tuple],
) -> List[Optional[Dict]]:
    """
    Analyze attributes for all detected faces.
    
    Parameters
    ----------
    image_bgr : np.ndarray
        Full image in BGR format
    face_boxes : List[tuple]
        List of (x, y, w, h) bounding boxes
    
    Returns
    -------
    List[Optional[Dict]]
        List of attribute dictionaries (one per face)
    """
    results = []
    for box in face_boxes:
        attrs = analyze_face_attributes(image_bgr, box, enforce_detection=False)
        results.append(attrs)
    return results

