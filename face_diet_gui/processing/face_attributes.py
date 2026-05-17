"""
Face attribute extraction module.

Extracts pose, age, gender, race, emotion, and distance for detected faces.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import tensorflow as tf
from deepface import DeepFace
from insightface.app import FaceAnalysis
from face_diet_gui.profiler import get_profiler

# Configure TensorFlow to use GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Set GPU as default device
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"TensorFlow configured to use GPU: {gpus[0]}")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")
else:
    print("Warning: No GPU devices found for TensorFlow")


def extract_pose_from_face(face_obj) -> Optional[Dict[str, float]]:
    """
    Extract head pose angles from InsightFace face object.
    
    Parameters
    ----------
    face_obj
        InsightFace face object with landmarks and pose information
    
    Returns
    -------
    dict or None
        Dictionary with 'yaw', 'pitch', 'roll' angles in degrees, or None if unavailable
    """
    # Try to use landmark-based pose estimation
    if hasattr(face_obj, 'landmark_2d_106') and face_obj.landmark_2d_106 is not None \
       and hasattr(face_obj, 'landmark_3d_68') and face_obj.landmark_3d_68 is not None:
        
        pts2d = np.asarray(face_obj.landmark_2d_106, dtype=np.float64)
        pts3d = np.asarray(face_obj.landmark_3d_68, dtype=np.float64)
        
        # Use stable landmark indices for PnP
        stable_indices = [30, 8, 36, 45, 48, 54]
        
        if pts2d.shape[0] >= 55 and pts3d.shape[0] >= 55:
            try:
                # Note: We need image dimensions for camera matrix
                # This will be handled by passing the image to extract_all_attributes
                pass
            except Exception:
                pass
    
    # Fallback: use pose provided by face object
    if hasattr(face_obj, 'pose') and face_obj.pose is not None and len(face_obj.pose) >= 3:
        pose_vals = face_obj.pose
        return {
            'pitch': float(pose_vals[0]),
            'yaw': float(pose_vals[1]),
            'roll': float(pose_vals[2]),
        }
    
    return None


def extract_pose_with_pnp(
    face_obj,
    image_bgr: np.ndarray
) -> Optional[Dict[str, float]]:
    """
    Extract head pose using PnP solve with landmarks.
    
    Parameters
    ----------
    face_obj
        InsightFace face object with landmarks
    image_bgr : np.ndarray
        Full image in BGR format (needed for camera matrix)
    
    Returns
    -------
    dict or None
        Dictionary with 'yaw', 'pitch', 'roll' angles in degrees
    """
    try:
        if not (hasattr(face_obj, 'landmark_2d_106') and face_obj.landmark_2d_106 is not None \
                and hasattr(face_obj, 'landmark_3d_68') and face_obj.landmark_3d_68 is not None):
            return extract_pose_from_face(face_obj)
        
        h, w = image_bgr.shape[:2]
        pts2d = np.asarray(face_obj.landmark_2d_106, dtype=np.float64)
        pts3d = np.asarray(face_obj.landmark_3d_68, dtype=np.float64)
        
        # Stable landmark indices
        stable_indices = [30, 8, 36, 45, 48, 54]
        
        if pts2d.shape[0] < 55 or pts3d.shape[0] < 55:
            return extract_pose_from_face(face_obj)
        
        # Camera intrinsics
        focal_length = float(w)
        center = (float(w) / 2.0, float(h) / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        
        image_points = np.array([pts2d[i] for i in stable_indices], dtype=np.float64)
        model_points = np.array([pts3d[i] for i in stable_indices], dtype=np.float64)
        
        success, rvec, tvec = cv2.solvePnP(
            model_points, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            rot_mat, _ = cv2.Rodrigues(rvec)
            rx, ry, rz = cv2.RQDecomp3x3(rot_mat)[-1]
            pitch = float(rx)
            yaw = float(ry)
            roll = float(rz)
            
            # Normalize angles to [-180, 180]
            def norm_angle(a: float) -> float:
                while a > 180.0:
                    a -= 360.0
                while a < -180.0:
                    a += 360.0
                return a
            
            return {
                'pitch': norm_angle(pitch),
                'yaw': norm_angle(yaw),
                'roll': norm_angle(roll),
            }
    except Exception:
        pass
    
    return extract_pose_from_face(face_obj)


def extract_age_gender_race_emotion(
    image_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Dict:
    """
    Extract age, gender, race, and emotion using DeepFace (single face).
    
    Parameters
    ----------
    image_bgr : np.ndarray
        Full image in BGR format
    bbox : tuple
        (x, y, w, h) bounding box of the face
    
    Returns
    -------
    dict
        Dictionary with 'age', 'gender', 'race', 'emotion' keys
        Returns None values if analysis fails
    """
    try:
        x, y, w, h = bbox
        # Crop face with padding
        pad = int(max(w, h) * 0.1)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image_bgr.shape[1], x + w + pad)
        y2 = min(image_bgr.shape[0], y + h + pad)
        
        face_crop = image_bgr[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return {'age': None, 'gender': None, 'race': None, 'emotion': None}
        
        # Run DeepFace analysis - use numpy array directly (faster than saving to disk)
        profiler = get_profiler()
        with profiler.time_block("deepface_analysis"):
            # DeepFace.analyze can accept numpy array directly, which is faster
            try:
                result = DeepFace.analyze(
                    img_path=face_crop,  # Can accept numpy array
                    actions=['age', 'gender', 'emotion', 'race'],
                    enforce_detection=False,
                    detector_backend='skip',
                    silent=True,
                )
            except Exception as e:
                # Fallback: try with image path if array fails
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    cv2.imwrite(tmp.name, face_crop)
                    try:
                        result = DeepFace.analyze(
                            img_path=tmp.name,
                            actions=['age', 'gender', 'emotion', 'race'],
                            enforce_detection=False,
                            detector_backend='skip',
                            silent=True,
                        )
                    finally:
                        if os.path.exists(tmp.name):
                            os.unlink(tmp.name)
        
        # DeepFace returns a list, get first result
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        
        # Extract dominant values
        attributes = {
            'age': result.get('age', None),
            'gender': max(result.get('gender', {}).items(), key=lambda x: x[1])[0] if result.get('gender') else None,
            'race': result.get('dominant_race', None),
            'emotion': result.get('dominant_emotion', None),
        }
        
        return attributes
        
    except Exception as e:
        print(f"Warning: Failed to analyze face attributes: {e}")
        return {'age': None, 'gender': None, 'race': None, 'emotion': None}


def extract_age_gender_race_emotion_batch(
    image_bgr: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    batch_size: int = 8,
) -> List[Dict]:
    """
    Extract age, gender, race, and emotion for multiple faces using DeepFace batch processing.
    
    This is more efficient than calling extract_age_gender_race_emotion multiple times
    because DeepFace can process multiple images in a single call, reducing model loading overhead.
    
    Parameters
    ----------
    image_bgr : np.ndarray
        Full image in BGR format
    bboxes : List[tuple]
        List of (x, y, w, h) bounding boxes
    batch_size : int
        Maximum number of faces to process in one batch
    
    Returns
    -------
    List[Dict]
        List of attribute dictionaries, one per bbox
    """
    if not bboxes:
        return []
    
    profiler = get_profiler()
    all_results = []
    
    # Process in batches
    for batch_start in range(0, len(bboxes), batch_size):
        batch_end = min(batch_start + batch_size, len(bboxes))
        batch_bboxes = bboxes[batch_start:batch_end]
        
        # Crop all faces in this batch
        face_crops = []
        valid_indices = []
        
        for i, (x, y, w, h) in enumerate(batch_bboxes):
            pad = int(max(w, h) * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(image_bgr.shape[1], x + w + pad)
            y2 = min(image_bgr.shape[0], y + h + pad)
            
            face_crop = image_bgr[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                face_crops.append(face_crop)
                valid_indices.append(i)
        
        if not face_crops:
            # No valid crops in this batch, return None for all
            all_results.extend([{'age': None, 'gender': None, 'race': None, 'emotion': None}] * len(batch_bboxes))
            continue
        
        # Run DeepFace batch analysis
        with profiler.time_block("deepface_batch_analysis"):
            try:
                # DeepFace.analyze can accept a list of images for batch processing
                results = DeepFace.analyze(
                    img_path=face_crops,  # List of numpy arrays
                    actions=['age', 'gender', 'emotion', 'race'],
                    enforce_detection=False,
                    detector_backend='skip',
                    silent=True,
                )
                
                # DeepFace returns a list of results
                if not isinstance(results, list):
                    results = [results]
                
                # Map results back to original bbox indices
                batch_results = [{'age': None, 'gender': None, 'race': None, 'emotion': None}] * len(batch_bboxes)
                
                for idx, result in zip(valid_indices, results):
                    if isinstance(result, dict):
                        batch_results[idx] = {
                            'age': result.get('age', None),
                            'gender': max(result.get('gender', {}).items(), key=lambda x: x[1])[0] if result.get('gender') else None,
                            'race': result.get('dominant_race', None),
                            'emotion': result.get('dominant_emotion', None),
                        }
                
                all_results.extend(batch_results)
                
            except Exception as e:
                # Fallback to individual processing if batch fails
                print(f"Warning: Batch DeepFace analysis failed, falling back to individual: {e}")
                for bbox in batch_bboxes:
                    result = extract_age_gender_race_emotion(image_bgr, bbox)
                    all_results.append(result)
    
    return all_results


# ---------------------------------------------------------------------------
# PPD-based distance estimation
# ---------------------------------------------------------------------------

# Hardcoded pixel-per-degree polynomial mapping (2nd-order, full basis).
# Coefficients derived from the wide-angle lens calibration.
_PPD_MAPPING: Dict = {
    "center_x":        959.5,
    "center_y":        539.5,
    "exponents":       [[0,0],[0,1],[0,2],[1,0],[1,1],[2,0]],
    "coefficients_x":  [
        14.05476210542249,
        -0.0876786050844629,
        -0.27582551113140186,
        -0.3989892392520565,
         0.8074276506823053,
        11.044886529750261,
    ],
    "coefficients_y":  [
        15.210399867098543,
         0.4202796457210294,
         2.076840035068318,
        -0.35104979073345033,
         0.6885119864666166,
         5.952656667352058,
    ],
}

# Physical bounding-box dimensions (metres) that InsightFace captures.
_FACE_WIDTH_M_FEMALE  = 0.1364
_FACE_HEIGHT_M_FEMALE = 0.1758
_FACE_WIDTH_M_MALE    = 0.1448
_FACE_HEIGHT_M_MALE   = 0.1908


def _ppd_at(x: float, y: float):
    """Return (ppd_x, ppd_y) at pixel position (x, y) using the hardcoded mapping."""
    cx, cy = _PPD_MAPPING["center_x"], _PPD_MAPPING["center_y"]
    xn, yn = (x - cx) / cx, (y - cy) / cy
    exps = _PPD_MAPPING["exponents"]
    row = [(xn ** ei) * (yn ** ej) for ei, ej in exps]
    ppd_x = sum(c * r for c, r in zip(_PPD_MAPPING["coefficients_x"], row))
    ppd_y = sum(c * r for c, r in zip(_PPD_MAPPING["coefficients_y"], row))
    return ppd_x, ppd_y


def _integrate_deg(a: float, b: float, fixed: float, axis: str, n: int = 60):
    """Trapezoidal integral of 1/ppd from a to b (returns degrees, or None)."""
    samples = np.linspace(a, b, n)
    inv_ppd = np.empty(n)
    for i, s in enumerate(samples):
        px, py = _ppd_at(s, fixed) if axis == "x" else _ppd_at(fixed, s)
        pv = px if axis == "x" else py
        if pv <= 0:
            return None
        inv_ppd[i] = 1.0 / pv
    result = float(np.trapz(inv_ppd, samples))
    return result if result > 0 else None


def estimate_distance(bbox: Tuple[int, int, int, int], gender: Optional[str] = None) -> float:
    """
    Estimate viewing distance to a face in metres using the PPD polynomial mapping.

    Converts the bounding-box pixel span to angular size via numerical
    integration of 1/ppd along each axis, applies the thin-lens formula to
    get per-axis distances, then returns the corrected geometric mean:
        sqrt(dist_w * dist_h) + 0.25 * ln(sqrt(dist_w * dist_h))

    Parameters
    ----------
    bbox : tuple
        (x, y, w, h) bounding box in pixels.
    gender : str or None
        ``"M"`` / ``"Man"`` / ``"Male"`` selects male face dimensions;
        anything else (including ``None``) uses female defaults.

    Returns
    -------
    float
        Estimated distance in metres, or ``float('inf')`` for degenerate inputs.
    """
    import math as _math

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return float("inf")

    is_male = isinstance(gender, str) and gender.strip().upper() in {"M", "MAN", "MALE"}
    fw = _FACE_WIDTH_M_MALE  if is_male else _FACE_WIDTH_M_FEMALE
    fh = _FACE_HEIGHT_M_MALE if is_male else _FACE_HEIGHT_M_FEMALE

    cx, cy = x + w / 2.0, y + h / 2.0
    theta_w = _integrate_deg(x, x + w, cy, "x")
    theta_h = _integrate_deg(y, y + h, cx, "y")

    def _to_dist(deg, physical_m):
        if deg is None or deg <= 0:
            return None
        return (physical_m / 2.0) / _math.tan(_math.radians(deg) / 2.0)

    dist_w = _to_dist(theta_w, fw)
    dist_h = _to_dist(theta_h, fh)

    if dist_w is not None and dist_h is not None:
        combined = _math.sqrt(dist_w * dist_h)
    else:
        combined = dist_w or dist_h

    if combined is None or combined <= 0:
        return float("inf")

    return float(combined + 0.25 * _math.log(combined))


def extract_all_attributes(
    face_obj,
    image_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Dict:
    """
    Extract all face attributes: pose, age, gender, race, emotion, distance.
    
    Parameters
    ----------
    face_obj
        InsightFace face object
    image_bgr : np.ndarray
        Full image in BGR format
    bbox : tuple
        (x, y, w, h) bounding box of the face
    
    Returns
    -------
    dict
        Dictionary with all attributes
    """
    profiler = get_profiler()
    
    # Extract pose
    with profiler.time_block("pose_extraction"):
        pose = extract_pose_with_pnp(face_obj, image_bgr)
    
    # Extract age, gender, race, emotion
    demographics = extract_age_gender_race_emotion(image_bgr, bbox)
    
    # Estimate distance (very fast, no need to profile)
    distance = estimate_distance(bbox)
    
    return {
        'pose': pose,
        'age': demographics['age'],
        'gender': demographics['gender'],
        'race': demographics['race'],
        'emotion': demographics['emotion'],
        'distance': distance,
    }


def extract_all_attributes_batch(
    face_objs: List,
    image_bgr: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    batch_size: int = 8,
) -> List[Dict]:
    """
    Extract all face attributes for multiple faces using batch processing.
    
    This is more efficient than calling extract_all_attributes multiple times
    because DeepFace can process multiple faces in a single batch call.
    
    Parameters
    ----------
    face_objs : List
        List of InsightFace face objects
    image_bgr : np.ndarray
        Full image in BGR format
    bboxes : List[tuple]
        List of (x, y, w, h) bounding boxes
    batch_size : int
        Maximum number of faces to process in one DeepFace batch
    
    Returns
    -------
    List[Dict]
        List of attribute dictionaries, one per face
    """
    if not face_objs or not bboxes:
        return []
    
    if len(face_objs) != len(bboxes):
        raise ValueError("face_objs and bboxes must have the same length")
    
    profiler = get_profiler()
    all_attributes = []
    
    # Extract poses (must be done individually)
    poses = []
    for face_obj in face_objs:
        with profiler.time_block("pose_extraction"):
            pose = extract_pose_with_pnp(face_obj, image_bgr)
        poses.append(pose)
    
    # Extract demographics in batch
    demographics_list = extract_age_gender_race_emotion_batch(image_bgr, bboxes, batch_size=batch_size)
    
    # Combine all attributes
    for i, (pose, demographics, bbox) in enumerate(zip(poses, demographics_list, bboxes)):
        distance = estimate_distance(bbox)
        all_attributes.append({
            'pose': pose,
            'age': demographics['age'],
            'gender': demographics['gender'],
            'race': demographics['race'],
            'emotion': demographics['emotion'],
            'distance': distance,
        })
    
    return all_attributes
