"""
Face detection, embedding extraction, and identity clustering module.

Implements two-pass face identity assignment:
1. Detect faces and extract embeddings
2. Cluster embeddings and assign consistent IDs
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN

from profiler import get_profiler
from utils import calculate_face_quality, cosine_similarity


def initialize_detector(
    model_name: str = "buffalo_l",
    use_gpu: bool = False,
) -> FaceAnalysis:
    """
    Initialize InsightFace detector.
    
    Parameters
    ----------
    model_name : str
        InsightFace model name
    use_gpu : bool
        Whether to use GPU
    
    Returns
    -------
    FaceAnalysis
        Initialized face analysis app
    """
    # Use buffalo_l as the default model (more commonly available)
    ctx_id = 0 if use_gpu else -1
    
    # Explicitly set providers for GPU to ensure CUDA is used
    if use_gpu:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        app = FaceAnalysis(name=model_name, providers=providers)
        print(f"InsightFace initialized with GPU support (CUDAExecutionProvider)")
    else:
        providers = ['CPUExecutionProvider']
        app = FaceAnalysis(name=model_name, providers=providers)
        print(f"InsightFace initialized with CPU only")
    
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def detect_faces_in_frame(
    detector: FaceAnalysis,
    image_bgr: np.ndarray,
) -> List[Dict]:
    """
    Detect all faces in a frame and extract embeddings.
    
    Parameters
    ----------
    detector : FaceAnalysis
        InsightFace detector
    image_bgr : np.ndarray
        Frame image in BGR format
    
    Returns
    -------
    List[Dict]
        List of face detections, each containing:
        - bbox: (x, y, w, h)
        - embedding: np.ndarray
        - confidence: float
        - face_obj: InsightFace face object (for attribute extraction)
    """
    profiler = get_profiler()
    with profiler.time_block("insightface_detection"):
        faces = detector.get(image_bgr)
    
    detections = []
    
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        
        detection = {
            'bbox': bbox,
            'embedding': face.embedding.astype(np.float32) if hasattr(face, 'embedding') else None,
            'confidence': float(face.det_score) if hasattr(face, 'det_score') else 0.0,
            'face_obj': face,  # Keep for later attribute extraction
        }
        
        detections.append(detection)
    
    return detections


def cluster_embeddings_threshold(
    embeddings: List[np.ndarray],
    threshold: float = 0.6,
) -> List[int]:
    """
    Cluster embeddings using threshold-based similarity.
    
    Uses greedy clustering: first embedding becomes cluster 0,
    subsequent embeddings join existing clusters if similarity > threshold,
    otherwise create new cluster.
    
    Parameters
    ----------
    embeddings : List[np.ndarray]
        List of face embeddings
    threshold : float
        Cosine similarity threshold for same identity
    
    Returns
    -------
    List[int]
        Cluster labels for each embedding
    """
    if not embeddings:
        return []
    
    cluster_labels = []
    cluster_centroids = []
    
    for emb in embeddings:
        if emb is None:
            cluster_labels.append(-1)  # Unknown
            continue
        
        # Find best matching cluster
        best_cluster = -1
        best_similarity = -1.0
        
        for cluster_id, centroid in enumerate(cluster_centroids):
            sim = cosine_similarity(emb, centroid)
            if sim > best_similarity:
                best_similarity = sim
                best_cluster = cluster_id
        
        # Assign to cluster if similarity exceeds threshold
        if best_similarity > threshold:
            cluster_labels.append(best_cluster)
            # Update centroid (running average)
            old_centroid = cluster_centroids[best_cluster]
            count = cluster_labels.count(best_cluster)
            new_centroid = (old_centroid * (count - 1) + emb) / count
            cluster_centroids[best_cluster] = new_centroid
        else:
            # Create new cluster
            new_cluster_id = len(cluster_centroids)
            cluster_labels.append(new_cluster_id)
            cluster_centroids.append(emb.copy())
    
    return cluster_labels


def cluster_embeddings_dbscan(
    embeddings: List[np.ndarray],
    eps: float = 0.4,
    min_samples: int = 2,
) -> List[int]:
    """
    Cluster embeddings using DBSCAN.
    
    DBSCAN uses distance metric (1 - cosine similarity) for clustering.
    
    Parameters
    ----------
    embeddings : List[np.ndarray]
        List of face embeddings
    eps : float
        Maximum distance between samples (1 - similarity threshold)
    min_samples : int
        Minimum samples in a neighborhood for core point
    
    Returns
    -------
    List[int]
        Cluster labels (-1 for noise/outliers)
    """
    if not embeddings:
        return []
    
    # Filter out None embeddings
    valid_embeddings = []
    valid_indices = []
    
    for i, emb in enumerate(embeddings):
        if emb is not None:
            valid_embeddings.append(emb)
            valid_indices.append(i)
    
    if not valid_embeddings:
        return [-1] * len(embeddings)
    
    # Stack embeddings
    X = np.vstack(valid_embeddings)
    
    # DBSCAN with cosine distance (use precomputed distance matrix)
    # Distance = 1 - cosine_similarity
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim
    similarity_matrix = sklearn_cosine_sim(X)
    distance_matrix = 1.0 - similarity_matrix
    
    # Run DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(distance_matrix)
    
    # Map back to original indices
    result = [-1] * len(embeddings)
    for i, idx in enumerate(valid_indices):
        result[idx] = int(labels[i])
    
    return result


def assign_face_ids(
    detections: List[Dict],
    clustering_method: str = 'threshold',
    similarity_threshold: float = 0.6,
    dbscan_eps: float = 0.4,
    dbscan_min_samples: int = 2,
) -> None:
    """
    Assign face IDs to all detections based on clustering.
    Modifies detections in-place to add 'face_id' field.
    
    Parameters
    ----------
    detections : List[Dict]
        List of all detections from Pass 1
    clustering_method : str
        'threshold' or 'dbscan'
    similarity_threshold : float
        Threshold for threshold-based clustering
    dbscan_eps : float
        Epsilon for DBSCAN
    dbscan_min_samples : int
        Min samples for DBSCAN
    """
    # Extract embeddings
    embeddings = [d.get('embedding') for d in detections]
    
    profiler = get_profiler()
    
    # Cluster embeddings
    if clustering_method == 'dbscan':
        with profiler.time_block("dbscan_clustering"):
            cluster_labels = cluster_embeddings_dbscan(
                embeddings,
                eps=dbscan_eps,
                min_samples=dbscan_min_samples
            )
    else:
        with profiler.time_block("threshold_clustering"):
            cluster_labels = cluster_embeddings_threshold(
                embeddings,
                threshold=similarity_threshold
            )
    
    # Assign face IDs
    for detection, label in zip(detections, cluster_labels):
        if label == -1:
            detection['face_id'] = 'UNKNOWN'
        else:
            detection['face_id'] = f'FACE_{label:03d}'


def find_representative_instances(detections: List[Dict]) -> Dict[str, Dict]:
    """
    Find the best representative instance for each face ID.
    
    Representative = detection closest to cluster centroid embedding.
    
    Parameters
    ----------
    detections : List[Dict]
        All detections with assigned face IDs
    
    Returns
    -------
    Dict[str, Dict]
        Mapping from face_id to representative detection
    """
    # Group detections by face_id
    face_groups = {}
    for detection in detections:
        face_id = detection.get('face_id', 'UNKNOWN')
        if face_id not in face_groups:
            face_groups[face_id] = []
        face_groups[face_id].append(detection)
    
    representatives = {}
    
    for face_id, group in face_groups.items():
        if face_id == 'UNKNOWN' or not group:
            continue
        
        # Calculate centroid embedding
        embeddings = [d['embedding'] for d in group if d.get('embedding') is not None]
        if not embeddings:
            continue
        
        centroid = np.mean(embeddings, axis=0)
        
        # Find detection closest to centroid
        best_detection = None
        best_distance = float('inf')
        
        for detection in group:
            emb = detection.get('embedding')
            if emb is None:
                continue
            
            # Use cosine distance (1 - similarity)
            distance = 1.0 - cosine_similarity(emb, centroid)
            
            if distance < best_distance:
                best_distance = distance
                best_detection = detection
        
        if best_detection:
            representatives[face_id] = best_detection
    
    return representatives


def calculate_quality_for_detection(
    detection: Dict,
    image_bgr: np.ndarray,
) -> float:
    """
    Calculate quality score for a detection.
    
    Parameters
    ----------
    detection : Dict
        Detection dictionary with bbox, confidence, pose
    image_bgr : np.ndarray
        Frame image
    
    Returns
    -------
    float
        Quality score [0.0, 1.0]
    """
    return calculate_face_quality(
        detection_confidence=detection.get('confidence', 0.0),
        bbox=detection['bbox'],
        pose=detection.get('pose'),
        image_bgr=image_bgr,
    )

