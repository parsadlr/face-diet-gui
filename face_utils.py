from typing import List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def initialize_detector(
    model_name: str = "buffalo_l",
    use_gpu: bool = False,
    allowed_modules: Optional[List[str]] = None,
) -> FaceAnalysis:
    providers = ["CPUExecutionProvider"]
    if allowed_modules is None:
        # Load detection + 2D/3D landmarks + recognition by default
        allowed_modules = [
            "detection",
            "landmark_2d_106",
            "landmark_3d_68",
            "recognition",
        ]
    app = FaceAnalysis(name=model_name, providers=providers, allowed_modules=allowed_modules)
    ctx_id = 0 if use_gpu else -1
    app.prepare(ctx_id=ctx_id)
    return app


def detect_faces(
    app: FaceAnalysis,
    image_bgr: np.ndarray,
    return_embeddings: bool = False,
) -> Tuple[List[Tuple[int, int, int, int]], List[float], Optional[List[np.ndarray]]]:
    faces = app.get(image_bgr)
    boxes: List[Tuple[int, int, int, int]] = []
    scores: List[float] = []
    embeddings: List[np.ndarray] = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        scores.append(float(f.det_score) if hasattr(f, "det_score") else 0.0)
        if return_embeddings and hasattr(f, "embedding"):
            embeddings.append(f.embedding.astype(np.float32))
    return boxes, scores, (embeddings if return_embeddings else None)


def extract_poses(app: FaceAnalysis, image_bgr: np.ndarray) -> List[Optional[dict]]:
    """
    Estimate head pose (yaw, pitch, roll in degrees) for each detected face using
    PnP solve with 2D (106) and 3D (68) landmarks when available. Falls back to
    face.pose if provided by the model.
    """
    h, w = image_bgr.shape[:2]
    faces = app.get(image_bgr)
    poses: List[Optional[dict]] = []

    # Camera intrinsics assuming focal length ~= image width
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

    # Indices for a stable subset: nose tip, chin, eye corners, mouth corners
    # These indices are valid for both 68- and 106-landmark schemes in InsightFace
    stable_indices = [30, 8, 36, 45, 48, 54]

    for f in faces:
        # Try landmark-based PnP first
        try:
            if hasattr(f, "landmark_2d_106") and f.landmark_2d_106 is not None \
               and hasattr(f, "landmark_3d_68") and f.landmark_3d_68 is not None:
                pts2d = np.asarray(f.landmark_2d_106, dtype=np.float64)
                pts3d = np.asarray(f.landmark_3d_68, dtype=np.float64)

                # Ensure indices are within bounds
                if pts2d.shape[0] >= 55 and pts3d.shape[0] >= 55:
                    image_points = np.array([pts2d[i] for i in stable_indices], dtype=np.float64)
                    model_points = np.array([pts3d[i] for i in stable_indices], dtype=np.float64)

                    success, rvec, tvec = cv2.solvePnP(
                        model_points, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    if success:
                        # Convert rotation vector to rotation matrix
                        rot_mat, _ = cv2.Rodrigues(rvec)
                        # Use RQDecomp3x3 which returns Euler angles (rx, ry, rz) in degrees
                        # Convention: rx=pitch (X), ry=yaw (Y), rz=roll (Z)
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
                        poses.append({
                            "pitch": norm_angle(pitch),
                            "yaw": norm_angle(yaw),
                            "roll": norm_angle(roll),
                        })
                        continue
        except Exception:
            # Fall through to other methods
            pass

        # Fallback: use pose provided by the face object if present
        if hasattr(f, "pose") and f.pose is not None and len(f.pose) >= 3:
            pose_vals = f.pose
            poses.append({
                "pitch": float(pose_vals[0]),
                "yaw": float(pose_vals[1]),
                "roll": float(pose_vals[2]),
            })
        else:
            poses.append(None)

    return poses


def draw_boxes(image_bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    out = image_bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out


def extract_landmarks(app: FaceAnalysis, image_bgr: np.ndarray) -> List[np.ndarray]:
    faces = app.get(image_bgr)
    landmarks_list: List[np.ndarray] = []
    for f in faces:
        # Prefer 5-point keypoints from detector (retinaface) if available
        if hasattr(f, "kps") and f.kps is not None:
            landmarks_list.append(f.kps.astype(np.int32))
            continue
        # Fallback to 106-point landmarks if the model is loaded
        if hasattr(f, "landmark_2d_106") and f.landmark_2d_106 is not None:
            landmarks_list.append(f.landmark_2d_106.astype(np.int32))
            continue
    return landmarks_list


def draw_landmarks(image_bgr: np.ndarray, landmarks_list: List[np.ndarray]) -> np.ndarray:
    out = image_bgr.copy()
    for landmarks in landmarks_list:
        for (px, py) in landmarks:
            cv2.circle(out, (int(px), int(py)), 2, (0, 200, 255), -1)
    return out


def _landmark_groups_106(num_points: int) -> List[Tuple[List[int], Tuple[int, int, int]]]:
    # Approximate grouping for InsightFace 106 points. Indices may vary by model.
    # We guard by num_points and clamp.
    groups = [
        (list(range(0, 33)), (0, 255, 0)),            # face contour
        (list(range(33, 39)), (0, 165, 255)),         # left eyebrow
        (list(range(39, 45)), (0, 140, 255)),         # right eyebrow
        (list(range(45, 55)), (255, 0, 0)),           # nose bridge/base
        (list(range(55, 61)), (255, 100, 0)),         # nose/nostrils detail
        (list(range(61, 69)), (255, 0, 255)),         # left eye
        (list(range(69, 77)), (180, 0, 255)),         # right eye
        (list(range(77, 89)), (0, 0, 255)),           # mouth outer
        (list(range(89, 106)), (128, 0, 128)),        # mouth inner/extra points
    ]
    clamped: List[Tuple[List[int], Tuple[int, int, int]]] = []
    max_idx = num_points - 1
    for idxs, color in groups:
        filtered = [i for i in idxs if 0 <= i <= max_idx]
        if filtered:
            clamped.append((filtered, color))
    return clamped


def draw_landmarks_colored(image_bgr: np.ndarray, landmarks_list: List[np.ndarray]) -> np.ndarray:
    out = image_bgr.copy()
    for landmarks in landmarks_list:
        n = int(landmarks.shape[0])
        if n >= 90:  # assume 106-point layout
            for idxs, color in _landmark_groups_106(n):
                for i in idxs:
                    px, py = landmarks[i]
                    cv2.circle(out, (int(px), int(py)), 2, color, -1)
        else:
            # 5-point fallback (uniform color)
            for (px, py) in landmarks:
                cv2.circle(out, (int(px), int(py)), 2, (0, 200, 255), -1)
    return out


