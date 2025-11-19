import argparse
import sys
from typing import List, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def initialize_app(model_name: str = "buffalo_l", use_gpu: bool = False) -> FaceAnalysis:
    """
    Initialize the InsightFace FaceAnalysis app.

    Parameters
    ----------
    model_name: str
        Name of the model pack to use (e.g., "buffalo_l").
    use_gpu: bool
        Whether to attempt GPU (ctx_id=0). Falls back to CPU if unavailable.
    """
    providers = ["CPUExecutionProvider"]
    app = FaceAnalysis(name=model_name, providers=providers)
    ctx_id = 0 if use_gpu else -1
    app.prepare(ctx_id=ctx_id)
    return app


def get_face_embedding(app: FaceAnalysis, image_path: str) -> np.ndarray:
    """
    Extract a single face embedding from an image.

    Raises ValueError if no face is found. If multiple faces are detected,
    the first one is used and a warning is printed to stderr.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = app.get(img)
    if len(faces) < 1:
        raise ValueError(f"No faces detected in the image: {image_path}")
    if len(faces) > 1:
        print(
            f"Warning: Multiple faces detected in {image_path}. Using the first detected face.",
            file=sys.stderr,
        )

    return faces[0].embedding.astype(np.float32)


def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    denom = (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
    if denom == 0:
        return 0.0
    return float(np.dot(emb_a, emb_b) / denom)




input_image_path = "examples/input_image.jpg"
target_image1_path = "examples/target_image1.jpg"
target_image2_path = "examples/target_image2.jpg"
target_image3_path = "examples/target_image3.jpg"
threshold = 0.65

app = initialize_app(model_name="buffalo_l", use_gpu=False)

input_emb = get_face_embedding(app, input_image_path)
targets = [
    ("Target 1", target_image1_path),
    ("Target 2", target_image2_path),
    ("Target 3", target_image3_path),
]

print(f"Input: {input_image_path}")
for name, path in targets:
    target_emb = get_face_embedding(app, path)
    sim = cosine_similarity(input_emb, target_emb)
    is_same = sim > threshold
    print(f"{name}: {path}")
    print(f"  Similarity: {sim:.4f}")
    print(f"  Same person? {'YES' if is_same else 'NO'}")

