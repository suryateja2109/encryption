"""
Face recognition embedding extraction and distance matching.
Reference: Ding et al., IEEE TCSVT 2025, Section IV-B, Table II.
"""

from typing import Tuple, List, Optional
import numpy as np

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except (ImportError, Exception):
    DEEPFACE_AVAILABLE = False


def extract_face_embedding(
    face_img: np.ndarray,
    model_name: str = "Facenet",
    detector_backend: str = "opencv",
) -> Optional[np.ndarray]:
    """
    Extract high-dimensional feature embedding vector for a face using DeepFace.

    Args:
        face_img: (H, W, 3) uint8 RGB face image.
        model_name: Embedding architecture ('Facenet', 'VGG-Face', 'ArcFace').
        detector_backend: Detection backend ('opencv', 'skip').

    Returns:
        1D normalized numpy array embedding vector (e.g. 128-d or 512-d), or None if failed.
    """
    if not DEEPFACE_AVAILABLE:
        return None

    try:
        results = DeepFace.represent(
            img_path=face_img,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=False,
            align=True,
        )
        if results and "embedding" in results[0]:
            emb = np.array(results[0]["embedding"], dtype=np.float64)
            # L2 normalize
            norm = np.linalg.norm(emb)
            if norm > 1e-8:
                emb = emb / norm
            return emb
    except Exception:
        pass
    return None


def compute_euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two L2-normalized embeddings:
    dist = ||emb1 - emb2||_2
    """
    return float(np.linalg.norm(emb1 - emb2))


def verify_face_match(
    emb1: np.ndarray,
    emb2: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[bool, float]:
    """
    Evaluate whether two face embeddings represent the same individual.
    Paper threshold = 0.5 (Section IV-B).

    Returns:
        (is_match, distance)
    """
    dist = compute_euclidean_distance(emb1, emb2)
    return (dist < threshold), dist
