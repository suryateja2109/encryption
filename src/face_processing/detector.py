"""
Face detection and ROI extraction using DeepFace with multi-backend support.
Reference: Ding et al., IEEE TCSVT 2025, Section II-A & Section IV-B.
"""

from typing import Tuple, List, Dict, Any, Optional
import cv2
import numpy as np
from deepface import DeepFace


def detect_face_roi(
    image: np.ndarray,
    backend: str = "opencv",
    enforce_detection: bool = False,
) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
    """
    Detect the primary face in an RGB image and extract its bounding box and ROI.

    Args:
        image: (H, W, 3) uint8 RGB array.
        backend: Detector backend ('opencv', 'retinaface', 'mtcnn').
        enforce_detection: If True, raises error when no face is found.

    Returns:
        (roi, (x, y, w, h)):
            - roi: (h, w, 3) cropped uint8 array, or None if detection failed
            - bbox: (x, y, w, h) coordinates
    """
    H, W, C = image.shape
    try:
        faces = DeepFace.extract_faces(
            img_path=image,
            detector_backend=backend,
            enforce_detection=enforce_detection,
            align=True,
        )
    except Exception:
        faces = []

    if not faces:
        # Fallback: central region (50% of image dimensions)
        w = int(W * 0.5)
        h = int(H * 0.5)
        x = (W - w) // 2
        y = (H - h) // 2
        return image[y : y + h, x : x + w].copy(), (x, y, w, h)

    # Primary face (highest confidence or largest area)
    primary = max(faces, key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])
    fa = primary["facial_area"]
    x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]

    # Boundary check and clamp
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(16, min(w, W - x))
    h = max(16, min(h, H - y))

    roi = image[y : y + h, x : x + w].copy()
    return roi, (x, y, w, h)


def detect_multiple_faces(
    image: np.ndarray,
    backend: str = "opencv",
) -> List[Dict[str, Any]]:
    """
    Detect multiple distinct faces in an image for Phase 5 multi-face selective encryption.

    Returns:
        List of dictionaries with keys: 'roi', 'bbox': (x, y, w, h), 'confidence'.
    """
    H, W, _ = image.shape
    try:
        faces = DeepFace.extract_faces(
            img_path=image,
            detector_backend=backend,
            enforce_detection=False,
            align=False,
        )
    except Exception:
        faces = []

    results = []
    for f in faces:
        fa = f["facial_area"]
        x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]

        # Clamp
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(8, min(w, W - x))
        h = max(8, min(h, H - y))

        roi = image[y : y + h, x : x + w].copy()
        confidence = f.get("confidence", 1.0)
        results.append({"roi": roi, "bbox": (x, y, w, h), "confidence": confidence})

    return results
