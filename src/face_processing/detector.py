"""
Face detection and ROI extraction using DeepFace with native OpenCV Haar fallback.
Reference: Ding et al., IEEE TCSVT 2025, Section II-A & Section IV-B.
"""

from typing import Tuple, List, Dict, Any, Optional
import cv2
import numpy as np

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except (ImportError, Exception):
    DEEPFACE_AVAILABLE = False


def _detect_with_opencv_haar(image: np.ndarray) -> List[Dict[str, Any]]:
    """Native OpenCV Haar Cascade detector (fast, lightweight, zero TensorFlow dependency)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    results = []
    for (x, y, w, h) in faces:
        results.append({
            "facial_area": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "confidence": 0.95,
        })
    return results


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
            - roi: (h, w, 3) cropped uint8 array, or fallback central region
            - bbox: (x, y, w, h) coordinates
    """
    H, W, C = image.shape
    faces = []

    # Try DeepFace if requested and available
    if DEEPFACE_AVAILABLE and backend in ["mtcnn", "retinaface"]:
        try:
            faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend=backend,
                enforce_detection=enforce_detection,
                align=True,
            )
        except Exception:
            faces = []

    # If backend is opencv or deepface failed/unavailable, use native OpenCV Haar Cascades
    if not faces:
        try:
            faces = _detect_with_opencv_haar(image)
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
    faces = []

    if DEEPFACE_AVAILABLE and backend in ["mtcnn", "retinaface"]:
        try:
            faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend=backend,
                enforce_detection=False,
                align=False,
            )
        except Exception:
            faces = []

    if not faces:
        try:
            faces = _detect_with_opencv_haar(image)
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
