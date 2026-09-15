"""
Face processing, detection, and database matching package.
"""

from .detector import detect_face_roi, detect_multiple_faces
from .matcher import extract_face_embedding, compute_euclidean_distance, verify_face_match
from .database import FaceDatabase

__all__ = [
    "detect_face_roi",
    "detect_multiple_faces",
    "extract_face_embedding",
    "compute_euclidean_distance",
    "verify_face_match",
    "FaceDatabase",
]
