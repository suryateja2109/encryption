"""
Unit tests for face detection, ROI extraction, and embedding distance matching.
"""

import os
import pytest
import cv2
import numpy as np
from src.face_processing.detector import detect_face_roi, detect_multiple_faces
from src.face_processing.matcher import extract_face_embedding, compute_euclidean_distance, verify_face_match
from src.face_processing.database import FaceDatabase


def test_face_detection_on_sample_image():
    """Verify face detection extracts a valid ROI from an LFW dataset sample."""
    sample_path = "archive/lfw-deepfunneled/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
    assert os.path.exists(sample_path), f"Sample image {sample_path} not found"

    bgr = cv2.imread(sample_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    roi, (x, y, w, h) = detect_face_roi(rgb, backend="opencv")

    assert roi is not None
    assert w > 20 and h > 20
    assert roi.shape[0] == h and roi.shape[1] == w and roi.shape[2] == 3
    assert 0 <= x < rgb.shape[1] and 0 <= y < rgb.shape[0]


def test_embedding_and_distance():
    """Verify feature embeddings and Euclidean distance metric."""
    sample_path = "archive/lfw-deepfunneled/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
    bgr = cv2.imread(sample_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    emb1 = extract_face_embedding(rgb, model_name="Facenet")
    if emb1 is not None:
        # Distance to self must be 0
        dist_self = compute_euclidean_distance(emb1, emb1)
        assert abs(dist_self) < 1e-6, f"Self distance should be 0, got {dist_self}"

        # Match to self must be True
        is_match, d = verify_face_match(emb1, emb1, threshold=0.5)
        assert is_match is True
        assert d < 0.5


def test_face_database_enroll_and_query():
    """Verify FaceDatabase enrollment and probe matching workflow."""
    sample1 = "archive/lfw-deepfunneled/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
    bgr = cv2.imread(sample1)
    rgb1 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    db = FaceDatabase(model_name="Facenet", detector_backend="opencv", threshold=0.5)
    success = db.enroll_image("Aaron_Eckhart", rgb1)

    if success:
        result = db.query_probe(rgb1)
        assert result["is_match"] is True
        assert result["matched_name"] == "Aaron_Eckhart"
        assert result["min_distance"] < 0.5
