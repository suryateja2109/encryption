"""
Unit tests for Video Selective Encryption and tracking pipeline.
"""

import pytest
import cv2
import numpy as np
from src.video_extension.tracker import FaceTracker
from src.video_extension.video_cipher import VideoSelectiveCipher


def test_face_tracker_initialization_and_motion():
    """Verify FaceTracker tracks moving template across synthetic frames."""
    H, W = 150, 150
    frame1 = np.full((H, W, 3), 128, dtype=np.uint8)
    # Add identifiable feature block at (40, 40, 30, 30)
    frame1[40:70, 40:70] = 200

    tracker = FaceTracker(confidence_threshold=0.4)
    tracker.init(frame1, (40, 40, 30, 30))
    assert tracker.is_tracking or tracker.last_bbox == (40, 40, 30, 30)

    # Frame 2: slightly shifted feature block at (43, 42, 30, 30)
    frame2 = np.full((H, W, 3), 128, dtype=np.uint8)
    frame2[42:72, 43:73] = 200

    success, bbox, conf = tracker.update(frame2)
    assert success is True
    assert conf > 0.4
    x, y, w, h = bbox
    assert abs(x - 43) <= 4 and abs(y - 42) <= 4


def test_video_cipher_pipeline_end_to_end():
    """Verify video encryption and decryption pipeline on a short video sequence."""
    np.random.seed(42)
    # Generate 6 synthetic video frames (250x250 RGB) simulating slight camera motion
    base_frame = cv2.imread("archive/lfw-deepfunneled/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")
    base_frame = cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB)

    frames = []
    for i in range(6):
        # Shift image slightly
        M = np.float32([[1, 0, i * 2], [0, 1, i * 1]])
        shifted = cv2.warpAffine(base_frame, M, (250, 250))
        frames.append(shifted)

    cipher_engine = VideoSelectiveCipher(gop_size=3, roundnum=200)
    report = cipher_engine.process_video_sequence(frames)

    assert report["total_frames"] == 6
    assert report["overall_fps"] > 0.0
    assert report["all_bit_exact"] is True, "Video decryption must be bit-exact across all frames"
    assert report["mean_entropy"] >= 7.8
