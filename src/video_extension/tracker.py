"""
Fast Face Tracking for Video Selective Encryption.
Tracks detected face across frames to bypass heavy deep learning face detection per frame.
Reference: Base Paper Future Work #2 ("apply it to video encryption").
"""

from typing import Tuple, Optional, Dict, Any
import cv2
import numpy as np


class FaceTracker:
    """
    Tracks a facial bounding box across video frames using MIL tracking with template correlation fallback.
    """

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold
        self.tracker = None
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None
        self.template: Optional[np.ndarray] = None
        self.is_tracking = False

    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        """
        Initialize tracker on a keyframe with detected bounding box (x, y, w, h).
        """
        x, y, w, h = bbox
        H, W = frame.shape[:2]

        # Clamp
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(16, min(w, W - x))
        h = max(16, min(h, H - y))
        self.last_bbox = (x, y, w, h)

        # Store template for verification
        self.template = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_RGB2GRAY)

        try:
            if hasattr(cv2, "TrackerMIL_create"):
                self.tracker = cv2.TrackerMIL_create()
            elif hasattr(cv2, "TrackerKCF_create"):
                self.tracker = cv2.TrackerKCF_create()
            else:
                self.tracker = None

            if self.tracker is not None:
                self.tracker.init(frame, (x, y, w, h))
                self.is_tracking = True
            else:
                self.is_tracking = False
        except Exception:
            self.is_tracking = False

    def update(self, frame: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int], float]:
        """
        Update tracker for the current frame.

        Returns:
            (success, (x, y, w, h), confidence)
        """
        if self.last_bbox is None or self.template is None:
            return False, (0, 0, 0, 0), 0.0

        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # 1. Try OpenCV tracker if initialized
        success = False
        bbox = self.last_bbox

        if self.is_tracking and self.tracker is not None:
            try:
                ok, box = self.tracker.update(frame)
                if ok:
                    x, y, w, h = [int(v) for v in box]
                    # Clamp
                    x = max(0, min(x, W - 1))
                    y = max(0, min(y, H - 1))
                    w = max(16, min(w, W - x))
                    h = max(16, min(h, H - y))
                    bbox = (x, y, w, h)
                    success = True
            except Exception:
                success = False

        # 2. Template correlation verification / fallback
        if not success:
            # Search within a localized window around last_bbox
            lx, ly, lw, lh = self.last_bbox
            pad_x = lw // 2
            pad_y = lh // 2
            sx = max(0, lx - pad_x)
            sy = max(0, ly - pad_y)
            sw = min(W - sx, lw + 2 * pad_x)
            sh = min(H - sy, lh + 2 * pad_y)

            search_region = gray[sy : sy + sh, sx : sx + sw]
            if (
                search_region.shape[0] >= self.template.shape[0]
                and search_region.shape[1] >= self.template.shape[1]
            ):
                res = cv2.matchTemplate(search_region, self.template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val >= self.confidence_threshold:
                    bx = sx + max_loc[0]
                    by = sy + max_loc[1]
                    bbox = (bx, by, self.template.shape[1], self.template.shape[0])
                    success = True

        # Compute confidence against template
        bx, by, bw, bh = bbox
        curr_crop = gray[by : by + bh, bx : bx + bw]
        if curr_crop.shape == self.template.shape:
            res_val = cv2.matchTemplate(curr_crop, self.template, cv2.TM_CCOEFF_NORMED)[0][0]
            confidence = float(max(0.0, res_val))
        else:
            confidence = 0.5 if success else 0.0

        if success and confidence >= self.confidence_threshold:
            self.last_bbox = bbox
            return True, bbox, confidence
        else:
            return False, bbox, confidence
