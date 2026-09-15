"""
Video Selective Encryption and Decryption Engine with GOP Key Reuse and Face Tracking.
Reference: Base Paper Future Work #2 ("apply it to video encryption").
"""

from typing import List, Dict, Any, Tuple, Optional
import time
import cv2
import numpy as np
from src.face_processing.detector import detect_face_roi
from src.optimization.baseline_pso import BaselinePSO
from src.encryption.cipher_pipeline import encrypt_full_image
from src.encryption.decryptor import decrypt_full_image
from src.cryptanalysis.metrics import compute_image_entropy
from src.cryptanalysis.correlation import evaluate_image_correlations
from .tracker import FaceTracker


class VideoSelectiveCipher:
    """
    Manages selective encryption for video streams using Group-of-Pictures (GOP) key reuse
    and fast face tracking to maximize throughput (FPS).
    """

    def __init__(
        self,
        gop_size: int = 20,
        roundnum: int = 500,
        tracking_threshold: float = 0.45,
    ) -> None:
        self.gop_size = gop_size
        self.roundnum = roundnum
        self.tracking_threshold = tracking_threshold

        self.tracker = FaceTracker(confidence_threshold=tracking_threshold)
        self.current_key: Optional[np.ndarray] = None
        self.current_bbox: Optional[Tuple[int, int, int, int]] = None
        self.frame_idx = 0

    def encrypt_frame(
        self,
        frame: np.ndarray,
        force_keyframe: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Encrypt a single video frame.

        - If keyframe (frame_idx % gop_size == 0 or force_keyframe or tracking lost):
          Runs face detection and derives/refreshes GOP key.
        - If intermediate frame:
          Tracks face using fast FaceTracker and reuses GOP key.

        Returns:
            (cipher_frame, frame_metadata)
        """
        t0 = time.perf_counter()
        H, W, _ = frame.shape
        is_keyframe = (self.frame_idx % self.gop_size == 0) or force_keyframe or (self.current_key is None)

        if not is_keyframe and self.current_bbox is not None:
            # Try fast tracking
            track_ok, bbox, conf = self.tracker.update(frame)
            if not track_ok:
                is_keyframe = True  # Tracking lost: trigger keyframe re-detection
            else:
                self.current_bbox = bbox
        else:
            is_keyframe = True

        stage_times = {}

        if is_keyframe:
            t_det = time.perf_counter()
            _, bbox = detect_face_roi(frame, backend="opencv")
            self.current_bbox = bbox
            self.tracker.init(frame, bbox)
            stage_times["detection"] = time.perf_counter() - t_det

            # Optimize key or use fast key derivation for GOP
            t_opt = time.perf_counter()
            x, y, w, h = bbox
            crop_64 = cv2.resize(frame[y : y + h, x : x + w], (64, 64))
            pso = BaselinePSO(n_particles=15, n_iterations=8, seed=42 + self.frame_idx)
            self.current_key, _, _ = pso.optimize(crop_64, eval_rounds=50)
            stage_times["optimization"] = time.perf_counter() - t_opt
        else:
            stage_times["detection"] = 0.0
            stage_times["optimization"] = 0.0

        # Encrypt frame ROI with current key
        t_enc = time.perf_counter()
        cipher_frame, meta = encrypt_full_image(
            frame,
            bbox=self.current_bbox,
            cimba_params=self.current_key,
            roundnum=self.roundnum,
        )
        stage_times["encryption"] = time.perf_counter() - t_enc

        total_time = time.perf_counter() - t0
        meta["frame_idx"] = self.frame_idx
        meta["is_keyframe"] = is_keyframe
        meta["total_time_sec"] = total_time
        meta["stage_times"] = stage_times
        meta["bbox"] = self.current_bbox

        self.frame_idx += 1
        return cipher_frame, meta

    def process_video_sequence(
        self,
        frames: List[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Process an entire sequence of frames, encrypting and decrypting each frame,
        and collecting performance metrics (throughput, average FPS, latency breakdown).
        """
        self.frame_idx = 0
        cipher_frames = []
        decrypted_frames = []
        metadata_list = []
        frame_times = []
        entropies = []
        correlations = []
        bit_exact_passes = []

        total_start = time.perf_counter()

        for idx, frame in enumerate(frames):
            cipher_frame, meta = self.encrypt_frame(frame)
            dec_frame = decrypt_full_image(cipher_frame, meta)

            cipher_frames.append(cipher_frame)
            decrypted_frames.append(dec_frame)
            metadata_list.append(meta)
            frame_times.append(meta["total_time_sec"])

            # Bit-exact check
            x, y, w, h = meta["bbox"]
            orig_roi = frame[y : y + h, x : x + w]
            dec_roi = dec_frame[y : y + h, x : x + w]
            diff = np.max(np.abs(orig_roi.astype(int) - dec_roi.astype(int)))
            bit_exact_passes.append(diff == 0)

            # Measure entropy and correlation on cipher ROI
            c_roi = cipher_frame[y : y + h, x : x + w]
            entropies.append(compute_image_entropy(c_roi)["mean"])
            corrs = evaluate_image_correlations(c_roi, n_samples=1000)
            correlations.append(corrs["horizontal"]["mean"])

        total_duration = time.perf_counter() - total_start
        n_frames = len(frames)
        overall_fps = n_frames / total_duration if total_duration > 0 else 0.0

        keyframe_times = [m["total_time_sec"] for m in metadata_list if m["is_keyframe"]]
        inter_times = [m["total_time_sec"] for m in metadata_list if not m["is_keyframe"]]

        return {
            "total_frames": n_frames,
            "total_duration_sec": total_duration,
            "overall_fps": overall_fps,
            "mean_frame_time_ms": float(np.mean(frame_times) * 1000.0),
            "keyframe_mean_time_ms": float(np.mean(keyframe_times) * 1000.0) if keyframe_times else 0.0,
            "intermediate_mean_time_ms": float(np.mean(inter_times) * 1000.0) if inter_times else 0.0,
            "intermediate_fps": float(1.0 / np.mean(inter_times)) if inter_times and np.mean(inter_times) > 0 else 0.0,
            "mean_entropy": float(np.mean(entropies)),
            "mean_correlation": float(np.mean(correlations)),
            "all_bit_exact": bool(all(bit_exact_passes)),
            "cipher_frames": cipher_frames,
            "decrypted_frames": decrypted_frames,
            "metadata_list": metadata_list,
        }
