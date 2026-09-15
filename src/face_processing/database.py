"""
Facial Database and selective encryption matching controller.
Reference: Ding et al., IEEE TCSVT 2025, Section IV-B, Table II.
"""

from typing import Dict, List, Tuple, Optional, Any
import os
import glob
import cv2
import numpy as np
from .detector import detect_face_roi
from .matcher import extract_face_embedding, compute_euclidean_distance, verify_face_match


class FaceDatabase:
    """
    Manages gallery identities, embedding index, and probe query verification.
    """

    def __init__(
        self,
        model_name: str = "Facenet",
        detector_backend: str = "opencv",
        threshold: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.threshold = threshold

        # In-memory index: name -> List[np.ndarray] (embeddings)
        self.gallery: Dict[str, List[np.ndarray]] = {}

    def enroll_image(self, name: str, image: np.ndarray) -> bool:
        """
        Enroll a new face image for individual 'name' using standardized ROI extraction.
        """
        roi, _ = detect_face_roi(image, backend=self.detector_backend)
        target_img = roi if roi is not None else image

        emb = extract_face_embedding(
            target_img,
            model_name=self.model_name,
            detector_backend="skip",
        )
        if emb is not None:
            if name not in self.gallery:
                self.gallery[name] = []
            self.gallery[name].append(emb)
            return True
        return False

    def build_from_dataset(
        self,
        archive_root: str = "archive",
        max_identities: int = 50,
    ) -> int:
        """
        Build gallery database by scanning identities from archive/lfw-deepfunneled.
        """
        lfw_dir = os.path.join(archive_root, "lfw-deepfunneled", "lfw-deepfunneled")
        if not os.path.exists(lfw_dir):
            return 0

        person_dirs = sorted([d for d in glob.glob(os.path.join(lfw_dir, "*")) if os.path.isdir(d)])
        enrolled_count = 0

        for p_dir in person_dirs:
            if enrolled_count >= max_identities:
                break
            name = os.path.basename(p_dir)
            jpgs = sorted(glob.glob(os.path.join(p_dir, "*.jpg")))
            if jpgs:
                # Enroll primary image
                bgr = cv2.imread(jpgs[0])
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    if self.enroll_image(name, rgb):
                        enrolled_count += 1

        return enrolled_count

    def query_probe(
        self,
        probe_image: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Query a probe image against the gallery database.
        Returns match decision, closest identity, minimum Euclidean distance, and detected bbox.
        """
        if threshold is None:
            threshold = self.threshold

        roi, bbox = detect_face_roi(probe_image, backend=self.detector_backend)
        target_img = roi if roi is not None else probe_image

        probe_emb = extract_face_embedding(
            target_img,
            model_name=self.model_name,
            detector_backend="skip",
        )

        if probe_emb is None or not self.gallery:
            return {
                "is_match": False,
                "matched_name": None,
                "min_distance": float("inf"),
                "bbox": bbox,
                "roi": roi,
            }

        min_dist = float("inf")
        best_name = None

        for name, emb_list in self.gallery.items():
            for emb in emb_list:
                dist = compute_euclidean_distance(probe_emb, emb)
                if dist < min_dist:
                    min_dist = dist
                    best_name = name

        is_match = (min_dist < threshold)
        return {
            "is_match": is_match,
            "matched_name": best_name if is_match else None,
            "min_distance": min_dist,
            "bbox": bbox,
            "roi": roi,
        }
