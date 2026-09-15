"""
Full Encryption Pipeline combining 3D-CIMBA, cyclic shift, and STP diffusion.
Reference: Ding et al., IEEE TCSVT 2025, Section IV-C.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from src.chaotic_map.cimba3d import CIMBAMap
from .cyclic_shift import cyclic_shift_scramble
from .stp_diffusion import construct_invertible_matrix, stp_diffuse


def derive_keystreams(
    cimba_params: np.ndarray,
    m: int,
    n: int,
    roundnum: int = 6000,
    discard: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, CIMBAMap]:
    """
    Derive line, row, and val sequences per Eq. (16).
    """
    # Total sequence needed
    val_needed = n * n + 2000
    seq_len = max(roundnum + 100, val_needed)

    a, b, delta, K, g, x1, y1, z1 = cimba_params
    cimba = CIMBAMap(a=a, b=b, delta=delta, K=K, g=g, x1=x1, y1=y1, z1=z1)
    xs, ys, zs = cimba.iterate(n_steps=seq_len, discard=discard)

    # Eq. (16)
    line = np.floor(np.mod(xs[:roundnum] * 1e5, 3 * n)).astype(np.int64)
    row = np.floor(np.mod(ys[:roundnum] * 1e5, m)).astype(np.int64)
    val = 2.0 * zs

    return line, row, val, cimba


def encrypt_face_roi(
    roi: np.ndarray,
    cimba_params: np.ndarray,
    roundnum: int = 6000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Encrypt a 3D facial ROI of shape (m, n, 3).

    Args:
        roi: (m, n, 3) uint8 image block.
        cimba_params: 8-element array [a, b, delta, K, g, x1, y1, z1].
        roundnum: Cyclic shift iterations (default 6000).

    Returns:
        (cipher_roi, metadata):
            - cipher_roi: (m, n, 3) encrypted uint8 array
            - metadata: dictionary containing recovery keys, R matrix, bounds, roundnum
    """
    m, n, c = roi.shape
    assert c == 3, f"Expected 3 color channels, got {c}"

    # Step 1: Reshape into (m, 3n)
    F1 = roi.reshape((m, 3 * n)).astype(np.float64)

    # Step 2: Derive keystreams
    line, row, val, _ = derive_keystreams(cimba_params, m, n, roundnum=roundnum)

    # Step 3: Cyclic shift scrambling
    F2 = cyclic_shift_scramble(F1, line, row, roundnum=roundnum)

    # Step 4: STP diffusion
    R, start_idx = construct_invertible_matrix(val, n=n)
    cipher_2d, stp_keys = stp_diffuse(F2, R)

    # Reshape cipher back to (m, n, 3)
    cipher_roi = cipher_2d.reshape((m, n, 3))

    metadata = {
        "m": m,
        "n": n,
        "roundnum": roundnum,
        "R": R,
        "stp_keys": stp_keys,
        "cimba_params": cimba_params.copy(),
        "start_idx": start_idx,
    }

    return cipher_roi, metadata


def encrypt_full_image(
    image: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    cimba_params: Optional[np.ndarray] = None,
    roundnum: int = 6000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Selectively encrypt an image within a given facial bounding box (x, y, w, h).
    If bbox is None, encrypts the entire image.

    Returns:
        (encrypted_image, cipher_metadata)
    """
    H, W, C = image.shape
    assert C == 3, "Input image must be RGB"

    if cimba_params is None:
        # Paper default settings (Section III-B & IV-C)
        cimba_params = np.array([10.0, 10.0, 0.4, 6.0, 10.0, 0.1, 0.2, 0.3], dtype=np.float64)

    cipher_img = image.copy()

    if bbox is not None:
        x, y, w, h = bbox
        # Clip to image boundaries
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))

        roi = image[y : y + h, x : x + w]
        cipher_roi, meta = encrypt_face_roi(roi, cimba_params, roundnum=roundnum)
        cipher_img[y : y + h, x : x + w] = cipher_roi
        meta["bbox"] = (x, y, w, h)
        meta["is_selective"] = True
    else:
        cipher_roi, meta = encrypt_face_roi(image, cimba_params, roundnum=roundnum)
        cipher_img = cipher_roi
        meta["bbox"] = (0, 0, W, H)
        meta["is_selective"] = False

    return cipher_img, meta
