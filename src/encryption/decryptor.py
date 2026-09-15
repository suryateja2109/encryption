"""
Full Decryption Pipeline: Exact inverse of cipher_pipeline.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from .cyclic_shift import cyclic_shift_descramble
from .stp_diffusion import stp_invert_diffuse
from .cipher_pipeline import derive_keystreams


def decrypt_face_roi(
    cipher_roi: np.ndarray,
    metadata: Dict[str, Any],
) -> np.ndarray:
    """
    Losslessly decrypt an encrypted 3D facial ROI of shape (m, n, 3).

    Args:
        cipher_roi: (m, n, 3) encrypted uint8 array.
        metadata: Metadata generated during encryption (R, stp_keys, cimba_params, roundnum).

    Returns:
        Decrypted (m, n, 3) uint8 array. Bit-exact inverse of plain ROI on clean channel.
    """
    m = metadata["m"]
    n = metadata["n"]
    roundnum = metadata["roundnum"]
    R = metadata["R"]
    stp_keys = metadata["stp_keys"]
    cimba_params = metadata["cimba_params"]

    # Step 1: Reshape cipher ROI to 2D (m, 3n)
    cipher_2d = cipher_roi.reshape((m, 3 * n))

    # Step 2: Inverse STP diffusion
    F2_rec = stp_invert_diffuse(cipher_2d, stp_keys, R)

    # Step 3: Derive keystreams
    line, row, _, _ = derive_keystreams(cimba_params, m, n, roundnum=roundnum)

    # Step 4: Reverse cyclic shift
    F1_rec = cyclic_shift_descramble(F2_rec, line, row, roundnum=roundnum)

    # Step 5: Reshape back to (m, n, 3)
    decrypted_roi = np.clip(np.round(F1_rec), 0, 255).astype(np.uint8).reshape((m, n, 3))

    return decrypted_roi


def decrypt_full_image(
    cipher_img: np.ndarray,
    metadata: Dict[str, Any],
) -> np.ndarray:
    """
    Decrypt an image encrypted with encrypt_full_image.
    Restores the decrypted ROI into the composite image.
    """
    decrypted_img = cipher_img.copy()

    if metadata.get("is_selective", False):
        x, y, w, h = metadata["bbox"]
        cipher_roi = cipher_img[y : y + h, x : x + w]
        decrypted_roi = decrypt_face_roi(cipher_roi, metadata)
        decrypted_img[y : y + h, x : x + w] = decrypted_roi
    else:
        decrypted_img = decrypt_face_roi(cipher_img, metadata)

    return decrypted_img
