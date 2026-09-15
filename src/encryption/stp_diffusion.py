"""
Semi-Tensor Product (STP) Diffusion and Decryption.
Reference: Ding et al., IEEE TCSVT 2025, Eq. (2)-(4), (17)-(18).
"""

from typing import Tuple, Dict
import numpy as np


def construct_invertible_matrix(
    val: np.ndarray,
    n: int,
    max_cond: float = 1e3,
) -> Tuple[np.ndarray, int]:
    """
    Construct an invertible n x n matrix R from chaotic sequence val.
    If ill-conditioned, advances by 100 elements iteratively per Section IV-C Step 5.

    Returns:
        (R, start_index): The n x n invertible matrix and the starting index used in val.
    """
    needed = n * n
    start_idx = 0

    while start_idx + needed <= len(val):
        candidate = val[start_idx : start_idx + needed].reshape((n, n)).astype(np.float64)
        cond = np.linalg.cond(candidate)
        if cond < max_cond and not np.isinf(cond) and not np.isnan(cond):
            return candidate, start_idx
        start_idx += 100

    # If sequence is exhausted without meeting condition threshold, regularize slightly
    candidate = val[:needed].reshape((n, n)).astype(np.float64)
    candidate += np.eye(n) * 1e-2
    return candidate, 0


def stp_diffuse(
    F2: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Semi-Tensor Product (STP) diffusion on scrambled matrix F2 (shape m x 3n) using R (shape n x n).

    Calculates:
        T = F2 ∝ R = F2 * (R ⊗ I3)
        cipher = mod(round(T), 256) (uint8 standard pixel range)
        keys = (T - cipher) / 256 (private recovery keystream per Eq. 18)

    Returns:
        (cipher, keys):
            - cipher: 2D uint8 array of shape (m, 3n)
            - keys: 2D float64 array of shape (m, 3n) representing exact private recovery keystream
    """
    m, cols = F2.shape
    n = R.shape[0]
    assert cols == 3 * n, f"Column count {cols} must equal 3 * n ({3*n})"

    # Semi-tensor product via Kronecker product with identity matrix I_3
    # M = R ⊗ I3, shape (3n, 3n)
    M = np.kron(R, np.eye(3, dtype=np.float64))
    T = F2.astype(np.float64) @ M

    # Standard uint8 cipher image pixels in [0, 255]
    cipher = np.mod(np.round(T), 256.0).astype(np.uint8)

    # Private recovery keystream preserving exact T = cipher + 256 * keys
    keys = (T - cipher.astype(np.float64)) / 256.0

    return cipher, keys


def stp_invert_diffuse(
    cipher: np.ndarray,
    keys: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """
    Exact inverse STP diffusion to recover F2.

    Calculates:
        T = cipher + 256 * keys
        F2 = T * (R^{-1} ⊗ I3)

    Returns:
        Recovered F2 of shape (m, 3n) as uint8.
    """
    m, cols = cipher.shape
    n = R.shape[0]

    T = cipher.astype(np.float64) + 256.0 * keys.astype(np.float64)
    R_inv = np.linalg.inv(R)
    M_inv = np.kron(R_inv, np.eye(3, dtype=np.float64))

    F2_rec = T @ M_inv
    # Rounding and clipping to uint8 ensures bit-exact restoration
    F2_uint8 = np.clip(np.round(F2_rec), 0, 255).astype(np.uint8)

    return F2_uint8
