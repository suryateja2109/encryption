"""
Fitness function evaluation for PSO key optimization (Eq. 13-15).
Reference: Ding et al., IEEE TCSVT 2025, Section IV-A.
"""

from typing import Tuple, Optional
import numpy as np
from src.encryption.cipher_pipeline import encrypt_face_roi


def compute_channel_psnr_variance(channel: np.ndarray) -> float:
    """
    Compute PSNR based on channel variance (Eq. 14).
    MSE = (1/n) * sum((x_i - mean(x))^2)
    PSNR = 10 * log10(255^2 / MSE)
    """
    mean_val = np.mean(channel)
    mse = np.mean((channel - mean_val) ** 2)
    if mse < 1e-8:
        return 50.0  # Max bound for zero variance
    psnr = 10.0 * np.log10((255.0 ** 2) / mse)
    return float(psnr)


def compute_information_entropy(data: np.ndarray) -> float:
    """
    Compute Shannon information entropy H(x) per Eq. (15).
    H(x) = sum(p(x_i) * log2(1 / p(x_i)))
    Theoretical maximum for 8-bit image is 8.0.
    """
    flat = data.flatten()
    if len(flat) == 0:
        return 0.0

    counts = np.bincount(flat, minlength=256)
    probs = counts[counts > 0] / len(flat)
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


def evaluate_key_fitness(
    candidate_key: np.ndarray,
    crop_64x64: np.ndarray,
    omega1: float = 0.3,
    omega2: float = 0.7,
    roundnum: int = 500,
) -> float:
    """
    Calculate fitness function F(p) for candidate key p on a 64x64 image crop (Eq. 13).

    F(p) = omega1 * sum_{i=1}^3 PSNR(P_i) + omega2 * (8 - |8 - H(P)|)

    Args:
        candidate_key: 8-element array [a, b, delta, K, g, x1, y1, z1].
        crop_64x64: (64, 64, 3) uint8 image crop.
        omega1: Weight for PSNR term (default 0.3).
        omega2: Weight for entropy term (default 0.7).
        roundnum: Number of cyclic shift rounds for optimization evaluation (default 500).

    Returns:
        Scalar fitness score (higher is better).
    """
    # Guard bounds
    # a, b, delta, K, g in [0, 10]; x1, y1, z1 in [0, 1]
    p = np.clip(
        candidate_key,
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001],
        [10.0, 10.0, 10.0, 10.0, 10.0, 0.999, 0.999, 0.999],
    )

    try:
        cipher_crop, _ = encrypt_face_roi(crop_64x64, p, roundnum=roundnum)
    except Exception:
        return -1e6

    # 1. PSNR across R, G, B channels
    psnr_sum = 0.0
    for c in range(3):
        psnr_sum += compute_channel_psnr_variance(cipher_crop[:, :, c])

    # 2. Information entropy of cipher crop
    entropy = compute_information_entropy(cipher_crop)

    # Fitness F(p) per Eq. 13 (rewarding entropy close to theoretical maximum 8)
    entropy_term = 8.0 - abs(8.0 - entropy)
    fitness = omega1 * psnr_sum + omega2 * entropy_term

    return float(fitness)
