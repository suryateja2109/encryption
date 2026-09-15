"""
Cryptographic metrics: Shannon entropy, histograms, and structural similarity (SSIM).
Reference: Ding et al., IEEE TCSVT 2025, Section V-B, Eq. (15).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func


def compute_entropy_channel(channel: np.ndarray) -> float:
    """
    Compute Shannon entropy H(x) of a single 2D channel per Eq. (15).
    """
    flat = channel.flatten()
    if len(flat) == 0:
        return 0.0
    counts = np.bincount(flat, minlength=256)
    probs = counts[counts > 0] / float(len(flat))
    return float(-np.sum(probs * np.log2(probs)))


def compute_image_entropy(image: np.ndarray) -> Dict[str, float]:
    """
    Compute Shannon entropy for an image across channels and combined.

    Returns:
        Dict with 'R', 'G', 'B', and 'mean' entropy.
    """
    if image.ndim == 2:
        val = compute_entropy_channel(image)
        return {"mean": val, "R": val, "G": val, "B": val}

    h_r = compute_entropy_channel(image[:, :, 0])
    h_g = compute_entropy_channel(image[:, :, 1])
    h_b = compute_entropy_channel(image[:, :, 2])
    return {
        "R": h_r,
        "G": h_g,
        "B": h_b,
        "mean": float((h_r + h_g + h_b) / 3.0),
    }


def compute_histogram_variance(image: np.ndarray) -> float:
    """
    Evaluate uniformity of pixel value distribution.
    For an ideally encrypted image, histogram variance is minimal.
    V = (1 / 256) * sum_{i=0}^{255} (N_i - N_avg)^2
    """
    flat = image.flatten()
    counts = np.bincount(flat, minlength=256)
    avg_count = float(len(flat)) / 256.0
    var = float(np.mean((counts - avg_count) ** 2))
    return var


def compute_ssim_psnr(plain: np.ndarray, recovered: np.ndarray) -> Tuple[float, float]:
    """
    Compute SSIM and PSNR between plain and recovered images.
    For bit-exact decryption: SSIM = 1.0, PSNR = inf.
    For cipher vs plain: SSIM ~ 0.0.
    """
    # SSIM
    if plain.ndim == 3:
        s = ssim_func(plain, recovered, channel_axis=2, data_range=255)
    else:
        s = ssim_func(plain, recovered, data_range=255)

    # PSNR
    p = psnr_func(plain, recovered, data_range=255)
    return float(s), float(p)
