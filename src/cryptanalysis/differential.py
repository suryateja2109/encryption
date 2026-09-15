"""
Differential cryptanalysis: NPCR and UACI with statistical critical value testing.
Reference: Ding et al., IEEE TCSVT 2025, Section V-E, Eq. (20)-(25).
"""

from typing import Tuple, Dict, Any
import numpy as np
from scipy.stats import norm


def compute_npcr_uaci(
    cipher1: np.ndarray,
    cipher2: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute NPCR (%) and UACI (%) between two encrypted images P1 and P2
    originating from plaintexts that differ by 1 bit (Eq. 20-21).

    Returns:
        (npcr_percent, uaci_percent)
    """
    assert cipher1.shape == cipher2.shape, "Cipher images must have identical shape"
    total_elements = float(cipher1.size)

    # NPCR: Fraction of differing pixels
    diff_mask = (cipher1 != cipher2)
    npcr = (np.sum(diff_mask) / total_elements) * 100.0

    # UACI: Mean absolute intensity change relative to 255
    abs_diff = np.abs(cipher1.astype(np.float64) - cipher2.astype(np.float64))
    uaci = (np.sum(abs_diff / 255.0) / total_elements) * 100.0

    return float(npcr), float(uaci)


def compute_critical_thresholds(
    m: int,
    n: int,
    channels: int = 3,
    alpha: float = 0.05,
    L: int = 256,
) -> Dict[str, float]:
    """
    Calculate statistical critical values for NPCR and UACI at significance level alpha (Eq. 22-25).

    Args:
        m: Height of image.
        n: Width of image.
        channels: Number of color channels (default 3).
        alpha: Significance level (default 0.05).
        L: Number of gray levels (default 256).

    Returns:
        Dict: {
            'npcr_critical': float (%),
            'uaci_lower': float (%),
            'uaci_upper': float (%),
            'uaci_mean': float (%),
            'npcr_ideal': float (%),
            'uaci_ideal': float (%),
        }
    """
    N = float(m * n * channels)

    # Ideal values
    npcr_ideal = ((L - 1) / float(L)) * 100.0  # 99.6094%

    # Critical NPCR (one-sided lower bound, Eq. 22 & Wu et al. 2011 standard)
    z_alpha = norm.ppf(1.0 - alpha)  # 1.6449 for alpha=0.05
    sigma_npcr = np.sqrt((L - 1.0) / ((L ** 2) * N))
    npcr_critical = (npcr_ideal / 100.0 - z_alpha * sigma_npcr) * 100.0

    # UACI expectations (Eq. 24-25)
    mu_u = (L + 2.0) / (3.0 * L + 3.0)  # ~0.3346355
    sigma_u_sq = ((L + 2.0) * (L ** 2 + 2.0 * L + 3.0)) / (18.0 * N * L * ((L + 1.0) ** 2))
    sigma_u = np.sqrt(sigma_u_sq)

    # Two-sided UACI critical interval (Eq. 23)
    z_alpha_half = norm.ppf(1.0 - alpha / 2.0)  # 1.96 for alpha=0.05
    uaci_lower = (mu_u - z_alpha_half * sigma_u) * 100.0
    uaci_upper = (mu_u + z_alpha_half * sigma_u) * 100.0
    uaci_ideal = mu_u * 100.0

    return {
        "npcr_ideal": float(npcr_ideal),
        "uaci_ideal": float(uaci_ideal),
        "npcr_critical": float(npcr_critical),
        "uaci_lower": float(uaci_lower),
        "uaci_upper": float(uaci_upper),
        "uaci_mean": float(mu_u * 100.0),
    }


def evaluate_differential_security(
    cipher1: np.ndarray,
    cipher2: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Evaluate differential security metrics and pass/fail statistical status.
    """
    m, n = cipher1.shape[:2]
    channels = cipher1.shape[2] if cipher1.ndim == 3 else 1

    npcr, uaci = compute_npcr_uaci(cipher1, cipher2)
    thresholds = compute_critical_thresholds(m, n, channels=channels, alpha=alpha)

    npcr_passed = npcr >= thresholds["npcr_critical"]
    uaci_passed = thresholds["uaci_lower"] <= uaci <= thresholds["uaci_upper"]

    return {
        "NPCR": npcr,
        "UACI": uaci,
        "NPCR_critical": thresholds["npcr_critical"],
        "UACI_lower": thresholds["uaci_lower"],
        "UACI_upper": thresholds["uaci_upper"],
        "NPCR_passed": bool(npcr_passed),
        "UACI_passed": bool(uaci_passed),
        "Overall_passed": bool(npcr_passed and uaci_passed),
    }
