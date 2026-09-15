"""
Adjacent pixel correlation analysis (Horizontal, Vertical, Diagonal).
Reference: Ding et al., IEEE TCSVT 2025, Section V-C, Eq. (19).
"""

from typing import Dict, Tuple
import numpy as np


def sample_adjacent_pixel_pairs(
    channel: np.ndarray,
    direction: str = "horizontal",
    n_samples: int = 5000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample adjacent pixel pairs (X, Y) from a 2D channel.

    Args:
        channel: 2D uint8 or float image channel (H, W).
        direction: 'horizontal', 'vertical', or 'diagonal'.
        n_samples: Number of pairs to sample.

    Returns:
        (x_vals, y_vals): Arrays of pixel pair values.
    """
    H, W = channel.shape
    rng = np.random.RandomState(seed)

    if direction == "horizontal":
        rows = rng.randint(0, H, size=n_samples)
        cols = rng.randint(0, W - 1, size=n_samples)
        x_vals = channel[rows, cols]
        y_vals = channel[rows, cols + 1]
    elif direction == "vertical":
        rows = rng.randint(0, H - 1, size=n_samples)
        cols = rng.randint(0, W, size=n_samples)
        x_vals = channel[rows, cols]
        y_vals = channel[rows + 1, cols]
    elif direction == "diagonal":
        rows = rng.randint(0, H - 1, size=n_samples)
        cols = rng.randint(0, W - 1, size=n_samples)
        x_vals = channel[rows, cols]
        y_vals = channel[rows + 1, cols + 1]
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return x_vals.astype(np.float64), y_vals.astype(np.float64)


def compute_correlation_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient rho per Eq. (19):
    rho = Cov(x, y) / (std(x) * std(y))
    """
    if len(x) < 2:
        return 0.0
    cov = np.cov(x, y)
    denom = np.std(x, ddof=1) * np.std(y, ddof=1)
    if denom < 1e-12:
        return 0.0
    rho = cov[0, 1] / denom
    return float(rho)


def evaluate_image_correlations(
    image: np.ndarray,
    n_samples: int = 5000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compute horizontal, vertical, and diagonal correlations for all channels.

    Returns:
        Dict: {
            'horizontal': {'R': float, 'G': float, 'B': float, 'mean': float},
            'vertical': ...,
            'diagonal': ...,
        }
    """
    directions = ["horizontal", "vertical", "diagonal"]
    results = {}

    for d in directions:
        d_dict = {}
        for c, c_name in enumerate(["R", "G", "B"]):
            channel = image[:, :, c] if image.ndim == 3 else image
            x, y = sample_adjacent_pixel_pairs(channel, direction=d, n_samples=n_samples, seed=seed + c)
            corr = compute_correlation_coefficient(x, y)
            d_dict[c_name] = corr
        d_dict["mean"] = float(np.mean([d_dict["R"], d_dict["G"], d_dict["B"]]))
        results[d] = d_dict

    return results
