"""
Sample Entropy (SampEn) computation and chaotic complexity comparisons.
Reference: Ding et al., IEEE TCSVT 2025, Section III-E, Fig. 4.
"""

from typing import Dict, Tuple, List
import numpy as np
from .cimba3d import CIMBAMap
from .bifurcation import classic_ikeda_trajectory


def compute_sample_entropy(
    data: np.ndarray,
    m: int = 2,
    r: float = 0.2,
) -> float:
    """
    Compute Sample Entropy (SampEn) of a 1D time series.

    Args:
        data: 1D numpy array of values.
        m: Embedding dimension (default 2).
        r: Tolerance threshold as fraction of standard deviation (default 0.2).

    Returns:
        Sample entropy value (float). Higher values denote greater irregularity/complexity.
    """
    x = np.asarray(data, dtype=np.float64)
    N = len(x)
    if N < 50:
        return 0.0

    sigma = np.std(x)
    if sigma == 0:
        return 0.0
    r_val = r * sigma

    # Construct m and (m+1) dimensional embedding vectors
    # Matrix of shape (N - m + 1, m)
    Xm = np.lib.stride_tricks.sliding_window_view(x, m)
    # Matrix of shape (N - m, m + 1)
    Xm1 = np.lib.stride_tricks.sliding_window_view(x, m + 1)

    Nm = len(Xm)
    Nm1 = len(Xm1)

    # Fast vectorization for moderate N (e.g. 1000 - 2000 points)
    # Chebyshev distance: max absolute difference
    # For memory efficiency, compute row by row or block-wise
    B_count = 0
    A_count = 0

    # Subsample if N is large to avoid O(N^2) memory bottleneck while maintaining precision
    max_eval = min(Nm1, 1500)
    for i in range(max_eval):
        # Compare Xm[i] with all other Xm[j] (j != i)
        diff_m = np.max(np.abs(Xm[:max_eval] - Xm[i]), axis=1)
        # Exclude self-match
        B_count += np.sum(diff_m <= r_val) - 1

        # Compare Xm1[i] with all other Xm1[j] (j != i)
        diff_m1 = np.max(np.abs(Xm1[:max_eval] - Xm1[i]), axis=1)
        A_count += np.sum(diff_m1 <= r_val) - 1

    if B_count == 0 or A_count == 0:
        return 0.0

    sampen = -np.log(A_count / B_count)
    return float(sampen)


def compare_chaotic_maps_entropy(
    n_points: int = 2000,
    m: int = 2,
    r: float = 0.2,
) -> Dict[str, float]:
    """
    Compare Sample Entropy across different chaotic maps:
    1. 3D-CIMBA (proposed map, Section III-E)
    2. Classic 2D Ikeda map
    3. Classic 1D Logistic map (x_{n+1} = 4 * x_n * (1 - x_n))

    Returns:
        Dictionary mapping map name to computed SampEn.
    """
    # 1. 3D-CIMBA
    cimba = CIMBAMap(a=10.0, b=10.0, delta=0.4, K=6.0, g=10.0)
    cimba_x, _, _ = cimba.iterate(n_steps=n_points, discard=1000)
    se_cimba = compute_sample_entropy(cimba_x, m=m, r=r)

    # 2. Classic Ikeda map (b=0.9, a=1.0)
    ikeda_x = classic_ikeda_trajectory(a=1.0, b=0.9, delta=0.4, K=6.0, n_steps=n_points, discard=1000)
    se_ikeda = compute_sample_entropy(ikeda_x, m=m, r=r)

    # 3. Logistic map (r=4.0)
    log_x = np.empty(n_points + 1000, dtype=np.float64)
    x = 0.3541
    for i in range(len(log_x)):
        x = 4.0 * x * (1.0 - x)
        log_x[i] = x
    se_logistic = compute_sample_entropy(log_x[1000:], m=m, r=r)

    return {
        "3D-CIMBA": se_cimba,
        "Classic Ikeda": se_ikeda,
        "Logistic Map": se_logistic,
    }
