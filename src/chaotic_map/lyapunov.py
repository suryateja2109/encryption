"""
Lyapunov exponent estimation and analysis for 3D-CIMBA.
Reference: Ding et al., IEEE TCSVT 2025, Section III-B, Eq. (9)-(12).
"""

from typing import Tuple, List, Dict
import numpy as np
from .cimba3d import CIMBAMap


def estimate_lyapunov_spectrum(
    cimba: CIMBAMap,
    n_iterations: int = 5000,
    discard: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the 3 Lyapunov exponents of the 3D-CIMBA map using continuous QR decomposition.

    Args:
        cimba: Instantiated CIMBAMap object.
        n_iterations: Number of evaluation iterations.
        discard: Initial transient iterations.

    Returns:
        (le_spectrum, le_history):
            - le_spectrum: Final [LE1, LE2, LE3] array sorted descending.
            - le_history: (n_iterations, 3) history of running mean LE estimates.
    """
    # Initialize state
    x, y, z = cimba.x1, cimba.y1, cimba.z1

    # Discard transient iterations
    for _ in range(discard):
        x, y, z = cimba.step(x, y, z)

    # Initialize orthonormal frame
    Q = np.eye(3, dtype=np.float64)
    le_sum = np.zeros(3, dtype=np.float64)
    le_history = np.empty((n_iterations, 3), dtype=np.float64)

    for i in range(n_iterations):
        J = cimba.jacobian(x, y, z)
        M = J @ Q
        Q, R = np.linalg.qr(M)

        # Accumulate log of diagonal scale factors
        diag_r = np.abs(np.diag(R))
        # Guard against zero/subnormal
        diag_r = np.maximum(diag_r, 1e-16)
        le_sum += np.log(diag_r)

        le_history[i] = le_sum / (i + 1)
        x, y, z = cimba.step(x, y, z)

    final_spectrum = np.sort(le_history[-1])[::-1]
    return final_spectrum, le_history


def theoretical_lyapunov(g: float) -> float:
    """
    Theoretical dominant Lyapunov exponent derived in Section III-B:
    When |b||1 + 4K| << exp(g), J ~ diag(exp(g), exp(g), exp(g)), so LE = ln(exp(g)) = g.
    """
    return float(g)


def sweep_parameter_le(
    param_name: str,
    param_values: np.ndarray,
    base_params: Dict[str, float] = None,
    n_iterations: int = 2000,
) -> np.ndarray:
    """
    Sweep a parameter (e.g. 'g', 'a', 'b', 'delta') and compute the LE spectrum across values.
    Reproduces Fig. 1(a-d) from the base paper.

    Returns:
        Array of shape (len(param_values), 3) with LE1, LE2, LE3 for each value.
    """
    if base_params is None:
        base_params = {
            "a": 10.0,
            "b": 10.0,
            "delta": 0.4,
            "K": 6.0,
            "g": 10.0,
            "x1": 0.1,
            "y1": 0.2,
            "z1": 0.3,
        }

    results = np.empty((len(param_values), 3), dtype=np.float64)

    for idx, val in enumerate(param_values):
        current_params = base_params.copy()
        current_params[param_name] = float(val)
        cimba = CIMBAMap(**current_params)
        spectrum, _ = estimate_lyapunov_spectrum(cimba, n_iterations=n_iterations, discard=300)
        results[idx] = spectrum

    return results
