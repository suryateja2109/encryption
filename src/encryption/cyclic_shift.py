"""
Row-column alternating cyclic shift scrambling (Algorithm 2).
Reference: Ding et al., IEEE TCSVT 2025, Section IV-C, Algorithm 2.
"""

from typing import Tuple
import numpy as np


def cyclic_shift_scramble(
    F: np.ndarray,
    line: np.ndarray,
    row: np.ndarray,
    roundnum: int = 6000,
) -> np.ndarray:
    """
    Perform row-column alternating cyclic shifting on image matrix F (Algorithm 2).

    Args:
        F: 2D numpy array of shape (m, 3n), representing reshaped facial ROI.
        line: 1D array of shift amounts for rows (derived from x_n chaotic sequence).
        row: 1D array of shift amounts for columns (derived from y_n chaotic sequence).
        roundnum: Total number of shifting rounds (default 6000).

    Returns:
        Scrambled 2D array of shape (m, 3n).
    """
    m, cols = F.shape
    F2 = F.copy().astype(np.float64)

    for i in range(1, roundnum + 1):
        ix = (i - 1) % m
        iy = (i - 1) % cols
        F2[ix, :] = np.roll(F2[ix, :], int(line[i - 1]))
        F2[:, iy] = np.roll(F2[:, iy], int(row[i - 1]))

    return F2


def cyclic_shift_descramble(
    F2: np.ndarray,
    line: np.ndarray,
    row: np.ndarray,
    roundnum: int = 6000,
) -> np.ndarray:
    """
    Exact inverse of Algorithm 2.
    Applies reverse cyclic shifts in reverse chronological order (roundnum down to 1).
    """
    m, cols = F2.shape
    F1 = F2.copy().astype(np.float64)

    for i in range(roundnum, 0, -1):
        ix = (i - 1) % m
        iy = (i - 1) % cols
        F1[:, iy] = np.roll(F1[:, iy], -int(row[i - 1]))
        F1[ix, :] = np.roll(F1[ix, :], -int(line[i - 1]))

    return F1
