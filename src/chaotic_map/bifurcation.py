"""
Bifurcation analysis for 3D-CIMBA and comparison with classic 2D Ikeda map.
Reference: Ding et al., IEEE TCSVT 2025, Section III-C, Fig. 2.
"""

from typing import Dict, Tuple, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
from .cimba3d import CIMBAMap


def classic_ikeda_trajectory(
    a: float,
    b: float,
    delta: float = 0.4,
    K: float = 6.0,
    x1: float = 0.1,
    y1: float = 0.2,
    n_steps: int = 200,
    discard: int = 500,
) -> np.ndarray:
    """
    Classic 2D Ikeda map trajectory (Eq. 6).
    """
    x, y = x1, y1
    total = n_steps + discard
    xs = np.empty(n_steps, dtype=np.float64)

    for i in range(total):
        denom = 1.0 + x * x + y * y
        phi = delta - K / denom
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)

        x_next = a + b * (x * cos_p - y * sin_p)
        y_next = b * (x * sin_p - y * cos_p)

        # Clip / guard against explosion in classic Ikeda
        if abs(x_next) > 1e4 or abs(y_next) > 1e4 or np.isnan(x_next) or np.isnan(y_next):
            x, y = 0.0, 0.0
        else:
            x, y = x_next, y_next

        if i >= discard:
            xs[i - discard] = x

    return xs


def compute_bifurcation_data(
    map_type: str = "cimba",
    param_name: str = "a",
    param_min: float = 0.0,
    param_max: float = 10.0,
    num_param_steps: int = 400,
    n_plot_points: int = 150,
    discard: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute bifurcation data: (param_values, state_matrix).
    state_matrix has shape (num_param_steps, n_plot_points).
    """
    param_values = np.linspace(param_min, param_max, num_param_steps)
    state_matrix = np.empty((num_param_steps, n_plot_points), dtype=np.float64)

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

    for i, val in enumerate(param_values):
        if map_type.lower() == "cimba":
            p = base_params.copy()
            p[param_name] = float(val)
            cimba = CIMBAMap(**p)
            xs, _, _ = cimba.iterate(n_steps=n_plot_points, discard=discard)
            state_matrix[i] = xs
        else:  # classic ikeda
            a_val = val if param_name == "a" else base_params["a"]
            b_val = val if param_name == "b" else base_params["b"]
            xs = classic_ikeda_trajectory(
                a=a_val,
                b=b_val,
                delta=base_params["delta"],
                K=base_params["K"],
                x1=base_params["x1"],
                y1=base_params["y1"],
                n_steps=n_plot_points,
                discard=discard,
            )
            state_matrix[i] = xs

    return param_values, state_matrix


def plot_bifurcation_diagram(
    param_values: np.ndarray,
    state_matrix: np.ndarray,
    xlabel: str,
    title: str,
    output_path: Optional[str] = None,
) -> None:
    """
    Render and optionally save a bifurcation diagram.
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    # Flatten for fast scatter plot
    num_steps, n_pts = state_matrix.shape
    x_coords = np.repeat(param_values, n_pts)
    y_coords = state_matrix.flatten()

    ax.scatter(x_coords, y_coords, s=0.2, c="#1f77b4", alpha=0.5, edgecolors="none")
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel("State variable $x_n$", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200)
    plt.close(fig)
