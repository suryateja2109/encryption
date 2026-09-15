"""
3D Coupled Ikeda Map with Bounded Amplitude (3D-CIMBA) and dynamic analysis module.
"""

from .cimba3d import CIMBAMap, generate_3d_cimba_sequences
from .lyapunov import estimate_lyapunov_spectrum, theoretical_lyapunov
from .bifurcation import compute_bifurcation_data, plot_bifurcation_diagram
from .sample_entropy import compute_sample_entropy, compare_chaotic_maps_entropy

__all__ = [
    "CIMBAMap",
    "generate_3d_cimba_sequences",
    "estimate_lyapunov_spectrum",
    "theoretical_lyapunov",
    "compute_bifurcation_data",
    "plot_bifurcation_diagram",
    "compute_sample_entropy",
    "compare_chaotic_maps_entropy",
]
