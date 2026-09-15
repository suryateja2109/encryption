"""
Unit tests for 3D-CIMBA chaotic map, Lyapunov estimation, bifurcation, and sample entropy.
"""

import pytest
import numpy as np
from src.chaotic_map.cimba3d import CIMBAMap, generate_3d_cimba_sequences
from src.chaotic_map.lyapunov import estimate_lyapunov_spectrum, theoretical_lyapunov
from src.chaotic_map.bifurcation import compute_bifurcation_data
from src.chaotic_map.sample_entropy import compute_sample_entropy, compare_chaotic_maps_entropy


def test_cimba_boundedness():
    """Verify that 3D-CIMBA sequences stay strictly bounded in [0, 1)."""
    cimba = CIMBAMap(a=10.0, b=10.0, delta=0.4, K=6.0, g=10.0)
    xs, ys, zs = cimba.iterate(n_steps=2000, discard=500)

    assert len(xs) == 2000
    assert len(ys) == 2000
    assert len(zs) == 2000

    assert np.all((xs >= 0.0) & (xs < 1.0)), "x sequence escaped [0, 1)"
    assert np.all((ys >= 0.0) & (ys < 1.0)), "y sequence escaped [0, 1)"
    assert np.all((zs >= 0.0) & (zs < 1.0)), "z sequence escaped [0, 1)"
    assert not np.any(np.isnan(xs) | np.isnan(ys) | np.isnan(zs)), "NaN in chaotic sequence"


def test_cimba_sensitivity():
    """Verify extreme sensitive dependence on initial conditions (butterfly effect)."""
    cimba1 = CIMBAMap(x1=0.1000000000000000)
    cimba2 = CIMBAMap(x1=0.1000000000000001)  # 1e-16 delta

    x1, _, _ = cimba1.iterate(n_steps=500, discard=50)
    x2, _, _ = cimba2.iterate(n_steps=500, discard=50)

    # After transient divergence, correlation should vanish
    corr = np.corrcoef(x1, x2)[0, 1]
    assert abs(corr) < 0.2, f"Expected near zero correlation between perturbed trajectories, got {corr}"


def test_lyapunov_exponent_matches_theory():
    """
    Verify that estimated Lyapunov exponents match Section III-B proposition:
    LE approximately equals g (within 10% relative error for large g).
    """
    g_val = 5.0
    cimba = CIMBAMap(a=10.0, b=10.0, delta=0.4, K=6.0, g=g_val)
    spectrum, _ = estimate_lyapunov_spectrum(cimba, n_iterations=1500, discard=200)

    theo = theoretical_lyapunov(g_val)
    # The dominant LE should be close to g_val
    dominant_le = spectrum[0]
    assert dominant_le > 0.0, f"Expected positive dominant LE, got {dominant_le}"
    rel_error = abs(dominant_le - theo) / theo
    assert rel_error < 0.20, f"LE {dominant_le} deviates significantly from theoretical g={theo}"


def test_bifurcation_computation():
    """Verify bifurcation data generation for 3D-CIMBA."""
    params, states = compute_bifurcation_data(
        map_type="cimba",
        param_name="a",
        param_min=1.0,
        param_max=5.0,
        num_param_steps=20,
        n_plot_points=30,
    )
    assert len(params) == 20
    assert states.shape == (20, 30)
    assert np.all((states >= 0.0) & (states <= 1.0))


def test_sample_entropy_comparison():
    """Verify sample entropy computation and ensure 3D-CIMBA exhibits high complexity."""
    results = compare_chaotic_maps_entropy(n_points=600)
    assert "3D-CIMBA" in results
    assert "Classic Ikeda" in results
    assert "Logistic Map" in results
    assert results["3D-CIMBA"] > 0.5, f"3D-CIMBA SampEn too low: {results['3D-CIMBA']}"
