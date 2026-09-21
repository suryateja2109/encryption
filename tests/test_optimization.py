"""
Unit tests for Baseline PSO, Phase 2 Enhanced APSO, and Quantum-behaved PSO optimizers.
"""

import pytest
import numpy as np
from src.optimization.fitness import evaluate_key_fitness, compute_information_entropy, compute_channel_psnr_variance
from src.optimization.baseline_pso import BaselinePSO
from src.optimization.enhanced_optimizer import ChaoticAdaptivePSO, QuantumPSO


def test_entropy_and_psnr_calculation():
    """Verify entropy and PSNR calculations on known distributions."""
    # Uniform random image
    np.random.seed(42)
    uniform_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    ent = compute_information_entropy(uniform_img)
    assert 7.8 < ent <= 8.0, f"Uniform noise entropy should be ~8.0, got {ent}"

    psnr = compute_channel_psnr_variance(uniform_img[:, :, 0])
    assert psnr > 0.0, "PSNR should be positive"


def test_baseline_pso_runs_and_improves():
    """Verify Baseline PSO executes and improves fitness over iterations."""
    np.random.seed(42)
    crop = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    pso = BaselinePSO(n_particles=10, n_iterations=8, seed=42)
    best_pos, best_fit, history = pso.optimize(crop, eval_rounds=50)

    assert len(best_pos) == 8
    assert len(history) == 8
    # Global best should never decrease
    assert np.all(np.diff(history) >= -1e-6), "Monotonicity violation in PSO gbest history"
    assert best_fit == history[-1]


def test_enhanced_apso_convergence():
    """Verify Chaotic-Adaptive PSO executes and attains valid parameters."""
    np.random.seed(42)
    crop = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    apso = ChaoticAdaptivePSO(n_particles=10, n_iterations=8, seed=42)
    best_pos, best_fit, history = apso.optimize(crop, eval_rounds=50)

    assert len(best_pos) == 8
    assert len(history) == 8
    assert np.all(best_pos >= apso.x_min)
    assert np.all(best_pos <= apso.x_max)
    assert best_fit == history[-1]


def test_apso_dynamic_parameter_adaptation():
    """Verify non-linear parabolic inertia decay and time-varying c1, c2 coefficients."""
    apso = ChaoticAdaptivePSO(
        n_particles=10,
        n_iterations=100,
        w_max=0.9,
        w_min=0.4,
        c1_max=2.5,
        c1_min=0.5,
        c2_max=2.5,
        c2_min=0.5,
        decay_mode="parabolic",
        seed=42,
    )

    # Initial iteration t=0
    w_0, c1_0, c2_0 = apso.get_time_varying_parameters(0, total_iter=100)
    assert 0.85 <= w_0 <= 0.95, f"Expected w(0) ~ 0.9, got {w_0}"
    assert 2.3 <= c1_0 <= 2.5, f"Expected c1(0) ~ 2.5, got {c1_0}"
    assert 0.5 <= c2_0 <= 0.7, f"Expected c2(0) ~ 0.5, got {c2_0}"

    # Final iteration t=100
    w_T, c1_T, c2_T = apso.get_time_varying_parameters(100, total_iter=100)
    assert 0.35 <= w_T <= 0.45, f"Expected w(T) ~ 0.4, got {w_T}"
    assert 0.5 <= c1_T <= 0.7, f"Expected c1(T) ~ 0.5, got {c1_T}"
    assert 2.3 <= c2_T <= 2.5, f"Expected c2(T) ~ 2.5, got {c2_T}"

    # Midpoint parabolic property: w(T/2) should be higher than linear decay (0.65)
    # Under parabolic w(t) = 0.9 - 0.5 * (0.5)^2 = 0.775
    w_mid, _, _ = apso.get_time_varying_parameters(50, total_iter=100)
    assert w_mid > 0.70, f"Parabolic decay should retain higher inertia at midpoint, got {w_mid}"


def test_apso_chaotic_perturbation_vector():
    """Verify 8-dimensional 3D-CIMBA chaotic perturbation vector generation."""
    apso = ChaoticAdaptivePSO(n_particles=10, n_iterations=20, seed=42)

    vec0 = apso.get_chaotic_perturbation_vector(t=5, particle_idx=0)
    vec1 = apso.get_chaotic_perturbation_vector(t=5, particle_idx=1)

    assert len(vec0) == 8
    assert len(vec1) == 8
    assert np.all(vec0 >= 0.0) and np.all(vec0 <= 1.0)
    # Different particles should receive distinct perturbation vectors
    assert not np.allclose(vec0, vec1)


def test_quantum_pso_execution():
    """Verify Quantum-behaved PSO (QPSO) execution and parameter bounds."""
    np.random.seed(42)
    crop = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    qpso = QuantumPSO(n_particles=10, n_iterations=6, seed=42)
    best_pos, best_fit, history = qpso.optimize(crop, eval_rounds=50)

    assert len(best_pos) == 8
    assert len(history) == 6
    assert np.all(best_pos >= qpso.x_min)
    assert np.all(best_pos <= qpso.x_max)
    assert np.all(np.diff(history) >= -1e-6)
    assert best_fit == history[-1]
