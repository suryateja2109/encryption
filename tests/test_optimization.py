"""
Unit tests for Baseline PSO and Phase 2 Enhanced APSO optimizers.
"""

import pytest
import numpy as np
from src.optimization.fitness import evaluate_key_fitness, compute_information_entropy, compute_channel_psnr_variance
from src.optimization.baseline_pso import BaselinePSO
from src.optimization.enhanced_optimizer import ChaoticAdaptivePSO


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
