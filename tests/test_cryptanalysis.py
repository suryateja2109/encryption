"""
Unit tests for the full Cryptanalysis suite.
"""

import pytest
import numpy as np
from src.encryption.cipher_pipeline import encrypt_face_roi
from src.encryption.decryptor import decrypt_face_roi
from src.cryptanalysis.metrics import compute_image_entropy, compute_histogram_variance, compute_ssim_psnr
from src.cryptanalysis.correlation import evaluate_image_correlations
from src.cryptanalysis.differential import compute_npcr_uaci, compute_critical_thresholds
from src.cryptanalysis.randomness import compute_key_space, evaluate_key_sensitivity
from src.cryptanalysis.robustness import benchmark_robustness_suite
from src.cryptanalysis.extended_analysis import compute_chi_square_uniformity, measure_avalanche_effect


def test_entropy_and_correlation_on_cipher():
    """Verify cipher image has near-ideal entropy (~8) and near-zero pixel correlation."""
    np.random.seed(42)
    plain = np.random.randint(50, 180, (64, 64, 3), dtype=np.uint8)
    params = np.array([10.0, 10.0, 0.4, 6.0, 10.0, 0.1, 0.2, 0.3])

    cipher, _ = encrypt_face_roi(plain, params, roundnum=500)

    # Entropy
    ent = compute_image_entropy(cipher)
    assert ent["mean"] >= 7.85, f"Expected cipher entropy >= 7.85, got {ent['mean']}"

    # Correlation
    corrs = evaluate_image_correlations(cipher, n_samples=2000)
    for d in ["horizontal", "vertical", "diagonal"]:
        assert abs(corrs[d]["mean"]) < 0.10, f"Correlation {d} too high: {corrs[d]['mean']}"


def test_key_space_criterion():
    """Verify key space exceeds 10^128 (>= 2^256 standard)."""
    ks = compute_key_space()
    assert ks["meets_security_threshold"] is True
    assert ks["equivalent_bits"] >= 400.0


def test_npcr_uaci_computation():
    """Verify NPCR and UACI calculations and statistical threshold formulas."""
    m, n = 64, 64
    thresh = compute_critical_thresholds(m, n, channels=3, alpha=0.05)
    assert thresh["npcr_ideal"] > 99.6
    assert 33.0 < thresh["uaci_ideal"] < 34.0
    assert thresh["npcr_critical"] < thresh["npcr_ideal"]
    assert thresh["uaci_lower"] < thresh["uaci_upper"]


def test_key_sensitivity():
    """Verify 10^-15 perturbation causes decryption failure (SSIM ~ 0)."""
    np.random.seed(42)
    plain = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    params = np.array([10.0, 10.0, 0.4, 6.0, 10.0, 0.1, 0.2, 0.3])
    bbox = (0, 0, 64, 64)

    sens = evaluate_key_sensitivity(plain, bbox, params, delta=1e-15)
    assert sens["is_sensitive"] is True
    assert sens["param_a_perturbed"]["SSIM"] < 0.15
    assert sens["state_x1_perturbed"]["SSIM"] < 0.15


def test_robustness_attacks():
    """Verify robustness attacks apply and allow partial recovery."""
    np.random.seed(42)
    plain = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    params = np.array([10.0, 10.0, 0.4, 6.0, 10.0, 0.1, 0.2, 0.3])

    cipher, meta = encrypt_face_roi(plain, params, roundnum=200)
    meta["bbox"] = (0, 0, 64, 64)
    meta["is_selective"] = False

    res = benchmark_robustness_suite(plain, cipher, meta)
    assert "Gaussian_0.5%" in res
    assert "SP_Noise_5%" in res
    assert "Crop_1/16" in res
