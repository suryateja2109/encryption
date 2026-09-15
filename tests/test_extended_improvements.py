"""
Unit tests for Phase 5 extended improvements:
- Multi-face selective encryption
- Local Shannon entropy
- Chi-square uniformity
- Machine learning statistical attack resistance
"""

import pytest
import numpy as np
from src.encryption.multi_face_cipher import encrypt_multi_face_image, decrypt_multi_face_image
from src.cryptanalysis.extended_analysis import (
    compute_local_shannon_entropy,
    compute_chi_square_uniformity,
    measure_avalanche_effect,
    evaluate_machine_learning_attack,
)
from src.encryption.cipher_pipeline import encrypt_face_roi


def test_multi_face_encryption_and_lossless_recovery():
    """Verify multiple faces can be independently encrypted and losslessly decrypted."""
    np.random.seed(42)
    # 200x300 image with 2 distinct face bounding boxes
    image = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
    params = np.array([10.0, 10.0, 0.4, 6.0, 10.0, 0.1, 0.2, 0.3])
    bboxes = [(20, 30, 64, 64), (180, 30, 64, 64)]

    cipher_img, meta_list = encrypt_multi_face_image(image, params, bboxes=bboxes, roundnum=200)

    assert len(meta_list) == 2
    # Verify both faces are modified in cipher
    for m in meta_list:
        x, y, w, h = m["bbox"]
        assert not np.array_equal(image[y : y + h, x : x + w], cipher_img[y : y + h, x : x + w])

    # Decrypt
    decrypted = decrypt_multi_face_image(cipher_img, meta_list)
    diff = np.max(np.abs(image.astype(int) - decrypted.astype(int)))
    assert diff == 0, f"Multi-face decryption not bit-exact: max error {diff}"


def test_local_shannon_entropy():
    """Verify local Shannon entropy computation across 8x8 blocks."""
    np.random.seed(42)
    cipher_block = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    res = compute_local_shannon_entropy(cipher_block, block_size=8)

    assert res["mean_local_entropy"] > 4.5
    assert res["std_local_entropy"] < 1.5


def test_chi_square_uniformity():
    """Verify Chi-Square goodness-of-fit test on cipher bytes."""
    np.random.seed(42)
    plain = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    params = np.array([10.0, 10.0, 0.4, 6.0, 10.0, 0.1, 0.2, 0.3])

    cipher, _ = encrypt_face_roi(plain, params, roundnum=500)
    chi_res = compute_chi_square_uniformity(cipher)

    assert "chi2_stat" in chi_res
    assert "p_value" in chi_res
    assert chi_res["p_value"] > 0.001


def test_ml_statistical_attack_resistance():
    """Verify machine learning classifier cannot distinguish cipher from true random noise."""
    np.random.seed(42)
    plain = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    params = np.array([10.0, 10.0, 0.4, 6.0, 10.0, 0.1, 0.2, 0.3])

    cipher, _ = encrypt_face_roi(plain, params, roundnum=500)
    ml_eval = evaluate_machine_learning_attack([cipher], n_samples_per_class=100, seed=42)

    assert ml_eval["attack_resisted"] is True
    # Classifier accuracy should be close to random chance (50%)
    assert ml_eval["accuracy"] < 0.65
