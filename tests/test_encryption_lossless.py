"""
Unit tests to prove bit-exact lossless recovery of the encryption and decryption pipeline.
Acceptance criterion: Decryption is a bit-exact inverse of encryption on a clean channel.
"""

import pytest
import numpy as np
from src.encryption.cipher_pipeline import encrypt_face_roi, encrypt_full_image
from src.encryption.decryptor import decrypt_face_roi, decrypt_full_image


def test_bit_exact_lossless_roi():
    """Verify bit-exact inverse recovery for an arbitrary synthetic face ROI."""
    np.random.seed(1234)
    # Test typical face ROI size: 64x64x3
    roi = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    # Chaotic parameters
    cimba_params = np.array([10.0, 10.0, 0.4, 6.0, 10.0, 0.1, 0.2, 0.3], dtype=np.float64)

    # Encrypt
    cipher_roi, metadata = encrypt_face_roi(roi, cimba_params, roundnum=500)

    # Assert cipher is scrambled (not equal to plaintext)
    assert not np.array_equal(roi, cipher_roi), "Cipher is identical to plaintext!"

    # Decrypt
    decrypted_roi = decrypt_face_roi(cipher_roi, metadata)

    # Assert bit-exact identity
    max_diff = np.max(np.abs(roi.astype(int) - decrypted_roi.astype(int)))
    assert max_diff == 0, f"Decryption not bit-exact: max error {max_diff}"
    assert np.array_equal(roi, decrypted_roi), "Recovered array does not match original uint8 bytes"


def test_bit_exact_lossless_selective_full_image():
    """Verify bit-exact selective encryption on a full 250x250 image."""
    np.random.seed(5678)
    image = np.random.randint(0, 256, (250, 250, 3), dtype=np.uint8)
    bbox = (50, 50, 64, 64)  # (x, y, w, h)

    cipher_img, metadata = encrypt_full_image(image, bbox=bbox, roundnum=300)

    # Outside the bounding box should be UNMODIFIED (selective encryption guarantee)
    assert np.array_equal(image[:50, :], cipher_img[:50, :])
    assert np.array_equal(image[114:, :], cipher_img[114:, :])

    # Inside bounding box should be encrypted
    assert not np.array_equal(image[50:114, 50:114], cipher_img[50:114, 50:114])

    # Decrypt
    decrypted_img = decrypt_full_image(cipher_img, metadata)

    max_diff = np.max(np.abs(image.astype(int) - decrypted_img.astype(int)))
    assert max_diff == 0, f"Full image selective decryption not bit-exact: max error {max_diff}"
