"""
Robustness analysis against noise and cropping attacks.
Reference: Ding et al., IEEE TCSVT 2025, Section V-F, Fig. 11.
"""

from typing import Dict, Tuple, Any
import numpy as np
from skimage.util import random_noise
from src.encryption.decryptor import decrypt_full_image
from .metrics import compute_ssim_psnr


def apply_gaussian_noise(image: np.ndarray, var: float = 0.005, seed: int = 42) -> np.ndarray:
    """
    Apply zero-mean Gaussian noise with specified variance.
    Default var = 0.005 (0.5%), 0.02 (2.0%).
    """
    img_float = image.astype(np.float64) / 255.0
    noisy_float = random_noise(img_float, mode="gaussian", var=var, rng=seed)
    noisy_uint8 = np.clip(np.round(noisy_float * 255.0), 0, 255).astype(np.uint8)
    return noisy_uint8


def apply_salt_and_pepper_noise(image: np.ndarray, amount: float = 0.05, seed: int = 42) -> np.ndarray:
    """
    Apply Salt-and-Pepper noise with specified density amount.
    Default amount = 0.05 (5%), 0.15 (15%).
    """
    img_float = image.astype(np.float64) / 255.0
    noisy_float = random_noise(img_float, mode="s&p", amount=amount, rng=seed)
    noisy_uint8 = np.clip(np.round(noisy_float * 255.0), 0, 255).astype(np.uint8)
    return noisy_uint8


def apply_cropping_attack(image: np.ndarray, crop_fraction: float = 0.0625) -> np.ndarray:
    """
    Apply cropping attack by zeroing out an area of size crop_fraction.
    Default crop_fraction = 1/16 (0.0625) or 1/4 (0.25).
    """
    H, W = image.shape[:2]
    side_fraction = np.sqrt(crop_fraction)
    h_cut = int(H * side_fraction)
    w_cut = int(W * side_fraction)

    # Central crop
    y0 = (H - h_cut) // 2
    x0 = (W - w_cut) // 2

    attacked = image.copy()
    attacked[y0 : y0 + h_cut, x0 : x0 + w_cut] = 0
    return attacked


def benchmark_robustness_suite(
    plain_image: np.ndarray,
    cipher_image: np.ndarray,
    cipher_metadata: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """
    Run full battery of noise and cropping attacks, decrypt, and measure PSNR / SSIM.
    """
    attacks = {
        "Gaussian_0.5%": apply_gaussian_noise(cipher_image, var=0.005),
        "Gaussian_2.0%": apply_gaussian_noise(cipher_image, var=0.02),
        "SP_Noise_5%": apply_salt_and_pepper_noise(cipher_image, amount=0.05),
        "SP_Noise_15%": apply_salt_and_pepper_noise(cipher_image, amount=0.15),
        "Crop_1/16": apply_cropping_attack(cipher_image, crop_fraction=1.0 / 16.0),
        "Crop_1/4": apply_cropping_attack(cipher_image, crop_fraction=1.0 / 4.0),
    }

    results = {}
    for name, attacked_cipher in attacks.items():
        decrypted = decrypt_full_image(attacked_cipher, cipher_metadata)
        ssim_val, psnr_val = compute_ssim_psnr(plain_image, decrypted)
        results[name] = {
            "SSIM": ssim_val,
            "PSNR": psnr_val,
            "attacked_cipher": attacked_cipher,
            "decrypted": decrypted,
        }

    return results
