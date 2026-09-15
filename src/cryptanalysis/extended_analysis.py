"""
Phase 5 Extended Cryptanalysis Improvements:
1. Block-wise local Shannon entropy
2. Chi-square randomness hypothesis test
3. Avalanche effect bit sensitivity
4. Machine learning classifier statistical attack test
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy.stats import chisquare
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from .metrics import compute_entropy_channel


def compute_local_shannon_entropy(
    image: np.ndarray,
    block_size: int = 8,
) -> Dict[str, float]:
    """
    Compute block-wise local Shannon entropy.
    Splits image into non-overlapping block_size x block_size blocks,
    computes entropy of each block, and returns the mean and variance.
    """
    channel = image[:, :, 0] if image.ndim == 3 else image
    H, W = channel.shape
    h_blocks = H // block_size
    w_blocks = W // block_size

    entropies = []
    for r in range(h_blocks):
        for c in range(w_blocks):
            block = channel[r * block_size : (r + 1) * block_size, c * block_size : (c + 1) * block_size]
            entropies.append(compute_entropy_channel(block))

    ent_arr = np.array(entropies)
    return {
        "mean_local_entropy": float(np.mean(ent_arr)),
        "std_local_entropy": float(np.std(ent_arr)),
        "min_local_entropy": float(np.min(ent_arr)),
        "max_local_entropy": float(np.max(ent_arr)),
    }


def compute_chi_square_uniformity(image: np.ndarray) -> Dict[str, Any]:
    """
    Perform Pearson Chi-Square Goodness-of-Fit test against uniform distribution.
    H0: Pixel values are uniformly distributed over [0, 255].
    alpha = 0.05, degrees of freedom = 255.
    Critical value chi2_crit ~ 293.25.
    """
    flat = image.flatten()
    N = len(flat)
    observed_counts = np.bincount(flat, minlength=256)
    expected_counts = np.full(256, N / 256.0, dtype=np.float64)

    chi2_stat, p_val = chisquare(observed_counts, f_exp=expected_counts)
    critical_val_005 = 293.25

    return {
        "chi2_stat": float(chi2_stat),
        "p_value": float(p_val),
        "critical_value_005": critical_val_005,
        "is_uniform": bool(chi2_stat < critical_val_005),
    }


def measure_avalanche_effect(
    cipher1: np.ndarray,
    cipher2: np.ndarray,
) -> float:
    """
    Measure bit-level Avalanche Effect (Strict Avalanche Criterion).
    Returns percentage of bits flipped between cipher1 and cipher2. Ideal ~ 50.0%.
    """
    assert cipher1.shape == cipher2.shape
    bits1 = np.unpackbits(cipher1.flatten())
    bits2 = np.unpackbits(cipher2.flatten())
    flipped = np.sum(bits1 != bits2)
    avalanche_rate = (flipped / float(len(bits1))) * 100.0
    return float(avalanche_rate)


def evaluate_machine_learning_attack(
    cipher_images: list,
    n_samples_per_class: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Machine learning statistical attack test:
    Train a Random Forest classifier to distinguish between blocks from cipher images
    and true pseudo-random uniform noise blocks.
    Demonstrates classifier cannot beat chance (accuracy ~ 50.0%).
    """
    rng = np.random.RandomState(seed)
    block_len = 64

    # Extract 1D feature vectors (histograms / moments / entropy)
    X = []
    y = []

    # Class 0: True uniform random noise
    for _ in range(n_samples_per_class):
        noise = rng.randint(0, 256, block_len, dtype=np.uint8)
        feats = [
            np.mean(noise),
            np.std(noise),
            compute_entropy_channel(noise),
            np.median(noise),
        ]
        X.append(feats)
        y.append(0)

    # Class 1: Cipher blocks
    for img in cipher_images:
        flat = img.flatten()
        if len(flat) >= block_len:
            for _ in range(n_samples_per_class // len(cipher_images)):
                idx = rng.randint(0, len(flat) - block_len)
                sub = flat[idx : idx + block_len]
                feats = [
                    np.mean(sub),
                    np.std(sub),
                    compute_entropy_channel(sub),
                    np.median(sub),
                ]
                X.append(feats)
                y.append(1)

    X = np.array(X)
    y = np.array(y)

    # Train / Test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

    clf = RandomForestClassifier(n_estimators=50, random_state=seed)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    return {
        "accuracy": float(accuracy),
        "random_chance_baseline": 0.50,
        "delta_from_chance": float(abs(accuracy - 0.50)),
        "attack_resisted": bool(accuracy < 0.60),  # Classifier cannot reliably distinguish cipher from noise
    }
