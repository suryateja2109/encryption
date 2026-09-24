"""
Cryptanalysis package covering all metric families from Section V and extended security tests.
"""

from .metrics import (
    compute_entropy_channel,
    compute_image_entropy,
    compute_histogram_variance,
    compute_ssim_psnr,
)
from .correlation import (
    sample_adjacent_pixel_pairs,
    compute_correlation_coefficient,
    evaluate_image_correlations,
)
from .differential import (
    compute_npcr_uaci,
    compute_critical_thresholds,
    evaluate_differential_security,
)
from .robustness import (
    apply_gaussian_noise,
    apply_salt_and_pepper_noise,
    apply_cropping_attack,
    benchmark_robustness_suite,
)
from .randomness import (
    compute_key_space,
    evaluate_key_sensitivity,
    benchmark_encryption_speed,
    run_nist_statistical_tests,
    compute_live_randomness_tests,
)
from .extended_analysis import (
    compute_local_shannon_entropy,
    compute_chi_square_uniformity,
    measure_avalanche_effect,
    evaluate_machine_learning_attack,
)

__all__ = [
    "compute_entropy_channel",
    "compute_image_entropy",
    "compute_histogram_variance",
    "compute_ssim_psnr",
    "sample_adjacent_pixel_pairs",
    "compute_correlation_coefficient",
    "evaluate_image_correlations",
    "compute_npcr_uaci",
    "compute_critical_thresholds",
    "evaluate_differential_security",
    "apply_gaussian_noise",
    "apply_salt_and_pepper_noise",
    "apply_cropping_attack",
    "benchmark_robustness_suite",
    "compute_key_space",
    "evaluate_key_sensitivity",
    "benchmark_encryption_speed",
    "run_nist_statistical_tests",
    "compute_live_randomness_tests",
    "compute_local_shannon_entropy",
    "compute_chi_square_uniformity",
    "measure_avalanche_effect",
    "evaluate_machine_learning_attack",
]
