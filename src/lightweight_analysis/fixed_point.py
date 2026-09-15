"""
Fixed-Point Quantization Simulation for 3D-CIMBA and STP Diffusion.
Reference: Base Paper Future Work #3 ("lightweight deployment using the FPGA platform").
"""

from typing import Dict, Any, Tuple
import numpy as np
from src.chaotic_map.cimba3d import CIMBAMap
from src.cryptanalysis.metrics import compute_entropy_channel


class FixedPointCIMBA:
    """
    Simulates fixed-point arithmetic (Q-format: Q(integer_bits).(fraction_bits))
    for 3D-CIMBA chaotic map to evaluate FPGA precision degradation.
    """

    def __init__(
        self,
        word_length: int = 24,
        fraction_bits: int = 16,
        a: float = 10.0,
        b: float = 10.0,
        delta: float = 0.4,
        K: float = 6.0,
        g: float = 10.0,
        x1: float = 0.1,
        y1: float = 0.2,
        z1: float = 0.3,
    ) -> None:
        self.word_length = word_length
        self.fraction_bits = fraction_bits
        self.scale = 1 << fraction_bits

        # Quantize control parameters
        self.a_q = int(round(a * self.scale))
        self.b_q = int(round(b * self.scale))
        self.delta_q = int(round(delta * self.scale))
        self.K_q = int(round(K * self.scale))
        self.exp_g_q = int(round(np.exp(g) * self.scale))

        # Initial states (scaled integers in [0, scale))
        self.x = int(round(x1 * self.scale))
        self.y = int(round(y1 * self.scale))
        self.z = int(round(z1 * self.scale))

    def _quantize_float(self, val: float) -> int:
        return int(round(val * self.scale))

    def _to_float(self, val_q: int) -> float:
        return float(val_q) / self.scale

    def step(self) -> Tuple[float, float, float]:
        """
        Fixed-point simulation of one 3D-CIMBA iteration.
        """
        # denom = 1 + x^2 + y^2 + z^2 in fixed point
        scale = self.scale
        x_sq = (self.x * self.x) >> self.fraction_bits
        y_sq = (self.y * self.y) >> self.fraction_bits
        z_sq = (self.z * self.z) >> self.fraction_bits
        denom = scale + x_sq + y_sq + z_sq

        # phi = delta - K / denom
        K_scaled = self.K_q << self.fraction_bits
        quotient = K_scaled // max(1, denom)
        phi_q = self.delta_q - quotient
        phi = float(phi_q) / scale

        cos_q = int(round(np.cos(phi) * scale))
        sin_q = int(round(np.sin(phi) * scale))

        # x_{n+1}
        u1 = (self.x * cos_q - self.y * sin_q) >> self.fraction_bits
        bu1 = (self.b_q * u1) >> self.fraction_bits
        eg_x = (self.exp_g_q * self.x) >> self.fraction_bits
        x_next = (self.a_q + bu1 + eg_x) % scale

        # y_{n+1}
        u2 = (self.x * sin_q - self.y * cos_q) >> self.fraction_bits
        bu2 = (self.b_q * u2) >> self.fraction_bits
        eg_y = (self.exp_g_q * self.y) >> self.fraction_bits
        y_next = (bu2 + eg_y) % scale

        # z_{n+1}
        u3 = (self.x * sin_q - self.y * sin_q) >> self.fraction_bits
        bu3 = (self.b_q * u3) >> self.fraction_bits
        eg_z = (self.exp_g_q * self.z) >> self.fraction_bits
        z_next = (bu3 + eg_z) % scale

        self.x, self.y, self.z = x_next, y_next, z_next
        return self._to_float(x_next), self._to_float(y_next), self._to_float(z_next)

    def iterate(self, n_steps: int, discard: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        for _ in range(discard):
            self.step()

        xs = np.empty(n_steps, dtype=np.float64)
        ys = np.empty(n_steps, dtype=np.float64)
        zs = np.empty(n_steps, dtype=np.float64)

        for i in range(n_steps):
            x, y, z = self.step()
            xs[i] = x
            ys[i] = y
            zs[i] = z

        return xs, ys, zs


def evaluate_precision_degradation(
    n_points: int = 5000,
) -> Dict[str, Dict[str, float]]:
    """
    Compare chaotic sequence properties between double precision (64-bit)
    and fixed-point quantization (16-bit Q8.8, 24-bit Q8.16, 32-bit Q12.20).

    Returns:
        Dictionary mapping precision name to entropy, mean, std, and correlation against double.
    """
    # 1. Baseline Double Precision (64-bit)
    cimba_double = CIMBAMap(a=10.0, b=10.0, delta=0.4, K=6.0, g=10.0)
    x_dbl, _, _ = cimba_double.iterate(n_steps=n_points, discard=500)
    bytes_dbl = np.floor(np.mod(x_dbl * 1e5, 256.0)).astype(np.uint8)
    ent_dbl = compute_entropy_channel(bytes_dbl)

    precisions = {
        "Double_64bit": {"word_length": 64, "fraction_bits": 52, "inst": None},
        "Fixed_32bit_Q12.20": {"word_length": 32, "fraction_bits": 20},
        "Fixed_24bit_Q8.16": {"word_length": 24, "fraction_bits": 16},
        "Fixed_16bit_Q8.8": {"word_length": 16, "fraction_bits": 8},
    }

    results = {
        "Double_64bit": {
            "entropy": ent_dbl,
            "mean": float(np.mean(x_dbl)),
            "std": float(np.std(x_dbl)),
            "corr_with_double": 1.0,
        }
    }

    for name, spec in precisions.items():
        if name == "Double_64bit":
            continue
        fp = FixedPointCIMBA(
            word_length=spec["word_length"],
            fraction_bits=spec["fraction_bits"],
            a=10.0,
            b=10.0,
            delta=0.4,
            K=6.0,
            g=10.0,
        )
        x_fp, _, _ = fp.iterate(n_steps=n_points, discard=500)
        bytes_fp = np.floor(np.mod(x_fp * 1e5, 256.0)).astype(np.uint8)
        ent_fp = compute_entropy_channel(bytes_fp)

        # Correlation between initial trajectories
        corr = np.corrcoef(x_dbl[:200], x_fp[:200])[0, 1]

        results[name] = {
            "entropy": float(ent_fp),
            "mean": float(np.mean(x_fp)),
            "std": float(np.std(x_fp)),
            "corr_with_double": float(corr) if not np.isnan(corr) else 0.0,
        }

    return results
