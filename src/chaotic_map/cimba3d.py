"""
3D Coupled Ikeda Map with Bounded Amplitude (3D-CIMBA).
Reference: Ding et al., IEEE TCSVT 2025, Eq. (7)-(8).
"""

from typing import Tuple, Optional
import numpy as np


class CIMBAMap:
    """
    Implements the 3-Dimensional Coupled Ikeda Map with Bounded Amplitude (3D-CIMBA).

    System Equations (Eq. 7-8):
        phi_n = delta - K / (1 + x_n^2 + y_n^2 + z_n^2)
        x_{n+1} = mod(a + b*(x_n*cos(phi_n) - y_n*sin(phi_n)) + exp(g)*x_n, 1)
        y_{n+1} = mod(b*(x_n*sin(phi_n) - y_n*cos(phi_n)) + exp(g)*y_n, 1)
        z_{n+1} = mod(b*(x_n*sin(phi_n) - y_n*sin(phi_n)) + exp(g)*z_n, 1)
    """

    def __init__(
        self,
        a: float = 10.0,
        b: float = 10.0,
        delta: float = 0.4,
        K: float = 6.0,
        g: float = 10.0,
        x1: float = 0.1,
        y1: float = 0.2,
        z1: float = 0.3,
    ) -> None:
        """
        Initialize the 3D-CIMBA map with control parameters and initial state.
        Default values match the base paper's simulation settings in Section III-B.
        """
        self.a = float(a)
        self.b = float(b)
        self.delta = float(delta)
        self.K = float(K)
        self.g = float(g)
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.z1 = float(z1)

        # Precompute exp(g)
        self._exp_g = float(np.exp(self.g))

    def step(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Perform a single iteration of the 3D-CIMBA map.
        Returns:
            (x_next, y_next, z_next) in [0, 1)
        """
        denom = 1.0 + x * x + y * y + z * z
        phi = self.delta - self.K / denom

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        x_next = (self.a + self.b * (x * cos_phi - y * sin_phi) + self._exp_g * x) % 1.0
        y_next = (self.b * (x * sin_phi - y * cos_phi) + self._exp_g * y) % 1.0
        z_next = (self.b * (x * sin_phi - y * sin_phi) + self._exp_g * z) % 1.0

        return float(x_next), float(y_next), float(z_next)

    def iterate(
        self,
        n_steps: int,
        discard: int = 500,
        init_state: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate trajectory of length n_steps after discarding initial transient states.

        Args:
            n_steps: Number of states to retain.
            discard: Number of initial transient states to discard (default 500 per Section IV-C Step 3).
            init_state: Optional (x, y, z) starting state. Defaults to (self.x1, self.y1, self.z1).

        Returns:
            Tuple of (x, y, z) numpy arrays of shape (n_steps,), with elements in [0, 1).
        """
        if init_state is not None:
            x, y, z = init_state
        else:
            x, y, z = self.x1, self.y1, self.z1

        total_steps = n_steps + discard

        # Allocate arrays
        xs = np.empty(total_steps, dtype=np.float64)
        ys = np.empty(total_steps, dtype=np.float64)
        zs = np.empty(total_steps, dtype=np.float64)

        a = self.a
        b = self.b
        delta = self.delta
        K = self.K
        exp_g = self._exp_g

        for i in range(total_steps):
            denom = 1.0 + x * x + y * y + z * z
            phi = delta - K / denom
            cos_p = np.cos(phi)
            sin_p = np.sin(phi)

            x = (a + b * (x * cos_p - y * sin_p) + exp_g * x) % 1.0
            y = (b * (x * sin_p - y * cos_p) + exp_g * y) % 1.0
            z = (b * (x * sin_p - y * sin_p) + exp_g * z) % 1.0

            xs[i] = x
            ys[i] = y
            zs[i] = z

        return xs[discard:], ys[discard:], zs[discard:]

    def jacobian(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute the 3x3 analytical Jacobian matrix J at state (x, y, z).
        Reference: Eq. (10)-(11).
        """
        denom = 1.0 + x * x + y * y + z * z
        phi = self.delta - self.K / denom
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)

        # Partial derivatives of phi: d(phi)/dx = 2*K*x / denom^2
        dphi_dx = (2.0 * self.K * x) / (denom * denom)
        dphi_dy = (2.0 * self.K * y) / (denom * denom)
        dphi_dz = (2.0 * self.K * z) / (denom * denom)

        # Row 1: x_{n+1} = a + b*(x*cos(phi) - y*sin(phi)) + exp(g)*x
        # dx_{n+1}/dx = b*(cos(phi) - (x*sin(phi) + y*cos(phi))*dphi_dx) + exp(g)
        u1 = x * sin_p + y * cos_p
        j11 = self.b * (cos_p - u1 * dphi_dx) + self._exp_g
        j12 = self.b * (-sin_p - u1 * dphi_dy)
        j13 = self.b * (-u1 * dphi_dz)

        # Row 2: y_{n+1} = b*(x*sin(phi) - y*cos(phi)) + exp(g)*y
        # dy_{n+1}/dx = b*(sin(phi) + (x*cos(phi) + y*sin(phi))*dphi_dx)
        u2 = x * cos_p + y * sin_p
        j21 = self.b * (sin_p + u2 * dphi_dx)
        j22 = self.b * (-cos_p + u2 * dphi_dy) + self._exp_g
        j23 = self.b * (u2 * dphi_dz)

        # Row 3: z_{n+1} = b*(x*sin(phi) - y*sin(phi)) + exp(g)*z
        # dz_{n+1}/dx = b*(sin(phi) + (x*cos(phi) - y*cos(phi))*dphi_dx)
        u3 = (x - y) * cos_p
        j31 = self.b * (sin_p + u3 * dphi_dx)
        j32 = self.b * (-sin_p + u3 * dphi_dy)
        j33 = self.b * (u3 * dphi_dz) + self._exp_g

        return np.array([[j11, j12, j13], [j21, j22, j23], [j31, j32, j33]], dtype=np.float64)


def generate_3d_cimba_sequences(
    params: np.ndarray,
    length: int,
    discard: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper for vector/optimizer key representations.

    Args:
        params: 8-element array [a, b, delta, K, g, x1, y1, z1].
        length: Required sequence length.
        discard: Transient states to discard.

    Returns:
        (x, y, z) arrays of length `length`.
    """
    a, b, delta, K, g, x1, y1, z1 = params
    cimba = CIMBAMap(a=a, b=b, delta=delta, K=K, g=g, x1=x1, y1=y1, z1=z1)
    return cimba.iterate(n_steps=length, discard=discard)
