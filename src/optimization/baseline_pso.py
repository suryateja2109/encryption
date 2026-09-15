"""
Baseline Particle Swarm Optimization (PSO) Key Optimizer.
Reference: Ding et al., IEEE TCSVT 2025, Section II-B, Eq. (1), Algorithm 1.
"""

from typing import Tuple, List, Optional
import numpy as np
from .fitness import evaluate_key_fitness


class BaselinePSO:
    """
    Standard Particle Swarm Optimization for 3D-CIMBA key generation (Algorithm 1).
    """

    def __init__(
        self,
        n_particles: int = 100,
        n_iterations: int = 100,
        dim: int = 8,
        w: float = 0.8,
        c1: float = 0.5,
        c2: float = 0.5,
        seed: Optional[int] = 42,
    ) -> None:
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.dim = dim
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.rng = np.random.RandomState(seed)

        # Bounds: a, b, delta, K, g in [0, 10]; x1, y1, z1 in [0, 1]
        self.x_min = np.array([0.1, 0.1, 0.05, 0.1, 1.0, 0.01, 0.01, 0.01], dtype=np.float64)
        self.x_max = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 0.99, 0.99, 0.99], dtype=np.float64)

        # Velocity limits: 15% of dynamic range
        self.v_max = 0.15 * (self.x_max - self.x_min)
        self.v_min = -self.v_max

    def optimize(
        self,
        crop_64x64: np.ndarray,
        omega1: float = 0.3,
        omega2: float = 0.7,
        eval_rounds: int = 200,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Execute Algorithm 1 to optimize the 8-parameter key vector.

        Returns:
            (g_best_position, g_best_fitness, convergence_history)
        """
        N = self.n_particles
        D = self.dim

        # Initialize positions and velocities
        positions = self.x_min + self.rng.rand(N, D) * (self.x_max - self.x_min)
        velocities = self.v_min + self.rng.rand(N, D) * (self.v_max - self.v_min)

        # Personal bests
        p_best = positions.copy()
        fp_best = np.full(N, -np.inf, dtype=np.float64)

        # Global best
        g_best = positions[0].copy()
        fg_best = -np.inf

        history = np.empty(self.n_iterations, dtype=np.float64)

        for it in range(self.n_iterations):
            # Evaluate swarm fitness
            for j in range(N):
                fit = evaluate_key_fitness(
                    positions[j],
                    crop_64x64,
                    omega1=omega1,
                    omega2=omega2,
                    roundnum=eval_rounds,
                )

                if fit > fp_best[j]:
                    fp_best[j] = fit
                    p_best[j] = positions[j].copy()

                if fit > fg_best:
                    fg_best = fit
                    g_best = positions[j].copy()

            # Record iteration global best
            history[it] = fg_best

            if verbose and (it + 1) % 10 == 0:
                print(f"[Baseline PSO] Iter {it+1}/{self.n_iterations} | Best Fitness: {fg_best:.4f}")

            # Update velocity and position (Eq. 1)
            r1 = self.rng.rand(N, D)
            r2 = self.rng.rand(N, D)

            velocities = (
                self.w * velocities
                + self.c1 * r1 * (p_best - positions)
                + self.c2 * r2 * (g_best - positions)
            )
            # Bound velocities
            velocities = np.clip(velocities, self.v_min, self.v_max)

            # Bound positions
            positions += velocities
            positions = np.clip(positions, self.x_min, self.x_max)

        return g_best, fg_best, history
