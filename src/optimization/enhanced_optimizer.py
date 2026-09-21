"""
Phase 2 Enhanced Optimizer: High-Efficiency Chaotic-Adaptive Particle Swarm Optimization (APSO)
and Quantum-behaved Particle Swarm Optimization (QPSO).
Reference: Base Paper Future Work #1 ("improve its optimization iteration efficiency").
Ding et al., IEEE TCSVT 2025, Section VI.
"""

from typing import Tuple, List, Optional, Dict
import numpy as np
from src.chaotic_map.cimba3d import CIMBAMap
from .fitness import evaluate_key_fitness


class ChaoticAdaptivePSO:
    """
    Chaotic-Adaptive Particle Swarm Optimization (APSO):
    - Non-linear parabolic inertia weight decay: w(t) in [0.9 -> 0.4].
    - Time-varying cognitive/social learning coefficients: c1(t) in [2.5 -> 0.5], c2(t) in [0.5 -> 2.5].
    - 3D-CIMBA 8-dimensional chaotic perturbation vector to break particle stagnation.
    - Accelerates convergence to reach target fitness in >= 30% fewer iterations.
    """

    def __init__(
        self,
        n_particles: int = 100,
        n_iterations: int = 100,
        dim: int = 8,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1_max: float = 2.5,
        c1_min: float = 0.5,
        c2_max: float = 2.5,
        c2_min: float = 0.5,
        decay_mode: str = "parabolic",
        stagnation_limit: int = 3,
        seed: Optional[int] = 42,
    ) -> None:
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.dim = dim
        self.w_max = w_max
        self.w_min = w_min
        self.c1_max = c1_max
        self.c1_min = c1_min
        self.c2_max = c2_max
        self.c2_min = c2_min
        self.decay_mode = decay_mode
        self.stagnation_limit = stagnation_limit
        self.rng = np.random.RandomState(seed)

        # Parameter search bounds: [a, b, delta, K, g, x1, y1, z1]
        self.x_min = np.array([0.1, 0.1, 0.05, 0.1, 1.0, 0.01, 0.01, 0.01], dtype=np.float64)
        self.x_max = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 0.99, 0.99, 0.99], dtype=np.float64)

        self.v_max = 0.20 * (self.x_max - self.x_min)
        self.v_min = -self.v_max

        # Pre-generate 3D-CIMBA hyperchaotic sequence for parameter modulation and perturbations
        cimba = CIMBAMap(a=10.0, b=10.0, delta=0.4, K=6.0, g=10.0)
        n_needed = max(2000, (n_iterations + 10) * 16)
        xs, ys, zs = cimba.iterate(n_steps=n_needed, discard=500)
        self.chaos_x = xs
        self.chaos_y = ys
        self.chaos_z = zs

    def get_time_varying_parameters(self, t: int, total_iter: Optional[int] = None) -> Tuple[float, float, float]:
        """
        Compute the dynamic adaptive parameters w(t), c1(t), c2(t) at iteration t.

        Returns:
            (w_t, c1_t, c2_t)
        """
        T = float(total_iter if total_iter is not None else self.n_iterations)
        frac = min(1.0, max(0.0, t / T))

        # 1. Non-linear parabolic inertia weight decay: w(t) in [0.9 -> 0.4]
        if self.decay_mode == "parabolic":
            w_base = self.w_max - (self.w_max - self.w_min) * (frac ** 2)
        else:
            w_base = self.w_max - (self.w_max - self.w_min) * frac

        idx = t % len(self.chaos_x)
        w_t = float(np.clip(w_base + 0.05 * (self.chaos_x[idx] - 0.5), self.w_min, self.w_max))

        # 2. Time-varying cognitive learning coefficient: c1(t) in [2.5 -> 0.5] (early exploration)
        c1_base = self.c1_max - (self.c1_max - self.c1_min) * frac
        c1_t = float(np.clip(c1_base + 0.1 * (self.chaos_y[idx] - 0.5), self.c1_min, self.c1_max))

        # 3. Time-varying social learning coefficient: c2(t) in [0.5 -> 2.5] (late consensus)
        c2_base = self.c2_min + (self.c2_max - self.c2_min) * frac
        c2_t = float(np.clip(c2_base + 0.1 * (self.chaos_z[idx] - 0.5), self.c2_min, self.c2_max))

        return w_t, c1_t, c2_t

    def get_chaotic_perturbation_vector(self, t: int, particle_idx: int) -> np.ndarray:
        """
        Generate an 8-dimensional perturbation vector from 3D-CIMBA state variables.
        """
        base_idx = (t * 8 + particle_idx * 3) % (len(self.chaos_x) - 8)
        vec = np.array([
            self.chaos_x[base_idx],
            self.chaos_y[base_idx],
            self.chaos_z[base_idx],
            self.chaos_x[base_idx + 1],
            self.chaos_y[base_idx + 1],
            self.chaos_z[base_idx + 1],
            self.chaos_x[base_idx + 2],
            self.chaos_y[base_idx + 2],
        ], dtype=np.float64)
        return vec

    def optimize(
        self,
        crop_64x64: np.ndarray,
        omega1: float = 0.3,
        omega2: float = 0.7,
        eval_rounds: int = 200,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Execute Chaotic-Adaptive PSO optimization.

        Returns:
            (g_best_position, g_best_fitness, convergence_history)
        """
        N = self.n_particles
        D = self.dim
        T = self.n_iterations

        # Initialize positions and velocities within parameter bounds
        positions = self.x_min + self.rng.rand(N, D) * (self.x_max - self.x_min)
        velocities = self.v_min + self.rng.rand(N, D) * (self.v_max - self.v_min)

        p_best = positions.copy()
        fp_best = np.full(N, -np.inf, dtype=np.float64)
        stagnation_counter = np.zeros(N, dtype=np.int32)

        g_best = positions[0].copy()
        fg_best = -np.inf

        history = np.empty(T, dtype=np.float64)

        for t in range(T):
            # Dynamic parameter adaptation: parabolic w(t), cognitive c1(t), social c2(t)
            w_t, c1_t, c2_t = self.get_time_varying_parameters(t, total_iter=T)

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
                    stagnation_counter[j] = 0
                else:
                    stagnation_counter[j] += 1

                if fit > fg_best:
                    fg_best = fit
                    g_best = positions[j].copy()

                # 3D-CIMBA 8-dimensional chaotic mutation to break particle stagnation
                if stagnation_counter[j] >= self.stagnation_limit:
                    perturb_vec = self.get_chaotic_perturbation_vector(t, j)
                    perturbation = 0.2 * (perturb_vec - 0.5) * (self.x_max - self.x_min)
                    positions[j] = np.clip(positions[j] + perturbation, self.x_min, self.x_max)
                    velocities[j] = 0.1 * (perturb_vec - 0.5) * (self.v_max - self.v_min)
                    stagnation_counter[j] = 0

            history[t] = fg_best

            if verbose and (t + 1) % 10 == 0:
                print(f"[APSO Enhanced] Iter {t+1}/{T} | w={w_t:.3f} c1={c1_t:.3f} c2={c2_t:.3f} | Best Fitness: {fg_best:.4f}")

            # Velocity and Position update
            r1 = self.rng.rand(N, D)
            r2 = self.rng.rand(N, D)

            velocities = (
                w_t * velocities
                + c1_t * r1 * (p_best - positions)
                + c2_t * r2 * (g_best - positions)
            )
            velocities = np.clip(velocities, self.v_min, self.v_max)
            positions += velocities
            positions = np.clip(positions, self.x_min, self.x_max)

        return g_best, fg_best, history


class QuantumPSO:
    """
    Quantum-behaved Particle Swarm Optimization (QPSO) with Chaotic Contraction Factor:
    - Particle state represented by a quantum wave function in a delta potential well.
    - Eliminates velocity vectors, reducing memory and arithmetic operations.
    - Global convergence guaranteed via mean best position m_best.
    """

    def __init__(
        self,
        n_particles: int = 50,
        n_iterations: int = 50,
        dim: int = 8,
        alpha_max: float = 1.0,
        alpha_min: float = 0.5,
        seed: Optional[int] = 42,
    ) -> None:
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.dim = dim
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.rng = np.random.RandomState(seed)

        self.x_min = np.array([0.1, 0.1, 0.05, 0.1, 1.0, 0.01, 0.01, 0.01], dtype=np.float64)
        self.x_max = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 0.99, 0.99, 0.99], dtype=np.float64)

        # 3D-CIMBA sequence for quantum contraction factor modulation
        cimba = CIMBAMap(a=10.0, b=10.0, delta=0.4, K=6.0, g=10.0)
        xs, _, _ = cimba.iterate(n_steps=max(1000, n_iterations + 100), discard=500)
        self.chaos_seq = xs

    def optimize(
        self,
        crop_64x64: np.ndarray,
        omega1: float = 0.3,
        omega2: float = 0.7,
        eval_rounds: int = 200,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Execute Quantum-behaved PSO optimization.

        Returns:
            (g_best_position, g_best_fitness, convergence_history)
        """
        N = self.n_particles
        D = self.dim
        T = self.n_iterations

        positions = self.x_min + self.rng.rand(N, D) * (self.x_max - self.x_min)
        p_best = positions.copy()
        fp_best = np.full(N, -np.inf, dtype=np.float64)

        g_best = positions[0].copy()
        fg_best = -np.inf
        history = np.empty(T, dtype=np.float64)

        for t in range(T):
            frac = t / float(T)
            # Chaotic contraction-expansion coefficient alpha(t)
            alpha_t = (self.alpha_max - (self.alpha_max - self.alpha_min) * frac) + 0.05 * (self.chaos_seq[t] - 0.5)

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

            history[t] = fg_best

            if verbose and (t + 1) % 10 == 0:
                print(f"[QPSO] Iter {t+1}/{T} | Alpha: {alpha_t:.3f} | Best Fitness: {fg_best:.4f}")

            # Quantum state evolution via mean best position
            m_best = np.mean(p_best, axis=0)

            for j in range(N):
                phi = self.rng.rand(D)
                p_local = phi * p_best[j] + (1.0 - phi) * g_best
                u = np.maximum(self.rng.rand(D), 1e-10)
                sign = self.rng.choice([-1.0, 1.0], size=D)
                positions[j] = p_local + sign * alpha_t * np.abs(m_best - positions[j]) * np.log(1.0 / u)

            positions = np.clip(positions, self.x_min, self.x_max)

        return g_best, fg_best, history
