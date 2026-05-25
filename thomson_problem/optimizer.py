"""
optimizer.py — Gradient Descent Engine for the Thomson Problem.

All internal coordinates are Cartesian (N,3).  After every gradient step we
renormalise each position vector to radius R, which automatically respects the
sphere constraint.  Spherical coordinates are used only for reporting and
visualisation.
"""

import numpy as np

from .energy import compute_energy_cartesian, compute_gradient_cartesian


class ThomsonOptimizer:
    """
    Gradient-descent optimizer for the Thomson Problem using Cartesian coordinates.

    Parameters
    ----------
    N : int
        Number of point charges.
    charges : np.ndarray, optional
        Charge array of shape (N,). Defaults to ones.
    R : float, default 1.0
        Sphere radius.
    lr : float, default 0.05
        Initial learning rate (Cartesian step size scale).
    momentum : float, default 0.9
        Momentum coefficient beta.
    lr_decay : float, default 0.001
        Learning rate decay factor.
    max_iter : int, default 50000
        Maximum iterations.
    tol : float, default 1e-10
        Convergence tolerance on energy change.
    patience : int, default 50
        Number of consecutive steps below tolerance before stopping.
    """

    def __init__(
        self,
        N: int,
        charges: np.ndarray = None,
        R: float = 1.0,
        lr: float = 0.05,
        momentum: float = 0.9,
        lr_decay: float = 0.001,
        max_iter: int = 50000,
        tol: float = 1e-10,
        patience: int = 50,
    ):
        self.N = N
        self.charges = charges if charges is not None else np.ones(N, dtype=float)
        self.R = float(R)
        self.lr0 = float(lr)
        self.beta = float(momentum)
        self.decay = float(lr_decay)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.patience = int(patience)

        # State: Cartesian positions (N,3)
        self.positions = None
        self.velocity = None

    def initialize(self, fix_north_pole: bool = True, seed: int = None) -> None:
        """
        Random initial positions uniformly on sphere surface.
        """
        rng = np.random.default_rng(seed)
        # Gaussian sampling then normalising gives uniform distribution on sphere surface
        pos = rng.standard_normal((self.N, 3))
        norms = np.linalg.norm(pos, axis=1, keepdims=True)
        pos = self.R * pos / norms

        if fix_north_pole and self.N > 0:
            pos[0] = np.array([0.0, 0.0, self.R])

        self.positions = pos.astype(float)
        self.velocity = np.zeros((self.N, 3), dtype=float)

    def _to_spherical(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert current Cartesian positions to (theta, phi)."""
        R = np.linalg.norm(self.positions, axis=1)
        theta = np.arccos(np.clip(self.positions[:, 2] / R, -1.0, 1.0))
        phi = np.mod(np.arctan2(self.positions[:, 1], self.positions[:, 0]), 2.0 * np.pi)
        return theta, phi

    def _energy(self) -> float:
        """Current energy."""
        return compute_energy_cartesian(self.positions, self.charges)

    def step(self, t: int, prev_energy: float = np.inf) -> float:
        """
        One gradient descent step with momentum and backtracking line search.
        Returns energy after the step.
        """
        grad_cart = compute_gradient_cartesian(self.positions, self.charges)

        # Project gradient onto tangent plane at each point:
        #   grad_tangent = grad_cart - (grad_cart · r_hat) r_hat
        r_hat = self.positions / self.R
        radial_component = np.sum(grad_cart * r_hat, axis=1, keepdims=True)
        grad_tangent = grad_cart - radial_component * r_hat

        # Momentum update (vectorial)
        self.velocity = self.beta * self.velocity + (1.0 - self.beta) * grad_tangent

        # Decaying learning rate
        lr = self.lr0 / (1.0 + self.decay * t)

        # Backtracking line search – try lr, lr/2, lr/4...
        best_E = np.inf
        best_pos = self.positions.copy()
        best_vel = self.velocity.copy()

        for attempt in range(6):
            pos_new = self.positions - lr * self.velocity
            # Renormalise to sphere
            norms = np.linalg.norm(pos_new, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)  # avoid division by zero
            pos_new = self.R * pos_new / norms
            E = compute_energy_cartesian(pos_new, self.charges)

            if E <= prev_energy + 1e-12:
                best_E = E
                best_pos = pos_new
                best_vel = self.velocity.copy()
                break
            lr *= 0.5

        self.positions = best_pos
        self.velocity = best_vel
        return float(best_E)

    def optimize(self, seed: int = None, trajectory_interval: int = 100) -> dict:
        """
        Full optimisation loop.

        Returns
        -------
        dict with keys:
            final_theta, final_phi, final_energy, energy_history, trajectory,
            converged, iterations.
        """
        self.initialize(fix_north_pole=True, seed=seed)

        energy_history = []
        trajectory = []

        prev_energy = np.inf
        below_tol_count = 0
        converged = False

        for t in range(self.max_iter):
            energy = self.step(t, prev_energy=prev_energy)
            energy_history.append(energy)

            if t % trajectory_interval == 0:
                theta, phi = self._to_spherical()
                trajectory.append((theta.copy(), phi.copy()))

            delta = abs(energy - prev_energy)
            if delta < self.tol:
                below_tol_count += 1
                if below_tol_count >= self.patience:
                    converged = True
                    break
            else:
                below_tol_count = 0

            prev_energy = energy

        # Final snapshot in trajectory
        theta, phi = self._to_spherical()
        trajectory.append((theta.copy(), phi.copy()))

        final_energy = self._energy()

        return {
            "final_theta": theta.copy(),
            "final_phi": phi.copy(),
            "final_energy": final_energy,
            "energy_history": energy_history,
            "trajectory": trajectory,
            "converged": converged,
            "iterations": len(energy_history),
        }

    def multi_start(self, n_runs: int = 100, seeds: list = None) -> tuple[dict, list[float]]:
        """
        Run optimise multiple times with different random seeds.
        """
        all_results = []
        all_energies = []

        rng = np.random.default_rng(42)
        for i in range(n_runs):
            seed = seeds[i] if seeds is not None else int(rng.integers(0, 1_000_000_000))
            result = self.optimize(seed=seed)
            all_results.append(result)
            all_energies.append(result["final_energy"])

        best_idx = int(np.argmin(all_energies))
        return all_results[best_idx], sorted(all_energies)
