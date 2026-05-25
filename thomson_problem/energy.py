"""
energy.py - Energy and Gradient Computation for the Thomson Problem.

This module implements fully vectorized computation of Coulomb potential energy
and its analytical gradient for point charges on a sphere. It supports both
Cartesian and spherical coordinate representations.

Physics:
- For N point charges q_i on a sphere of radius R, total Coulomb potential is:
  E = sum_{i<j} (k * q_i * q_j / d_ij)
- Energies computed in cartesian coordinates to avoid issues with local minima.

Gradient:
- Cartesian gradient: dE/dr_i = sum_{j!=i} (k * q_i * q_j / d^3) * (r_j - r_i)
- Spherical gradient: dE/dtheta_i = (dE/dr_i) . e_theta_i, dE/dphi_i = (dE/dr_i) . e_phi_i
"""

import numpy as np

# Constants
epsilon = 1e-12


def compute_cos_gamma(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Compute the matrix of cos(gamma_ij) for all pairs (i, j).
    Uses spherical coordinates : theta (polar) and phi (azimuthal).
    
    cos_gamma_ij = sin(theta_i) * sin(theta_j) * cos(phi_i - phi_j) + cos(theta_i) * cos(theta_j)
    """
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)

    cos_delta_phi = cos_p[:, None] * cos_p[None, :] + sin_p[:, None] * sin_p[None, :]
    cos_gamma = sin_t[:, None] * sin_t[None, :] * cos_delta_phi + cos_t[:, None] * cos_t[None, :]
    np.clip(cos_gamma, -1.0, 1.0, out=cos_gamma)
    return cos_gamma


def compute_distances(cos_gamma: np.ndarray, R: float = 1.0) -> np.ndarray:
    """
    Compute pairwise chord distances with a soft core to avoid 0/inf issues.
    When particles coincide, 1 - cos_gamma = 0, so we add epsilon to avoid div by zero.
    """
    distances = R * np.sqrt(2.0 * (1.0 - cos_gamma) + epsilon)
    np.fill_diagonal(distances, np.inf)
    return distances


def compute_energy(distances: np.ndarray, charges: np.ndarray, k: float = 1.0) -> float:
    """
    Compute total Coulomb potential energy from precomputed distances.
    
    Parameters
    ----------
    distances : np.ndarray, shape (N, N)
        Pairwise distance matrix. Diagonal should be infinity.
    charges : np.ndarray, shape (N,)
        Charge values of each particle.
    k : float, default 1.0
        Coulomb constant (absorbed into units for normalized comparisons).

    Returns
    -------
    float
        Total energy E = sum_{i < j} (k * q_i * q_j / d_ij).
    """
    q_matrix = charges[:, None] * charges[None, :]
    energies = k * q_matrix / distances
    return float(np.sum(np.triu(energies, k=1)))


def compute_energy_spherical(theta: np.ndarray, phi: np.ndarray, charges: np.ndarray, R: float = 1.0, k: float = 1.0) -> float:
    """
    Compute total energy from spherical coordinates.
    """
    cos_gamma = compute_cos_gamma(theta, phi)
    distances = compute_distances(cos_gamma, R)
    return compute_energy(distances, charges, k)


def compute_energy_cartesian(positions: np.ndarray, charges: np.ndarray, k: float = 1.0) -> float:
    """
    Compute total Coulomb energy from Cartesian coordinates (N,3).
    
    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Cartesian coordinates of N particles on a sphere.
    charges : np.ndarray, shape (N,)
        Charge values of each particle.
    k : float, default 1.0
        Coulomb constant.

    Returns
    -------
    float
        Total energy.
    """
    diff = positions[None, :, :] - positions[:, None, :]   # (N,N,3)
    d2 = np.sum(diff ** 2, axis=2) + epsilon
    d = np.sqrt(d2)
    np.fill_diagonal(d, np.inf)
    q_matrix = charges[:, None] * charges[None, :]
    energies = k * q_matrix / d
    return float(np.sum(np.triu(energies, k=1)))


def compute_gradient(theta: np.ndarray, phi: np.ndarray, charges: np.ndarray, R: float = 1.0, k: float = 1.0) -> tuple:
    """
    Compute the analytical gradient of energy with respect to theta and phi.

    Derivation:
        E = sum_{i < j} k q_i q_j / d_ij
        d_ij = R sqrt(2(1 - cos_gamma_ij))

        dE/d(cos_gamma_kj) = -k q_k q_j / (R * d_kj^3) * R^2
                           = -k q_k q_j * R^2 / d_kj^3

        d(cos_gamma_kj)/d(theta_k) =
            cos(theta_k) sin(theta_j) cos(phi_k - phi_j) - sin(theta_k) cos(theta_j)
        d(cos_gamma_kj)/d(phi_k) =
            -sin(theta_k) sin(theta_j) sin(phi_k - phi_j)

        dE/d(theta_k) = sum_{j != k} dE/d(cos_gamma_kj) * d(cos_gamma_kj)/d(theta_k)
        dE/d(phi_k)   = sum_{j != k} dE/d(cos_gamma_kj) * d(cos_gamma_kj)/d(phi_k)
    """
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)

    delta_phi = phi[:, None] - phi[None, :]
    cos_delta_phi = np.cos(delta_phi)
    sin_delta_phi = np.sin(delta_phi)

    cos_gamma = (
        sin_t[:, None] * sin_t[None, :] * cos_delta_phi
        + cos_t[:, None] * cos_t[None, :]
    )
    np.clip(cos_gamma, -1.0, 1.0, out=cos_gamma)

    d_ij = R * np.sqrt(2.0 * (1.0 - cos_gamma) + epsilon)
    np.fill_diagonal(d_ij, np.inf)

    q_matrix = charges[:, None] * charges[None, :]
    dE_dcos = k * q_matrix * (R ** 2) / (d_ij ** 3)
    np.fill_diagonal(dE_dcos, 0.0)

    dcos_dtheta = (
        cos_t[:, None] * sin_t[None, :] * cos_delta_phi
        - sin_t[:, None] * cos_t[None, :]
    )

    dcos_dphi = -sin_t[:, None] * sin_t[None, :] * sin_delta_phi

    grad_theta = np.sum(dE_dcos * dcos_dtheta, axis=1)
    grad_phi = np.sum(dE_dcos * dcos_dphi, axis=1)

    return grad_theta, grad_phi


def _positions(theta, phi, R=1.0):
    """Spherical -> Cartesian coordinates for gradient computation."""
    st = np.sin(theta)
    x = R * st * np.cos(phi)
    y = R * st * np.sin(phi)
    z = R * np.cos(theta)
    return np.column_stack([x, y, z])


def compute_gradient_cartesian(positions, charges, k=1.0):
    """
    Compute the Cartesian gradient of the energy.
    
    positions : np.ndarray of shape (N, 3)
    charges   : np.ndarray of shape (N,)
    
    Returns np.ndarray of shape (N, 3).
    """
    diff = positions[None, :, :] - positions[:, None, :]
    d2 = np.sum(diff ** 2, axis=2) + epsilon
    d = np.sqrt(d2)
    np.fill_diagonal(d, np.inf)

    q_matrix = charges[:, None] * charges[None, :]
    factor = k * q_matrix / (d ** 3)
    np.fill_diagonal(factor, 0.0)

    grad = np.sum(factor[:, :, None] * diff, axis=1)
    return grad