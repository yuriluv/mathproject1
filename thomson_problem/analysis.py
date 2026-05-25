"""
analysis.py — Results Analysis for the Thomson Problem.

Provides utility functions to compute bond angles, pairwise distances,
validate computed energies against theoretical values, and identify
the symmetry of converged structures.
"""

import numpy as np
from .energy import compute_cos_gamma


def compute_bond_angles(theta: np.ndarray, phi: np.ndarray, R: float = 1.0) -> np.ndarray:
    """
    Compute all unique pairwise central angles in degrees.

    Parameters
    ----------
    theta : np.ndarray, shape (N,)
    phi : np.ndarray, shape (N,)
    R : float, default 1.0
        Sphere radius (does not affect the angle).

    Returns
    -------
    np.ndarray, shape (M,)
        Sorted array of unique pairwise angles in degrees, where M = N*(N-1)/2.
    """
    cos_gamma = compute_cos_gamma(theta, phi)
    np.clip(cos_gamma, -1.0, 1.0, out=cos_gamma)
    angles = np.arccos(cos_gamma) * 180.0 / np.pi
    # Extract unique upper-triangle entries (i < j)
    iu = np.triu_indices(theta.shape[0], k=1)
    return angles[iu]


def compute_pairwise_distances(theta: np.ndarray, phi: np.ndarray, R: float = 1.0) -> np.ndarray:
    """
    Compute all unique pairwise chord distances and return them sorted.

    Parameters
    ----------
    theta : np.ndarray, shape (N,)
    phi : np.ndarray, shape (N,)
    R : float, default 1.0

    Returns
    -------
    np.ndarray
        Sorted array of unique pairwise distances.
    """
    cos_gamma = compute_cos_gamma(theta, phi)
    distances = R * np.sqrt(2.0 * (1.0 - cos_gamma))
    iu = np.triu_indices(theta.shape[0], k=1)
    d = distances[iu]
    return np.sort(d)


def validate_against_theory(N: int, computed_energy: float, theoretical_energy: float) -> float:
    """
    Print and return the relative error between computed and theoretical energy.

    Parameters
    ----------
    N : int
        Number of charges.
    computed_energy : float
        Optimized energy value.
    theoretical_energy : float
        Known theoretical minimum energy.

    Returns
    -------
    float
        Relative error in percent.
    """
    if theoretical_energy == 0:
        rel_error = 0.0
    else:
        rel_error = abs((computed_energy - theoretical_energy) / theoretical_energy) * 100.0
    print(f"  N={N}: Theoretical E = {theoretical_energy:.6f}, Computed E = {computed_energy:.6f}, "
          f"Relative Error = {rel_error:.4f}%")
    return rel_error


def identify_structure(theta: np.ndarray, phi: np.ndarray) -> str:
    """
    Identify the approximate geometry by comparing computed bond angles
    against known structures for N = 2..6.

    Parameters
    ----------
    theta : np.ndarray, shape (N,)
    phi : np.ndarray, shape (N,)

    Returns
    -------
    str
        Description of the identified geometry.
    """
    N = theta.shape[0]
    angles = compute_bond_angles(theta, phi)
    # Round to nearest integer for matching
    rounded = np.round(angles)

    def count_near(target, tol=3.0):
        return int(np.sum(np.abs(angles - target) < tol))

    if N == 2:
        return "Linear (180°)"
    elif N == 3:
        if count_near(120.0) >= 2:
            return "Equilateral triangle (120°)"
        return "Triangle (irregular)"
    elif N == 4:
        if count_near(109.5) >= 5:
            return "Tetrahedron (109.47°)"
        return "Tetrahedron-like"
    elif N == 5:
        cnt90 = count_near(90.0)
        cnt120 = count_near(120.0)
        cnt180 = count_near(180.0)
        if cnt90 >= 4 and cnt120 >= 2 and cnt180 >= 1:
            return "Trigonal bipyramid (90°, 120°, 180°)"
        return "Trigonal bipyramid-like"
    elif N == 6:
        cnt90 = count_near(90.0)
        cnt180 = count_near(180.0)
        if cnt90 >= 12 and cnt180 >= 3:
            return "Octahedron (90°, 180°)"
        return "Octahedron-like"
    else:
        return "Unknown / higher-order structure"
