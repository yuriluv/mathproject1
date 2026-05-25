"""
experiments.py — Experiment Runner.

Implements the four experiments specified in the development plan:
1. Validation against theoretical minima (N=2..6).
2. Exploration of higher N geometries (N=7..12).
3. Symmetry breaking (lone-pair weight effect on bond angles).
4. Local minima analysis (histogram for N=8).
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from .optimizer import ThomsonOptimizer
from . import analysis as an
from . import visualize as vis


def _ensure_output_dir(path="output"):
    os.makedirs(path, exist_ok=True)


THEORETICAL_ENERGIES = {
    2: 0.500,
    3: 1.732,
    4: 3.674,
    5: 6.475,
    6: 9.985,
}


def run_experiment_1_validation(N_range=(2, 3, 4, 5, 6), output_dir="output"):
    """
    For each N, run multi_start(100), compare to theory, save 3D and convergence plots.

    Returns
    -------
    list of dict
        One dict per N with keys: N, theoretical_E, computed_E, rel_error, structure, angles.
    """
    _ensure_output_dir(output_dir)
    results = []
    print("\n=== Experiment 1: Validation (N=2..6) ===")
    for N in N_range:
        print(f"\nRunning N={N} ...")
        opt = ThomsonOptimizer(N=N)
        best, all_energies = opt.multi_start(n_runs=100)
        E = best["final_energy"]
        theory = THEORETICAL_ENERGIES[N]
        rel_err = an.validate_against_theory(N, E, theory)
        struct = an.identify_structure(best["final_theta"], best["final_phi"])
        angles = an.compute_bond_angles(best["final_theta"], best["final_phi"])
        print(f"  Structure: {struct}")
        print(f"  Bond angles (deg): {np.round(angles, 2).tolist()}")

        # 3D plot
        fig3d = vis.plot_sphere_with_charges(
            best["final_theta"], best["final_phi"], title=f"N={N} — {struct}"
        )
        try:
            fig3d.write_image(os.path.join(output_dir, f"exp1_N{N}_3d.png"), scale=2)
        except Exception:
            # Fallback if kaleido is not available; save HTML instead
            fig3d.write_html(os.path.join(output_dir, f"exp1_N{N}_3d.html"))

        # Convergence plot
        fig_conv = vis.plot_energy_convergence(
            best["energy_history"], theoretical_E=theory, title=f"Energy Convergence (N={N})"
        )
        fig_conv.savefig(os.path.join(output_dir, f"exp1_N{N}_convergence.png"), dpi=300)
        plt.close(fig_conv)

        results.append({
            "N": N,
            "theoretical_E": theory,
            "computed_E": E,
            "rel_error_percent": rel_err,
            "structure": struct,
            "bond_angles_deg": angles.tolist(),
            "iterations": best["iterations"],
            "converged": best["converged"],
        })
    return results


def run_experiment_2_exploration(N_range=(7, 8, 9, 10, 11, 12), output_dir="output"):
    """
    For each N, run multi_start(200), visualize best structure, report bond angles.

    Returns
    -------
    list of dict
        One dict per N with keys: N, computed_E, structure, angles.
    """
    _ensure_output_dir(output_dir)
    results = []
    print("\n=== Experiment 2: Exploration (N=7..12) ===")
    for N in N_range:
        print(f"\nRunning N={N} ...")
        opt = ThomsonOptimizer(N=N)
        best, all_energies = opt.multi_start(n_runs=200)
        E = best["final_energy"]
        struct = an.identify_structure(best["final_theta"], best["final_phi"])
        angles = an.compute_bond_angles(best["final_theta"], best["final_phi"])
        print(f"  Best energy: {E:.6f}")
        print(f"  Structure: {struct}")
        print(f"  Bond angles (deg): {np.round(angles, 2).tolist()}")

        # 3D plot
        fig3d = vis.plot_sphere_with_charges(
            best["final_theta"], best["final_phi"], title=f"N={N} — {struct}"
        )
        try:
            fig3d.write_image(os.path.join(output_dir, f"exp2_N{N}_3d.png"), scale=2)
        except Exception:
            fig3d.write_html(os.path.join(output_dir, f"exp2_N{N}_3d.html"))

        results.append({
            "N": N,
            "computed_E": E,
            "structure": struct,
            "bond_angles_deg": angles.tolist(),
        })
    return results


def run_experiment_3_symmetry_breaking(
    N=4, n_lone=2, alpha_range=None, output_dir="output"
):
    """
    Vary lone-pair weight alpha and record bond angle between bonding pairs.

    Returns
    -------
    dict
        alpha_list, angle_list, best_alpha_for_104_5.
    """
    _ensure_output_dir(output_dir)
    if alpha_range is None:
        alpha_range = np.arange(1.00, 1.31, 0.01)
    alpha_values = []
    bond_angles = []

    print("\n=== Experiment 3: Symmetry Breaking ===")
    for alpha in alpha_range:
        charges = np.array([1.0, 1.0, float(alpha), float(alpha)])
        opt = ThomsonOptimizer(N=N, charges=charges)
        best, _ = opt.multi_start(n_runs=50)
        # Bond angle between the two unit charges (indices 0 and 1)
        theta = best["final_theta"]
        phi = best["final_phi"]
        cos_gamma = np.sin(theta[0]) * np.sin(theta[1]) * np.cos(phi[0] - phi[1]) + np.cos(theta[0]) * np.cos(theta[1])
        np.clip(cos_gamma, -1.0, 1.0)
        angle = float(np.arccos(cos_gamma) * 180.0 / np.pi)
        alpha_values.append(float(alpha))
        bond_angles.append(angle)
        if len(alpha_values) % 10 == 0:
            print(f"  alpha={alpha:.2f} -> bond angle={angle:.3f}°")

    fig = vis.plot_symmetry_breaking(alpha_values, bond_angles)
    fig.savefig(os.path.join(output_dir, "exp3_symmetry_breaking.png"), dpi=300)
    plt.close(fig)

    # Find alpha closest to 104.5°
    arr_alpha = np.array(alpha_values)
    arr_angles = np.array(bond_angles)
    idx = int(np.argmin(np.abs(arr_angles - 104.5)))
    best_alpha = float(arr_alpha[idx])
    print(f"\n  Alpha closest to 104.5°: alpha={best_alpha:.2f} -> angle={arr_angles[idx]:.3f}°")

    return {
        "alpha_list": alpha_values,
        "angle_list": bond_angles,
        "best_alpha_for_104_5": best_alpha,
    }


def run_experiment_4_local_minima(N=8, n_runs=200, output_dir="output"):
    """
    Run multi_start for N=8, collect energies, plot histogram, count distinct levels.

    Returns
    -------
    dict
        energies, distinct_count (based on a simple clustering heuristic).
    """
    _ensure_output_dir(output_dir)
    print(f"\n=== Experiment 4: Local Minima (N={N}, {n_runs} runs) ===")
    opt = ThomsonOptimizer(N=N)
    best, all_energies = opt.multi_start(n_runs=n_runs)

    print(f"  Global minimum energy: {best['final_energy']:.6f}")

    # Count distinct energy levels using a tolerance cluster
    tol = 1e-3
    clustered = []
    for e in sorted(all_energies):
        if not clustered or abs(e - clustered[-1]) > tol:
            clustered.append(e)
    distinct_count = len(clustered)
    print(f"  Distinct energy levels (tol={tol}): {distinct_count}")

    fig = vis.plot_local_minima_histogram(all_energies, N, title=f"Local Minima Histogram (N={N})")
    fig.savefig(os.path.join(output_dir, f"exp4_local_minima_N{N}.png"), dpi=300)
    plt.close(fig)

    return {
        "N": N,
        "n_runs": n_runs,
        "energies": all_energies,
        "distinct_count": distinct_count,
        "global_minimum": best["final_energy"],
    }
