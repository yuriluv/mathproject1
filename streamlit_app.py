"""Streamlit app for Thomson Problem interactive exploration."""

import os
import sys
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from thomson_problem.optimizer import ThomsonOptimizer
from thomson_problem.analysis import compute_bond_angles, validate_against_theory
from thomson_problem import visualize as vis

# ── Page config ──
st.set_page_config(
    page_title="Thomson Problem Explorer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar controls ──
st.sidebar.title("⚙️ Controls")
mode = st.sidebar.radio(
    "Choose mode:",
    ["Interactive optimizer", "Experiment 1: Validation", "Experiment 2: Exploration",
     "Experiment 3: Symmetry breaking", "Experiment 4: Local minima"],
)

# ── Helper: cached optimizer wrapper ──
@st.cache_data(show_spinner=False)
def run_optimization(N, seed, charges_tuple, R, lr, momentum, max_iter):
    """Run a single optimization and return serializable dict."""
    charges = np.array(charges_tuple, dtype=float) if charges_tuple else np.ones(N, dtype=float)
    opt = ThomsonOptimizer(N=N, charges=charges, R=R, lr=lr, momentum=momentum, max_iter=max_iter)
    result = opt.optimize(seed=seed)
    return result


@st.cache_data(show_spinner=False)
def run_multi(N, n_runs, charges_tuple, R, lr, momentum, max_iter):
    """Run multi-start and return best + all energies."""
    charges = np.array(charges_tuple, dtype=float) if charges_tuple else np.ones(N, dtype=float)
    opt = ThomsonOptimizer(N=N, charges=charges, R=R, lr=lr, momentum=momentum, max_iter=max_iter)
    best, all_energies = opt.multi_start(n_runs=n_runs, seeds=None)
    return best, all_energies


# ── Mode 1: Interactive optimizer ──
if mode == "Interactive optimizer":
    st.header("🔬 Interactive Thomson Optimizer")
    col1, col2 = st.columns([1, 2])

    with col1:
        N = st.slider("Number of charges (N)", min_value=2, max_value=20, value=4, step=1)
        seed = st.number_input("Random seed", value=42, step=1)
        R = st.slider("Sphere radius", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        lr = st.slider("Learning rate", min_value=0.001, max_value=0.5, value=0.05, step=0.001, format="%.3f")
        momentum = st.slider("Momentum", min_value=0.0, max_value=0.99, value=0.9, step=0.01)
        max_iter = st.slider("Max iterations", min_value=1000, max_value=100000, value=50000, step=1000)
        run_btn = st.button("▶️ Run optimization")

    if run_btn:
        with st.spinner("Optimizing …"):
            result = run_optimization(N, int(seed), tuple(np.ones(N)), R, lr, momentum, max_iter)
        theta, phi = result["final_theta"], result["final_phi"]
        angles = compute_bond_angles(theta, phi, R)

        with col2:
            st.subheader("3D Structure")
            fig3d = vis.plot_sphere_with_charges(theta, phi, R=R, title=f"N={N}  E={result['final_energy']:.6f}")
            st.plotly_chart(fig3d, use_container_width=True)

            st.subheader("Energy Convergence")
            fig_conv = vis.plot_energy_convergence(
                result["energy_history"],
                theoretical_E=None,
                title=f"Convergence (N={N})",
            )
            st.pyplot(fig_conv)
            plt.close(fig_conv)

            st.subheader("Bond angles (deg)")
            st.write(np.round(angles, 2).tolist())
            st.write(f"**Iterations:** {result['iterations']}  |  **Converged:** {result['converged']}  |  **Final energy:** {result['final_energy']:.6f}")


# ── Mode 2: Experiment 1 ──
elif mode == "Experiment 1: Validation":
    st.header("✅ Experiment 1 – Validation against Theory")
    st.markdown("Compare optimized energies for N=2..6 with known theoretical minima.")

    THEORETICAL = {2: 0.5, 3: 1.732, 4: 3.674, 5: 6.475, 6: 9.985}

    if st.button("▶️ Run validation"):
        results = []
        progress = st.progress(0)
        for idx, N in enumerate([2, 3, 4, 5, 6]):
            with st.spinner(f"Running N={N} …"):
                best, _ = run_multi(N, 100, tuple(np.ones(N)), 1.0, 0.05, 0.9, 50000)
                E = best["final_energy"]
                theory = THEORETICAL[N]
                rel = validate_against_theory(N, E, theory)
                angles = compute_bond_angles(best["final_theta"], best["final_phi"])
                results.append({
                    "N": N, "theory": theory, "computed": E,
                    "rel_error": rel, "angles": np.round(angles, 2).tolist(),
                })
                progress.progress((idx + 1) / 5)
        progress.empty()

        st.subheader("Summary table")
        st.table([
            {
                "N": r["N"],
                "Theoretical E": f"{r['theory']:.6f}",
                "Computed E": f"{r['computed']:.6f}",
                "Rel. Error (%)": f"{r['rel_error']:.4f}",
                "Bond angles (deg)": r["angles"],
            }
            for r in results
        ])


# ── Mode 3: Experiment 2 ──
elif mode == "Experiment 2: Exploration":
    st.header("🔭 Experiment 2 – Exploration (N=7..12)")
    N = st.selectbox("Select N", [7, 8, 9, 10, 11, 12])
    n_runs = st.slider("Multi-start runs", min_value=10, max_value=500, value=200, step=10)

    if st.button("▶️ Run exploration"):
        with st.spinner(f"Running N={N} with {n_runs} starts …"):
            best, all_E = run_multi(N, n_runs, tuple(np.ones(N)), 1.0, 0.05, 0.9, 50000)
        theta, phi = best["final_theta"], best["final_phi"]
        angles = compute_bond_angles(theta, phi)

        st.subheader("3D Structure")
        fig3d = vis.plot_sphere_with_charges(theta, phi, title=f"N={N}  E={best['final_energy']:.6f}")
        st.plotly_chart(fig3d, use_container_width=True)

        st.subheader("Bond angles (deg)")
        st.write(np.round(angles, 2).tolist())
        st.write(f"**Best energy:** {best['final_energy']:.6f}  |  **All energies range:** {min(all_E):.6f} – {max(all_E):.6f}")


# ── Mode 4: Experiment 3 ──
elif mode == "Experiment 3: Symmetry breaking":
    st.header("💧 Experiment 3 – Symmetry Breaking (lone-pair effect)")
    st.markdown("Vary lone-pair weight α and observe the bond angle between two unit charges.")

    N = 4
    n_lone = 2
    alpha_min = st.slider("Min α", min_value=1.0, max_value=2.0, value=1.0, step=0.01)
    alpha_max = st.slider("Max α", min_value=1.0, max_value=2.0, value=1.31, step=0.01)
    n_steps = st.slider("Steps", min_value=5, max_value=100, value=31, step=1)
    n_runs = st.slider("Multi-start per α", min_value=10, max_value=100, value=50, step=5)

    if st.button("▶️ Run symmetry breaking"):
        alphas = np.linspace(alpha_min, alpha_max, n_steps)
        angles = []
        progress = st.progress(0)
        for idx, alpha in enumerate(alphas):
            charges = tuple([1.0, 1.0, float(alpha), float(alpha)])
            best, _ = run_multi(N, n_runs, charges, 1.0, 0.05, 0.9, 50000)
            theta, phi = best["final_theta"], best["final_phi"]
            cos_g = (
                np.sin(theta[0]) * np.sin(theta[1]) * np.cos(phi[0] - phi[1])
                + np.cos(theta[0]) * np.cos(theta[1])
            )
            cos_g = np.clip(cos_g, -1.0, 1.0)
            angle = float(np.arccos(cos_g) * 180.0 / np.pi)
            angles.append(angle)
            progress.progress((idx + 1) / len(alphas))
        progress.empty()

        fig = vis.plot_symmetry_breaking(alphas.tolist(), angles)
        st.pyplot(fig)
        plt.close(fig)

        # Find closest to 104.5
        arr_alpha = np.array(alphas)
        arr_angles = np.array(angles)
        idx = int(np.argmin(np.abs(arr_angles - 104.5)))
        st.success(f"Closest to 104.5°: α={arr_alpha[idx]:.2f} → angle={arr_angles[idx]:.3f}°")


# ── Mode 5: Experiment 4 ──
elif mode == "Experiment 4: Local minima":
    st.header("📊 Experiment 4 – Local Minima Analysis")
    st.markdown("Run many random starts for a fixed N and inspect energy distribution.")

    N = st.slider("N", min_value=3, max_value=15, value=8, step=1)
    n_runs = st.slider("Number of runs", min_value=50, max_value=1000, value=200, step=50)

    if st.button("▶️ Run local minima analysis"):
        with st.spinner(f"Running {n_runs} optimizations …"):
            best, all_E = run_multi(N, n_runs, tuple(np.ones(N)), 1.0, 0.05, 0.9, 50000)

        # Cluster distinct levels
        tol = 1e-3
        clustered = []
        for e in sorted(all_E):
            if not clustered or abs(e - clustered[-1]) > tol:
                clustered.append(e)
        distinct = len(clustered)

        st.subheader("Histogram")
        fig = vis.plot_local_minima_histogram(all_E, N, title=f"Local Minima Histogram (N={N})")
        st.pyplot(fig)
        plt.close(fig)

        st.write(f"**Global minimum:** {best['final_energy']:.6f}")
        st.write(f"**Distinct energy levels (tol={tol}):** {distinct}")
        st.write(f"**All energies range:** {min(all_E):.6f} – {max(all_E):.6f}")


# ── Footer ──
st.sidebar.markdown("---")
st.sidebar.info(
    "This app uses a gradient-descent optimizer with momentum and backtracking line search.\n\n"
    "Results are cached so you can switch tabs without re-running."
)
