"""
visualize.py — 3D Visualization and Animation Utilities.

Functions to generate publication-quality static and interactive plots
using Plotly (3D) and Matplotlib (2D).
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from .energy import compute_cos_gamma


def _spherical_to_cartesian(theta, phi, R=1.0):
    """Convert spherical coordinates to Cartesian."""
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)
    return x, y, z


def _create_sphere_wireframe(R=1.0, n_theta=40, n_phi=40):
    """Generate x, y, z wireframe arrays for a sphere."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta, phi = np.meshgrid(theta, phi)
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)
    return x, y, z


def plot_sphere_with_charges(theta, phi, R=1.0, title=""):
    """
    Create an interactive Plotly 3D figure showing charges on a sphere.

    Parameters
    ----------
    theta : np.ndarray, shape (N,)
    phi : np.ndarray, shape (N,)
    R : float, default 1.0
    title : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    x, y, z = _spherical_to_cartesian(theta, phi, R)
    N = len(theta)

    fig = go.Figure()

    # Semi-transparent wireframe sphere surface (light)
    xs, ys, zs = _create_sphere_wireframe(R)
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        opacity=0.15,
        colorscale=[[0, 'lightblue'], [1, 'lightblue']],
        showscale=False,
        hoverinfo='skip',
    ))

    # Charges as colored spheres
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=8, color=z, colorscale='Viridis', showscale=True, colorbar=dict(title='z')),
        name='Charges',
    ))

    # Lines between pairs, color-coded by distance
    if N >= 2:
        cos_gamma = compute_cos_gamma(theta, phi)
        distances = R * np.sqrt(2.0 * (1.0 - cos_gamma))
        np.fill_diagonal(distances, np.nan)
        # We use upper triangle to avoid duplicate lines
        for i in range(N):
            for j in range(i + 1, N):
                d = distances[i, j]
                # Continuous color mapping based on distance
                # Normalize distance for color: 0 to 2R
                norm = d / (2.0 * R)
                fig.add_trace(go.Scatter3d(
                    x=[x[i], x[j]],
                    y=[y[i], y[j]],
                    z=[z[i], z[j]],
                    mode='lines',
                    line=dict(
                        color=plt.cm.plasma(norm),
                        width=3,
                    ),
                    hoverinfo='skip',
                    showlegend=False,
                ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis=dict(title='x', range=[-1.2 * R, 1.2 * R]),
            yaxis=dict(title='y', range=[-1.2 * R, 1.2 * R]),
            zaxis=dict(title='z', range=[-1.2 * R, 1.2 * R]),
            aspectmode='cube',
        ),
        width=700,
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


def create_optimization_animation(trajectory, energy_history, N, R=1.0, filename="animation.html"):
    """
    Create a Plotly HTML animation of the optimization trajectory.

    Parameters
    ----------
    trajectory : list of (theta, phi) tuples
    energy_history : list of float
    N : int
    R : float, default 1.0
    filename : str
        Path to save the HTML animation.
    """
    n_frames = len(trajectory)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}]],
        subplot_titles=('Charge Positions', 'Energy Convergence'),
    )

    # Static wireframe sphere (trace 0)
    xs, ys, zs = _create_sphere_wireframe(R, n_theta=30, n_phi=30)
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        opacity=0.10,
        colorscale=[[0, 'lightblue'], [1, 'lightblue']],
        showscale=False,
        hoverinfo='skip',
    ), row=1, col=1)

    # Initial charge positions (trace 1)
    theta0, phi0 = trajectory[0]
    x0, y0, z0 = _spherical_to_cartesian(theta0, phi0, R)
    fig.add_trace(go.Scatter3d(
        x=x0, y=y0, z=z0,
        mode='markers',
        marker=dict(size=7, color='red'),
        name='Charges',
    ), row=1, col=1)

    # Full energy curve (trace 2) — stays static
    iters = np.arange(len(energy_history))
    fig.add_trace(go.Scatter(
        x=iters, y=energy_history,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Energy',
    ), row=1, col=2)

    # Moving energy dot (trace 3)
    fig.add_trace(go.Scatter(
        x=[0], y=[energy_history[0]],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Current Iter',
    ), row=1, col=2)

    frames = []
    for idx, (theta_i, phi_i) in enumerate(trajectory):
        xi, yi, zi = _spherical_to_cartesian(theta_i, phi_i, R)
        iter_idx = idx * 100
        if iter_idx >= len(energy_history):
            iter_idx = len(energy_history) - 1

        frame = go.Frame(
            data=[
                go.Scatter3d(x=xi, y=yi, z=zi, mode='markers', marker=dict(size=7, color='red')),
                go.Scatter(x=iters, y=energy_history, mode='lines', line=dict(color='blue', width=2)),
                go.Scatter(x=[iter_idx], y=[energy_history[iter_idx]], mode='markers', marker=dict(color='red', size=10)),
            ],
            traces=[1, 2, 3],
            name=str(idx),
        )
        frames.append(frame)

    fig.frames = frames

    # Slider
    steps = []
    for k in range(n_frames):
        step = dict(
            method='animate',
            args=[[str(k)], dict(mode='immediate', frame=dict(duration=100, redraw=True), transition=dict(duration=0))],
            label=str(k * 100),
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        steps=steps,
        x=0.1, y=-0.05,
        len=0.8,
        currentvalue=dict(visible=True, prefix='Iteration: ', font=dict(size=14)),
    )]

    fig.update_layout(
        sliders=sliders,
        width=1000,
        height=500,
        scene=dict(
            aspectmode='cube',
            xaxis=dict(range=[-1.2 * R, 1.2 * R]),
            yaxis=dict(range=[-1.2 * R, 1.2 * R]),
            zaxis=dict(range=[-1.2 * R, 1.2 * R]),
        ),
        margin=dict(l=0, r=0, b=50, t=30),
    )

    fig.write_html(filename)


def plot_energy_convergence(energy_history, theoretical_E=None, title="Energy Convergence"):
    """
    Create a Matplotlib figure of energy vs iteration.

    Parameters
    ----------
    energy_history : list or np.ndarray
    theoretical_E : float, optional
        If provided, a horizontal dashed line is drawn and a log-scale
        subplot shows |E - E_theory|.
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2 if theoretical_E is not None else 1, figsize=(12, 4))
    if theoretical_E is not None:
        ax_main, ax_log = axes
    else:
        ax_main = axes

    iters = np.arange(len(energy_history))
    ax_main.plot(iters, energy_history, lw=1.5)
    if theoretical_E is not None:
        ax_main.axhline(theoretical_E, color='red', linestyle='--', label=f'Theoretical E={theoretical_E:.3f}')
        ax_main.legend()
    ax_main.set_xlabel('Iteration')
    ax_main.set_ylabel('Energy')
    ax_main.set_title(title)
    ax_main.grid(True, alpha=0.3)

    if theoretical_E is not None:
        diff = np.abs(np.array(energy_history) - theoretical_E)
        diff = np.clip(diff, 1e-16, None)  # avoid log(0)
        ax_log.semilogy(iters, diff, lw=1.5, color='green')
        ax_log.set_xlabel('Iteration')
        ax_log.set_ylabel('|E - E_theoretical|')
        ax_log.set_title('Energy Difference (log scale)')
        ax_log.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_local_minima_histogram(energies, N, title="Local Minima Histogram"):
    """
    Create a Matplotlib histogram of final energies from multi-start runs.

    Parameters
    ----------
    energies : list or np.ndarray
    N : int
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(energies, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(min(energies), color='red', linestyle='--', linewidth=2, label=f'Global minimum: {min(energies):.4f}')
    ax.set_xlabel('Final Energy')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_symmetry_breaking(alpha_values, bond_angles, title="Symmetry Breaking"):
    """
    Create a Matplotlib plot of bond angle vs lone-pair weight alpha.

    Parameters
    ----------
    alpha_values : list or np.ndarray
    bond_angles : list or np.ndarray
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alpha_values, bond_angles, marker='o', markersize=4, lw=2, color='darkgreen')
    # Reference lines
    ax.axhline(109.5, color='blue', linestyle='--', label='Tetrahedral (109.5°)')
    ax.axhline(104.5, color='red', linestyle='--', label='Water target (104.5°)')
    ax.axvline(1.0, color='gray', linestyle=':', label='α=1.0 (ideal)')
    ax.set_xlabel('Lone-pair weight α')
    ax.set_ylabel('Bond angle (degrees)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
