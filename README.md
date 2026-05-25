# Thomson Problem Explorer

An interactive simulation and visualization dashboard for the **Thomson Problem** (a.k.a. Coulomb repulsion on a sphere) using gradient-descent optimization, symmetry analysis, and a Streamlit web interface.

**Live App:** [https://mathproject1.streamlit.app/](https://mathproject1.streamlit.app/)

---

## What is the Thomson Problem?

The Thomson problem asks how to arrange N identical point charges on the surface of a sphere so that the total electrostatic potential energy is minimized. It is a classic problem in physics and computational geometry with applications in molecular modeling, crystallography, and geometry.

This project implements:
- A **gradient-descent optimizer** with momentum and backtracking line search
- **Symmetry analysis** (bond angles, structure identification)
- **4 reproducible experiments** covering validation, exploration, symmetry breaking, and local minima analysis
- An **interactive web dashboard** via Streamlit

---

## Experiments Summary

| Experiment | Description | Key Results |
|------------|-------------|-------------|
| **Exp 1: Validation (N=2..6)** | Compare optimized energies against theoretical minima | Matched within < 0.01%. Structures: linear, triangle, tetrahedron, bipyramid, octahedron. |
| **Exp 2: Exploration (N=7..12)** | Multi-start optimization for higher N geometries | Best energies captured for N=7..12; 3D visualizations generated. |
| **Exp 3: Symmetry Breaking** | Vary lone-pair weight α and observe bond-angle shift | At α=1.24, bond angle ≈ 104.576° — closely matches water molecule (104.5°). |
| **Exp 4: Local Minima** | Histogram of energies from 200 random starts (N=8) | Only 1 distinct energy level found (global minimum = 19.675288). |

---

## Quick Start

### Local Setup
```bash
# Clone the repository
git clone https://github.com/yuriluv/mathproject1.git
cd mathproject1

# Install dependencies
pip install -r requirements.txt

# Run experiments (optional)
python -m thomson_problem.main

# Launch the interactive dashboard
streamlit run streamlit_app.py
```

### Dependencies
- Python ≥ 3.9
- numpy, matplotlib, plotly
- streamlit (for the web app)

---

## Folder Structure

```
mathproject1/
├── thomson_problem/
│   ├── __init__.py
│   ├── optimizer.py       # Gradient-descent engine
│   ├── energy.py          # Coulomb potential computation
│   ├── analysis.py        # Bond angles, structure ID, validation
│   ├── visualize.py       # Plotly 3D & Matplotlib 2D plots
│   ├── experiments.py     # 4 experiment runners
│   ├── main.py            # Entry point (batch mode)
│   └── output/            # Experiment results (PNGs + JSON)
├── streamlit_app.py       # Interactive web dashboard
├── requirements.txt
└── README.md
```

---

## Streamlit Dashboard Features

The web app includes three modes:

1. **Interactive Optimizer**
   - Adjust N, radius, learning rate, momentum, and max iterations
   - See real-time 3D structure and energy convergence plots

2. **Results Gallery**
   - Displays pre-computed experiment results (summary tables + output images)

3. **Run Experiments**
   - Re-run any experiment directly from the browser with custom parameters
   - Cached so repeated runs are fast

---

## Experiment Outputs

All generated plots and data are saved under `thomson_problem/output/`:

- `exp1_N{2..6}_3d.png` / `exp1_N{2..6}_convergence.png`
- `exp2_N{7..12}_3d.png`
- `exp3_symmetry_breaking.png`
- `exp4_local_minima_N8.png`
- `results_summary.json`

---

## Author

Built with Sisyphus 🪨
