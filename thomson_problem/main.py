"""
main.py — Entry Point.

Runs all four experiments sequentially, saves figures and data, and prints
a summary table.
"""

import os
import json

from .experiments import (
    run_experiment_1_validation,
    run_experiment_2_exploration,
    run_experiment_3_symmetry_breaking,
    run_experiment_4_local_minima,
)
from .optimizer import ThomsonOptimizer
from . import visualize as vis


def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Experiment 1
    exp1_results = run_experiment_1_validation(N_range=(2, 3, 4, 5, 6), output_dir=output_dir)

    # Experiment 2
    exp2_results = run_experiment_2_exploration(N_range=(7, 8, 9, 10, 11, 12), output_dir=output_dir)

    # Experiment 3
    exp3_results = run_experiment_3_symmetry_breaking(output_dir=output_dir)

    # Experiment 4
    exp4_results = run_experiment_4_local_minima(N=8, n_runs=200, output_dir=output_dir)

    # Summary table (Experiment 1)
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'N':>3} {'Theoretical E':>15} {'Computed E':>15} {'Rel. Error (%)':>16} {'Structure':>25}")
    print("-" * 80)
    for r in exp1_results:
        print(
            f"{r['N']:>3} {r['theoretical_E']:>15.6f} {r['computed_E']:>15.6f} "
            f"{r['rel_error_percent']:>16.4f} {r['structure']:>25}"
        )
    print("=" * 80)

    # Save JSON summary
    summary = {
        "experiment_1_validation": exp1_results,
        "experiment_2_exploration": exp2_results,
        "experiment_3_symmetry_breaking": exp3_results,
        "experiment_4_local_minima": exp4_results,
    }
    json_path = os.path.join(output_dir, "results_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results summary to {json_path}")

    # Optional animations for Exp1 N=4 and N=5 (use a single run to build trajectory)
    for N_anim in (4, 5):
        print(f"\nGenerating animation for N={N_anim} ...")
        opt = ThomsonOptimizer(N=N_anim)
        result = opt.optimize(seed=42 + N_anim)
        vis.create_optimization_animation(
            result["trajectory"],
            result["energy_history"],
            N=N_anim,
            filename=os.path.join(output_dir, f"exp1_N{N_anim}_animation.html"),
        )


if __name__ == "__main__":
    main()
