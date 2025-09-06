import pickle
from pathlib import Path
from typing import Dict, Iterable, Tuple
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def plot_deferral_curves(
    curves: Dict[str, Iterable[Tuple[float, float]]],
    save_path: str | None = None,
    *,
    show: bool = True,
) -> None:

    plt.style.use("seaborn-v0_8-whitegrid") 

    fig, ax = plt.subplots(figsize=(5.1, 3.4), dpi=150)

    colors = {
        "CSCR": "#1f77b4",  # blue
        "parametric": "#ff7f0e",  # orange
        "pareto_random": "#2ca02c",  # green
        "random": "#d62728",  # red
        "oracle": "#9467bd",  # purple
        "umr": "#8c564b",  # brown
        "softmoe": "#e377c2",  # pink
        "thompson": "#7f7f7f",  # gray
    }

    if "routerbench" in save_path:
        for router_name, curve in curves.items():
            costs, accs = curve
            costs = np.array(costs)
            curves[router_name] = (costs, accs)
    

    for router_name, curve in curves.items():
        # if router_name == "oracle":
        #     continue
        costs, accs = curve
        ax.plot(
            costs,
            accs,
            marker="o",
            color=colors.get(router_name, "#1f77b4"),
            markersize=4,
            linewidth=2.0 if router_name == "CSCR" else 1.3,
            label=router_name.upper(),
        )
    ax.set_xlabel("# Params (B)" if "routerbench" not in save_path else "Cost (USD) / 1k queries")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False, fontsize=8)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.7)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved ➜ {save_path}")

    if show:
        plt.show()

def build_cost_grid(all_costs, N_grid=50):
    min_c, max_c = all_costs.min(), all_costs.max()
    return np.linspace(min_c, max_c, N_grid)

def interp_to_grid(costs, accs, grid):
    """Piecewise‑linear interpolation; extrapolate w/ edge values."""
    f = interp1d(costs, accs,
                 kind="linear",
                 bounds_error=False,
                 fill_value=(accs[0], accs[-1]))
    return f(grid)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot deferral curves.")
    parser.add_argument("--pickle_file", help="Path to a pickle file with the curves data.")
    parser.add_argument("--save", dest="save", help="File to save the plot to.")
    parser.add_argument("--no-show", action="store_true", help="Suppress interactive window.")
    args = parser.parse_args()

    with open(args.pickle_file, 'rb') as f:
        curves = pickle.load(f)
    
    all_costs_concat = np.concatenate([v[0] for v in curves.values()])
    grid = build_cost_grid(all_costs_concat, N_grid=20)

    audc = {}
    interp_curves = {}
    peak_acc_dict = {}
    qnc_dict = {}
    for r, (c, a) in curves.items():
        a_grid = interp_to_grid(c, a, grid)
        interp_curves[r] = (grid, a_grid)
        audc[r] = np.trapz(a_grid, grid) / (grid[-1] - grid[0])
        peak_idx = np.argmax(a)
        peak_acc_dict[r] = float(a[peak_idx])
        qnc_dict[r] = float(c[peak_idx])  

    plot_deferral_curves(
        curves,
        global_min=0,
        global_max=70,
        save_path=args.save,
        show=not args.no_show,
    )

    print("=== Normalized Deferral Curves ===")
    header = f"{'Router':>10s} | {'AUDC':>7s} | {'QNC':>8s} | {'PeakAcc':>8s}"
    print(header)
    print('-' * len(header))
    for r in sorted(curves.keys()):
        print(f"{r:>10s} | {audc[r]:7.4f} | {qnc_dict[r]:8.3f} | {peak_acc_dict[r]:8.3f}")
        