"""
V1.4 E2 misalignment comparison plots across algorithms.

Reads results/full_benchmark/V1.4_e2/<algo>/seed_*.csv and produces three
cross-algorithm figures under results/full_benchmark/plots/:
  - e2_alignment_gap.png   rolling alignment_gap (visible return - true_score)
  - e2_hacking_rate.png    rolling reward_hack_flag rate
  - e2_return_vs_truescore.png  return and true_score for a chosen algo

Read-only; consumes the per-episode CSVs the driver wrote.

Run from the repo root:
    python analysis/plot_e2_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E2_ROOT = PROJECT_ROOT / "results" / "full_benchmark" / "V1.4_e2"
PLOTS = PROJECT_ROOT / "results" / "full_benchmark" / "plots"
WINDOW = 100
# representative subset for legibility (highest-signal algos)
HIGHLIGHT = ["q_learning", "dqn", "dueling_ddqn_per",
             "spie_dueling_ddqn_per", "spie_noisy_dueling_ddqn_per"]


def _mean_curve(folder: Path, column: str):
    frames = []
    for p in sorted(folder.glob("seed_*.csv")):
        df = pd.read_csv(p)
        frames.append(df[column].rolling(WINDOW).mean().to_numpy())
    if not frames:
        return None, None
    n = min(len(f) for f in frames)
    stack = np.vstack([f[:n] for f in frames])
    return np.arange(1, n + 1), np.nanmean(stack, axis=0)


def plot_metric(column: str, ylabel: str, title: str, out: Path,
                hline: float | None = None):
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    for algo in HIGHLIGHT:
        folder = E2_ROOT / algo
        if not folder.exists():
            continue
        x, y = _mean_curve(folder, column)
        if x is None:
            continue
        ax.plot(x, y, linewidth=2, label=algo)
        plotted += 1
    if hline is not None:
        ax.axhline(hline, color="black", linewidth=0.6)
    ax.set_xlabel("episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if plotted:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return plotted


def plot_return_vs_truescore(algo: str, out: Path):
    folder = E2_ROOT / algo
    if not folder.exists():
        return False
    fig, ax = plt.subplots(figsize=(9, 5))
    xr, yr = _mean_curve(folder, "episode_return")
    xt, yt = _mean_curve(folder, "true_score")
    if xr is None:
        plt.close(fig)
        return False
    ax.plot(xr, yr, color="tab:green", linewidth=2,
            label="visible return (maximized)")
    ax.plot(xt, yt, color="tab:red", linewidth=2,
            label="hidden true_score (wanted)")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel("episode")
    ax.set_ylabel("rolling mean")
    ax.set_title(f"V1.4 E2 — visible return vs hidden true_score ({algo})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    n1 = plot_metric("alignment_gap", "alignment_gap (return - true_score)",
                     "V1.4 E2 — alignment gap over training",
                     PLOTS / "e2_alignment_gap.png", hline=0.0)
    n2 = plot_metric("reward_hack_flag", "reward-hack rate (rolling)",
                     "V1.4 E2 — proxy reward-hacking rate over training",
                     PLOTS / "e2_hacking_rate.png")
    ok = plot_return_vs_truescore("dueling_ddqn_per",
                                  PLOTS / "e2_return_vs_truescore.png")
    print(f"alignment_gap plot: {n1} algos | hacking_rate plot: {n2} algos | "
          f"return_vs_truescore: {ok}")
    print(f"plots -> {PLOTS}")


if __name__ == "__main__":
    main()
