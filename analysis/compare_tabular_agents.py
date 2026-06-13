"""
Compare tabular agents on Mini Research World from their CSV result logs.

Reads the per-episode CSV logs produced by the agent scripts, builds rolling
learning curves and final-performance bar charts, and prints (and saves) a
compact comparison table.

Run from the project root:

    python analysis/compare_tabular_agents.py
    python analysis/compare_tabular_agents.py --window 100
    python analysis/compare_tabular_agents.py --results-dir results --plots-dir results/plots

Expected input files (missing ones are skipped gracefully):
    results/random_baseline_1k.csv
    results/sarsa_10k.csv
    results/q_learning_10k.csv
    results/expected_sarsa_10k.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

# Use a non-interactive backend so the script runs headless (CI, servers).
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------
# Order here controls plot ordering, legend order, and bar order.
# Each entry: (display label, csv filename, plot color)

AGENTS = [
    ("Random Agent", "random_baseline_1k.csv", "#888888"),
    ("SARSA", "sarsa_10k.csv", "#1f77b4"),
    ("Expected SARSA", "expected_sarsa_10k.csv", "#2ca02c"),
    ("Q-learning", "q_learning_10k.csv", "#d62728"),
]


# Progress milestones tracked by every agent, in pipeline order.
# Each entry: (column name, short label for the x-axis)
MILESTONES = [
    ("has_key", "Key"),
    ("door_open", "Door"),
    ("terminal_activated", "Terminal"),
    ("has_portal_core", "Core"),
    ("portal_open", "Portal"),
    ("has_lab_key", "Lab key"),
    ("machine_used", "Machine"),
    ("has_result", "Result"),
]


# ---------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------

def to_float_series(series):
    """
    Safely convert a column to float.

    Handles real booleans, "True"/"False" strings, and already-numeric data.
    """

    if series.dtype == bool:
        return series.astype(float)

    if series.dtype == object:
        mapping = {"True": 1.0, "False": 0.0, True: 1.0, False: 0.0}
        mapped = series.map(mapping)
        # Fall back to numeric parsing for anything not in the map.
        return mapped.fillna(pd.to_numeric(series, errors="coerce")).astype(float)

    return series.astype(float)


def load_results(results_dir, window):
    """
    Load every available agent CSV.

    Returns an ordered list of (label, color, dataframe). Missing files are
    reported and skipped. Each dataframe is sorted by episode and given:
        - an "algorithm" column
        - float versions of the success/truncated flags
        - rolling metric columns based on the given window
    """

    results_dir = Path(results_dir)
    loaded = []

    for label, filename, color in AGENTS:
        csv_path = results_dir / filename

        if not csv_path.exists():
            print(f"[skip] Missing file: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df = df.sort_values("episode").reset_index(drop=True)
        df["algorithm"] = label

        # Normalize the flags we rely on into clean float columns.
        df["success_f"] = to_float_series(df["success"])
        df["truncated_f"] = to_float_series(df["truncated"])

        # Rolling metrics (min_periods=1 so the curves start at episode 1).
        roll = lambda s: s.rolling(window=window, min_periods=1).mean()
        df["roll_reward"] = roll(df["total_reward"])
        df["roll_success"] = roll(df["success_f"]) * 100.0
        df["roll_steps"] = roll(df["steps"])
        df["roll_timeout"] = roll(df["truncated_f"]) * 100.0

        print(f"[load] {label:>16}: {len(df):>6} episodes  ({csv_path})")
        loaded.append((label, color, df))

    return loaded


# ---------------------------------------------------------------------
# Rolling curve plots
# ---------------------------------------------------------------------

def plot_rolling_curve(loaded, metric_col, title, ylabel, filename, plots_dir, window):
    """
    Plot one rolling metric over episodes, one line per algorithm.
    """

    fig, ax = plt.subplots(figsize=(11, 6))

    for label, color, df in loaded:
        ax.plot(
            df["episode"],
            df[metric_col],
            label=label,
            color=color,
            linewidth=1.8,
        )

    ax.set_title(f"{title}  (rolling window = {window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)

    return _save(fig, filename, plots_dir)


# ---------------------------------------------------------------------
# Final-performance bar plots
# ---------------------------------------------------------------------

def _final_window(df, window):
    """Return the last `window` episodes of a dataframe."""
    return df.tail(min(window, len(df)))


def plot_final_bar(loaded, value_fn, title, ylabel, filename, plots_dir, fmt="{:.1f}"):
    """
    Generic single-value-per-algorithm bar chart with value labels on top.
    """

    labels = [label for label, _, _ in loaded]
    colors = [color for _, color, _ in loaded]
    values = [value_fn(df) for _, _, df in loaded]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)

    # Annotate each bar with its value.
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.margins(y=0.12)

    return _save(fig, filename, plots_dir)


def plot_milestone_bar(loaded, plots_dir):
    """
    Grouped bar chart: milestone completion rate (%) per algorithm.
    """

    milestone_cols = [col for col, _ in MILESTONES]
    milestone_names = [name for _, name in MILESTONES]

    x = np.arange(len(MILESTONES))
    num_agents = len(loaded)
    width = 0.8 / max(num_agents, 1)

    fig, ax = plt.subplots(figsize=(12, 6.5))

    for i, (label, color, df) in enumerate(loaded):
        rates = [to_float_series(df[col]).mean() * 100.0 for col in milestone_cols]
        offset = (i - (num_agents - 1) / 2) * width
        ax.bar(x + offset, rates, width, label=label, color=color,
               edgecolor="black", linewidth=0.4)

    ax.set_title("Milestone Completion Rate by Algorithm (full run)")
    ax.set_xlabel("Milestone (in task order)")
    ax.set_ylabel("Completion rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(milestone_names, rotation=20, ha="right")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)

    return _save(fig, "milestone_completion_bar.png", plots_dir)


def _save(fig, filename, plots_dir):
    """Save a figure with consistent settings and report the path."""
    output_path = Path(plots_dir) / filename
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------

def build_comparison_table(loaded, window):
    """
    Build a one-row-per-algorithm summary dataframe.
    """

    rows = []

    for label, _, df in loaded:
        final = _final_window(df, window)

        rows.append({
            "algorithm": label,
            "episodes": len(df),
            "avg_reward": df["total_reward"].mean(),
            "avg_steps": df["steps"].mean(),
            "success_rate": df["success_f"].mean() * 100.0,
            "timeout_rate": df["truncated_f"].mean() * 100.0,
            f"final_{window}_avg_reward": final["total_reward"].mean(),
            f"final_{window}_success_rate": final["success_f"].mean() * 100.0,
            f"final_{window}_avg_steps": final["steps"].mean(),
            "best_reward": df["total_reward"].max(),
            "has_result_rate": to_float_series(df["has_result"]).mean() * 100.0,
        })

    return pd.DataFrame(rows)


def print_comparison_table(table):
    """Pretty-print the comparison table with rounded numbers."""

    rounded = table.copy()
    for col in rounded.columns:
        if col not in ("algorithm", "episodes"):
            rounded[col] = rounded[col].round(2)

    print("\n========== TABULAR AGENT COMPARISON ==========")
    print(rounded.to_string(index=False))
    print("==============================================\n")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare tabular RL agents on Mini Research World from CSV logs."
    )

    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Rolling window size for curves and final-N metrics. Default: 100.",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing the agent CSV logs. Default: results.",
    )

    parser.add_argument(
        "--plots-dir",
        type=str,
        default="results/plots",
        help="Directory to write the comparison plots. Default: results/plots.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    if args.window <= 0:
        raise SystemExit("--window must be greater than 0")

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nComparing tabular agents")
    print(f"results_dir = {args.results_dir}")
    print(f"plots_dir   = {plots_dir}")
    print(f"window      = {args.window}\n")

    loaded = load_results(args.results_dir, args.window)

    if not loaded:
        raise SystemExit(
            f"No CSV result files found in '{args.results_dir}'. Nothing to compare."
        )

    print()

    # --- Rolling learning curves ---
    plot_rolling_curve(
        loaded, "roll_reward",
        "Rolling Average Reward", "Average total reward",
        "rolling_reward_curve.png", plots_dir, args.window,
    )
    plot_rolling_curve(
        loaded, "roll_success",
        "Rolling Success Rate", "Success rate (%)",
        "rolling_success_rate_curve.png", plots_dir, args.window,
    )
    plot_rolling_curve(
        loaded, "roll_steps",
        "Rolling Average Steps", "Average steps per episode",
        "rolling_steps_curve.png", plots_dir, args.window,
    )
    plot_rolling_curve(
        loaded, "roll_timeout",
        "Rolling Timeout Rate", "Timeout / truncation rate (%)",
        "rolling_timeout_rate_curve.png", plots_dir, args.window,
    )

    # --- Final-performance bars (last `window` episodes) ---
    plot_final_bar(
        loaded,
        lambda df: _final_window(df, args.window)["success_f"].mean() * 100.0,
        f"Final {args.window}-Episode Success Rate", "Success rate (%)",
        "final_success_rate_bar.png", plots_dir, fmt="{:.1f}%",
    )
    plot_final_bar(
        loaded,
        lambda df: _final_window(df, args.window)["total_reward"].mean(),
        f"Final {args.window}-Episode Average Reward", "Average total reward",
        "final_avg_reward_bar.png", plots_dir, fmt="{:.2f}",
    )
    plot_final_bar(
        loaded,
        lambda df: _final_window(df, args.window)["steps"].mean(),
        f"Final {args.window}-Episode Average Steps", "Average steps",
        "final_avg_steps_bar.png", plots_dir, fmt="{:.1f}",
    )

    # --- Milestone completion ---
    plot_milestone_bar(loaded, plots_dir)

    # --- Comparison table ---
    table = build_comparison_table(loaded, args.window)
    print_comparison_table(table)

    table_path = plots_dir / "comparison_summary.csv"
    table.to_csv(table_path, index=False)
    print(f"[table] Saved: {table_path}\n")

    return table


if __name__ == "__main__":
    main()
