"""
Multi-seed evaluation for Mini Research World tabular agents.

Running an agent on a single seed tells you almost nothing: RL training is
noisy, and one lucky/unlucky seed can swing the numbers a lot. This script
trains each agent across many seeds and reports the distribution of outcomes
(mean +/- std and the IQR), which is what makes a result credible.

It drives the existing training functions directly, so the learning logic is
identical to running the agent scripts themselves.

Run from the project root:

    python analysis/multi_seed_eval.py
    python analysis/multi_seed_eval.py --num-seeds 17 --episodes 10000
    python analysis/multi_seed_eval.py --num-seeds 17 --episodes 5000 --no-plots
    python analysis/multi_seed_eval.py --agents sarsa q_learning --num-seeds 18

Outputs (written to --out-dir, default results/multi_seed):
    per_seed_results.csv      one row per (agent, seed)
    aggregate_summary.csv     mean/std/median/IQR per (agent, metric)
    band_reward_curve.png     mean rolling reward +/- std across seeds
    band_success_rate_curve.png
"""

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Project import setup
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.sarsa_agent import train_sarsa
from agents.expected_sarsa_agent import train_expected_sarsa
from agents.q_learning_agent import train_q_learning


# ---------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------
# key -> (display label, train function, plot color)

AGENT_REGISTRY = {
    "sarsa": ("SARSA", train_sarsa, "#1f77b4"),
    "expected_sarsa": ("Expected SARSA", train_expected_sarsa, "#2ca02c"),
    "q_learning": ("Q-learning", train_q_learning, "#d62728"),
}

DEFAULT_AGENTS = ["sarsa", "expected_sarsa", "q_learning"]


# Metrics computed per seed, then aggregated across seeds.
# Each: (key, human label, format spec)
METRICS = [
    ("final_avg_reward", "Final-N avg reward", "{:.2f}"),
    ("final_success_rate", "Final-N success rate (%)", "{:.2f}"),
    ("final_avg_steps", "Final-N avg steps", "{:.2f}"),
    ("overall_success_rate", "Overall success rate (%)", "{:.2f}"),
    ("timeout_rate", "Timeout rate (%)", "{:.2f}"),
    ("best_reward", "Best episode reward", "{:.2f}"),
    ("has_result_rate", "Produced-result rate (%)", "{:.2f}"),
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def to_float_series(series):
    """Safely convert a (possibly boolean) column to float."""
    if series.dtype == bool:
        return series.astype(float)
    if series.dtype == object:
        mapping = {"True": 1.0, "False": 0.0, True: 1.0, False: 0.0}
        mapped = series.map(mapping)
        return mapped.fillna(pd.to_numeric(series, errors="coerce")).astype(float)
    return series.astype(float)


def summarize_seed(results_df, window):
    """Compute the per-seed scalar metrics from a training results dataframe."""

    final = results_df.tail(min(window, len(results_df)))

    return {
        "final_avg_reward": final["total_reward"].mean(),
        "final_success_rate": to_float_series(final["success"]).mean() * 100.0,
        "final_avg_steps": final["steps"].mean(),
        "overall_success_rate": to_float_series(results_df["success"]).mean() * 100.0,
        "timeout_rate": to_float_series(results_df["truncated"]).mean() * 100.0,
        "best_reward": results_df["total_reward"].max(),
        "has_result_rate": to_float_series(results_df["has_result"]).mean() * 100.0,
    }


def rolling_curves(results_df, window):
    """Return rolling reward and rolling success-rate(%) arrays for one seed."""
    reward = (
        results_df["total_reward"].rolling(window, min_periods=1).mean().to_numpy()
    )
    success = (
        to_float_series(results_df["success"])
        .rolling(window, min_periods=1)
        .mean()
        .to_numpy()
        * 100.0
    )
    return reward, success


# ---------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------

def run_experiment(agent_keys, seeds, train_kwargs, window, collect_curves):
    """
    Train every selected agent on every seed.

    Returns:
        per_seed_rows: list of dicts (one per agent/seed) for the raw CSV
        curves: dict agent_key -> {"reward": 2D array, "success": 2D array}
                (only populated when collect_curves is True)
    """

    per_seed_rows = []
    curves = {key: {"reward": [], "success": []} for key in agent_keys}

    for key in agent_keys:
        label, train_fn, _ = AGENT_REGISTRY[key]
        print(f"\n=== {label}: {len(seeds)} seeds ===")

        for i, seed in enumerate(seeds, start=1):
            start = time.time()

            # Silence the per-run training progress prints; we print our own.
            with contextlib.redirect_stdout(io.StringIO()):
                _, results_df = train_fn(seed=seed, **train_kwargs)

            metrics = summarize_seed(results_df, window)
            row = {"agent": label, "agent_key": key, "seed": seed}
            row.update(metrics)
            per_seed_rows.append(row)

            if collect_curves:
                reward_curve, success_curve = rolling_curves(results_df, window)
                curves[key]["reward"].append(reward_curve)
                curves[key]["success"].append(success_curve)

            elapsed = time.time() - start
            print(
                f"  seed {seed:>4} ({i:>2}/{len(seeds)}) | "
                f"final-{window} success={metrics['final_success_rate']:6.2f}% | "
                f"final-{window} reward={metrics['final_avg_reward']:6.2f} | "
                f"{elapsed:5.1f}s"
            )

    # Stack curve lists into 2D arrays [num_seeds, num_episodes].
    if collect_curves:
        for key in agent_keys:
            curves[key]["reward"] = np.vstack(curves[key]["reward"])
            curves[key]["success"] = np.vstack(curves[key]["success"])

    return per_seed_rows, curves


# ---------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------

def build_aggregate(per_seed_df, agent_keys):
    """
    Aggregate per-seed metrics into mean/std/median/IQR per (agent, metric).
    Returns a tidy long-form dataframe.
    """

    rows = []

    for key in agent_keys:
        label = AGENT_REGISTRY[key][0]
        agent_df = per_seed_df[per_seed_df["agent_key"] == key]

        for metric_key, metric_label, _ in METRICS:
            values = agent_df[metric_key].to_numpy(dtype=float)
            q25, median, q75 = np.percentile(values, [25, 50, 75])

            rows.append({
                "agent": label,
                "metric": metric_label,
                "mean": values.mean(),
                "std": values.std(ddof=1) if len(values) > 1 else 0.0,
                "median": median,
                "iqr_low": q25,
                "iqr_high": q75,
                "min": values.min(),
                "max": values.max(),
                "n_seeds": len(values),
            })

    return pd.DataFrame(rows)


def print_headline(per_seed_df, agent_keys, window):
    """Print the compact mean +/- std headline table for the key metrics."""

    print("\n========== MULTI-SEED HEADLINE (mean +/- std) ==========")
    header = (
        f"{'algorithm':>16} | "
        f"{'final-' + str(window) + ' success':>20} | "
        f"{'final-' + str(window) + ' reward':>20} | "
        f"{'overall success':>18}"
    )
    print(header)
    print("-" * len(header))

    for key in agent_keys:
        label = AGENT_REGISTRY[key][0]
        d = per_seed_df[per_seed_df["agent_key"] == key]

        def ms(col):
            v = d[col].to_numpy(dtype=float)
            std = v.std(ddof=1) if len(v) > 1 else 0.0
            return v.mean(), std

        sr_m, sr_s = ms("final_success_rate")
        rw_m, rw_s = ms("final_avg_reward")
        ov_m, ov_s = ms("overall_success_rate")

        print(
            f"{label:>16} | "
            f"{sr_m:7.2f}% +/- {sr_s:5.2f}  | "
            f"{rw_m:7.2f}  +/- {rw_s:5.2f}  | "
            f"{ov_m:6.2f}% +/- {ov_s:4.2f}"
        )

    print("========================================================\n")


def print_full_aggregate(aggregate_df):
    """Print the full per-metric aggregate, grouped by agent."""

    print("========== FULL AGGREGATE (per metric) ==========")
    for agent, group in aggregate_df.groupby("agent", sort=False):
        print(f"\n{agent}")
        view = group[["metric", "mean", "std", "median", "iqr_low", "iqr_high"]].copy()
        for col in ["mean", "std", "median", "iqr_low", "iqr_high"]:
            view[col] = view[col].round(2)
        print(view.to_string(index=False))
    print("\n=================================================\n")


# ---------------------------------------------------------------------
# Band plots
# ---------------------------------------------------------------------

def plot_band(curves, agent_keys, metric, title, ylabel, filename, out_dir, window):
    """
    Plot mean rolling curve across seeds with a +/- std shaded band, one band
    per agent.
    """

    fig, ax = plt.subplots(figsize=(11, 6))

    for key in agent_keys:
        label, _, color = AGENT_REGISTRY[key]
        arr = curves[key][metric]
        episodes = np.arange(1, arr.shape[1] + 1)

        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)

        ax.plot(episodes, mean, color=color, linewidth=1.8, label=label)
        ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.20)

    n_seeds = curves[agent_keys[0]][metric].shape[0]
    ax.set_title(f"{title}  (rolling={window}, {n_seeds} seeds, band = mean +/- std)")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)

    output_path = Path(out_dir) / filename
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-seed evaluation of tabular agents on Mini Research World."
    )

    parser.add_argument(
        "--agents",
        nargs="+",
        choices=list(AGENT_REGISTRY.keys()),
        default=DEFAULT_AGENTS,
        help="Which agents to evaluate. Default: all three.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=17,
        help="Number of seeds to run per agent. Default: 17.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Seeds used are base_seed .. base_seed + num_seeds - 1. Default: 0.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Training episodes per run. Default: 10000.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Max steps per episode. Default: 100.",
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument(
        "--epsilon-start", type=float, default=1.0, help="Initial epsilon."
    )
    parser.add_argument(
        "--epsilon-end", type=float, default=0.05, help="Minimum epsilon."
    )
    parser.add_argument(
        "--epsilon-decay", type=float, default=0.995, help="Epsilon decay per episode."
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Window for rolling curves and final-N metrics. Default: 100.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/multi_seed",
        help="Directory for outputs. Default: results/multi_seed.",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate confidence-band plots. Use --no-plots to skip.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    if args.num_seeds <= 0:
        raise SystemExit("--num-seeds must be greater than 0")

    seeds = list(range(args.base_seed, args.base_seed + args.num_seeds))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs = dict(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        render_every=None,
    )

    print("\nMulti-seed evaluation")
    print(f"agents     = {[AGENT_REGISTRY[k][0] for k in args.agents]}")
    print(f"seeds      = {seeds[0]}..{seeds[-1]}  ({len(seeds)} seeds)")
    print(f"episodes   = {args.episodes}")
    print(f"window     = {args.window}")
    print(f"plots      = {args.plots}")
    print(f"out_dir    = {out_dir}")
    print(f"total runs = {len(args.agents) * len(seeds)}")

    overall_start = time.time()

    per_seed_rows, curves = run_experiment(
        agent_keys=args.agents,
        seeds=seeds,
        train_kwargs=train_kwargs,
        window=args.window,
        collect_curves=args.plots,
    )

    per_seed_df = pd.DataFrame(per_seed_rows)
    aggregate_df = build_aggregate(per_seed_df, args.agents)

    # Reports
    print_headline(per_seed_df, args.agents, args.window)
    print_full_aggregate(aggregate_df)

    # Save tables
    per_seed_path = out_dir / "per_seed_results.csv"
    aggregate_path = out_dir / "aggregate_summary.csv"
    per_seed_df.to_csv(per_seed_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)
    print(f"[table] Saved: {per_seed_path}")
    print(f"[table] Saved: {aggregate_path}")

    # Band plots
    if args.plots:
        plot_band(
            curves, args.agents, "reward",
            "Rolling Average Reward", "Average total reward",
            "band_reward_curve.png", out_dir, args.window,
        )
        plot_band(
            curves, args.agents, "success",
            "Rolling Success Rate", "Success rate (%)",
            "band_success_rate_curve.png", out_dir, args.window,
        )

    print(f"\nDone in {time.time() - overall_start:.1f}s")
    return per_seed_df, aggregate_df


if __name__ == "__main__":
    main()
