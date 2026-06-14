"""
Build the consolidated R0->R2 anti-reward-hacking summary artifacts: a few
lightweight plots + consolidated CSV tables under results/r0_r2_summary/.

Read-only over the existing experiment CSVs; writes only into the new summary
folder. No training, no new mechanism.

    python scripts/build_r0_r2_summary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "r0_r2_summary"
PLOTS = OUT / "plots"


def _bars(ax, labels, values, title, ylabel, rotate=20):
    ax.bar(range(len(labels)), values, color="tab:blue")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotate, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)


def plot_r0_ablation(ab):
    order = ["noisy_dueling_ddqn", "noisy_dueling_ddqn_per",
             "noisy_dueling_ddqn_spie", "noisy_dueling_ddqn_per_spie"]
    short = ["noisy", "+PER", "+SPIE", "+PER+SPIE"]
    a = ab.set_index("variant").reindex(order)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _bars(axes[0], short, a["mean_trap_hits_mean"].tolist(),
          "R0 ablation (no-demo): trap_hits", "mean trap_hits")
    _bars(axes[1], short, a["mean_alignment_gap_mean"].tolist(),
          "R0 ablation (no-demo): alignment_gap", "gap (higher=more hacking)")
    fig.suptitle("R0: naive SPIE and PER+SPIE amplify proxy hacking", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "r0_ablation.png", dpi=150)
    plt.close(fig)


def plot_demo_ladder(ks):
    order = ["plain_noisy", "naive_spie", "pg_spie",
             "pg_spie_r2_k4", "pg_spie_r2_k8"]
    short = ["plain", "naive_spie", "pg_spie(R1)", "R2 k=4", "R2 k=8"]
    a = ks.set_index("variant").reindex(order)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    _bars(axes[0], short, a["mean_trap_hits_mean"].tolist(),
          "demo-seeded: trap_hits", "mean trap_hits")
    _bars(axes[1], short, a["mean_alignment_gap_mean"].tolist(),
          "demo-seeded: alignment_gap", "gap (higher=more hacking)")
    _bars(axes[2], short, a["success_rate_mean"].tolist(),
          "demo-seeded: success", "success rate")
    axes[2].set_ylim(0, 1)
    fig.suptitle("Anti-hacking ladder (demo-seeded, 5 seeds): "
                 "R2 removes hacking at no success cost", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "demo_ladder.png", dpi=150)
    plt.close(fig)


def plot_ksweep(ks):
    r1 = ks.set_index("variant").loc["pg_spie"]
    rows = []
    for k in (1, 4, 8):
        v = f"pg_spie_r2_k{k}"
        if v in ks["variant"].values:
            r = ks.set_index("variant").loc[v]
            rows.append((k, r["mean_trap_hits_mean"], r["mean_trap_hits_std"],
                         r["mean_alignment_gap_mean"], r["mean_alignment_gap_std"],
                         r["success_rate_mean"]))
    ksv = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].errorbar(ksv, [r[1] for r in rows], yerr=[r[2] for r in rows],
                     marker="o", capsize=3, label="R2")
    axes[0].axhline(r1["mean_trap_hits_mean"], color="tab:red", ls="--",
                    label="R1 (pg_spie)")
    axes[0].set_title("trap_hits vs kappa"); axes[0].set_xlabel("kappa")
    axes[0].set_ylabel("mean trap_hits"); axes[0].legend(fontsize=8)
    axes[1].errorbar(ksv, [r[3] for r in rows], yerr=[r[4] for r in rows],
                     marker="o", capsize=3, color="tab:green", label="R2")
    axes[1].axhline(r1["mean_alignment_gap_mean"], color="tab:red", ls="--",
                    label="R1")
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_title("alignment_gap vs kappa"); axes[1].set_xlabel("kappa")
    axes[1].set_ylabel("gap (higher=hacking)"); axes[1].legend(fontsize=8)
    axes[2].plot(ksv, [r[5] for r in rows], marker="o", color="tab:purple",
                 label="R2")
    axes[2].axhline(r1["success_rate_mean"], color="tab:red", ls="--", label="R1")
    axes[2].set_ylim(0, 1); axes[2].set_title("success vs kappa")
    axes[2].set_xlabel("kappa"); axes[2].set_ylabel("success rate")
    axes[2].legend(fontsize=8)
    fig.suptitle("R2 kappa-sweep (demo-seeded, 5 seeds): hacking falls to ~0, "
                 "variance collapses, success preserved", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "ksweep_trend.png", dpi=150)
    plt.close(fig)


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    ab = pd.read_csv(ROOT / "results/ablation_noisy_per_spi/ablation_summary.csv")
    ks = pd.read_csv(ROOT / "results/pg_spie_r2_ksweep/ksweep_summary.csv")
    runs = pd.read_csv(ROOT / "results/pg_spie_r2_ksweep/ksweep_runs.csv")

    plot_r0_ablation(ab)
    plot_demo_ladder(ks)
    plot_ksweep(ks)

    # consolidated tables (CSV) for the report
    keep_ab = ["variant", "success_rate_mean", "mean_trap_hits_mean",
               "reward_hack_rate_mean", "mean_alignment_gap_mean",
               "mean_intrinsic_contribution_mean", "mean_td_abs_mean"]
    ab[keep_ab].to_csv(OUT / "table_r0_ablation.csv", index=False)
    keep_ks = ["variant", "success_rate_mean", "mean_trap_hits_mean",
               "mean_trap_hits_std", "reward_hack_rate_mean",
               "mean_alignment_gap_mean", "mean_alignment_gap_std",
               "mean_true_score_mean", "proxy_penalty_count_mean",
               "mean_proxy_penalty_mean"]
    ks[[c for c in keep_ks if c in ks.columns]].to_csv(
        OUT / "table_ksweep.csv", index=False)

    # per-seed (esp. seed 0) robustness for the gated/penalized variants
    perseed = runs[runs.variant.isin(
        ["pg_spie", "pg_spie_r2_k1", "pg_spie_r2_k4", "pg_spie_r2_k8"])]
    perseed = perseed[["variant", "seed", "mean_trap_hits",
                       "mean_alignment_gap", "success_rate"]].sort_values(
        ["variant", "seed"])
    perseed.to_csv(OUT / "table_per_seed.csv", index=False)

    print("wrote plots:", sorted(p.name for p in PLOTS.glob("*.png")))
    print("wrote tables:", sorted(p.name for p in OUT.glob("*.csv")))


if __name__ == "__main__":
    main()
