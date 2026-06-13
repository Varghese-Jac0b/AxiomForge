"""
Aggregate the full-scale benchmark under results/full_benchmark/ into
statistical tables (mean / std / median / best / worst across seeds),
generalization and misalignment tables, and the Phase-5 rankings.

Read-only: consumes the per-seed CSVs and summary.csv the multi-config
driver wrote; never re-runs training. Writes CSV tables under
results/full_benchmark/tables/ and prints a console digest.

Run from the repo root:
    python analysis/aggregate_full_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROOT = PROJECT_ROOT / "results" / "full_benchmark"
TABLES = ROOT / "tables"
TABULAR = ["q_learning", "sarsa", "expected_sarsa", "spie_q"]
DEEP = ["dqn", "ddqn", "dueling_dqn", "dueling_ddqn_per",
        "noisy_dueling_ddqn_per", "spie_dqn", "spie_ddqn",
        "spie_dueling_dqn", "spie_dueling_ddqn_per",
        "spie_noisy_dueling_ddqn_per"]
ALL = TABULAR + DEEP
VERSIONS = ["V1.1", "V1.2", "V1.3", "V1.4"]
FINAL_N = 200  # episodes used for E2 steady-state means


def _stats(series: pd.Series) -> dict:
    a = series.to_numpy(dtype=float)
    return dict(mean=np.mean(a), std=np.std(a), median=np.median(a),
                best=np.max(a), worst=np.min(a), n=len(a))


# ---------------------------------------------------------------------
# Generalization (train / held-out greedy success per version x algo)
# ---------------------------------------------------------------------


def generalization_tables():
    rows = []
    for version in VERSIONS:
        for algo in ALL:
            summ = ROOT / version / algo / "summary.csv"
            if not summ.exists():
                continue
            df = pd.read_csv(summ)
            tr = _stats(df["train_greedy_success"])
            ho = _stats(df["heldout_greedy_success"])
            rows.append({
                "version": version, "algo": algo,
                "kind": "tabular" if algo in TABULAR else "deep",
                "seeds": tr["n"],
                "train_mean": tr["mean"], "train_std": tr["std"],
                "train_median": tr["median"], "train_best": tr["best"],
                "train_worst": tr["worst"],
                "heldout_mean": ho["mean"], "heldout_std": ho["std"],
                "heldout_median": ho["median"], "heldout_best": ho["best"],
                "heldout_worst": ho["worst"],
                "gen_gap": tr["mean"] - ho["mean"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Sample efficiency proxy (episodes to rolling-100 training success >= 0.9)
# ---------------------------------------------------------------------


def _episodes_to_90(seed_csv: Path) -> float | None:
    df = pd.read_csv(seed_csv)
    roll = df["success"].rolling(100).mean()
    hit = roll[roll >= 0.9]
    return float(hit.index[0] + 1) if len(hit) else None


def sample_efficiency_table():
    rows = []
    for version in VERSIONS:
        for algo in ALL:
            folder = ROOT / version / algo
            if not folder.exists():
                continue
            vals = [_episodes_to_90(p) for p in sorted(folder.glob("seed_*.csv"))]
            solved = [v for v in vals if v is not None]
            rows.append({
                "version": version, "algo": algo,
                "seeds_reaching_90pct_train": len(solved),
                "median_episodes_to_90pct":
                    np.median(solved) if solved else None,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Misalignment / E2 (V1.4 vanilla)
# ---------------------------------------------------------------------


def e2_table():
    rows = []
    e2_root = ROOT / "V1.4_e2"
    if not e2_root.exists():
        return pd.DataFrame()
    for algo in ALL:
        folder = e2_root / algo
        seed_csvs = sorted(folder.glob("seed_*.csv")) if folder.exists() else []
        if not seed_csvs:
            continue
        per_seed = []
        for p in seed_csvs:
            df = pd.read_csv(p).tail(FINAL_N)
            per_seed.append(dict(
                ret=df["episode_return"].mean(),
                true_score=df["true_score"].mean(),
                align_gap=df["alignment_gap"].mean(),
                hack=df["reward_hack_flag"].mean(),
                claims=df["proxy_claim_count"].mean(),
                success=df["success"].mean()))
        s = pd.DataFrame(per_seed)
        rows.append({
            "algo": algo, "kind": "tabular" if algo in TABULAR else "deep",
            "seeds": len(seed_csvs),
            "success_mean": s["success"].mean(),
            "return_mean": s["ret"].mean(),
            "true_score_mean": s["true_score"].mean(),
            "align_gap_mean": s["align_gap"].mean(),
            "align_gap_std": s["align_gap"].std(),
            "hack_rate_mean": s["hack"].mean(),
            "proxy_claims_mean": s["claims"].mean(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Rankings (Phase 5)
# ---------------------------------------------------------------------


def rankings(gen: pd.DataFrame, e2: pd.DataFrame, sample: pd.DataFrame):
    out = []
    # best generalization = highest held-out mean over all version x algo
    if not gen.empty:
        g = gen.sort_values("heldout_mean", ascending=False).iloc[0]
        out.append(("Best generalization (highest held-out greedy)",
                    f"{g['algo']} on {g['version']} "
                    f"(held-out {g['heldout_mean']:.2f})"))
        # best tabular / deep by held-out averaged across versions
        for kind in ("tabular", "deep"):
            sub = gen[gen["kind"] == kind]
            if sub.empty:
                continue
            by_algo = sub.groupby("algo")["heldout_mean"].mean()
            top = by_algo.idxmax()
            out.append((f"Best {kind} algorithm (mean held-out across versions)",
                        f"{top} ({by_algo.max():.2f})"))
        # best SPIE variant
        spie = gen[gen["algo"].str.startswith("spie")]
        if not spie.empty:
            by_algo = spie.groupby("algo")["heldout_mean"].mean()
            out.append(("Best SPIE algorithm (mean held-out across versions)",
                        f"{by_algo.idxmax()} ({by_algo.max():.2f})"))
    # best alignment-preserving = lowest E2 alignment gap
    if not e2.empty:
        a = e2.sort_values("align_gap_mean").iloc[0]
        out.append(("Best alignment-preserving (lowest V1.4 align_gap)",
                    f"{a['algo']} (gap {a['align_gap_mean']:.2f}, "
                    f"hack {a['hack_rate_mean']:.2f})"))
        worst = e2.sort_values("align_gap_mean").iloc[-1]
        out.append(("Worst (most reward-hacking)",
                    f"{worst['algo']} (gap {worst['align_gap_mean']:.2f}, "
                    f"hack {worst['hack_rate_mean']:.2f})"))
    # most sample efficient (fewest median episodes to train-90%)
    if not sample.empty:
        s = sample.dropna(subset=["median_episodes_to_90pct"])
        if not s.empty:
            top = s.sort_values("median_episodes_to_90pct").iloc[0]
            out.append(("Most sample-efficient (fewest episodes to train-90%)",
                        f"{top['algo']} on {top['version']} "
                        f"({top['median_episodes_to_90pct']:.0f} episodes)"))
    return out


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    gen = generalization_tables()
    sample = sample_efficiency_table()
    e2 = e2_table()

    if not gen.empty:
        gen.round(3).to_csv(TABLES / "generalization.csv", index=False)
    if not sample.empty:
        sample.to_csv(TABLES / "sample_efficiency.csv", index=False)
    if not e2.empty:
        e2.round(3).to_csv(TABLES / "misalignment_e2.csv", index=False)

    print("=" * 70)
    print("FULL BENCHMARK AGGREGATION")
    print("=" * 70)
    if not gen.empty:
        print("\n--- Generalization: held-out greedy (mean +/- std) by version ---")
        pivot = gen.pivot_table(index="algo", columns="version",
                                values="heldout_mean")
        pivot = pivot.reindex([a for a in ALL if a in pivot.index])
        print(pivot.round(2).to_string())
        print("\n--- Generalization: train greedy (mean) by version ---")
        pivot_t = gen.pivot_table(index="algo", columns="version",
                                  values="train_mean")
        pivot_t = pivot_t.reindex([a for a in ALL if a in pivot_t.index])
        print(pivot_t.round(2).to_string())
    if not e2.empty:
        print("\n--- V1.4 E2 misalignment (vanilla, final-200-episode means) ---")
        print(e2[["algo", "kind", "seeds", "success_mean", "return_mean",
                  "true_score_mean", "align_gap_mean", "hack_rate_mean",
                  "proxy_claims_mean"]].round(2).to_string(index=False))
    print("\n--- RANKINGS ---")
    for label, val in rankings(gen, e2, sample):
        print(f"  {label}:\n      {val}")
    print(f"\ntables -> {TABLES}")


if __name__ == "__main__":
    main()
