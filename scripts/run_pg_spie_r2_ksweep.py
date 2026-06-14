"""
R2 consolidation: kappa-sweep + 5-seed tightening (demo-seeded).

R2 showed a directionally positive but seed-noisy improvement over R1 in the
demo-seeded regime, with a tiny penalty magnitude (mean_pen ~0.046 at kappa=1).
This run consolidates that result by (a) sweeping the penalty coefficient kappa
and (b) using 5 seeds for tighter statistics.

Scope = demo-seeded ONLY, on purpose: in the no-demo regime the PG gate already
keeps the agent off the proxy (gate_proxy == 0), so the R2 penalty never fires
and R2 is bit-identical to R1 for every kappa — that regime was already a clean,
unambiguous win and is kappa-invariant. (Pass --regimes to override.)

Variants (Noisy Dueling DDQN backbone, PER off):
    plain_noisy / naive_spie / pg_spie  -- kappa-invariant references (run once)
    pg_spie_r2_k{K}                      -- R2 at each kappa in the sweep

Reuses the verified train_seed_deep + run_one from run_pg_spie_r2 (same gate,
same proxy penalty, observable-only; true_score is reporting-only). Writes to a
SEPARATE folder/report so all earlier artifacts are untouched.

    python scripts/run_pg_spie_r2_ksweep.py --kappas 1 4 8 --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.axiom_forge_baselines_common import ROLLING_WINDOW
from scripts.run_pg_spie_r2 import REGIME_DEMO, build_args, run_one

REFERENCE_VARIANTS = [
    ("plain_noisy", "noisy_dueling_ddqn"),
    ("naive_spie",  "spie_noisy_dueling_ddqn"),
    ("pg_spie",     "pg_spie_noisy_dueling_ddqn"),
]
R2_KEY = "pg_spie_r2_noisy_dueling_ddqn"

REPORT_PATH = PROJECT_ROOT / "reports" / "pg_spie_r2_ksweep.md"
RESULTS_START = "<!-- RESULTS:START -->"
RESULTS_END = "<!-- RESULTS:END -->"


def _ktag(k: float) -> str:
    return f"k{int(k)}" if float(k).is_integer() else f"k{k}".replace(".", "p")


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def variant_order(kappas):
    return [d for d, _ in REFERENCE_VARIANTS] + \
        [f"pg_spie_r2_{_ktag(k)}" for k in kappas]


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = ["success_rate", "heldout_greedy_success", "mean_trap_hits",
               "reward_hack_rate", "mean_alignment_gap", "mean_true_score",
               "mean_final_reward", "proxy_penalty_count", "proxy_penalty_total",
               "mean_proxy_penalty"]
    agg = {m: ["mean", "std"] for m in metrics}
    g = runs.groupby(["regime", "variant"], sort=False).agg(agg)
    g.columns = ["_".join(c).rstrip("_") for c in g.columns]
    n = runs.groupby(["regime", "variant"], sort=False).size().rename("n_seeds")
    return g.join(n).reset_index()


def _table(summary, regime, order):
    sub = summary[summary["regime"] == regime].set_index("variant").reindex(order)
    cols = [
        ("success_rate_mean", "succ"), ("success_rate_std", "succ_sd"),
        ("mean_trap_hits_mean", "trap_hits"), ("mean_trap_hits_std", "trap_sd"),
        ("reward_hack_rate_mean", "hack"),
        ("mean_alignment_gap_mean", "align_gap"),
        ("mean_alignment_gap_std", "gap_sd"),
        ("mean_true_score_mean", "true_score"),
        ("proxy_penalty_count_mean", "pen_n"),
        ("mean_proxy_penalty_mean", "mean_pen"),
    ]
    header = "| variant | " + " | ".join(c for _, c in cols) + " |"
    sep = "|" + "|".join("---" for _ in range(len(cols) + 1)) + "|"
    lines = [header, sep]
    for v in order:
        if v not in sub.index:
            continue
        row = sub.loc[v]
        lines.append(f"| {v} | " + " | ".join(_fmt(row.get(k)) for k, _ in cols) + " |")
    return "\n".join(lines)


def _interpretation(summary, regime, kappas):
    sub = summary[summary["regime"] == regime].set_index("variant")

    def g(v, c):
        return float(sub.loc[v, c]) if v in sub.index else float("nan")

    lines = ["**kappa trend (R2 vs R1 reference):**", ""]
    r1_trap = g("pg_spie", "mean_trap_hits_mean")
    r1_gap = g("pg_spie", "mean_alignment_gap_mean")
    plain_succ = g("plain_noisy", "success_rate_mean")
    for k in kappas:
        v = f"pg_spie_r2_{_ktag(k)}"
        lines.append(
            f"- kappa={k}: trap_hits {_fmt(g(v,'mean_trap_hits_mean'))} "
            f"(R1 {_fmt(r1_trap)}), align_gap {_fmt(g(v,'mean_alignment_gap_mean'))} "
            f"(R1 {_fmt(r1_gap)}), success {_fmt(g(v,'success_rate_mean'))} "
            f"(plain {_fmt(plain_succ)}), mean_pen {_fmt(g(v,'mean_proxy_penalty_mean'))}.")
    lines += ["",
              "Read: lower trap_hits/align_gap as kappa rises = the penalty is "
              "biting; a success drop toward 0 = over-conservative. Compare the "
              "deltas against the std columns to judge whether the effect now "
              "exceeds seed noise at 5 seeds."]
    return "\n".join(lines)


def fill_report(summary, cli, order):
    if not REPORT_PATH.exists():
        print(f"[warn] {REPORT_PATH} missing; skipping report fill.")
        return
    text = REPORT_PATH.read_text()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    blocks = [RESULTS_START, "",
              f"_Generated {stamp} — V1.4, {cli.episodes} episodes, seeds "
              f"{cli.seeds}, kappa sweep {cli.kappas}, Noisy Dueling DDQN (PER "
              f"off). success = final-{ROLLING_WINDOW}-ep training success; "
              "align_gap = visible return − hidden true_score (reporting-only)._",
              ""]
    for regime in [r for r in REGIME_DEMO if (summary["regime"] == r).any()]:
        label = ("demo-seeded (--demo-configs 8)" if regime == "demo_seeded"
                 else "no-demo (misalignment/E2)")
        blocks += [f"### {regime} — {label}", "", _table(summary, regime, order),
                   "", _interpretation(summary, regime, cli.kappas), ""]
    blocks.append(RESULTS_END)
    block = "\n".join(blocks)
    if RESULTS_START in text and RESULTS_END in text:
        text = text.split(RESULTS_START)[0] + block + text.split(RESULTS_END)[1]
    else:
        text = text.rstrip() + "\n\n## Results\n\n" + block + "\n"
    REPORT_PATH.write_text(text)
    print(f"report updated -> {REPORT_PATH}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", default="v1_4")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--kappas", type=float, nargs="+", default=[1.0, 4.0, 8.0])
    p.add_argument("--regimes", nargs="+", default=["demo_seeded"],
                   choices=["no_demo", "demo_seeded"])
    p.add_argument("--train-configs", type=int, default=16)
    p.add_argument("--heldout-configs", type=int, default=8)
    p.add_argument("--demo-sweeps", type=int, default=5)
    p.add_argument("--eval-window", type=int, default=ROLLING_WINDOW)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--target-update", type=int, default=1000)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--pretrain-steps", type=int, default=2000)
    p.add_argument("--demo-fraction", type=float, default=0.25)
    p.add_argument("--margin", type=float, default=0.8)
    p.add_argument("--margin-weight", type=float, default=1.0)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.999)
    p.add_argument("--mode", default="full")
    p.add_argument("--abstraction", default="milestone")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--beta-decay", type=float, default=0.999)
    p.add_argument("--beta-end", type=float, default=0.0)
    p.add_argument("--sr-alpha", type=float, default=0.1)
    p.add_argument("--sr-gamma", type=float, default=0.95)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--results-dir", type=str, default="results/pg_spie_r2_ksweep")
    p.add_argument("--no-report", action="store_true")
    cli = p.parse_args()
    cli.kappa = cli.kappas[0]   # placeholder so build_args() is happy

    out_dir = PROJECT_ROOT / cli.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    order = variant_order(cli.kappas)
    runs = []
    for regime in cli.regimes:
        demo = REGIME_DEMO[regime]
        base_args = build_args(cli, demo)
        # kappa-invariant references (run once)
        for display, key in REFERENCE_VARIANTS:
            for seed in cli.seeds:
                print(f"\n=== [{regime}] {display} | seed {seed} ===", flush=True)
                row = run_one(cli.version, regime, display, key, seed,
                              base_args, out_dir, cli.eval_window)
                runs.append(row)
                print(f"  succ={row['success_rate']:.2f} "
                      f"trap_hits={row['mean_trap_hits']:.2f} "
                      f"gap={row['mean_alignment_gap']:.2f}", flush=True)
        # R2 swept over kappa
        for k in cli.kappas:
            args_k = build_args(cli, demo)
            args_k.kappa = k
            display = f"pg_spie_r2_{_ktag(k)}"
            for seed in cli.seeds:
                print(f"\n=== [{regime}] {display} | seed {seed} (kappa={k}) ===",
                      flush=True)
                row = run_one(cli.version, regime, display, R2_KEY, seed,
                              args_k, out_dir, cli.eval_window)
                runs.append(row)
                print(f"  succ={row['success_rate']:.2f} "
                      f"trap_hits={row['mean_trap_hits']:.2f} "
                      f"gap={row['mean_alignment_gap']:.2f} "
                      f"pen(n/total)={row['proxy_penalty_count']}/"
                      f"{row['proxy_penalty_total']:.1f}", flush=True)

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(out_dir / "ksweep_runs.csv", index=False)
    summary = summarize(runs_df)
    summary.to_csv(out_dir / "ksweep_summary.csv", index=False)
    print("\n========== R2 KAPPA-SWEEP SUMMARY ==========")
    print(summary.to_string(index=False))
    print(f"\nartifacts -> {out_dir}")
    if not cli.no_report:
        fill_report(summary, cli, order)


if __name__ == "__main__":
    main()
