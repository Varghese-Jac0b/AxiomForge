"""
R3 — Trap-aware PER comparison runner.

R3 is NOT a rescue step. R2 (kappa=4) already solves the anti-hacking objective
without PER. R3 is an orthogonal extension asking:

    "Can trap-aware PER recover/reintroduce replay benefits WITHOUT re-igniting
     the proxy-trap amplification seen in the earlier PER/SPIE ablation?"

One new mechanism: trap-aware PER (priority of observable proxy transitions
hard-capped at the running batch median). Base = R2 kappa=4. No PSFA, no env
change, observable-only (true_score is reporting-only).

Variants (Noisy Dueling DDQN backbone), demo-seeded, kappa=4:
    plain_noisy              noisy_dueling_ddqn
    per_spie                 spie_noisy_dueling_ddqn_per          (old PER+SPIE hacker)
    r2_k4_no_per             pg_spie_r2_noisy_dueling_ddqn        (R2, the bar)
    r2_k4_naive_per          pg_spie_r2_noisy_dueling_ddqn_per    (PER re-ignition test)
    r2_k4_trap_aware_per     pg_spie_r2_taper_noisy_dueling_ddqn_per  (R3)

    python scripts/run_pg_spie_r3_trap_aware_per.py --episodes 400 \
        --seeds 0 1 2 3 4 --kappa 4 --regimes demo_seeded
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
from scripts.run_pg_spie_r2 import REGIME_DEMO, build_args
from scripts.train_axiom_forge import DEEP_ALGOS, VERSIONS, train_seed_deep

VARIANTS = [
    ("plain_noisy",          "noisy_dueling_ddqn"),
    ("per_spie",             "spie_noisy_dueling_ddqn_per"),
    ("r2_k4_no_per",         "pg_spie_r2_noisy_dueling_ddqn"),
    ("r2_k4_naive_per",      "pg_spie_r2_noisy_dueling_ddqn_per"),
    ("r2_k4_trap_aware_per", "pg_spie_r2_taper_noisy_dueling_ddqn_per"),
]

REPORT_PATH = PROJECT_ROOT / "reports" / "pg_spie_r3_trap_aware_per.md"
RESULTS_START = "<!-- RESULTS:START -->"
RESULTS_END = "<!-- RESULTS:END -->"


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def run_one(version, regime, display, algo_key, seed, args, out_dir, window):
    sink: list[dict] = []
    df, agent, final_train, final_held, ev = train_seed_deep(
        version, algo_key, seed, args, metrics_sink=sink)
    vdir = out_dir / regime / display
    vdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(vdir / f"seed_{seed}.csv", index=False)
    pd.DataFrame(sink).to_csv(vdir / f"signals_seed_{seed}.csv", index=False)

    tail = df.tail(window)
    sw = DEEP_ALGOS[algo_key]
    ps = agent.buffer.per_stats() if hasattr(agent.buffer, "per_stats") else {}
    return {
        "regime": regime, "variant": display, "algo_key": algo_key, "seed": seed,
        "use_per": sw["per"], "use_gate": bool(sw.get("pg", False)),
        "use_penalty": bool(sw.get("pp", False)),
        "use_trap_aware": bool(sw.get("taper", False)), "kappa": args.kappa,
        # task + misalignment
        "success_rate": float(tail["success"].mean()),
        "heldout_greedy_success": float(final_held["success"].mean()),
        "mean_final_reward": float(tail["episode_return"].mean()),
        "mean_episode_length": float(df["steps"].mean()),
        "mean_trap_hits": float(df["proxy_claim_count"].mean()),
        "reward_hack_rate": float(df["reward_hack_flag"].mean()),
        "mean_true_score": float(_num(df["true_score"]).mean()),
        "mean_alignment_gap": float(_num(df["alignment_gap"]).mean()),
        # PER priority diagnostics (0 for non-PER variants)
        "mean_trap_priority": float(ps.get("mean_trap_priority", 0.0)),
        "mean_nontrap_priority": float(ps.get("mean_nontrap_priority", 0.0)),
        "trap_replay_ratio": float(ps.get("trap_replay_ratio", 0.0)),
        "n_capped_trap": int(ps.get("n_capped_trap", 0)),
        "cap_median_value": float(ps.get("cap_median_value", 0.0)),
        "learn_steps": int(getattr(agent, "learn_steps", 0)),
    }


def summarize(runs):
    metrics = ["success_rate", "heldout_greedy_success", "mean_trap_hits",
               "reward_hack_rate", "mean_alignment_gap", "mean_true_score",
               "mean_final_reward", "mean_trap_priority", "mean_nontrap_priority",
               "trap_replay_ratio", "n_capped_trap"]
    agg = {m: ["mean", "std"] for m in metrics}
    g = runs.groupby(["regime", "variant"], sort=False).agg(agg)
    g.columns = ["_".join(c).rstrip("_") for c in g.columns]
    n = runs.groupby(["regime", "variant"], sort=False).size().rename("n_seeds")
    return g.join(n).reset_index()


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _table(summary, regime):
    order = [d for d, _ in VARIANTS]
    sub = summary[summary.regime == regime].set_index("variant").reindex(order)
    cols = [
        ("success_rate_mean", "succ"), ("mean_trap_hits_mean", "trap_hits"),
        ("mean_trap_hits_std", "trap_sd"), ("reward_hack_rate_mean", "hack"),
        ("mean_alignment_gap_mean", "align_gap"),
        ("mean_alignment_gap_std", "gap_sd"),
        ("mean_true_score_mean", "true_score"),
        ("mean_trap_priority_mean", "trap_prio"),
        ("mean_nontrap_priority_mean", "nontrap_prio"),
        ("trap_replay_ratio_mean", "trap_replay"),
        ("n_capped_trap_mean", "n_capped"),
    ]
    head = "| variant | " + " | ".join(c for _, c in cols) + " |"
    sep = "|" + "|".join("---" for _ in range(len(cols) + 1)) + "|"
    lines = [head, sep]
    for v in order:
        if v in sub.index:
            r = sub.loc[v]
            lines.append(f"| {v} | " + " | ".join(_fmt(r.get(k)) for k, _ in cols) + " |")
    return "\n".join(lines)


def _interpretation(summary, regime):
    sub = summary[summary.regime == regime].set_index("variant")

    def g(v, c):
        return float(sub.loc[v, c]) if v in sub.index else float("nan")

    r2, naive, taper = "r2_k4_no_per", "r2_k4_naive_per", "r2_k4_trap_aware_per"
    return "\n".join([
        f"- **Does naive PER re-ignite?** naive_per vs R2: trap_hits "
        f"{_fmt(g(naive,'mean_trap_hits_mean'))} vs {_fmt(g(r2,'mean_trap_hits_mean'))}, "
        f"hack {_fmt(g(naive,'reward_hack_rate_mean'))} vs {_fmt(g(r2,'reward_hack_rate_mean'))}, "
        f"align_gap {_fmt(g(naive,'mean_alignment_gap_mean'))} vs {_fmt(g(r2,'mean_alignment_gap_mean'))} "
        "(higher for naive_per => PER re-ignites hacking).",
        f"- **Does trap-aware PER prevent it?** trap_aware vs naive_per: trap_hits "
        f"{_fmt(g(taper,'mean_trap_hits_mean'))} vs {_fmt(g(naive,'mean_trap_hits_mean'))}, "
        f"align_gap {_fmt(g(taper,'mean_alignment_gap_mean'))} vs {_fmt(g(naive,'mean_alignment_gap_mean'))}.",
        f"- **Stays safe vs R2?** trap_aware vs R2: trap_hits "
        f"{_fmt(g(taper,'mean_trap_hits_mean'))} vs {_fmt(g(r2,'mean_trap_hits_mean'))}, success "
        f"{_fmt(g(taper,'success_rate_mean'))} vs {_fmt(g(r2,'success_rate_mean'))}.",
        f"- **Priority cap working?** trap/non-trap priority — naive_per "
        f"{_fmt(g(naive,'mean_trap_priority_mean'))}/{_fmt(g(naive,'mean_nontrap_priority_mean'))} "
        f"vs trap_aware {_fmt(g(taper,'mean_trap_priority_mean'))}/"
        f"{_fmt(g(taper,'mean_nontrap_priority_mean'))}; capped trap transitions "
        f"(trap_aware) {_fmt(g(taper,'n_capped_trap_mean'))}.",
    ])


def fill_report(summary, cli):
    if not REPORT_PATH.exists():
        print(f"[warn] {REPORT_PATH} missing; skipping report fill.")
        return
    text = REPORT_PATH.read_text()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    blocks = [RESULTS_START, "",
              f"_Generated {stamp} — V1.4, {cli.episodes} episodes, seeds "
              f"{cli.seeds}, kappa={cli.kappa}, Noisy Dueling DDQN. success = "
              f"final-{ROLLING_WINDOW}-ep training success; align_gap = visible "
              "return − hidden true_score (reporting-only)._", ""]
    for regime in [r for r in REGIME_DEMO if (summary.regime == r).any()]:
        label = ("demo-seeded" if regime == "demo_seeded" else "no-demo (E2)")
        blocks += [f"### {regime} — {label}", "", _table(summary, regime), "",
                   _interpretation(summary, regime), ""]
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
    p.add_argument("--kappa", type=float, default=4.0)
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
    p.add_argument("--results-dir", type=str,
                   default="results/pg_spie_r3_trap_aware_per")
    p.add_argument("--no-report", action="store_true")
    cli = p.parse_args()

    out_dir = PROJECT_ROOT / cli.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for regime in cli.regimes:
        args = build_args(cli, REGIME_DEMO[regime])
        for display, key in VARIANTS:
            for seed in cli.seeds:
                print(f"\n=== [{regime}] {display} | seed {seed} ===", flush=True)
                row = run_one(cli.version, regime, display, key, seed, args,
                              out_dir, cli.eval_window)
                runs.append(row)
                print(f"  succ={row['success_rate']:.2f} "
                      f"trap_hits={row['mean_trap_hits']:.2f} "
                      f"hack={row['reward_hack_rate']:.2f} "
                      f"gap={row['mean_alignment_gap']:.2f} "
                      f"trap_prio/nontrap={row['mean_trap_priority']:.3f}/"
                      f"{row['mean_nontrap_priority']:.3f} "
                      f"capped={row['n_capped_trap']}", flush=True)

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(out_dir / "pg_spie_r3_runs.csv", index=False)
    summary = summarize(runs_df)
    summary.to_csv(out_dir / "pg_spie_r3_summary.csv", index=False)
    print("\n========== PG-SPIE R3 SUMMARY ==========")
    print(summary.to_string(index=False))
    print(f"\nartifacts -> {out_dir}")
    if not cli.no_report:
        fill_report(summary, cli)


if __name__ == "__main__":
    main()
