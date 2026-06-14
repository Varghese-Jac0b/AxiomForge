"""
R2 — PG-SPIE + proxy-coupled penalty (Trap-Sentinel penalty) comparison runner.

Builds directly on R1. Same observable milestone gate; the ONE new knob is a
proxy-coupled negative response: on a proxy_attempt/claim step the SPIE bonus is
turned into -kappa*bonus instead of merely suppressed. No PER (R3), no PSFA.
Nothing reads true_score / HiddenContext in training, policy, replay, or the
gate; true_score is REPORTED only.

Four variants on the same Noisy Dueling DDQN backbone (PER off):
    plain_noisy   noisy_dueling_ddqn
    naive_spie    spie_noisy_dueling_ddqn
    pg_spie       pg_spie_noisy_dueling_ddqn       (R1)
    pg_spie_r2    pg_spie_r2_noisy_dueling_ddqn     (R2: +proxy penalty)

Two regimes: no_demo (E2) and demo_seeded (--demo-configs 8). Separate output:
results/pg_spie_r2/ and reports/pg_spie_r2.md.

    python scripts/run_pg_spie_r2.py --episodes 400 --seeds 0 1 2 --kappa 1.0
    python scripts/run_pg_spie_r2.py --episodes 30 --seeds 0 --train-configs 4 \
        --heldout-configs 4 --regimes no_demo   # smoke
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.axiom_forge_baselines_common import ROLLING_WINDOW
from scripts.train_axiom_forge import DEEP_ALGOS, VERSIONS, train_seed_deep

VARIANTS = [
    ("plain_noisy", "noisy_dueling_ddqn"),
    ("naive_spie",  "spie_noisy_dueling_ddqn"),
    ("pg_spie",     "pg_spie_noisy_dueling_ddqn"),
    ("pg_spie_r2",  "pg_spie_r2_noisy_dueling_ddqn"),
]
REGIME_DEMO = {"no_demo": 0, "demo_seeded": 8}

REPORT_PATH = PROJECT_ROOT / "reports" / "pg_spie_r2.md"
RESULTS_START = "<!-- RESULTS:START -->"
RESULTS_END = "<!-- RESULTS:END -->"


def build_args(cli, demo_configs) -> Namespace:
    return Namespace(
        episodes=cli.episodes,
        gamma=cli.gamma, lr=cli.lr,
        batch_size=cli.batch_size, buffer_size=cli.buffer_size,
        target_update=cli.target_update, train_freq=cli.train_freq,
        pretrain_steps=(cli.pretrain_steps if demo_configs > 0 else 0),
        demo_configs=demo_configs, demo_sweeps=cli.demo_sweeps,
        demo_fraction=cli.demo_fraction, margin=cli.margin,
        margin_weight=cli.margin_weight,
        train_configs=cli.train_configs, heldout_configs=cli.heldout_configs,
        epsilon_start=cli.epsilon_start, epsilon_end=cli.epsilon_end,
        epsilon_decay=cli.epsilon_decay,
        mode=cli.mode, abstraction=cli.abstraction,
        beta=cli.beta, beta_decay=cli.beta_decay, beta_end=cli.beta_end,
        sr_alpha=cli.sr_alpha, sr_gamma=cli.sr_gamma, kappa=cli.kappa,
        device=cli.device, shaping=False, e2=(demo_configs == 0),
    )


def _num(series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def run_one(version, regime, display, algo_key, seed, args, out_dir, window):
    sink: list[dict] = []
    df, agent, final_train, final_held, ev = train_seed_deep(
        version, algo_key, seed, args, metrics_sink=sink)

    vdir = out_dir / regime / display
    vdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(vdir / f"seed_{seed}.csv", index=False)
    sk = pd.DataFrame(sink)
    sk.to_csv(vdir / f"signals_seed_{seed}.csv", index=False)

    tail = df.tail(window)
    sw = DEEP_ALGOS[algo_key]
    g_adv = int(sk["gate_advance"].sum())
    g_prx = int(sk["gate_proxy"].sum())
    allowed = float(sk["allowed_intrinsic"].sum())
    suppressed = float(sk["suppressed_intrinsic"].sum())
    pen_total = float(sk["proxy_penalty_total"].sum())
    pen_count = int(sk["proxy_penalty_count"].sum())
    return {
        "regime": regime, "variant": display, "algo_key": algo_key, "seed": seed,
        "use_spie": sw["spie"], "use_gate": bool(sw.get("pg", False)),
        "use_penalty": bool(sw.get("pp", False)), "kappa": args.kappa,
        # task performance
        "success_rate": float(tail["success"].mean()),
        "heldout_greedy_success": float(final_held["success"].mean()),
        "final_correct_submissions": int(df["success"].sum()),
        "mean_final_reward": float(tail["episode_return"].mean()),
        "mean_episode_length": float(df["steps"].mean()),
        # misalignment signals
        "mean_trap_hits": float(df["proxy_claim_count"].mean()),
        "reward_hack_rate": float(df["reward_hack_flag"].mean()),
        "mean_true_score": float(_num(df["true_score"]).mean()),
        "mean_alignment_gap": float(_num(df["alignment_gap"]).mean()),
        # gate / penalty internals
        "mean_intrinsic": float(sk["intrinsic_contribution"].mean()),
        "gate_advance_count": g_adv,
        "gate_proxy_count": g_prx,
        "allowed_spie_bonus": allowed,
        "suppressed_spie_bonus": suppressed,
        "proxy_penalty_total": pen_total,
        "proxy_penalty_count": pen_count,
        "mean_proxy_penalty": pen_total / pen_count if pen_count else 0.0,
        "learn_steps": int(getattr(agent, "learn_steps", 0)),
    }


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = ["success_rate", "heldout_greedy_success", "mean_trap_hits",
               "reward_hack_rate", "mean_alignment_gap", "mean_true_score",
               "mean_final_reward", "mean_intrinsic", "gate_advance_count",
               "gate_proxy_count", "proxy_penalty_total", "proxy_penalty_count",
               "mean_proxy_penalty"]
    agg = {m: ["mean", "std"] for m in metrics}
    g = runs.groupby(["regime", "variant"], sort=False).agg(agg)
    g.columns = ["_".join(c).rstrip("_") for c in g.columns]
    n = runs.groupby(["regime", "variant"], sort=False).size().rename("n_seeds")
    return g.join(n).reset_index()


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _regime_table(summary, regime):
    order = [d for d, _ in VARIANTS]
    sub = summary[summary["regime"] == regime].set_index("variant").reindex(order)
    cols = [
        ("success_rate_mean", "succ"), ("mean_trap_hits_mean", "trap_hits"),
        ("reward_hack_rate_mean", "hack_rate"),
        ("mean_alignment_gap_mean", "align_gap"),
        ("mean_true_score_mean", "true_score"),
        ("mean_final_reward_mean", "final_ret"),
        ("mean_intrinsic_mean", "net_intrinsic"),
        ("gate_proxy_count_mean", "gate_proxy"),
        ("proxy_penalty_count_mean", "pen_count"),
        ("proxy_penalty_total_mean", "pen_total"),
        ("mean_proxy_penalty_mean", "mean_pen"),
    ]
    header = "| variant | " + " | ".join(c for _, c in cols) + " |"
    sep = "|" + "|".join("---" for _ in range(len(cols) + 1)) + "|"
    lines = [header, sep]
    for v in order:
        row = sub.loc[v]
        lines.append(f"| {v} | " + " | ".join(_fmt(row.get(k)) for k, _ in cols) + " |")
    return "\n".join(lines)


def _interpretation(summary, regime):
    rows = {d: summary[(summary.regime == regime) & (summary.variant == d)]
            for d, _ in VARIANTS}

    def g(v, c):
        s = rows[v]
        return float(s[c].iloc[0]) if len(s) else float("nan")

    return "\n".join([
        f"- **R2 vs R1** (trap_hits {_fmt(g('pg_spie_r2','mean_trap_hits_mean'))} "
        f"vs {_fmt(g('pg_spie','mean_trap_hits_mean'))}; hack_rate "
        f"{_fmt(g('pg_spie_r2','reward_hack_rate_mean'))} vs "
        f"{_fmt(g('pg_spie','reward_hack_rate_mean'))}; align_gap "
        f"{_fmt(g('pg_spie_r2','mean_alignment_gap_mean'))} vs "
        f"{_fmt(g('pg_spie','mean_alignment_gap_mean'))}) — lower for R2 means the "
        "active penalty added value beyond R1's passive suppression.",
        f"- **R2 vs naive SPIE** (trap_hits "
        f"{_fmt(g('pg_spie_r2','mean_trap_hits_mean'))} vs "
        f"{_fmt(g('naive_spie','mean_trap_hits_mean'))}; align_gap "
        f"{_fmt(g('pg_spie_r2','mean_alignment_gap_mean'))} vs "
        f"{_fmt(g('naive_spie','mean_alignment_gap_mean'))}).",
        f"- **Cost check** (success R2 {_fmt(g('pg_spie_r2','success_rate_mean'))} "
        f"vs plain {_fmt(g('plain_noisy','success_rate_mean'))}) — if success "
        "collapses while alignment improves, the penalty is over-conservative.",
    ])


def fill_report(summary, cli):
    if not REPORT_PATH.exists():
        print(f"[warn] {REPORT_PATH} missing; skipping report fill.")
        return
    text = REPORT_PATH.read_text()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    blocks = [RESULTS_START, "",
              f"_Generated {stamp} — V1.4, {cli.episodes} episodes, "
              f"seeds {cli.seeds}, kappa={cli.kappa}, backbone Noisy Dueling DDQN "
              f"(PER off). success = final-{ROLLING_WINDOW}-episode training "
              "success; align_gap = visible return − hidden true_score (higher = "
              "more hacking, reporting-only)._", ""]
    for regime in [r for r in REGIME_DEMO if (summary["regime"] == r).any()]:
        label = ("no-demo (misalignment/E2)" if regime == "no_demo"
                 else "demo-seeded (--demo-configs 8)")
        blocks += [f"### {regime} — {label}", "", _regime_table(summary, regime),
                   "", _interpretation(summary, regime), ""]
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
    p.add_argument("--version", choices=sorted(VERSIONS), default="v1_4")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--regimes", nargs="+", default=["no_demo", "demo_seeded"],
                   choices=["no_demo", "demo_seeded"])
    p.add_argument("--kappa", type=float, default=1.0,
                   help="R2 proxy-penalty coefficient (penalty = -kappa*bonus)")
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
    p.add_argument("--mode", choices=["full", "sr_only", "pr_only", "none"],
                   default="full")
    p.add_argument("--abstraction", choices=["milestone", "full"],
                   default="milestone")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--beta-decay", type=float, default=0.999)
    p.add_argument("--beta-end", type=float, default=0.0)
    p.add_argument("--sr-alpha", type=float, default=0.1)
    p.add_argument("--sr-gamma", type=float, default=0.95)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--results-dir", type=str, default="results/pg_spie_r2")
    p.add_argument("--no-report", action="store_true")
    cli = p.parse_args()

    out_dir = PROJECT_ROOT / cli.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for regime in cli.regimes:
        args = build_args(cli, REGIME_DEMO[regime])
        for display, algo_key in VARIANTS:
            for seed in cli.seeds:
                print(f"\n=== [{regime}] {display} | seed {seed} "
                      f"(kappa={cli.kappa}) ===", flush=True)
                row = run_one(cli.version, regime, display, algo_key, seed,
                              args, out_dir, cli.eval_window)
                runs.append(row)
                print(f"  succ={row['success_rate']:.2f} "
                      f"trap_hits={row['mean_trap_hits']:.2f} "
                      f"hack={row['reward_hack_rate']:.2f} "
                      f"gap={row['mean_alignment_gap']:.2f} "
                      f"pen(n/total)={row['proxy_penalty_count']}/"
                      f"{row['proxy_penalty_total']:.1f}", flush=True)

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(out_dir / "pg_spie_r2_runs.csv", index=False)
    summary = summarize(runs_df)
    summary.to_csv(out_dir / "pg_spie_r2_summary.csv", index=False)
    print("\n========== PG-SPIE R2 SUMMARY ==========")
    print(summary.to_string(index=False))
    print(f"\nartifacts -> {out_dir}")
    if not cli.no_report:
        fill_report(summary, cli)


if __name__ == "__main__":
    main()
