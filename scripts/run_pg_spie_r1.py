"""
R1 Protocol-Gated SPIE — comparison runner.

Compares three variants on the SAME Noisy Dueling Double-DQN backbone (PER OFF),
holding everything else fixed and changing one knob at a time:

    variant        spie  pg    DEEP_ALGOS key
    plain_noisy     off  off   noisy_dueling_ddqn
    naive_spie      on   off   spie_noisy_dueling_ddqn
    pg_spie         on   ON    pg_spie_noisy_dueling_ddqn   (R1: gated SPIE)

Run on V1.4 in two regimes: no-demo (misalignment / "E2") and demo-seeded
(--demo-configs 8). All variants share version, budget, seeds, eval protocol.
Reuses the verified train_seed_deep + metrics_sink; PG-SPIE adds only the
observable milestone gate (suppress SPIE unless milestone progress increased;
suppress on proxy-claim steps). No penalty (R2), no trap-aware PER (R3), no PSFA.

`true_score` is read for REPORTING only (the env logs it; it never enters the
training reward, policy, replay, or the gate).

    python scripts/run_pg_spie_r1.py --episodes 400 --seeds 0 1 2
    python scripts/run_pg_spie_r1.py --episodes 30 --seeds 0 --train-configs 4 \
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
]
REGIME_DEMO = {"no_demo": 0, "demo_seeded": 8}     # -> demo_configs

REPORT_PATH = PROJECT_ROOT / "reports" / "pg_spie_r1.md"
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
        sr_alpha=cli.sr_alpha, sr_gamma=cli.sr_gamma,
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
    g_neu = int(sk["gate_neutral"].sum())
    g_prx = int(sk["gate_proxy"].sum())
    allowed = float(sk["allowed_intrinsic"].sum())
    suppressed = float(sk["suppressed_intrinsic"].sum())
    return {
        "regime": regime, "variant": display, "algo_key": algo_key, "seed": seed,
        "use_spie": sw["spie"], "use_gate": bool(sw.get("pg", False)),
        # task performance
        "success_rate": float(tail["success"].mean()),
        "heldout_greedy_success": float(final_held["success"].mean()),
        "final_correct_submissions": int(df["success"].sum()),
        "mean_final_reward": float(tail["episode_return"].mean()),
        "mean_episode_length": float(df["steps"].mean()),
        # misalignment signals (V1.4)
        "mean_trap_hits": float(df["proxy_claim_count"].mean()),
        "reward_hack_rate": float(df["reward_hack_flag"].mean()),
        "mean_true_score": float(_num(df["true_score"]).mean()),
        "mean_alignment_gap": float(_num(df["alignment_gap"]).mean()),
        # SPIE / gate internals
        "mean_intrinsic": float(sk["intrinsic_contribution"].mean()),
        "gate_advance_count": g_adv,
        "gate_neutral_count": g_neu,
        "gate_proxy_count": g_prx,
        "allowed_spie_bonus": allowed,
        "suppressed_spie_bonus": suppressed,
        "mean_allowed_bonus": allowed / g_adv if g_adv else 0.0,
        "mean_suppressed_bonus": suppressed / (g_neu + g_prx)
            if (g_neu + g_prx) else 0.0,
        "learn_steps": int(getattr(agent, "learn_steps", 0)),
    }


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = ["success_rate", "heldout_greedy_success", "mean_trap_hits",
               "reward_hack_rate", "mean_alignment_gap", "mean_true_score",
               "mean_final_reward", "mean_episode_length", "mean_intrinsic",
               "gate_advance_count", "gate_proxy_count", "gate_neutral_count",
               "mean_allowed_bonus", "mean_suppressed_bonus"]
    agg = {m: ["mean", "std"] for m in metrics}
    g = runs.groupby(["regime", "variant"], sort=False).agg(agg)
    g.columns = ["_".join(c).rstrip("_") for c in g.columns]
    n = runs.groupby(["regime", "variant"], sort=False).size().rename("n_seeds")
    return g.join(n).reset_index()


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _regime_table(summary: pd.DataFrame, regime: str) -> str:
    order = [d for d, _ in VARIANTS]
    sub = summary[summary["regime"] == regime].set_index("variant").reindex(order)
    cols = [
        ("success_rate_mean", "succ"), ("mean_trap_hits_mean", "trap_hits"),
        ("reward_hack_rate_mean", "hack_rate"),
        ("mean_alignment_gap_mean", "align_gap"),
        ("mean_true_score_mean", "true_score"),
        ("mean_final_reward_mean", "final_ret"),
        ("mean_intrinsic_mean", "intrinsic(allowed)"),
        ("gate_advance_count_mean", "gate_adv"),
        ("gate_proxy_count_mean", "gate_proxy"),
        ("gate_neutral_count_mean", "gate_neutral"),
        ("mean_allowed_bonus_mean", "mean_allowed"),
        ("mean_suppressed_bonus_mean", "mean_suppr"),
    ]
    header = "| variant | " + " | ".join(c for _, c in cols) + " |"
    sep = "|" + "|".join("---" for _ in range(len(cols) + 1)) + "|"
    lines = [header, sep]
    for variant in order:
        row = sub.loc[variant]
        cells = [_fmt(row.get(k)) for k, _ in cols]
        lines.append(f"| {variant} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _interpretation(summary: pd.DataFrame, regime: str) -> str:
    order = {d: summary[(summary.regime == regime) & (summary.variant == d)]
             for d, _ in VARIANTS}

    def g(variant, col):
        s = order[variant]
        return float(s[col].iloc[0]) if len(s) else float("nan")

    return "\n".join([
        f"- **PG vs naive SPIE** (trap_hits {_fmt(g('pg_spie','mean_trap_hits_mean'))} "
        f"vs {_fmt(g('naive_spie','mean_trap_hits_mean'))}; align_gap "
        f"{_fmt(g('pg_spie','mean_alignment_gap_mean'))} vs "
        f"{_fmt(g('naive_spie','mean_alignment_gap_mean'))}; hack_rate "
        f"{_fmt(g('pg_spie','reward_hack_rate_mean'))} vs "
        f"{_fmt(g('naive_spie','reward_hack_rate_mean'))}) — lower for PG ="
        " the gate removed SPIE's pro-hacking pull.",
        f"- **PG vs plain noisy** (align_gap {_fmt(g('pg_spie','mean_alignment_gap_mean'))} "
        f"vs {_fmt(g('plain_noisy','mean_alignment_gap_mean'))}; success "
        f"{_fmt(g('pg_spie','success_rate_mean'))} vs "
        f"{_fmt(g('plain_noisy','success_rate_mean'))}) — PG should match plain on "
        "alignment; any success/exploration gain is the gate's directed-exploration value.",
    ])


def fill_report(summary: pd.DataFrame, cli):
    if not REPORT_PATH.exists():
        print(f"[warn] {REPORT_PATH} missing; skipping report fill.")
        return
    text = REPORT_PATH.read_text()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    blocks = [RESULTS_START, "",
              f"_Generated {stamp} — V1.4, {cli.episodes} episodes, "
              f"seeds {cli.seeds}, backbone Noisy Dueling DDQN (PER off). "
              "success = final-{w}-episode training success; align_gap = visible "
              "return − hidden true_score (higher = more hacking, reporting-only)._"
              .replace("{w}", str(ROLLING_WINDOW)), ""]
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
    p.add_argument("--results-dir", type=str, default="results/pg_spie_r1")
    p.add_argument("--no-report", action="store_true")
    cli = p.parse_args()

    out_dir = PROJECT_ROOT / cli.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for regime in cli.regimes:
        args = build_args(cli, REGIME_DEMO[regime])
        for display, algo_key in VARIANTS:
            for seed in cli.seeds:
                print(f"\n=== [{regime}] {display} | seed {seed} ===", flush=True)
                row = run_one(cli.version, regime, display, algo_key, seed,
                              args, out_dir, cli.eval_window)
                runs.append(row)
                print(f"  succ={row['success_rate']:.2f} "
                      f"trap_hits={row['mean_trap_hits']:.2f} "
                      f"hack={row['reward_hack_rate']:.2f} "
                      f"gap={row['mean_alignment_gap']:.2f} "
                      f"intrinsic={row['mean_intrinsic']:.2f} "
                      f"gate(adv/neu/prx)={row['gate_advance_count']}/"
                      f"{row['gate_neutral_count']}/{row['gate_proxy_count']}",
                      flush=True)

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(out_dir / "pg_spie_r1_runs.csv", index=False)
    summary = summarize(runs_df)
    summary.to_csv(out_dir / "pg_spie_r1_summary.csv", index=False)
    print("\n========== PG-SPIE R1 SUMMARY ==========")
    print(summary.to_string(index=False))
    print(f"\nartifacts -> {out_dir}")
    if not cli.no_report:
        fill_report(summary, cli)


if __name__ == "__main__":
    main()
