"""
Noisy x PER x SPIE ablation runner for AxiomForge.

A focused 2x2 ablation over a FIXED Noisy Dueling Double-DQN backbone
(double=ON, dueling=ON, noisy=ON), toggling only two exploration/learning
signals -- Prioritized Experience Replay (PER) and the SPIE/SBI intrinsic
bonus:

    variant                         use_per  use_spie   DEEP_ALGOS key
    1 noisy_dueling_ddqn              off       off      noisy_dueling_ddqn
    2 noisy_dueling_ddqn_per          on        off      noisy_dueling_ddqn_per
    3 noisy_dueling_ddqn_spie         off       on       spie_noisy_dueling_ddqn
    4 noisy_dueling_ddqn_per_spie     on        on       spie_noisy_dueling_ddqn_per

This is NOT a new algorithm. It drives the SAME configurable DQNAgent and the
SAME verified `train_seed_deep` training recipe used by the main benchmark; it
only fixes the backbone, sweeps PER/SPIE, and adds per-episode logging of the
two signals that `make_episode_row` cannot see (SPIE intrinsic contribution and
PER TD-error magnitude) via `train_seed_deep`'s opt-in `metrics_sink`.

Default environment is V1.4 in the misalignment ("E2") regime (no demos), the
only version that exposes the proxy-reward trap, hidden `true_score`, and
`alignment_gap` the hypothesis is about. All four variants share the same
version, training budget, seeds, and evaluation protocol; results are saved
separately per variant plus combined run/summary tables.

Run from the repo root, e.g.:

    # quick smoke (pipeline check, ~1 seed, tiny budget)
    python scripts/run_noisy_per_spi_ablation.py --episodes 30 --seeds 0 \
        --train-configs 4 --heldout-configs 4

    # the real ablation (3 seeds, V1.4 misalignment regime)
    python scripts/run_noisy_per_spi_ablation.py --episodes 400 --seeds 0 1 2
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

# Display name -> underlying DEEP_ALGOS key. The fixed backbone is
# double+dueling+noisy for all four; only per/spie vary.
VARIANTS = [
    ("noisy_dueling_ddqn",          "noisy_dueling_ddqn"),
    ("noisy_dueling_ddqn_per",      "noisy_dueling_ddqn_per"),
    ("noisy_dueling_ddqn_spie",     "spie_noisy_dueling_ddqn"),
    ("noisy_dueling_ddqn_per_spie", "spie_noisy_dueling_ddqn_per"),
]

REPORT_PATH = PROJECT_ROOT / "reports" / "noisy_per_spi_ablation.md"
RESULTS_START = "<!-- RESULTS:START -->"
RESULTS_END = "<!-- RESULTS:END -->"


def assert_backbone_is_fixed():
    """Guard: every variant must share double+dueling+noisy and differ ONLY
    in per/spie. Fails loudly if DEEP_ALGOS ever drifts from the spec."""
    for display, key in VARIANTS:
        sw = DEEP_ALGOS[key]
        assert sw["double"] and sw["dueling"] and sw["noisy"], (
            f"{display}: backbone must be double+dueling+noisy, got {sw}")
    by_signal = {(DEEP_ALGOS[k]["per"], DEEP_ALGOS[k]["spie"]): d
                 for d, k in VARIANTS}
    assert set(by_signal) == {(False, False), (True, False),
                              (False, True), (True, True)}, (
        "the four variants must be exactly the 2x2 of (per, spie)")


def build_args(cli) -> Namespace:
    """Assemble the Namespace that train_seed_deep consumes. Backbone-level
    knobs are fixed here; only the experiment budget is exposed on the CLI so
    all four variants are guaranteed identical except per/spie."""
    return Namespace(
        episodes=cli.episodes,
        gamma=cli.gamma, lr=cli.lr,
        batch_size=cli.batch_size, buffer_size=cli.buffer_size,
        target_update=cli.target_update, train_freq=cli.train_freq,
        pretrain_steps=cli.pretrain_steps,
        demo_configs=cli.demo_configs, demo_sweeps=cli.demo_sweeps,
        demo_fraction=cli.demo_fraction, margin=cli.margin,
        margin_weight=cli.margin_weight,
        train_configs=cli.train_configs, heldout_configs=cli.heldout_configs,
        epsilon_start=cli.epsilon_start, epsilon_end=cli.epsilon_end,
        epsilon_decay=cli.epsilon_decay,
        # SPIE/SBI intrinsic knobs (only consumed when a variant has spie ON).
        mode=cli.mode, abstraction=cli.abstraction,
        beta=cli.beta, beta_decay=cli.beta_decay, beta_end=cli.beta_end,
        sr_alpha=cli.sr_alpha, sr_gamma=cli.sr_gamma,
        device=cli.device, shaping=False, e2=True,
    )


def _num(series) -> pd.Series:
    """Coerce a possibly-object column (true_score is None pre-V1.4) to float."""
    return pd.to_numeric(series, errors="coerce")


def run_one(version, display, algo_key, seed, args, out_dir, window):
    """Train one (variant, seed) and return a flat metrics row."""
    sink: list[dict] = []
    df, agent, final_train, final_held, ev = train_seed_deep(
        version, algo_key, seed, args, metrics_sink=sink)

    vdir = out_dir / display
    vdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(vdir / f"seed_{seed}.csv", index=False)
    ev.to_csv(vdir / f"eval_history_seed_{seed}.csv", index=False)
    sink_df = pd.DataFrame(sink)
    sink_df.to_csv(vdir / f"signals_seed_{seed}.csv", index=False)

    tail = df.tail(window)
    switches = DEEP_ALGOS[algo_key]
    return {
        "variant": display,
        "algo_key": algo_key,
        "seed": seed,
        "use_per": switches["per"],
        "use_spie": switches["spie"],
        "episodes": args.episodes,
        # --- task performance ---
        # headline success = fraction of late-training episodes ending in a
        # correct final submission (greedy held-out is reported alongside).
        "success_rate": float(tail["success"].mean()),
        "train_greedy_success": float(final_train["success"].mean()),
        "heldout_greedy_success": float(final_held["success"].mean()),
        "final_correct_submissions": int(df["success"].sum()),
        # --- reward / behaviour ---
        "mean_final_reward": float(tail["episode_return"].mean()),
        "mean_episode_return": float(df["episode_return"].mean()),
        "mean_episode_length": float(df["steps"].mean()),
        # --- misalignment signals (V1.4) ---
        "mean_trap_hits": float(df["proxy_claim_count"].mean()),
        "total_trap_hits": int(df["proxy_claim_count"].sum()),
        "reward_hack_rate": float(df["reward_hack_flag"].mean()),
        "safety_violation_rate": float(df["safety_violation"].mean()),
        "mean_true_score": float(_num(df["true_score"]).mean()),
        "mean_alignment_gap": float(_num(df["alignment_gap"]).mean()),
        # --- exploration / learning internals ---
        "mean_intrinsic_contribution":
            float(sink_df["intrinsic_contribution"].mean())
            if not sink_df.empty else 0.0,
        "mean_td_abs": float(_num(sink_df["td_abs_mean"]).mean())
            if not sink_df.empty else float("nan"),
        "learn_steps": int(getattr(agent, "learn_steps", 0)),
    }


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-variant mean/std across seeds, preserving variant order."""
    agg = {
        "use_per": "first", "use_spie": "first",
        "success_rate": ["mean", "std"],
        "heldout_greedy_success": ["mean", "std"],
        "mean_trap_hits": ["mean", "std"],
        "reward_hack_rate": ["mean"],
        "mean_final_reward": ["mean", "std"],
        "mean_episode_length": ["mean"],
        "mean_true_score": ["mean"],
        "mean_alignment_gap": ["mean", "std"],
        "mean_intrinsic_contribution": ["mean"],
        "mean_td_abs": ["mean"],
    }
    order = [d for d, _ in VARIANTS]
    g = runs.groupby("variant", sort=False).agg(agg)
    g.columns = ["_".join(c).rstrip("_") for c in g.columns]
    g = g.reindex(order)
    g.insert(0, "n_seeds", runs.groupby("variant", sort=False).size()
             .reindex(order).astype(int))
    return g.reset_index()


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def summary_markdown(summary: pd.DataFrame) -> str:
    cols = [
        ("variant", "variant_name"), ("use_per_first", "use_per"),
        ("use_spie_first", "use_spi/sbi"),
        ("success_rate_mean", "mean_success"), ("success_rate_std", "std_success"),
        ("mean_trap_hits_mean", "mean_trap_hits"),
        ("reward_hack_rate_mean", "hack_rate"),
        ("mean_final_reward_mean", "mean_final_reward"),
        ("mean_episode_length_mean", "mean_ep_len"),
        ("mean_true_score_mean", "mean_true_score"),
        ("mean_alignment_gap_mean", "mean_align_gap"),
        ("mean_intrinsic_contribution_mean", "mean_intrinsic"),
        ("mean_td_abs_mean", "mean_td_abs"),
    ]
    header = "| " + " | ".join(label for _, label in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [header, sep]
    for _, row in summary.iterrows():
        cells = [_fmt(row.get(key)) for key, _ in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def interpretation_markdown(summary: pd.DataFrame) -> str:
    """Auto-derive the guided comparisons (v3-v1, v2-v1, v4-v3) so the report
    states what the run actually showed, mapped to the interpretation guide."""
    s = summary.set_index("variant")

    def get(variant, col):
        try:
            return float(s.loc[variant, col])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    v1, v2 = "noisy_dueling_ddqn", "noisy_dueling_ddqn_per"
    v3, v4 = "noisy_dueling_ddqn_spie", "noisy_dueling_ddqn_per_spie"

    def delta(a, b, col):
        return get(a, col) - get(b, col)

    lines = [
        "**Guided comparisons (computed from this run):**",
        "",
        f"- **SPIE effect** (v3 - v1): success "
        f"{_fmt(delta(v3, v1, 'success_rate_mean'))}, "
        f"align_gap {_fmt(delta(v3, v1, 'mean_alignment_gap_mean'))}, "
        f"trap_hits {_fmt(delta(v3, v1, 'mean_trap_hits_mean'))} "
        "(positive success => SPIE adds useful structure-seeking exploration).",
        f"- **PER on baseline** (v2 - v1): success "
        f"{_fmt(delta(v2, v1, 'success_rate_mean'))}, "
        f"align_gap {_fmt(delta(v2, v1, 'mean_alignment_gap_mean'))}, "
        f"trap_hits {_fmt(delta(v2, v1, 'mean_trap_hits_mean'))} "
        "(positive success => PER helps the noisy dueling DDQN baseline).",
        f"- **PER on SPIE** (v4 - v3): success "
        f"{_fmt(delta(v4, v3, 'success_rate_mean'))}, "
        f"align_gap {_fmt(delta(v4, v3, 'mean_alignment_gap_mean'))}, "
        f"trap_hits {_fmt(delta(v4, v3, 'mean_trap_hits_mean'))} "
        "(v3 > v4 on success, or v4 trap_hits > v3 => PER may be interfering "
        "with SPIE / over-prioritizing high-error trap transitions).",
    ]
    return "\n".join(lines)


def fill_report(summary: pd.DataFrame, cli, out_dir):
    """Replace the RESULTS block in reports/noisy_per_spi_ablation.md with the
    live summary table + auto interpretation. Leaves the explanatory prose
    (sections 1-4 + interpretation guide) untouched."""
    if not REPORT_PATH.exists():
        print(f"[warn] {REPORT_PATH} missing; skipping report fill.")
        return
    text = REPORT_PATH.read_text()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    rel = out_dir.relative_to(PROJECT_ROOT)
    block = "\n".join([
        RESULTS_START,
        "",
        f"_Generated {stamp} — version `{cli.version}`, "
        f"{cli.episodes} episodes, seeds {cli.seeds}, "
        f"demo_configs={cli.demo_configs} "
        f"({'misalignment/E2 no-demo' if cli.demo_configs == 0 else 'demo-seeded'} "
        f"regime). mean_success = final-{ROLLING_WINDOW}-episode training "
        "success; mean_align_gap = visible return − hidden true_score "
        "(higher = more hacking)._",
        "",
        summary_markdown(summary),
        "",
        interpretation_markdown(summary),
        "",
        f"Per-run rows: `{rel}/ablation_runs.csv` · "
        f"per-variant summary: `{rel}/ablation_summary.csv` · "
        f"per-variant episode/eval/signal CSVs under `{rel}/<variant>/`.",
        "",
        RESULTS_END,
    ])
    if RESULTS_START in text and RESULTS_END in text:
        pre = text.split(RESULTS_START)[0]
        post = text.split(RESULTS_END)[1]
        text = pre + block + post
    else:
        text = text.rstrip() + "\n\n## Results\n\n" + block + "\n"
    REPORT_PATH.write_text(text)
    print(f"report updated -> {REPORT_PATH}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", choices=sorted(VERSIONS), default="v1_4",
                   help="env version (default v1_4 — the only one with "
                        "proxy traps / true_score / alignment_gap)")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--train-configs", type=int, default=16)
    p.add_argument("--heldout-configs", type=int, default=8)
    p.add_argument("--demo-configs", type=int, default=0,
                   help="0 = misalignment/E2 regime (recommended); >0 seeds "
                        "demos from that many train configs")
    p.add_argument("--demo-sweeps", type=int, default=5)
    p.add_argument("--eval-window", type=int, default=ROLLING_WINDOW,
                   help="final-episode window for success_rate / mean reward")
    # backbone / optimizer knobs (shared by all four variants)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--target-update", type=int, default=1000)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--pretrain-steps", type=int, default=0)
    p.add_argument("--demo-fraction", type=float, default=0.25)
    p.add_argument("--margin", type=float, default=0.8)
    p.add_argument("--margin-weight", type=float, default=1.0)
    # noisy nets explore via parameter noise -> epsilon schedule is inert,
    # but train_seed_deep still reads these; keep them defined.
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.999)
    # SPIE/SBI knobs (consumed only by the two spie variants)
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
    p.add_argument("--results-dir", type=str,
                   default="results/ablation_noisy_per_spi")
    p.add_argument("--no-report", action="store_true",
                   help="skip updating reports/noisy_per_spi_ablation.md")
    cli = p.parse_args()

    assert_backbone_is_fixed()
    args = build_args(cli)
    out_dir = PROJECT_ROOT / cli.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for display, algo_key in VARIANTS:
        sw = DEEP_ALGOS[algo_key]
        for seed in cli.seeds:
            print(f"\n=== {cli.version} | {display} "
                  f"(per={sw['per']} spie={sw['spie']}) | seed {seed} ===",
                  flush=True)
            row = run_one(cli.version, display, algo_key, seed, args,
                          out_dir, cli.eval_window)
            runs.append(row)
            print(f"  success={row['success_rate']:.2f} "
                  f"heldout={row['heldout_greedy_success']:.2f} "
                  f"trap_hits={row['mean_trap_hits']:.2f} "
                  f"align_gap={row['mean_alignment_gap']:.2f} "
                  f"intrinsic={row['mean_intrinsic_contribution']:.3f} "
                  f"td_abs={row['mean_td_abs']:.3f}", flush=True)

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(out_dir / "ablation_runs.csv", index=False)
    summary = summarize(runs_df)
    summary.to_csv(out_dir / "ablation_summary.csv", index=False)

    print("\n========== NOISY x PER x SPIE ABLATION SUMMARY ==========")
    print(summary.to_string(index=False))
    print(f"\nartifacts -> {out_dir}")

    if not cli.no_report:
        fill_report(summary, cli, out_dir)


if __name__ == "__main__":
    main()
