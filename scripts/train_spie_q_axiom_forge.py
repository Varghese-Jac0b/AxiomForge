"""
SPIE-Q on AxiomForge V1.0: Successor-Predecessor Intrinsic Exploration
added to the verified tabular Q-learning baseline.

Design: docs/spie_q_design_note.md. The headline experiment is the
NO-DEMONSTRATION arm (--demo-sweeps 0 --mode full): can structure-aware
intrinsic exploration cross the V1.0 exploration wall that defeats
vanilla and PBRS-shaped Q-learning?

Discipline (frozen ROADMAP rules):
- intrinsic reward is TRAINING-ONLY: the CSV episode_return column logs
  the EXTRINSIC return; greedy evaluation reuses the canonical
  evaluate_greedy (clean config, no intrinsic, shaping off);
- beta anneals multiplicatively per episode toward --beta-end;
- `done` for bootstrapping is termination-only (truncation bootstraps);
- --mode none runs the IDENTICAL code path with zero bonus, so it is
  bit-for-bit vanilla Q-learning (the ablation control).

Ablation arms (Section: design note):
    --mode full | sr_only | pr_only | none
    --abstraction milestone | full

Run from the repo root:
    python scripts/train_spie_q_axiom_forge.py --episodes 5000 --demo-sweeps 0
    python scripts/train_spie_q_axiom_forge.py --mode none   # = baseline
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.spie_q_agent import SuccessorPredecessorTables, make_abstraction
from environments.axiom_forge_configs import make_v1_0_config
from environments.axiom_forge_env import AxiomForgeEnv
from scripts.axiom_forge_baselines_common import (
    ROLLING_WINDOW,
    TARGET_SUCCESS_RATE,
    encode_state,
    episodes_to_target,
    epsilon_greedy_action,
    evaluate_greedy,
    get_q_values,
    make_episode_row,
    plot_learning_curves_labeled,
)
from scripts.train_q_learning_axiom_forge import (
    generate_demo_transitions,
    seed_q_table_from_demo,
)

# The six observable milestones, in causal-chain order, read from the obs
# the same way the abstraction reads them (for first-discovery logging).
MILESTONE_NAMES = (
    "manifest_read", "archive_read", "diagnostics_read",
    "sample_present", "refined", "analyzed",
)


def milestone_flags(obs) -> tuple[bool, ...]:
    return (
        bool(int(obs["work_order"][0])),
        bool(int(obs["knowledge_bits"][0])),
        bool(int(obs["knowledge_bits"][1])),
        int(obs["sample_state"][0]) > 0,
        int(obs["sample_state"][1]) == 2,
        bool(int(obs["sample_state"][5])),
    )


def train_seed(seed, args):
    cfg = make_v1_0_config()
    if args.shaping:
        cfg = dataclasses.replace(cfg, shaping_enabled=True)
    env = AxiomForgeEnv(cfg)
    num_actions = env.action_space.n
    q_table: dict[tuple, np.ndarray] = {}
    if args.demo_sweeps > 0:
        transitions = generate_demo_transitions(cfg)
        seed_q_table_from_demo(q_table, transitions, args.gamma,
                               num_actions, args.demo_sweeps)

    abstract = make_abstraction(args.abstraction, encode_state)
    tables = SuccessorPredecessorTables(
        sr_alpha=args.sr_alpha, sr_gamma=args.sr_gamma,
        pr_alpha=args.sr_alpha, pr_gamma=args.sr_gamma,
    )

    rng = np.random.default_rng(seed)
    epsilon = args.epsilon_start
    beta = args.beta
    rows = []
    first_milestone_episode: dict[str, int | None] = {
        name: None for name in MILESTONE_NAMES}

    progress_interval = max(1, args.episodes // 10)

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset()
        state = encode_state(obs)
        abstract_state = abstract(obs)
        episode_return = 0.0          # EXTRINSIC only (CSV column)
        intrinsic_sum = 0.0
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action = epsilon_greedy_action(
                q_table, state, num_actions, epsilon, rng)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = encode_state(obs)
            next_abstract = abstract(obs)
            episode_return += reward
            steps += 1

            # SPIE: update SR/PR on the abstract chain, then add the
            # training-only bonus for ARRIVING in the next abstract state.
            tables.update(abstract_state, next_abstract)
            r_int = tables.bonus(next_abstract, mode=args.mode)
            intrinsic_sum += r_int
            r_total = reward + beta * r_int

            # Termination-only bootstrap (truncation bootstraps through).
            if terminated:
                target = r_total
            else:
                target = r_total + args.gamma * get_q_values(
                    q_table, next_state, num_actions).max()
            q_values = get_q_values(q_table, state, num_actions)
            q_values[action] += args.alpha * (target - q_values[action])
            state, abstract_state = next_state, next_abstract

            for name, hit in zip(MILESTONE_NAMES, milestone_flags(obs)):
                if hit and first_milestone_episode[name] is None:
                    first_milestone_episode[name] = episode

        rows.append(make_episode_row(
            episode_idx=episode, seed=seed, cfg=cfg,
            episode_return=episode_return, terminated=terminated,
            truncated=truncated, steps=steps, final_obs=obs, info=info,
        ))
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
        beta = max(args.beta_end, beta * args.beta_decay)

        if episode % progress_interval == 0 or episode == args.episodes:
            recent = rows[-min(ROLLING_WINDOW, len(rows)):]
            recent_success = np.mean([r["success"] for r in recent]) * 100
            recent_return = np.mean([r["episode_return"] for r in recent])
            print(
                f"seed {seed} | ep {episode:5d}/{args.episodes} | "
                f"eps={epsilon:.3f} beta={beta:.4f} | last-{len(recent)} "
                f"success={recent_success:5.1f}% return={recent_return:6.2f}"
                f" | q={len(q_table)} | sr_states={len(tables.M)} | "
                f"r_int/ep={intrinsic_sum:6.2f}",
                flush=True,
            )

    return pd.DataFrame(rows), q_table, tables, first_milestone_episode


def main():
    parser = argparse.ArgumentParser(
        description="Train SPIE-Q on AxiomForge V1.0.")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.998)
    # SPIE knobs
    parser.add_argument("--mode", choices=["full", "sr_only", "pr_only",
                                           "none"], default="full")
    parser.add_argument("--abstraction", choices=["milestone", "full"],
                        default="milestone")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Initial intrinsic-reward scale.")
    parser.add_argument("--beta-decay", type=float, default=0.999)
    parser.add_argument("--beta-end", type=float, default=0.0)
    parser.add_argument("--sr-alpha", type=float, default=0.1)
    parser.add_argument("--sr-gamma", type=float, default=0.95)
    # harness
    parser.add_argument("--shaping", action="store_true")
    parser.add_argument("--demo-sweeps", type=int, default=0,
                        help="0 = the headline no-demonstration arm.")
    parser.add_argument("--results-dir", type=str,
                        default="results/V1.0/spie_q")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    seed_frames: dict[int, pd.DataFrame] = {}
    summary = []
    for seed in args.seeds:
        print(f"\n=== training SPIE-Q seed {seed} "
              f"(mode={args.mode}, abstraction={args.abstraction}, "
              f"demo_sweeps={args.demo_sweeps}) ===", flush=True)
        df, q_table, tables, milestones = train_seed(seed, args)
        csv_path = results_dir / f"seed_{seed}.csv"
        df.to_csv(csv_path, index=False)
        final = df.tail(ROLLING_WINDOW)
        target_ep = episodes_to_target(df)
        eval_success = evaluate_greedy(q_table)   # beta=0, clean config
        entry = {
            "seed": seed,
            "mode": args.mode,
            "abstraction": args.abstraction,
            "demo_sweeps": args.demo_sweeps,
            "final_success_rate": final["success"].mean(),
            "final_mean_return": final["episode_return"].mean(),
            "final_mean_steps": final["steps"].mean(),
            "episodes_to_90pct": target_ep,
            "env_steps_to_90pct":
                int(df["steps"][:target_ep].sum()) if target_ep else None,
            "greedy_eval_success": eval_success,
            "q_table_size": len(q_table),
            "sr_states": len(tables.M),
        }
        for name, ep in milestones.items():
            entry[f"first_{name}_ep"] = ep
        summary.append(entry)
        seed_frames[seed] = df
        print(f"saved {csv_path} | greedy eval {eval_success:.2f} | "
              f"milestone discovery: {milestones}")

    plot_path = results_dir / "learning_curve.png"
    plot_learning_curves_labeled(
        seed_frames, plot_path,
        f"SPIE-Q ({args.mode}/{args.abstraction})")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(results_dir / "summary.csv", index=False)

    print("\n========== SPIE-Q V1.0 SUMMARY ==========")
    print(summary_df.to_string(index=False))
    reached = summary_df["final_success_rate"] > TARGET_SUCCESS_RATE
    print(f"\nseeds reaching >90% final success: {int(reached.sum())}"
          f"/{len(summary_df)}")
    print(f"learning curves -> {plot_path}")


if __name__ == "__main__":
    main()
