"""
AxiomForge V1.0 tabular SARSA baseline.

Same harness as the verified Q-learning baseline (Section-7 state key,
Section-15 CSV schema, optional PBRS shaping, demonstration seeding,
greedy evaluation, 5-seed runner) with the on-policy SARSA update:

    target = r + gamma * Q(s', a')        a' = the action actually taken next

Bootstrap-through-truncation (audit fix, Build Record Section 3) applies:
hitting max_steps is a time limit, not a terminal state, so the update
bootstraps from Q(s', a') where a' is the epsilon-greedy action that
WOULD have been taken; only true termination (submission at G) writes
target = reward.

Demonstration seeding replays the manual-solver trajectory in reverse
sweeps with the on-policy target along the demo: the "next action" is
literally the next demonstrated action, which is exactly what SARSA
would bootstrap from when following the demo policy.

Run from the repo root:
    python scripts/train_sarsa_axiom_forge.py --episodes 3000 --demo-sweeps 50
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

from environments.axiom_forge_configs import make_v1_0_config
from environments.axiom_forge_env import AxiomForgeEnv
from scripts.axiom_forge_baselines_common import (
    DEMO_SEED_ALPHA,
    ROLLING_WINDOW,
    TARGET_SUCCESS_RATE,
    encode_state,
    episodes_to_target,
    epsilon_greedy_action,
    evaluate_greedy,
    generate_demo_transitions_encoded,
    get_q_values,
    make_episode_row,
    plot_learning_curves_labeled,
)


def seed_q_table_from_demo_sarsa(q_table, transitions, gamma, num_actions,
                                 sweeps):
    """Reverse sweeps over the demo with the SARSA target: bootstrap from
    the NEXT DEMONSTRATED action (the on-policy choice along the demo).
    The final transition is true termination (submission), so its target
    is the bare reward."""
    for _ in range(sweeps):
        for i in range(len(transitions) - 1, -1, -1):
            state, action, reward, next_state, terminated, _ = transitions[i]
            if terminated:
                target = reward
            else:
                next_action = transitions[i + 1][1]
                target = reward + gamma * get_q_values(
                    q_table, next_state, num_actions,
                )[next_action]
            q_values = get_q_values(q_table, state, num_actions)
            q_values[action] += DEMO_SEED_ALPHA * (target - q_values[action])


def train_seed(
    seed, num_episodes, alpha, gamma,
    epsilon_start, epsilon_end, epsilon_decay, shaping=False,
    demo_sweeps=0,
):
    cfg = make_v1_0_config()
    if shaping:
        cfg = dataclasses.replace(cfg, shaping_enabled=True)
    env = AxiomForgeEnv(cfg)
    num_actions = env.action_space.n
    q_table: dict[tuple[int, ...], np.ndarray] = {}
    if demo_sweeps > 0:
        transitions = generate_demo_transitions_encoded(cfg, encode_state)
        seed_q_table_from_demo_sarsa(q_table, transitions, gamma, num_actions,
                                     demo_sweeps)
    rng = np.random.default_rng(seed)
    epsilon = epsilon_start
    rows = []

    progress_interval = max(1, num_episodes // 10)

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        state = encode_state(obs)
        # On-policy: the action is committed before the step and the next
        # action is chosen (and then executed) inside the loop.
        action = epsilon_greedy_action(q_table, state, num_actions, epsilon,
                                       rng)
        episode_return = 0.0
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = encode_state(obs)
            episode_return += reward
            steps += 1

            if terminated:
                target = reward
                next_action = 0  # unused
            else:
                # Bootstrap through truncation: choose a' epsilon-greedily
                # and bootstrap from Q(s', a') even when the clock expires.
                next_action = epsilon_greedy_action(
                    q_table, next_state, num_actions, epsilon, rng,
                )
                target = reward + gamma * get_q_values(
                    q_table, next_state, num_actions,
                )[next_action]
            q_values = get_q_values(q_table, state, num_actions)
            q_values[action] += alpha * (target - q_values[action])
            state, action = next_state, next_action

        rows.append(make_episode_row(
            episode_idx=episode, seed=seed, cfg=cfg,
            episode_return=episode_return, terminated=terminated,
            truncated=truncated, steps=steps, final_obs=obs, info=info,
        ))
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % progress_interval == 0 or episode == num_episodes:
            recent = rows[-min(ROLLING_WINDOW, len(rows)):]
            recent_success = np.mean([r["success"] for r in recent]) * 100
            recent_return = np.mean([r["episode_return"] for r in recent])
            print(
                f"seed {seed} | ep {episode:5d}/{num_episodes} | "
                f"eps={epsilon:.3f} | last-{len(recent)} "
                f"success={recent_success:5.1f}% return={recent_return:6.2f} | "
                f"q_table={len(q_table)}",
                flush=True,
            )

    return pd.DataFrame(rows), q_table


def main():
    parser = argparse.ArgumentParser(
        description="Train tabular SARSA on AxiomForge V1.0.",
    )
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.998)
    parser.add_argument("--shaping", action="store_true",
                        help="Enable potential-based shaping during training "
                             "(greedy eval always runs with shaping off).")
    parser.add_argument("--demo-sweeps", type=int, default=0,
                        help="Seed the Q-table with N reverse sweeps over "
                             "one manual-solver demonstration before "
                             "training (0 = vanilla SARSA).")
    parser.add_argument("--results-dir", type=str,
                        default="results/V1.0/sarsa")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    seed_frames: dict[int, pd.DataFrame] = {}
    summary = []
    for seed in args.seeds:
        print(f"\n=== training seed {seed} ===", flush=True)
        df, q_table = train_seed(
            seed=seed, num_episodes=args.episodes, alpha=args.alpha,
            gamma=args.gamma, epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end, epsilon_decay=args.epsilon_decay,
            shaping=args.shaping, demo_sweeps=args.demo_sweeps,
        )
        csv_path = results_dir / f"seed_{seed}.csv"
        df.to_csv(csv_path, index=False)
        final = df.tail(ROLLING_WINDOW)
        target_ep = episodes_to_target(df)
        eval_success = evaluate_greedy(q_table)
        summary.append({
            "seed": seed,
            "final_success_rate": final["success"].mean(),
            "final_mean_return": final["episode_return"].mean(),
            "final_mean_steps": final["steps"].mean(),
            "episodes_to_90pct": target_ep,
            "env_steps_to_90pct":
                int(df["steps"][:target_ep].sum()) if target_ep else None,
            "greedy_eval_success": eval_success,
            "q_table_size": len(q_table),
        })
        seed_frames[seed] = df
        print(f"saved {csv_path}")

    plot_path = results_dir / "learning_curve.png"
    plot_learning_curves_labeled(seed_frames, plot_path, "SARSA")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(results_dir / "summary.csv", index=False)

    print("\n========== SARSA V1.0 SUMMARY ==========")
    print(summary_df.to_string(index=False))
    reached = summary_df["final_success_rate"] > TARGET_SUCCESS_RATE
    print(f"\nseeds reaching >90% final success: {int(reached.sum())}"
          f"/{len(summary_df)}")
    print(f"learning curves -> {plot_path}")


if __name__ == "__main__":
    main()
