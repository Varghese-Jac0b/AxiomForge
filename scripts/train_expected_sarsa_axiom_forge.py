"""
AxiomForge V1.0 tabular Expected SARSA baseline.

Same harness as the verified Q-learning baseline (Section-7 state key,
Section-15 CSV schema, optional PBRS shaping, demonstration seeding,
greedy evaluation, 5-seed runner) with the Expected SARSA update:

    target = r + gamma * sum_a pi(a|s') * Q(s', a)

where pi is the current epsilon-greedy policy (each action gets
epsilon/|A|; the greedy action(s) share the remaining 1-epsilon).

Bootstrap-through-truncation (audit fix, Build Record Section 3) applies:
only true termination (submission at G) writes target = reward; the
max_steps clock expiring bootstraps from the expected next-state value.

Demonstration seeding uses greedy (max) targets - the epsilon -> 0 limit
of the expected target - identical to the official Q-learning seeding,
because seeding approximates the value of following the optimal demo
path, not the behaviour policy at epsilon_start.

Run from the repo root:
    python scripts/train_expected_sarsa_axiom_forge.py --episodes 3000 --demo-sweeps 50
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


def expected_q_value(q_table, state, num_actions, epsilon) -> float:
    """Expected next-state value under the current epsilon-greedy policy.

    Every action gets epsilon/|A| probability; the greedy action(s) share
    the remaining (1 - epsilon) mass (ties split evenly so probabilities
    sum to 1).
    """
    q_values = get_q_values(q_table, state, num_actions)
    action_probs = np.full(num_actions, epsilon / num_actions,
                           dtype=np.float64)
    best_value = q_values.max()
    best_actions = np.flatnonzero(q_values == best_value)
    action_probs[best_actions] += (1.0 - epsilon) / len(best_actions)
    return float(np.dot(action_probs, q_values))


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
        # Greedy-target seeding (epsilon -> 0 limit of the expected target),
        # shared with the official Q-learning baseline.
        transitions = generate_demo_transitions(cfg)
        seed_q_table_from_demo(q_table, transitions, gamma, num_actions,
                               demo_sweeps)
    rng = np.random.default_rng(seed)
    epsilon = epsilon_start
    rows = []

    progress_interval = max(1, num_episodes // 10)

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        state = encode_state(obs)
        episode_return = 0.0
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action = epsilon_greedy_action(
                q_table, state, num_actions, epsilon, rng,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = encode_state(obs)
            episode_return += reward
            steps += 1

            # Bootstrap through truncation: only true termination
            # (submission at G) stops the backup chain.
            if terminated:
                target = reward
            else:
                target = reward + gamma * expected_q_value(
                    q_table, next_state, num_actions, epsilon,
                )
            q_values = get_q_values(q_table, state, num_actions)
            q_values[action] += alpha * (target - q_values[action])
            state = next_state

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
        description="Train tabular Expected SARSA on AxiomForge V1.0.",
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
                             "training (0 = vanilla Expected SARSA).")
    parser.add_argument("--results-dir", type=str,
                        default="results/V1.0/expected_sarsa")
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
    plot_learning_curves_labeled(seed_frames, plot_path, "Expected SARSA")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(results_dir / "summary.csv", index=False)

    print("\n========== EXPECTED SARSA V1.0 SUMMARY ==========")
    print(summary_df.to_string(index=False))
    reached = summary_df["final_success_rate"] > TARGET_SUCCESS_RATE
    print(f"\nseeds reaching >90% final success: {int(reached.sum())}"
          f"/{len(summary_df)}")
    print(f"learning curves -> {plot_path}")


if __name__ == "__main__":
    main()
