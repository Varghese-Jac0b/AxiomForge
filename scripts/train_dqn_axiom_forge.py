"""
AxiomForge V1.0 deep Q-learning baselines: DQN, DDQN, Dueling DQN, and
Dueling DDQN + PER, selected with --variant.

    --variant dqn               vanilla DQN (Mnih et al. 2015)
    --variant ddqn              Double DQN target
    --variant dueling_dqn       Dueling architecture, vanilla target
    --variant dueling_ddqn_per  Dueling + Double + prioritized replay

Same harness as the tabular baselines: Section-15 CSV schema per episode,
multi-seed runner, learning-curve plot, greedy evaluation with shaping
off. Demonstrations follow the official V1.0 baseline recipe (the
exploration wall makes vanilla deep RL fail exactly like vanilla
Q-learning): the manual-solver trajectory is injected into a protected
replay region, the agent pretrains on it with a DQfD margin loss, and a
demo share persists in every batch (guaranteed fraction for uniform
replay, priority bonus for PER). Pass --demo-repeats 0 for the vanilla
no-demonstration arm.

`done` for bootstrapping is termination only - truncation (max_steps)
bootstraps through, matching the audited tabular fix.

Run from the repo root:
    python scripts/train_dqn_axiom_forge.py --variant dqn --episodes 300
    python scripts/train_dqn_axiom_forge.py --variant dueling_ddqn_per \
        --episodes 300 --seeds 0 1 2
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

from agents.dqn_axiom_forge import DQNAgent, make_obs_encoder
from environments.axiom_forge_configs import make_v1_0_config
from environments.axiom_forge_env import AxiomForgeEnv
from scripts.axiom_forge_baselines_common import (
    ROLLING_WINDOW,
    TARGET_SUCCESS_RATE,
    episodes_to_target,
    generate_demo_transitions_encoded,
    make_episode_row,
    plot_learning_curves_labeled,
)

# variant -> (double, dueling, per)
VARIANTS: dict[str, tuple[bool, bool, bool]] = {
    "dqn": (False, False, False),
    "ddqn": (True, False, False),
    "dueling_dqn": (False, True, False),
    "dueling_ddqn_per": (True, True, True),
}

VARIANT_LABELS = {
    "dqn": "DQN",
    "ddqn": "Double DQN",
    "dueling_dqn": "Dueling DQN",
    "dueling_ddqn_per": "Dueling DDQN + PER",
}


def evaluate_greedy_net(agent: DQNAgent, encode, episodes: int = 20) -> float:
    """Greedy (eps=0) evaluation on the clean config: shaping always OFF
    so eval numbers are uncontaminated (Section 11 hard rule)."""
    env = AxiomForgeEnv(make_v1_0_config())
    successes = 0
    for _ in range(episodes):
        obs, info = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(encode(obs), epsilon=0.0)
            obs, _, terminated, truncated, info = env.step(action)
        successes += int(info["success"])
    return successes / episodes


def train_seed(seed: int, args) -> tuple[pd.DataFrame, DQNAgent, object]:
    double, dueling, per = VARIANTS[args.variant]
    cfg = make_v1_0_config()
    if args.shaping:
        cfg = dataclasses.replace(cfg, shaping_enabled=True)
    env = AxiomForgeEnv(cfg)
    encode, obs_dim = make_obs_encoder(env.observation_space)
    num_actions = env.action_space.n

    demo_transitions = []
    if args.demo_repeats > 0:
        demo_transitions = generate_demo_transitions_encoded(cfg, encode)
    demo_capacity = len(demo_transitions) * args.demo_repeats

    agent = DQNAgent(
        obs_dim, num_actions,
        double=double, dueling=dueling, per=per,
        gamma=args.gamma, lr=args.lr,
        buffer_capacity=args.buffer_size, demo_capacity=demo_capacity,
        batch_size=args.batch_size,
        target_update_interval=args.target_update,
        demo_fraction=args.demo_fraction, margin=args.margin,
        margin_weight=args.margin_weight,
        device=args.device, seed=seed,
    )

    for _ in range(args.demo_repeats):
        for state, action, reward, next_state, terminated, _ in \
                demo_transitions:
            agent.store_demo(state, action, reward, next_state, terminated)

    if demo_capacity > 0 and args.pretrain_steps > 0:
        for step in range(args.pretrain_steps):
            agent.learn()
        print(f"seed {seed} | pretrained {args.pretrain_steps} steps on "
              f"{demo_capacity} demo transitions | greedy eval "
              f"{evaluate_greedy_net(agent, encode, 5):.2f}", flush=True)

    epsilon = args.epsilon_start
    rows = []
    progress_interval = max(1, args.episodes // 10)

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset()
        state = encode(obs)
        episode_return = 0.0
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.act(state, epsilon)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = encode(obs)
            episode_return += reward
            steps += 1
            # Store termination only: truncation bootstraps through.
            agent.store(state, action, reward, next_state, terminated)
            if steps % args.train_freq == 0:
                agent.learn()
            state = next_state

        rows.append(make_episode_row(
            episode_idx=episode, seed=seed, cfg=cfg,
            episode_return=episode_return, terminated=terminated,
            truncated=truncated, steps=steps, final_obs=obs, info=info,
        ))
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        if episode % progress_interval == 0 or episode == args.episodes:
            recent = rows[-min(ROLLING_WINDOW, len(rows)):]
            recent_success = np.mean([r["success"] for r in recent]) * 100
            recent_return = np.mean([r["episode_return"] for r in recent])
            print(
                f"seed {seed} | ep {episode:5d}/{args.episodes} | "
                f"eps={epsilon:.3f} | last-{len(recent)} "
                f"success={recent_success:5.1f}% return={recent_return:6.2f} | "
                f"learn_steps={agent.learn_steps}",
                flush=True,
            )

    return pd.DataFrame(rows), agent, encode


def main():
    parser = argparse.ArgumentParser(
        description="Train the DQN family on AxiomForge V1.0.",
    )
    parser.add_argument("--variant", choices=sorted(VARIANTS),
                        default="dqn")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--target-update", type=int, default=1000,
                        help="Hard target-network sync interval "
                             "(in learn steps).")
    parser.add_argument("--train-freq", type=int, default=1,
                        help="Gradient step every N env steps.")
    parser.add_argument("--epsilon-start", type=float, default=0.2,
                        help="Default assumes demonstrations; use 1.0 for "
                             "the vanilla no-demo arm.")
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--shaping", action="store_true",
                        help="Enable potential-based shaping during training "
                             "(greedy eval always runs with shaping off).")
    parser.add_argument("--demo-repeats", type=int, default=5,
                        help="Times the manual-solver trajectory is "
                             "injected into the protected replay region "
                             "(0 = vanilla, no demonstrations).")
    parser.add_argument("--pretrain-steps", type=int, default=2000,
                        help="Gradient steps on demo-only replay before "
                             "any environment interaction.")
    parser.add_argument("--demo-fraction", type=float, default=0.25,
                        help="Guaranteed demo share per batch (uniform "
                             "replay only; PER uses its priority bonus).")
    parser.add_argument("--margin", type=float, default=0.8,
                        help="DQfD large-margin value.")
    parser.add_argument("--margin-weight", type=float, default=1.0,
                        help="Weight of the margin loss (demo samples only).")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu",
                        help='"cpu", "mps", or "auto".')
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Default: results/V1.0/<variant>")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / (args.results_dir
                                  or f"results/V1.0/{args.variant}")
    results_dir.mkdir(parents=True, exist_ok=True)
    label = VARIANT_LABELS[args.variant]

    seed_frames: dict[int, pd.DataFrame] = {}
    summary = []
    for seed in args.seeds:
        print(f"\n=== training {label} seed {seed} ===", flush=True)
        df, agent, encode = train_seed(seed, args)
        csv_path = results_dir / f"seed_{seed}.csv"
        df.to_csv(csv_path, index=False)
        final = df.tail(ROLLING_WINDOW)
        target_ep = episodes_to_target(df)
        eval_success = evaluate_greedy_net(agent, encode, args.eval_episodes)
        summary.append({
            "seed": seed,
            "variant": args.variant,
            "final_success_rate": final["success"].mean(),
            "final_mean_return": final["episode_return"].mean(),
            "final_mean_steps": final["steps"].mean(),
            "episodes_to_90pct": target_ep,
            "env_steps_to_90pct":
                int(df["steps"][:target_ep].sum()) if target_ep else None,
            "greedy_eval_success": eval_success,
            "learn_steps": agent.learn_steps,
        })
        seed_frames[seed] = df
        print(f"saved {csv_path} | greedy eval success {eval_success:.2f}")

    plot_path = results_dir / "learning_curve.png"
    plot_learning_curves_labeled(seed_frames, plot_path, label)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(results_dir / "summary.csv", index=False)

    print(f"\n========== {label.upper()} V1.0 SUMMARY ==========")
    print(summary_df.to_string(index=False))
    reached = summary_df["final_success_rate"] > TARGET_SUCCESS_RATE
    print(f"\nseeds reaching >90% final success: {int(reached.sum())}"
          f"/{len(summary_df)}")
    print(f"learning curves -> {plot_path}")


if __name__ == "__main__":
    main()
