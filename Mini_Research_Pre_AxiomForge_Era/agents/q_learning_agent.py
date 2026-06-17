"""
Tabular Q-learning agent for Mini Research World.

Run from the project root:

    python agents/q_learning_agent.py
    python agents/q_learning_agent.py --episodes 1000 --seed 42
    python -m agents.q_learning_agent --episodes 1000 --seed 42
    python agents/q_learning_agent.py --episodes 1000 --save-csv results/q_learning_training.csv
    python agents/q_learning_agent.py --episodes 5000 --save-qtable results/q_learning_qtable.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Project import setup
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from environments.mini_world_env import MiniResearchWorldEnv, ACTION_NAMES


# ---------------------------------------------------------------------
# Argument validation helpers
# ---------------------------------------------------------------------

def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def non_negative_float(value):
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def probability(value):
    parsed = float(value)
    if parsed < 0 or parsed > 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


# ---------------------------------------------------------------------
# Q-table helpers
# ---------------------------------------------------------------------

def get_q_values(q_table, state, num_actions):
    """
    Return the Q-value vector for a state.

    The Q-table maps state tuples to NumPy arrays of action values.
    If the state is new, initialize it with zeros.
    """

    if state not in q_table:
        q_table[state] = np.zeros(num_actions, dtype=np.float64)

    return q_table[state]


def epsilon_greedy_action(q_table, state, env, epsilon, rng):
    """
    Choose an action using epsilon-greedy policy.

    With probability epsilon, pick a random action.
    Otherwise pick the action with the highest Q-value.
    Ties are broken randomly so the agent does not get stuck on one action.
    """

    num_actions = env.action_space.n

    # Explore: random action.
    if rng.random() < epsilon:
        return int(rng.integers(0, num_actions))

    # Exploit: best action(s), with random tie-breaking.
    q_values = get_q_values(q_table, state, num_actions)
    best_value = q_values.max()
    best_actions = np.flatnonzero(q_values == best_value)

    return int(rng.choice(best_actions))


# ---------------------------------------------------------------------
# Single episode training
# ---------------------------------------------------------------------

def train_q_learning_episode(
    env, q_table, alpha, gamma, epsilon, rng, render=False
):
    """
    Train one Q-learning episode and update the Q-table on every step.

    Q-learning is off-policy: it bootstraps from the greedy (best) action
    of the next state, regardless of which action the policy actually takes:

        target = reward + gamma * max_a Q(next_state, a)

    At the terminal step, there is no future reward:

        target = reward

    Returns:
        dict containing episode-level statistics.
    """

    obs, info = env.reset()
    state = tuple(obs)
    num_actions = env.action_space.n

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    if render:
        print("\nInitial state:")
        env.render()

    while not terminated and not truncated:
        # Choose the action for the current state.
        action = epsilon_greedy_action(q_table, state, env, epsilon, rng)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = tuple(next_obs)

        total_reward += reward
        steps += 1

        done = terminated or truncated

        if not done:
            # Q-learning: bootstrap from the best next action (greedy),
            # not the action the policy will actually take.
            next_q = get_q_values(q_table, next_state, num_actions)
            target = reward + gamma * next_q.max()
        else:
            # Terminal step: no future reward to bootstrap from.
            target = reward

        # Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a))
        q_values = get_q_values(q_table, state, num_actions)
        q_values[action] += alpha * (target - q_values[action])

        if render:
            action_name = ACTION_NAMES.get(action, str(action))
            print(
                f"\nStep {steps:03d} | "
                f"action={action_name:>8} | "
                f"reward={reward:>6.2f} | "
                f"total_reward={total_reward:>7.2f} | "
                f"terminated={terminated} | "
                f"truncated={truncated}"
            )
            env.render()

        # Move to the next state.
        state = next_state

    success = bool(terminated and info.get("has_result", False))

    return {
        "total_reward": total_reward,
        "steps": steps,
        "terminated": terminated,
        "truncated": truncated,
        "success": success,

        # Final location
        "final_layer": info["layer"],
        "final_position": info["position"],

        # Progress flags
        "has_key": info["has_key"],
        "door_open": info["door_open"],
        "terminal_activated": info["terminal_activated"],
        "has_portal_core": info["has_portal_core"],
        "portal_open": info["portal_open"],
        "has_lab_key": info["has_lab_key"],
        "machine_used": info["machine_used"],
        "has_result": info["has_result"],

        "epsilon": epsilon,
    }


# ---------------------------------------------------------------------
# Multi-episode training
# ---------------------------------------------------------------------

def train_q_learning(
    num_episodes=5000,
    max_steps=100,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    seed=None,
    render_every=None,
):
    """
    Train a tabular Q-learning agent over many episodes.

    Epsilon starts high for exploration, then decays after each episode:
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    Returns:
        q_table: dict mapping state tuples to Q-value arrays
        results_df: pandas DataFrame with one row per episode
    """

    render_mode = "human" if render_every is not None else None
    env = MiniResearchWorldEnv(render_mode=render_mode, max_steps=max_steps)

    q_table = {}
    rng = np.random.default_rng(seed)
    results = []

    epsilon = epsilon_start

    # Print progress every 500 episodes, or every 10% for shorter runs.
    if num_episodes >= 5000:
        progress_interval = 500
    else:
        progress_interval = max(1, num_episodes // 10)

    for episode in range(1, num_episodes + 1):
        should_render = render_every is not None and episode % render_every == 0

        episode_result = train_q_learning_episode(
            env=env,
            q_table=q_table,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            rng=rng,
            render=should_render,
        )

        episode_result["episode"] = episode
        results.append(episode_result)

        # Decay exploration rate after each episode.
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % progress_interval == 0 or episode == num_episodes:
            recent = results[-min(100, len(results)) :]
            recent_reward = np.mean([r["total_reward"] for r in recent])
            recent_success = np.mean([r["success"] for r in recent]) * 100
            print(
                f"Episode {episode:5d}/{num_episodes} | "
                f"epsilon={epsilon:.4f} | "
                f"last-{len(recent)} avg reward={recent_reward:.2f} | "
                f"last-{len(recent)} success={recent_success:.1f}% | "
                f"q_table size={len(q_table)}"
            )

    env.close()

    results_df = pd.DataFrame(results)

    # Put episode column first.
    cols = ["episode"] + [col for col in results_df.columns if col != "episode"]
    results_df = results_df[cols]

    return q_table, results_df


# ---------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------

def print_summary(results_df):
    """
    Print a clean Q-learning training summary.
    """

    num_episodes = len(results_df)
    last_n = min(100, num_episodes)
    recent = results_df.tail(last_n)

    avg_reward = results_df["total_reward"].mean()
    avg_steps = results_df["steps"].mean()

    success_rate = results_df["success"].mean() * 100
    termination_rate = results_df["terminated"].mean() * 100
    timeout_rate = results_df["truncated"].mean() * 100

    final_avg_reward = recent["total_reward"].mean()
    final_success_rate = recent["success"].mean() * 100
    final_avg_steps = recent["steps"].mean()
    best_reward = results_df["total_reward"].max()

    key_rate = results_df["has_key"].mean() * 100
    door_rate = results_df["door_open"].mean() * 100
    terminal_rate = results_df["terminal_activated"].mean() * 100
    core_rate = results_df["has_portal_core"].mean() * 100
    portal_rate = results_df["portal_open"].mean() * 100
    lab_key_rate = results_df["has_lab_key"].mean() * 100
    machine_rate = results_df["machine_used"].mean() * 100
    result_rate = results_df["has_result"].mean() * 100

    print("\n========== Q-LEARNING AGENT TRAINING SUMMARY ==========")
    print(f"Episodes run                  : {num_episodes}")
    print(f"Average reward                : {avg_reward:.2f}")
    print(f"Average steps                 : {avg_steps:.1f}")
    print(f"Success rate                  : {success_rate:.1f}%")
    print(f"Termination rate              : {termination_rate:.1f}%")
    print(f"Timeout/truncation rate       : {timeout_rate:.1f}%")
    print(f"Final {last_n} avg reward       : {final_avg_reward:.2f}")
    print(f"Final {last_n} success rate     : {final_success_rate:.1f}%")
    print(f"Final {last_n} avg steps        : {final_avg_steps:.1f}")
    print(f"Best episode reward           : {best_reward:.2f}")

    print("\n---------- Progress Milestones ----------")
    print(f"Collected Layer-0 key         : {key_rate:.1f}%")
    print(f"Opened door                   : {door_rate:.1f}%")
    print(f"Activated terminal            : {terminal_rate:.1f}%")
    print(f"Collected portal core         : {core_rate:.1f}%")
    print(f"Used portal                   : {portal_rate:.1f}%")
    print(f"Collected lab key             : {lab_key_rate:.1f}%")
    print(f"Used machine                  : {machine_rate:.1f}%")
    print(f"Produced final result         : {result_rate:.1f}%")
    print("======================================================\n")


# ---------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a tabular Q-learning agent on Mini Research World."
    )

    parser.add_argument(
        "--episodes",
        type=positive_int,
        default=5000,
        help="Number of training episodes. Default: 5000.",
    )

    parser.add_argument(
        "--max-steps",
        type=positive_int,
        default=100,
        help="Maximum steps per episode before timeout. Default: 100.",
    )

    parser.add_argument(
        "--alpha",
        type=non_negative_float,
        default=0.1,
        help="Learning rate. Default: 0.1.",
    )

    parser.add_argument(
        "--gamma",
        type=probability,
        default=0.99,
        help="Discount factor. Default: 0.99.",
    )

    parser.add_argument(
        "--epsilon-start",
        type=probability,
        default=1.0,
        help="Initial exploration rate. Default: 1.0.",
    )

    parser.add_argument(
        "--epsilon-end",
        type=probability,
        default=0.05,
        help="Minimum exploration rate. Default: 0.05.",
    )

    parser.add_argument(
        "--epsilon-decay",
        type=probability,
        default=0.995,
        help="Epsilon multiplier after each episode. Default: 0.995.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. Default: None.",
    )

    parser.add_argument(
        "--render-every",
        type=positive_int,
        default=None,
        help="Render one episode every N episodes. Default: None.",
    )

    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Optional path to save episode results as CSV.",
    )

    parser.add_argument(
        "--save-qtable",
        type=str,
        default=None,
        help="Optional path to save the trained Q-table as a pickle file.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    print("\nTraining Q-learning agent...")
    print(f"episodes       = {args.episodes}")
    print(f"max_steps      = {args.max_steps}")
    print(f"alpha          = {args.alpha}")
    print(f"gamma          = {args.gamma}")
    print(f"epsilon_start  = {args.epsilon_start}")
    print(f"epsilon_end    = {args.epsilon_end}")
    print(f"epsilon_decay  = {args.epsilon_decay}")
    print(f"seed           = {args.seed}")
    print(f"render_every   = {args.render_every}")

    q_table, results_df = train_q_learning(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
        render_every=args.render_every,
    )

    print_summary(results_df)

    print("First 10 episode results:")
    print(results_df.head(10).to_string(index=False))

    if args.save_csv:
        output_path = Path(args.save_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nSaved episode results to: {output_path}")

    if args.save_qtable:
        output_path = Path(args.save_qtable)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(q_table, f)
        print(f"Saved Q-table to: {output_path}")

    return q_table, results_df


if __name__ == "__main__":
    main()
