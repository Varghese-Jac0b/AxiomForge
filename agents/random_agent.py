"""
Random agent baseline for Mini Research World.

Run from the project root:

    python agents/random_agent.py
    python agents/random_agent.py --episodes 100 --seed 42
    python agents/random_agent.py --episodes 20 --render
    python -m agents.random_agent --episodes 100 --save-csv tests/random_baseline.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Project import setup
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from environments.mini_world_env import MiniResearchWorldEnv, ACTION_NAMES


# ---------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------

def run_random_episode(env, seed=None, render=False):
    """
    Run one full episode using random actions.

    The agent does not learn.
    At every step, it samples an action from env.action_space.

    Returns:
        dict containing episode-level statistics.
    """

    obs, info = env.reset(seed=seed)

    # Important:
    # env.reset(seed=seed) seeds the environment.
    # But env.action_space.sample() has its own RNG.
    # So we also seed the action space for reproducible random actions.
    if seed is not None:
        env.action_space.seed(seed)

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    if render:
        print("\nInitial state:")
        env.render()

    while not terminated and not truncated:
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

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

    # In the current environment:
    # terminated=True happens only when the agent successfully submits at G.
    # Still, using has_result makes the success condition more explicit.
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
    }


# ---------------------------------------------------------------------
# Multi-episode baseline
# ---------------------------------------------------------------------

def run_random_baseline(num_episodes=100, max_steps=100, seed=None, render=False):
    """
    Run multiple random-agent episodes.

    Args:
        num_episodes: number of episodes to run.
        max_steps: maximum steps per episode before truncation.
        seed: optional base seed for reproducibility.
        render: whether to print environment state at every step.

    Returns:
        pandas DataFrame with one row per episode.
    """

    render_mode = "human" if render else None
    env = MiniResearchWorldEnv(render_mode=render_mode, max_steps=max_steps)

    results = []

    for episode in range(1, num_episodes + 1):
        episode_seed = None if seed is None else seed + episode

        episode_result = run_random_episode(
            env=env,
            seed=episode_seed,
            render=render,
        )

        episode_result["episode"] = episode
        results.append(episode_result)

    env.close()

    results_df = pd.DataFrame(results)

    # Put episode column first.
    cols = ["episode"] + [col for col in results_df.columns if col != "episode"]
    results_df = results_df[cols]

    return results_df


# ---------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------

def print_summary(results_df):
    """
    Print a clean random-agent performance summary.
    """

    num_episodes = len(results_df)

    avg_reward = results_df["total_reward"].mean()
    avg_steps = results_df["steps"].mean()

    success_rate = results_df["success"].mean() * 100
    termination_rate = results_df["terminated"].mean() * 100
    timeout_rate = results_df["truncated"].mean() * 100

    key_rate = results_df["has_key"].mean() * 100
    door_rate = results_df["door_open"].mean() * 100
    terminal_rate = results_df["terminal_activated"].mean() * 100
    core_rate = results_df["has_portal_core"].mean() * 100
    portal_rate = results_df["portal_open"].mean() * 100
    lab_key_rate = results_df["has_lab_key"].mean() * 100
    machine_rate = results_df["machine_used"].mean() * 100
    result_rate = results_df["has_result"].mean() * 100

    print("\n========== RANDOM AGENT BASELINE SUMMARY ==========")
    print(f"Episodes run             : {num_episodes}")
    print(f"Average reward           : {avg_reward:.2f}")
    print(f"Average steps            : {avg_steps:.1f}")
    print(f"Success rate             : {success_rate:.1f}%")
    print(f"Termination rate         : {termination_rate:.1f}%")
    print(f"Timeout/truncation rate  : {timeout_rate:.1f}%")

    print("\n---------- Progress Milestones ----------")
    print(f"Collected Layer-0 key    : {key_rate:.1f}%")
    print(f"Opened door              : {door_rate:.1f}%")
    print(f"Activated terminal       : {terminal_rate:.1f}%")
    print(f"Collected portal core    : {core_rate:.1f}%")
    print(f"Used portal              : {portal_rate:.1f}%")
    print(f"Collected lab key        : {lab_key_rate:.1f}%")
    print(f"Used machine             : {machine_rate:.1f}%")
    print(f"Produced final result    : {result_rate:.1f}%")
    print("===================================================\n")


# ---------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a random-action baseline on Mini Research World."
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to run. Default: 100.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps per episode before timeout. Default: 100.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for reproducibility. Default: None.",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during each episode.",
    )

    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Optional path to save episode results as CSV.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    print("\nRunning random agent baseline...")
    print(f"episodes  = {args.episodes}")
    print(f"max_steps = {args.max_steps}")
    print(f"seed      = {args.seed}")
    print(f"render    = {args.render}")

    results_df = run_random_baseline(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        render=args.render,
    )

    print_summary(results_df)

    print("First 10 episode results:")
    print(results_df.head(10).to_string(index=False))

    if args.save_csv:
        output_path = Path(args.save_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nSaved episode results to: {output_path}")

    return results_df


if __name__ == "__main__":
    main()