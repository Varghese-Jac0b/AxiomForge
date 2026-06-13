"""
AxiomForge random smoke test (Prompt F / contract Section 2.6).

Runs N episodes of uniform-random actions on a given config and asserts:
no exception, every observation within observation_space, every episode
terminates or truncates within max_steps. Prints summary stats.

No success expectation (random agents should fail) and no learning.

Run from the repo root:
    python scripts/random_smoke_test_axiom_forge.py [--episodes 100] [--seed 0]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environments.axiom_forge_configs import make_v1_0_config
from environments.axiom_forge_env import AxiomForgeEnv


def run_smoke_test(episodes: int, seed: int) -> None:
    env = AxiomForgeEnv(make_v1_0_config())
    rng = random.Random(seed)

    successes = 0
    episode_lengths: list[int] = []
    episode_returns: list[float] = []
    min_step_reward = float("inf")
    max_step_reward = float("-inf")
    terminated_count = 0
    truncated_count = 0

    for episode_idx in range(episodes):
        obs, info = env.reset()
        assert env.observation_space.contains(obs), (
            f"episode {episode_idx}: reset obs outside observation_space"
        )
        episode_return = 0.0
        steps = 0
        while True:
            action = rng.randrange(env.action_space.n)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            episode_return += reward
            min_step_reward = min(min_step_reward, reward)
            max_step_reward = max(max_step_reward, reward)
            assert env.observation_space.contains(obs), (
                f"episode {episode_idx} step {steps}: obs outside space"
            )
            assert steps <= env.cfg.max_steps, (
                f"episode {episode_idx}: ran past max_steps without ending"
            )
            if terminated or truncated:
                terminated_count += int(terminated)
                truncated_count += int(truncated)
                break
        if info["success"]:
            successes += 1
        episode_lengths.append(steps)
        episode_returns.append(episode_return)

    success_rate = successes / episodes
    mean_length = sum(episode_lengths) / episodes
    mean_return = sum(episode_returns) / episodes

    print(f"random smoke test: {episodes} episodes, seed {seed} - no crashes")
    print(f"success rate       = {success_rate:.3f} (expected ~0)")
    print(f"mean episode steps = {mean_length:.1f}")
    print(f"episode length     = min {min(episode_lengths)}, "
          f"max {max(episode_lengths)}")
    print(f"terminated/truncated = {terminated_count}/{truncated_count}")
    print(f"step reward range  = [{min_step_reward:.2f}, {max_step_reward:.2f}]")
    print(f"episode return     = mean {mean_return:.2f}, "
          f"min {min(episode_returns):.2f}, max {max(episode_returns):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_smoke_test(args.episodes, args.seed)
