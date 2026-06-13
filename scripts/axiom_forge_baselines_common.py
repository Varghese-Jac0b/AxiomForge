"""
Shared helpers for the AxiomForge V1.0 baseline-ladder scripts
(SARSA, Expected SARSA, and the DQN family).

The tabular Q-learning script (scripts/train_q_learning_axiom_forge.py)
is the verified Prompt G artifact and stays untouched; this module
re-exports its canonical pieces (state encoder, Q-table helpers, the
Section-15 row schema, greedy evaluation) and adds the two things the
other baselines need:

- generate_demo_transitions_encoded(): the same BFS manual-solver demo
  rollout, but parameterized over the state encoder so deep agents can
  record float-vector transitions instead of tabular keys. It returns
  (state, action, reward, next_state, terminated, truncated) tuples.
- plot_learning_curves_labeled(): the same two-panel learning-curve plot
  with the algorithm name as a parameter instead of hardcoded
  "Q-learning" titles.

Demonstration seeding/usage is part of the OFFICIAL V1.0 baseline recipe
(Build Record Section 4): vanilla undirected exploration never crosses
the exploration wall, and that result is a documented finding, not a bug.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.axiom_forge_env import AxiomForgeEnv
from environments.axiom_forge_maps import find_symbol, is_wall, make_layers
from environments.axiom_forge_objects import ACTION_CYCLE, ACTION_INTERACT

# Canonical implementations live in the verified Q-learning script.
from scripts.train_q_learning_axiom_forge import (  # noqa: F401  (re-exports)
    DEMO_SEED_ALPHA,
    ROLLING_WINDOW,
    TARGET_SUCCESS_RATE,
    _MOVE_DELTAS,
    encode_state,
    episodes_to_target,
    epsilon_greedy_action,
    evaluate_greedy,
    get_q_values,
    make_episode_row,
)


# ---------------------------------------------------------------------
# Demonstration rollout, parameterized over the state encoder
# ---------------------------------------------------------------------


def generate_demo_transitions_encoded(
    cfg, encode: Callable[[dict], object],
) -> list[tuple]:
    """Run the Section 12 manual plan once, recording
    (state, action, reward, next_state, terminated, truncated) transitions
    with `encode` applied to both endpoint observations.

    Navigation is BFS over the static maps; decisions come only from
    observations. The plan must end in success (asserted) - it is the
    same 61-step P0 path the manual solver and the Q-learning demo
    seeding use.
    """
    env = AxiomForgeEnv(cfg)
    grids = make_layers()
    obs, _ = env.reset()
    transitions: list[tuple] = []
    last_info: dict = {}

    def do(action):
        nonlocal obs, last_info
        state = encode(obs)
        obs, reward, terminated, truncated, last_info = env.step(action)
        transitions.append(
            (state, action, reward, encode(obs), terminated, truncated)
        )

    def goto(symbol):
        layer = int(obs["agent"][0])
        grid = grids[layer]
        start = (int(obs["agent"][1]), int(obs["agent"][2]))
        target = find_symbol(grid, symbol)
        came_from = {start: None}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current == target:
                break
            for action, (dr, dc) in _MOVE_DELTAS.items():
                nxt = (current[0] + dr, current[1] + dc)
                if nxt not in came_from and not is_wall(grid, *nxt):
                    came_from[nxt] = (current, action)
                    queue.append(nxt)
        actions = []
        node = target
        while came_from[node] is not None:
            node, action = came_from[node]
            actions.append(action)
        for action in reversed(actions):
            do(action)

    goto("M"); do(ACTION_INTERACT)
    goto("H"); do(ACTION_INTERACT)
    goto("D"); do(ACTION_INTERACT)
    goto("R"); do(ACTION_INTERACT)
    goto("C"); do(ACTION_CYCLE); do(ACTION_CYCLE); do(ACTION_INTERACT)
    goto("E"); do(ACTION_INTERACT)
    goto("U"); do(ACTION_INTERACT)
    goto("T"); do(ACTION_CYCLE); do(ACTION_INTERACT)
    goto("I"); do(ACTION_INTERACT)
    goto("N"); do(ACTION_INTERACT)
    goto("G"); do(ACTION_INTERACT)
    assert last_info["success"] is True, "demo plan must succeed"
    return transitions


# ---------------------------------------------------------------------
# Learning-curve plot with a parameterized algorithm label
# ---------------------------------------------------------------------


def plot_learning_curves_labeled(
    seed_frames: dict[int, pd.DataFrame], out_path: Path, algo_label: str,
):
    """Two-panel rolling success/return plot (same layout as the verified
    Q-learning plot) with `algo_label` in the titles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_success, ax_return) = plt.subplots(1, 2, figsize=(13, 5))
    curves = []
    for seed, df in seed_frames.items():
        rolling_success = df["success"].rolling(ROLLING_WINDOW).mean()
        rolling_return = df["episode_return"].rolling(ROLLING_WINDOW).mean()
        ax_success.plot(df["episode_idx"], rolling_success, alpha=0.35,
                        label=f"seed {seed}")
        ax_return.plot(df["episode_idx"], rolling_return, alpha=0.35)
        curves.append(rolling_success.to_numpy())
    stacked = np.vstack(curves)
    mean_curve = np.full(stacked.shape[1], np.nan)
    valid = ~np.all(np.isnan(stacked), axis=0)
    mean_curve[valid] = np.nanmean(stacked[:, valid], axis=0)
    episodes = seed_frames[next(iter(seed_frames))]["episode_idx"]
    ax_success.plot(episodes, mean_curve, color="black", linewidth=2,
                    label="mean")
    ax_success.axhline(TARGET_SUCCESS_RATE, color="red", linestyle="--",
                       linewidth=1, label="90% target")
    ax_success.set_xlabel("episode")
    ax_success.set_ylabel(f"success rate (rolling {ROLLING_WINDOW})")
    ax_success.set_title(f"AxiomForge V1.0 - {algo_label} success rate")
    ax_success.set_ylim(-0.02, 1.02)
    ax_success.legend()
    ax_return.set_xlabel("episode")
    ax_return.set_ylabel(f"return (rolling {ROLLING_WINDOW})")
    ax_return.set_title(f"AxiomForge V1.0 - {algo_label} return")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
