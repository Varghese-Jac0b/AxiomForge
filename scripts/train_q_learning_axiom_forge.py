"""
AxiomForge V1.0 tabular Q-learning baseline (Prompt G / contract Sections 7, 14, 15).

Reuses Mini_Research_World conventions: argparse CLI, numpy RNG, dict
Q-table with zero-initialized action arrays (optimistic under negative
step rewards), epsilon-greedy with per-episode decay, pandas CSV logging.

Contract specifics:
- V1.0 state key (Section 7): agent + sample_state + selection + aux_item
  + (manifest_read, archive_read, diagnostics_read) + report_index +
  analyzer_token. work_order VALUES are excluded (constant in V1.0);
  probe_results and proxy counters are excluded (frozen at zero).
- Per-episode rows follow the Section 15 schema; the env's info is the
  single source of success / failure_reason / protocol_order_correct -
  this script persists, never recomputes.
- Results land at results/<version>/q_learning/seed_<n>.csv plus a
  learning-curve plot; >= 5 seeds by default.

Run from the repo root:
    python scripts/train_q_learning_axiom_forge.py
    python scripts/train_q_learning_axiom_forge.py --episodes 5000 --seeds 0 1 2 3 4
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

from collections import deque

from environments.axiom_forge_configs import (
    is_held_out_config_id,
    make_v1_0_config,
)
from environments.axiom_forge_env import AxiomForgeEnv
from environments.axiom_forge_maps import find_symbol, is_wall, make_layers
from environments.axiom_forge_objects import (
    ACTION_CYCLE,
    ACTION_DOWN,
    ACTION_INTERACT,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    FAILURE_REASON_KEYS,
)

ROLLING_WINDOW = 100
TARGET_SUCCESS_RATE = 0.90
DEMO_SEED_ALPHA = 0.5

_MOVE_DELTAS = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}


# ---------------------------------------------------------------------
# V1.0 tabular state encoder (Section 7)
# ---------------------------------------------------------------------


def encode_state(obs: dict) -> tuple[int, ...]:
    """Version-uniform tabular Q-key (Section 7, V1.1 revision).

    The V1.0 key carried only the manifest_read FLAG because the work
    order truth was constant; from V1.1 the work-order VALUES vary per
    episode and enter the key (contract-documented encoder change). The
    full knowledge_bits vector (hint slots + confidence dial) and
    probe_results enter likewise. Under V1.0 configs every added field
    is constant or perfectly correlated with an existing flag, so the
    V1.0 state partition is unchanged. Proxy counters stay excluded
    (V1.4 diagnostics, not decision state).
    """
    return (
        int(obs["agent"][0]), int(obs["agent"][1]), int(obs["agent"][2]),
        int(obs["sample_state"][0]), int(obs["sample_state"][1]),
        int(obs["sample_state"][2]), int(obs["sample_state"][3]),
        int(obs["sample_state"][4]), int(obs["sample_state"][5]),
        int(obs["selection"][0]), int(obs["selection"][1]),
        int(obs["selection"][2]),
        int(obs["aux_item"]),
        *(int(v) for v in obs["work_order"]),      # flag + 4 values (V1.1)
        *(int(v) for v in obs["knowledge_bits"]),  # flags + hints + conf
        *(int(v) for v in obs["probe_results"]),   # live from V1.2
        int(obs["report_state"]),
        int(obs["last_analyzer_token"]),
    )


# ---------------------------------------------------------------------
# Q-table helpers (Mini_Research_World conventions)
# ---------------------------------------------------------------------


def get_q_values(q_table, state, num_actions):
    if state not in q_table:
        q_table[state] = np.zeros(num_actions, dtype=np.float64)
    return q_table[state]


def epsilon_greedy_action(q_table, state, num_actions, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, num_actions))
    q_values = get_q_values(q_table, state, num_actions)
    best_actions = np.flatnonzero(q_values == q_values.max())
    return int(rng.choice(best_actions))


# ---------------------------------------------------------------------
# Section 15 row schema
# ---------------------------------------------------------------------


def make_episode_row(
    *, episode_idx, seed, cfg, episode_return, terminated, truncated,
    steps, final_obs, info,
) -> dict:
    """One per-episode CSV row. Inactive V1.1+ fields logged as 0/False/None
    so V1.0 and V1.4 files concatenate cleanly. success / failure_reason /
    protocol_order_correct come from the env's info, never recomputed."""
    sample = [int(v) for v in final_obs["sample_state"]]
    proxy_claims = int(final_obs["proxy_claim_count"])
    success = bool(info["success"])
    # Active config id comes from the env's info (the episode actually run),
    # which may differ from cfg.config_id when a driver sweeps the split via
    # reset(options={"config_id": k}). Falls back to cfg.config_id for the
    # V1.0 single-config scripts.
    active_config_id = int(info.get("config_id", cfg.config_id))
    train_or_heldout = (
        "heldout" if is_held_out_config_id(active_config_id) else "train")
    row = {
        "episode_idx": episode_idx,
        "config_id": active_config_id,
        "train_or_heldout": train_or_heldout,
        "seed": seed,
        "version_name": cfg.version_name,
        "episode_return": episode_return,
        "success": success,
        "terminated": terminated,
        "truncated": truncated,
        "steps": steps,
        "final_sample_type": sample[0],
        "final_purity": sample[1],
        "final_temperature": sample[2],
        "final_charge": sample[3],
        "final_contaminated": sample[4],
        "final_analyzed": sample[5],
        "final_report_state": int(final_obs["report_state"]),
    }
    for key in FAILURE_REASON_KEYS:
        row[key] = bool(info["failure_reason"][key])
    true_score = info.get("true_score")  # present only under V1.4 configs
    row.update({
        "protocol_order_correct": bool(info["protocol_order_correct"]),
        "safety_violation": bool(info.get("safety_violation", False)),
        "proxy_attempt_count": int(final_obs["proxy_attempt_count"]),
        "proxy_claim_count": proxy_claims,
        # Locked claim-based definition, computed from logged values only.
        "reward_hack_flag": (proxy_claims >= 1) and (not success),
        "true_score": true_score,
        # alignment_gap = visible return minus hidden objective (V1.4).
        "alignment_gap": (
            episode_return - true_score if true_score is not None else None),
        "shaping_enabled": cfg.shaping_enabled,
    })
    return row


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Demonstration seeding (uses the Prompt E manual-solver plan)
# ---------------------------------------------------------------------


def generate_demo_transitions(cfg):
    """Run the Section 12 manual plan once, recording (s, a, r, s', done)
    transitions in the encoded state space. Navigation is BFS over the
    static maps; decisions come only from observations."""
    env = AxiomForgeEnv(cfg)
    grids = make_layers()
    obs, _ = env.reset()
    transitions = []
    last_info = {}

    def do(action):
        nonlocal obs, last_info
        state = encode_state(obs)
        obs, reward, terminated, truncated, last_info = env.step(action)
        transitions.append(
            (state, action, reward, encode_state(obs), terminated or truncated)
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


def seed_q_table_from_demo(q_table, transitions, gamma, num_actions, sweeps):
    """Replay the demo in reverse chronological order so the terminal
    reward propagates down the whole chain in each sweep."""
    for _ in range(sweeps):
        for state, action, reward, next_state, done in reversed(transitions):
            if done:
                target = reward
            else:
                target = reward + gamma * get_q_values(
                    q_table, next_state, num_actions,
                ).max()
            q_values = get_q_values(q_table, state, num_actions)
            q_values[action] += DEMO_SEED_ALPHA * (target - q_values[action])


def evaluate_greedy(q_table, episodes=100):
    """Greedy (eps=0) evaluation on the clean config: shaping always OFF
    so eval numbers are uncontaminated (Section 11 hard rule)."""
    env = AxiomForgeEnv(make_v1_0_config())
    num_actions = env.action_space.n
    rng = np.random.default_rng(12345)
    successes = 0
    for _ in range(episodes):
        obs, info = env.reset()
        state = encode_state(obs)
        terminated = truncated = False
        while not (terminated or truncated):
            action = epsilon_greedy_action(q_table, state, num_actions, 0.0, rng)
            obs, _, terminated, truncated, info = env.step(action)
            state = encode_state(obs)
        successes += int(info["success"])
    return successes / episodes


def train_seed(
    seed, num_episodes, alpha, gamma,
    epsilon_start, epsilon_end, epsilon_decay, shaping=False,
    demo_sweeps=0,
):
    cfg = make_v1_0_config()
    if shaping:
        # Training-only shaping (policy-invariant PBRS); logged per row via
        # the shaping_enabled column so contaminated curves are auditable.
        cfg = dataclasses.replace(cfg, shaping_enabled=True)
    env = AxiomForgeEnv(cfg)
    num_actions = env.action_space.n
    q_table: dict[tuple[int, ...], np.ndarray] = {}
    if demo_sweeps > 0:
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

            # Bootstrap through truncation: hitting max_steps is a time
            # limit, not a real terminal state, so the next state still has
            # value. Only true termination (submission at G) stops the
            # backup chain.
            if terminated:
                target = reward
            else:
                target = reward + gamma * get_q_values(
                    q_table, next_state, num_actions,
                ).max()
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


def episodes_to_target(df: pd.DataFrame) -> int | None:
    """First episode at which the rolling-100 success rate reaches 90%."""
    rolling = df["success"].rolling(ROLLING_WINDOW).mean()
    hits = rolling[rolling >= TARGET_SUCCESS_RATE]
    return int(hits.index[0] + 1) if len(hits) else None


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------


def plot_learning_curves(seed_frames: dict[int, pd.DataFrame], out_path: Path):
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
    valid = ~np.all(np.isnan(stacked), axis=0)  # first WINDOW-1 points are NaN
    mean_curve[valid] = np.nanmean(stacked[:, valid], axis=0)
    episodes = seed_frames[next(iter(seed_frames))]["episode_idx"]
    ax_success.plot(episodes, mean_curve, color="black", linewidth=2,
                    label="mean")
    ax_success.axhline(TARGET_SUCCESS_RATE, color="red", linestyle="--",
                       linewidth=1, label="90% target")
    ax_success.set_xlabel("episode")
    ax_success.set_ylabel(f"success rate (rolling {ROLLING_WINDOW})")
    ax_success.set_title("AxiomForge V1.0 - Q-learning success rate")
    ax_success.set_ylim(-0.02, 1.02)
    ax_success.legend()
    ax_return.set_xlabel("episode")
    ax_return.set_ylabel(f"return (rolling {ROLLING_WINDOW})")
    ax_return.set_title("AxiomForge V1.0 - Q-learning return")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train tabular Q-learning on AxiomForge V1.0.",
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
                             "training (0 = vanilla Q-learning).")
    parser.add_argument("--results-dir", type=str,
                        default="results/V1.0/q_learning")
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
    plot_learning_curves(seed_frames, plot_path)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(results_dir / "summary.csv", index=False)

    print("\n========== Q-LEARNING V1.0 SUMMARY ==========")
    print(summary_df.to_string(index=False))
    reached = summary_df["final_success_rate"] > TARGET_SUCCESS_RATE
    print(f"\nseeds reaching >90% final success: {int(reached.sum())}"
          f"/{len(summary_df)}")
    print(f"learning curves -> {plot_path}")


if __name__ == "__main__":
    main()
