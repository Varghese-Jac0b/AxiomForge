"""
AxiomForge V1.1-V1.4 multi-config training driver (the generalization
harness behind the Build Record version gates).

Where the per-algorithm V1.0 scripts train on the single fixed config,
this driver SWEEPS the train split: every episode draws a train config_id
(even ids; held-out ids are odd and never trained on), resets the env via
reset(options={"config_id": k}), and logs the ACTIVE config from info.
After training it measures the train-vs-held-out generalization gap with
per-protocol and per-fault breakdowns, and for V1.4 produces the E2
figure: visible return vs hidden true_score.

Algorithms (tabular value-based ladder, shared encode_state Q-key):
    q_learning | sarsa | expected_sarsa | spie_q

Demonstrations come from the version's reference solver (the same
branching/probe-first/fault-aware/legal solvers verified in tests),
recorded in encoded space and replayed in reverse value sweeps - exactly
the official V1.0 recipe, generalized across configs.

Discipline (locked rules):
- intrinsic reward (spie_q) is TRAINING-ONLY; episode_return logs the
  EXTRINSIC return; greedy evaluation uses no intrinsic and shaping off;
- `done` for bootstrapping is termination-only (truncation bootstraps);
- shaping stays OFF for every evaluation/headline run;
- training never touches held-out config_ids.

Run from the repo root:
    python scripts/train_axiom_forge.py --version v1_1 --algo q_learning \
        --episodes 4000 --seeds 0 1 2 --demo-configs 12 --demo-sweeps 20
    python scripts/train_axiom_forge.py --version v1_4 --algo q_learning \
        --episodes 4000 --demo-configs 0 --e2     # the misalignment headline
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
from agents.spie_q_agent import SuccessorPredecessorTables, make_abstraction
from environments.axiom_forge_configs import (
    held_out_config_ids,
    make_v1_1_config,
    make_v1_2_config,
    make_v1_3_config,
    make_v1_4_config,
    train_config_ids,
)
from environments.axiom_forge_env import AxiomForgeEnv
from scripts.axiom_forge_baselines_common import (
    ROLLING_WINDOW,
    TARGET_SUCCESS_RATE,
    encode_state,
    epsilon_greedy_action,
    get_q_values,
    make_episode_row,
)
from scripts.manual_solve_axiom_forge_v1_1 import BranchingSolver
from scripts.manual_solve_axiom_forge_v1_2 import ProbeFirstSolver
from scripts.manual_solve_axiom_forge_v1_3 import FaultAwareSolver
from scripts.manual_solve_axiom_forge_v1_4 import V14LegalSolver
from scripts.train_expected_sarsa_axiom_forge import expected_q_value

# version -> (config factory, reference solver class for demos)
VERSIONS = {
    "v1_1": (make_v1_1_config, BranchingSolver),
    "v1_2": (make_v1_2_config, ProbeFirstSolver),
    "v1_3": (make_v1_3_config, FaultAwareSolver),
    "v1_4": (make_v1_4_config, V14LegalSolver),
}
ALGOS = ("q_learning", "sarsa", "expected_sarsa", "spie_q")

# Deep algorithms: ONE configurable DQNAgent, selected by a clean switch
# set (double / dueling / per / noisy) plus a driver-level `spie` flag that
# adds the intrinsic bonus to the stored reward (never to the env reward or
# to greedy evaluation). Explicit table over name-parsing for inspectability.
DEEP_ALGOS = {
    "dqn":                          dict(double=False, dueling=False, per=False, noisy=False, spie=False),
    "ddqn":                         dict(double=True,  dueling=False, per=False, noisy=False, spie=False),
    "dueling_dqn":                  dict(double=False, dueling=True,  per=False, noisy=False, spie=False),
    "dueling_ddqn_per":             dict(double=True,  dueling=True,  per=True,  noisy=False, spie=False),
    "noisy_dueling_ddqn_per":       dict(double=True,  dueling=True,  per=True,  noisy=True,  spie=False),
    "spie_dqn":                     dict(double=False, dueling=False, per=False, noisy=False, spie=True),
    "spie_ddqn":                    dict(double=True,  dueling=False, per=False, noisy=False, spie=True),
    "spie_dueling_dqn":             dict(double=False, dueling=True,  per=False, noisy=False, spie=True),
    "spie_dueling_ddqn_per":        dict(double=True,  dueling=True,  per=True,  noisy=False, spie=True),
    "spie_noisy_dueling_ddqn_per":  dict(double=True,  dueling=True,  per=True,  noisy=True,  spie=True),
}
ALL_ALGOS = ALGOS + tuple(DEEP_ALGOS)


# ---------------------------------------------------------------------
# Demonstration recording (drive a reference solver, capture transitions)
# ---------------------------------------------------------------------


def record_demo_transitions(version, seed, config_id, encode):
    """Run the version's reference solver on (seed, config_id) and return
    the encoded transitions (state, action, reward, next_state,
    terminated). Wraps the solver's _step so every primitive action is
    captured; asserts the demo succeeds."""
    factory, solver_cls = VERSIONS[version]
    env = AxiomForgeEnv(factory(seed=seed, config_id=config_id))
    solver = solver_cls(env, options={"config_id": config_id})
    transitions: list[tuple] = []
    original_step = solver._step

    def recording_step(action):
        state = encode(solver.obs)
        reward, terminated, truncated = original_step(action)
        transitions.append(
            (state, action, reward, encode(solver.obs), terminated))
        return reward, terminated, truncated

    solver._step = recording_step
    solver.solve()
    assert solver.last_info["success"] is True, (
        f"demo for {version} config {config_id} failed: "
        f"{solver.last_info['failure_reason']}")
    return transitions


def seed_q_table(algo, q_table, transitions, gamma, num_actions, sweeps,
                 alpha=1.0):
    """Reverse value sweeps over one demo trajectory. Off-policy algos
    (q_learning / expected_sarsa / spie_q) use the greedy (max) target -
    the epsilon->0 limit, i.e. the value of following the optimal demo;
    SARSA bootstraps from the NEXT DEMONSTRATED action (on-policy).

    alpha defaults to 1.0 (full Bellman backup): one reverse sweep then
    propagates the terminal value through the whole chain, so the
    demonstrated action is the greedy argmax at every state on the path -
    necessary because V1.1-V1.4 demos run 60-140 steps and the damped
    alpha=0.5 V1.0 recipe would need ~step-count sweeps to fill them."""
    n = len(transitions)
    for _ in range(sweeps):
        for i in range(n - 1, -1, -1):
            state, action, reward, next_state, terminated = transitions[i]
            if terminated:
                target = reward
            elif algo == "sarsa":
                next_action = transitions[i + 1][1]
                target = reward + gamma * get_q_values(
                    q_table, next_state, num_actions)[next_action]
            else:
                target = reward + gamma * get_q_values(
                    q_table, next_state, num_actions).max()
            q_values = get_q_values(q_table, state, num_actions)
            q_values[action] += alpha * (target - q_values[action])


# ---------------------------------------------------------------------
# TD target dispatch
# ---------------------------------------------------------------------


def td_target(algo, q_table, num_actions, reward, next_state, gamma,
              epsilon, next_action):
    """Non-terminal bootstrap target. `reward` already includes any
    intrinsic bonus (spie_q). `next_action` is used only by SARSA."""
    if algo == "sarsa":
        return reward + gamma * get_q_values(
            q_table, next_state, num_actions)[next_action]
    if algo == "expected_sarsa":
        return reward + gamma * expected_q_value(
            q_table, next_state, num_actions, epsilon)
    return reward + gamma * get_q_values(
        q_table, next_state, num_actions).max()


# ---------------------------------------------------------------------
# Greedy evaluation across a config split (with protocol/fault labels)
# ---------------------------------------------------------------------


def evaluate_split(version, seed, policy_fn, config_ids):
    """Greedy rollout on each config_id under `policy_fn(obs) -> action`;
    returns a DataFrame with one row per config: success + the latent
    protocol/fault (read from a debug env, eval only). Intrinsic rewards
    are NOT involved here - the policy is queried purely greedily and the
    only thing scored is the environment's success flag. Shaping is
    irrelevant (eval configs build from the clean factory)."""
    factory, _ = VERSIONS[version]
    rows = []
    for config_id in config_ids:
        cfg = dataclasses.replace(
            factory(seed=seed, config_id=config_id), debug_mode=True)
        env = AxiomForgeEnv(cfg)
        obs, info = env.reset(options={"config_id": config_id})
        hidden = info["debug"]["hidden_context"]
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy_fn(obs)
            obs, _, terminated, truncated, info = env.step(action)
        rows.append({
            "config_id": config_id,
            "protocol": hidden.protocol.name,
            "fault": hidden.fault.name,
            "success": int(info["success"]),
        })
    return pd.DataFrame(rows)


def tabular_policy(q_table, num_actions):
    """Greedy (eps=0) tabular policy over the encoded Q-key."""
    rng = np.random.default_rng(99)

    def policy(obs):
        return epsilon_greedy_action(
            q_table, encode_state(obs), num_actions, 0.0, rng)

    return policy


def deep_policy(agent, encode_fn):
    """Greedy, deterministic deep policy: no epsilon, noise zeroed for
    noisy agents, no intrinsic reward - just argmax Q on the env obs."""
    def policy(obs):
        return agent.act(encode_fn(obs), epsilon=0.0, deterministic=True)

    return policy


# ---------------------------------------------------------------------
# One training seed
# ---------------------------------------------------------------------


def train_seed(version, algo, seed, args):
    factory, _ = VERSIONS[version]
    cfg = factory(seed=seed, config_id=0)
    if args.shaping:
        cfg = dataclasses.replace(cfg, shaping_enabled=True)
    env = AxiomForgeEnv(cfg)
    num_actions = env.action_space.n
    train_ids = train_config_ids(args.train_configs)
    held_ids = held_out_config_ids(args.heldout_configs)

    q_table: dict[tuple, np.ndarray] = {}
    if args.demo_configs > 0:
        for config_id in train_config_ids(args.demo_configs):
            transitions = record_demo_transitions(
                version, seed, config_id, encode_state)
            seed_q_table(algo, q_table, transitions, args.gamma,
                         num_actions, args.demo_sweeps)

    # SPIE-Q exploration tables (unused by the other algos).
    spie = SuccessorPredecessorTables(
        sr_alpha=args.sr_alpha, sr_gamma=args.sr_gamma,
        pr_alpha=args.sr_alpha, pr_gamma=args.sr_gamma)
    abstract = make_abstraction(args.abstraction, encode_state)

    rng = np.random.default_rng(seed)
    epsilon = args.epsilon_start
    beta = args.beta
    rows = []
    eval_history = []
    progress_interval = max(1, args.episodes // 10)

    for episode in range(1, args.episodes + 1):
        config_id = int(rng.choice(train_ids))
        obs, info = env.reset(options={"config_id": config_id})
        state = encode_state(obs)
        abstract_state = abstract(obs)
        episode_return = 0.0          # EXTRINSIC only
        steps = 0
        terminated = truncated = False
        # SARSA commits its first action before the loop (on-policy).
        action = epsilon_greedy_action(q_table, state, num_actions,
                                       epsilon, rng)

        while not (terminated or truncated):
            if algo != "sarsa":
                action = epsilon_greedy_action(
                    q_table, state, num_actions, epsilon, rng)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = encode_state(obs)
            episode_return += reward
            steps += 1

            r_total = reward
            if algo == "spie_q":
                next_abstract = abstract(obs)
                spie.update(abstract_state, next_abstract)
                r_total = reward + beta * spie.bonus(next_abstract,
                                                     mode=args.mode)
                abstract_state = next_abstract

            # On-policy SARSA needs a' before the update.
            next_action = None
            if algo == "sarsa" and not terminated:
                next_action = epsilon_greedy_action(
                    q_table, next_state, num_actions, epsilon, rng)

            if terminated:
                target = r_total
            else:
                target = td_target(algo, q_table, num_actions, r_total,
                                   next_state, args.gamma, epsilon,
                                   next_action)
            q_values = get_q_values(q_table, state, num_actions)
            q_values[action] += args.alpha * (target - q_values[action])
            state = next_state
            if algo == "sarsa":
                action = next_action if next_action is not None else 0

        rows.append(make_episode_row(
            episode_idx=episode, seed=seed, cfg=cfg,
            episode_return=episode_return, terminated=terminated,
            truncated=truncated, steps=steps, final_obs=obs, info=info))
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
        beta = max(args.beta_end, beta * args.beta_decay)

        if episode % progress_interval == 0 or episode == args.episodes:
            policy = tabular_policy(q_table, num_actions)
            tr = evaluate_split(version, seed, policy, train_ids)
            ho = evaluate_split(version, seed, policy, held_ids)
            eval_history.append({
                "episode": episode,
                "train_greedy": tr["success"].mean(),
                "heldout_greedy": ho["success"].mean(),
            })
            print(
                f"[{version}/{algo}] seed {seed} | ep {episode:5d}/"
                f"{args.episodes} | eps={epsilon:.3f} beta={beta:.3f} | "
                f"train_succ={tr['success'].mean():.2f} "
                f"heldout_succ={ho['success'].mean():.2f} | q={len(q_table)}",
                flush=True)

    policy = tabular_policy(q_table, num_actions)
    final_train = evaluate_split(version, seed, policy, train_ids)
    final_held = evaluate_split(version, seed, policy, held_ids)
    return (pd.DataFrame(rows), q_table, final_train, final_held,
            pd.DataFrame(eval_history))


# ---------------------------------------------------------------------
# One training seed - DEEP (DQN family, noisy, SPIE-deep)
# ---------------------------------------------------------------------


def train_seed_deep(version, algo, seed, args):
    """Deep training path: one configurable DQNAgent (double/dueling/per/
    noisy) plus an optional driver-level SPIE intrinsic bonus. Mirrors the
    tabular train_seed contract (same return tuple, same make_episode_row
    schema, same train/held-out greedy evaluation). The intrinsic bonus is
    added ONLY to the reward stored in replay - episode_return logs the
    extrinsic return, and greedy evaluation never sees it."""
    switches = DEEP_ALGOS[algo]
    factory, _ = VERSIONS[version]
    cfg = factory(seed=seed, config_id=0)
    if args.shaping:
        cfg = dataclasses.replace(cfg, shaping_enabled=True)
    env = AxiomForgeEnv(cfg)
    encode_fn, obs_dim = make_obs_encoder(env.observation_space)
    num_actions = env.action_space.n
    train_ids = train_config_ids(args.train_configs)
    held_ids = held_out_config_ids(args.heldout_configs)

    # Demonstrations: record encoded transitions from the version solver on
    # each demo config, store them in the protected replay region, pretrain.
    demo_transitions = []
    if args.demo_configs > 0:
        for config_id in train_config_ids(args.demo_configs):
            demo_transitions += record_demo_transitions(
                version, seed, config_id, encode_fn)
    demo_capacity = len(demo_transitions)

    agent = DQNAgent(
        obs_dim, num_actions,
        double=switches["double"], dueling=switches["dueling"],
        per=switches["per"], noisy=switches["noisy"],
        gamma=args.gamma, lr=args.lr, buffer_capacity=args.buffer_size,
        demo_capacity=demo_capacity, batch_size=args.batch_size,
        target_update_interval=args.target_update,
        demo_fraction=args.demo_fraction, margin=args.margin,
        margin_weight=args.margin_weight, device=args.device, seed=seed)

    for state, action, reward, next_state, terminated in demo_transitions:
        agent.store_demo(state, action, reward, next_state, terminated)
    if demo_capacity > 0 and args.pretrain_steps > 0:
        for _ in range(args.pretrain_steps):
            agent.learn()

    spie = SuccessorPredecessorTables(
        sr_alpha=args.sr_alpha, sr_gamma=args.sr_gamma,
        pr_alpha=args.sr_alpha, pr_gamma=args.sr_gamma)
    abstract = make_abstraction(args.abstraction, encode_state)

    cfg_rng = np.random.default_rng(seed + 10_000)  # config sweep only
    # Noisy nets explore through learned noise -> no epsilon schedule.
    epsilon = 0.0 if switches["noisy"] else args.epsilon_start
    beta = args.beta
    rows = []
    eval_history = []
    progress_interval = max(1, args.episodes // 10)

    for episode in range(1, args.episodes + 1):
        config_id = int(cfg_rng.choice(train_ids))
        obs, info = env.reset(options={"config_id": config_id})
        state = encode_fn(obs)
        abstract_state = abstract(obs)
        episode_return = 0.0           # EXTRINSIC only (CSV column)
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.act(state, epsilon)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = encode_fn(obs)
            episode_return += reward
            steps += 1

            r_train = reward
            if switches["spie"]:
                next_abstract = abstract(obs)
                spie.update(abstract_state, next_abstract)
                r_train = reward + beta * spie.bonus(
                    next_abstract, mode=args.mode)
                abstract_state = next_abstract

            # Store termination only: truncation bootstraps through.
            agent.store(state, action, r_train, next_state, terminated)
            if steps % args.train_freq == 0:
                agent.learn()
            state = next_state

        rows.append(make_episode_row(
            episode_idx=episode, seed=seed, cfg=cfg,
            episode_return=episode_return, terminated=terminated,
            truncated=truncated, steps=steps, final_obs=obs, info=info))
        if not switches["noisy"]:
            epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
        beta = max(args.beta_end, beta * args.beta_decay)

        if episode % progress_interval == 0 or episode == args.episodes:
            policy = deep_policy(agent, encode_fn)
            tr = evaluate_split(version, seed, policy, train_ids)
            ho = evaluate_split(version, seed, policy, held_ids)
            eval_history.append({
                "episode": episode,
                "train_greedy": tr["success"].mean(),
                "heldout_greedy": ho["success"].mean(),
            })
            print(
                f"[{version}/{algo}] seed {seed} | ep {episode:5d}/"
                f"{args.episodes} | eps={epsilon:.3f} beta={beta:.3f} | "
                f"train_succ={tr['success'].mean():.2f} "
                f"heldout_succ={ho['success'].mean():.2f} | "
                f"learn_steps={agent.learn_steps}", flush=True)

    policy = deep_policy(agent, encode_fn)
    final_train = evaluate_split(version, seed, policy, train_ids)
    final_held = evaluate_split(version, seed, policy, held_ids)
    return (pd.DataFrame(rows), agent, final_train, final_held,
            pd.DataFrame(eval_history))


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------


def plot_generalization(eval_frames, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for seed, ev in eval_frames.items():
        ax.plot(ev["episode"], ev["train_greedy"], color="tab:blue",
                alpha=0.4)
        ax.plot(ev["episode"], ev["heldout_greedy"], color="tab:orange",
                alpha=0.4)
    # mean curves
    all_ep = eval_frames[next(iter(eval_frames))]["episode"]
    train_stack = np.vstack([ev["train_greedy"] for ev in
                             eval_frames.values()])
    held_stack = np.vstack([ev["heldout_greedy"] for ev in
                            eval_frames.values()])
    ax.plot(all_ep, train_stack.mean(0), color="tab:blue", linewidth=2,
            label="train (mean)")
    ax.plot(all_ep, held_stack.mean(0), color="tab:orange", linewidth=2,
            label="held-out (mean)")
    ax.axhline(TARGET_SUCCESS_RATE, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("episode")
    ax.set_ylabel("greedy success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_e2(seed_frames, out_path, title):
    """The misalignment headline: rolling visible return vs hidden
    true_score (and the alignment gap) over training."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for df in seed_frames.values():
        ret = df["episode_return"].rolling(ROLLING_WINDOW).mean()
        ts = df["true_score"].rolling(ROLLING_WINDOW).mean()
        ax.plot(df["episode_idx"], ret, color="tab:green", alpha=0.35)
        ax.plot(df["episode_idx"], ts, color="tab:red", alpha=0.35)
    any_df = seed_frames[next(iter(seed_frames))]
    ep = any_df["episode_idx"]
    ret_mean = np.vstack([
        d["episode_return"].rolling(ROLLING_WINDOW).mean().to_numpy()
        for d in seed_frames.values()])
    ts_mean = np.vstack([
        d["true_score"].rolling(ROLLING_WINDOW).mean().to_numpy()
        for d in seed_frames.values()])
    ax.plot(ep, np.nanmean(ret_mean, 0), color="tab:green", linewidth=2,
            label="visible return (what the agent maximizes)")
    ax.plot(ep, np.nanmean(ts_mean, 0), color="tab:red", linewidth=2,
            label="hidden true_score (what we actually want)")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel("episode")
    ax.set_ylabel("rolling mean")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Multi-config AxiomForge training driver (V1.1-V1.4).")
    parser.add_argument("--version", choices=sorted(VERSIONS), required=True)
    parser.add_argument("--algo", choices=ALL_ALGOS, default="q_learning")
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--train-configs", type=int, default=32,
                        help="number of train (even) config_ids to sweep")
    parser.add_argument("--heldout-configs", type=int, default=16,
                        help="number of held-out (odd) config_ids to test")
    parser.add_argument("--demo-configs", type=int, default=12,
                        help="train configs to seed demos from (0 = vanilla)")
    parser.add_argument("--demo-sweeps", type=int, default=5,
                        help="full-backup (alpha=1.0) reverse sweeps per demo")
    # spie_q knobs
    parser.add_argument("--mode", choices=["full", "sr_only", "pr_only",
                                           "none"], default="full")
    parser.add_argument("--abstraction", choices=["milestone", "full"],
                        default="milestone")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--beta-decay", type=float, default=0.999)
    parser.add_argument("--beta-end", type=float, default=0.0)
    parser.add_argument("--sr-alpha", type=float, default=0.1)
    parser.add_argument("--sr-gamma", type=float, default=0.95)
    # deep (DQN-family) knobs - ignored by tabular algos
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--pretrain-steps", type=int, default=2000)
    parser.add_argument("--demo-fraction", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.8)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu",
                        help='"cpu", "mps", or "auto"')
    parser.add_argument("--shaping", action="store_true")
    parser.add_argument("--e2", action="store_true",
                        help="V1.4: also write the return-vs-true_score plot")
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / (
        args.results_dir or f"results/{args.version.upper().replace('_', '.')}"
        f"/{args.algo}")
    results_dir.mkdir(parents=True, exist_ok=True)

    seed_frames = {}
    eval_frames = {}
    summary = []
    per_config_train = []
    per_config_held = []
    is_deep = args.algo in DEEP_ALGOS
    trainer = train_seed_deep if is_deep else train_seed
    for seed in args.seeds:
        print(f"\n=== {args.version}/{args.algo} seed {seed} "
              f"(demo_configs={args.demo_configs}, deep={is_deep}) ===",
              flush=True)
        df, model, final_train, final_held, ev = trainer(
            args.version, args.algo, seed, args)
        df.to_csv(results_dir / f"seed_{seed}.csv", index=False)
        seed_frames[seed] = df
        eval_frames[seed] = ev
        final_train["seed"] = seed
        final_held["seed"] = seed
        per_config_train.append(final_train)
        per_config_held.append(final_held)
        gap = final_train["success"].mean() - final_held["success"].mean()
        summary.append({
            "seed": seed,
            "version": args.version,
            "algo": args.algo,
            "train_greedy_success": final_train["success"].mean(),
            "heldout_greedy_success": final_held["success"].mean(),
            "generalization_gap": gap,
            "final_train_episode_success":
                df.tail(ROLLING_WINDOW)["success"].mean(),
            # tabular: visited Q-keys; deep: gradient steps taken.
            "q_table_size": len(model) if isinstance(model, dict) else 0,
            "learn_steps": 0 if isinstance(model, dict)
                else getattr(model, "learn_steps", 0),
        })
        print(f"seed {seed} | train_greedy={final_train['success'].mean():.2f}"
              f" heldout_greedy={final_held['success'].mean():.2f} "
              f"gap={gap:.2f}")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(results_dir / "summary.csv", index=False)

    # per-protocol / per-fault generalization breakdown (held-out)
    held_all = pd.concat(per_config_held, ignore_index=True)
    by_protocol = held_all.groupby("protocol")["success"].mean()
    by_fault = held_all.groupby("fault")["success"].mean()
    breakdown = pd.DataFrame({
        "heldout_success_by_protocol": by_protocol})
    breakdown.to_csv(results_dir / "heldout_by_protocol.csv")
    by_fault.to_frame("heldout_success_by_fault").to_csv(
        results_dir / "heldout_by_fault.csv")

    plot_generalization(
        eval_frames, results_dir / "generalization_curve.png",
        f"AxiomForge {args.version} {args.algo}: train vs held-out")
    if args.e2 and "true_score" in seed_frames[args.seeds[0]].columns \
            and seed_frames[args.seeds[0]]["true_score"].notna().any():
        plot_e2(seed_frames, results_dir / "e2_return_vs_true_score.png",
                f"AxiomForge {args.version} {args.algo}: "
                f"visible return vs hidden true_score")

    print(f"\n========== {args.version}/{args.algo} SUMMARY ==========")
    print(summary_df.to_string(index=False))
    print("\nHeld-out success by protocol:")
    print(by_protocol.round(3).to_string())
    print("\nHeld-out success by fault:")
    print(by_fault.round(3).to_string())
    print(f"\nartifacts -> {results_dir}")


if __name__ == "__main__":
    main()
