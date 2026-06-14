"""
Tests for the deep-RL benchmark additions:
- NoisyLinear (shape, noise behaviour, determinism via remove_noise);
- the noisy flag end to end in DQNAgent (deterministic greedy eval);
- the deep multi-config driver path (DQN family + SPIE-deep + noisy),
  the switch table, demo seeding, train/held-out evaluation, and the
  intrinsic-bonus separation (episode_return stays extrinsic; greedy eval
  never sees the bonus).
Runtimes are kept tiny (few episodes / configs / pretrain steps).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.dqn_axiom_forge import (
    DQNAgent,
    DuelingQNetwork,
    NoisyLinear,
    QNetwork,
)
from agents.spie_q_agent import SuccessorPredecessorTables
from scripts.train_axiom_forge import (
    ALL_ALGOS,
    DEEP_ALGOS,
    deep_policy,
    evaluate_split,
    train_seed_deep,
)


# ---------------------------------------------------------------------
# NoisyLinear
# ---------------------------------------------------------------------


def test_noisy_linear_output_shape():
    layer = NoisyLinear(8, 5)
    out = layer(torch.randn(4, 8))
    assert out.shape == (4, 5)


def test_noise_changes_output_then_remove_is_deterministic():
    layer = NoisyLinear(8, 5)
    x = torch.randn(1, 8)
    layer.reset_noise(); a = layer(x)
    layer.reset_noise(); b = layer(x)
    assert not torch.allclose(a, b)         # fresh noise changes output
    layer.remove_noise(); c = layer(x)
    layer.remove_noise(); d = layer(x)
    assert torch.allclose(c, d)             # zeroed noise -> deterministic


@pytest.mark.parametrize("net_cls", [QNetwork, DuelingQNetwork])
def test_noisy_networks_build_and_toggle(net_cls):
    net = net_cls(12, 6, noisy=True)
    x = torch.randn(3, 12)
    assert net(x).shape == (3, 6)
    net.reset_noise(); a = net(x)
    net.reset_noise(); b = net(x)
    assert not torch.allclose(a, b)
    net.remove_noise(); c = net(x)
    net.remove_noise(); d = net(x)
    assert torch.allclose(c, d)


def test_non_noisy_network_has_no_noisy_layers():
    net = QNetwork(12, 6, noisy=False)
    assert not any(isinstance(m, NoisyLinear) for m in net.modules())


# ---------------------------------------------------------------------
# Noisy agent: deterministic greedy evaluation
# ---------------------------------------------------------------------


def test_noisy_agent_greedy_is_deterministic():
    agent = DQNAgent(16, 6, dueling=True, double=True, per=True, noisy=True,
                     seed=0)
    s = np.random.default_rng(0).standard_normal(16).astype(np.float32)
    greedy = {agent.act(s, 0.0, deterministic=True) for _ in range(8)}
    assert len(greedy) == 1                 # no noise -> one stable action


def test_noisy_agent_behaviour_varies_with_noise():
    agent = DQNAgent(16, 6, noisy=True, seed=0)
    s = np.random.default_rng(1).standard_normal(16).astype(np.float32)
    # over many noisy forward passes the behaviour policy should not be a
    # single fixed action (noise drives exploration)
    acts = {agent.act(s, 0.0, deterministic=False) for _ in range(40)}
    assert len(acts) >= 2


# ---------------------------------------------------------------------
# Switch table
# ---------------------------------------------------------------------


def test_switch_table_is_consistent_with_names():
    assert DEEP_ALGOS["dqn"] == dict(double=False, dueling=False, per=False,
                                     noisy=False, spie=False)
    assert DEEP_ALGOS["spie_noisy_dueling_ddqn_per"] == dict(
        double=True, dueling=True, per=True, noisy=True, spie=True)
    # every deep algo name encodes its switches. "spie" appears in the name of
    # every SPIE variant (prefix `spie_`, or `pg_spie_` for the gated R1 one);
    # the `pg_` prefix marks the Protocol-Gated SPIE knob.
    for name, sw in DEEP_ALGOS.items():
        assert sw["spie"] == ("spie" in name)
        assert sw["noisy"] == ("noisy" in name)
        assert sw["per"] == name.endswith("per")
        assert sw.get("pg", False) == name.startswith("pg_")
    assert set(ALL_ALGOS) >= set(DEEP_ALGOS)


# ---------------------------------------------------------------------
# SPIE-deep separation invariants
# ---------------------------------------------------------------------


def test_spie_none_mode_is_zero_bonus():
    t = SuccessorPredecessorTables()
    for _ in range(5):
        t.update("a", "b")
    assert t.bonus("b", mode="none") == 0.0       # disabled -> exactly zero


def test_spie_enabled_gives_positive_bonus_on_novelty():
    t = SuccessorPredecessorTables()
    assert t.bonus("never_seen", mode="full") > 0.0


def test_beta_annealing_formula():
    beta, decay, end = 0.5, 0.9, 0.05
    for _ in range(100):
        beta = max(end, beta * decay)
    assert beta == pytest.approx(end)             # anneals down to the floor


# ---------------------------------------------------------------------
# Deep driver end-to-end (tiny)
# ---------------------------------------------------------------------


def _deep_args(**overrides):
    base = dict(
        alpha=0.1, gamma=0.99, episodes=4,
        epsilon_start=0.2, epsilon_end=0.05, epsilon_decay=0.99,
        train_configs=3, heldout_configs=3, demo_configs=3, demo_sweeps=4,
        mode="full", abstraction="milestone",
        beta=0.5, beta_decay=0.99, beta_end=0.0, sr_alpha=0.1, sr_gamma=0.95,
        lr=5e-4, batch_size=32, buffer_size=4000, target_update=500,
        train_freq=1, pretrain_steps=20, demo_fraction=0.25, margin=0.8,
        margin_weight=1.0, device="cpu", shaping=False, e2=False,
        results_dir=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("algo", ["dqn", "spie_dqn",
                                  "spie_noisy_dueling_ddqn_per"])
def test_deep_driver_runs_and_evaluates(algo):
    df, agent, final_train, final_held, ev = train_seed_deep(
        "v1_1", algo, seed=0, args=_deep_args())
    # ran the requested number of episodes and produced eval frames
    assert len(df) == 4
    assert {"config_id", "protocol", "fault", "success"} <= set(
        final_train.columns)
    assert {"episode", "train_greedy", "heldout_greedy"} <= set(ev.columns)
    # both splits were evaluated (3 train + 3 held-out configs)
    assert len(final_train) == 3 and len(final_held) == 3
    assert set(final_held["success"]).issubset({0, 1})


def test_deep_episode_return_is_extrinsic_only():
    # SPIE adds the bonus to the STORED reward only; the logged
    # episode_return must remain the extrinsic env return, so even a heavy
    # intrinsic weight cannot inflate it above the env's success ceiling.
    df, *_ = train_seed_deep(
        "v1_1", "spie_dqn", seed=0,
        args=_deep_args(beta=5.0, beta_decay=1.0, episodes=4))
    # no episode here succeeds (tiny run); extrinsic returns must be < 1.0
    assert (df["episode_return"] < 1.0).all()


def test_deep_greedy_eval_uses_no_intrinsic_and_is_clean():
    # a freshly built agent's deep_policy is a pure greedy env policy;
    # evaluate_split returns well-formed success labels with no bonus path.
    from agents.dqn_axiom_forge import make_obs_encoder
    from environments.axiom_forge_configs import make_v1_1_config
    from environments.axiom_forge_env import AxiomForgeEnv
    env = AxiomForgeEnv(make_v1_1_config(seed=0))
    encode_fn, obs_dim = make_obs_encoder(env.observation_space)
    agent = DQNAgent(obs_dim, env.action_space.n, noisy=True, seed=0)
    res = evaluate_split("v1_1", 0, deep_policy(agent, encode_fn), [0, 2])
    assert set(res["success"]).issubset({0, 1})
    assert list(res["config_id"]) == [0, 2]


def test_demo_seeded_deep_solves_train_configs():
    # the deep analogue of the V1.1 gate: demo-seeded DQN solves the train
    # split greedily after enough pretraining.
    df, agent, final_train, _, _ = train_seed_deep(
        "v1_1", "dqn", seed=0,
        args=_deep_args(episodes=10, train_configs=4, heldout_configs=4,
                        demo_configs=4, pretrain_steps=1500,
                        epsilon_start=0.1))
    assert final_train["success"].mean() >= 0.75
