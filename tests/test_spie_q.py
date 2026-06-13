"""
Tests for SPIE-Q (agents/spie_q_agent.py + the training script):
SR/PR TD arithmetic, bonus behavior (novelty decay, bottleneck term,
mode ablations), the milestone abstraction, the beta=0/none reduction
to vanilla Q-learning (bit-identical), and CSV-schema compatibility
with the canonical baseline.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.spie_q_agent import (
    SuccessorPredecessorTables,
    make_abstraction,
    milestone_abstraction,
)
from environments.axiom_forge_configs import make_v1_0_config
from environments.axiom_forge_env import AxiomForgeEnv
from scripts.train_q_learning_axiom_forge import (
    encode_state,
    train_seed as q_train_seed,
)
from scripts.train_spie_q_axiom_forge import (
    MILESTONE_NAMES,
    train_seed as spie_train_seed,
)


# ---------------------------------------------------------------------
# SR / PR arithmetic
# ---------------------------------------------------------------------


def test_sr_td_update_arithmetic():
    t = SuccessorPredecessorTables(sr_alpha=0.5, sr_gamma=0.9)
    t.update("a", "b")
    # M[a] starts {a:1}; target = onehot(a) + 0.9*M[b]ie {a:1, b:0.9}
    # M[a] <- {a:1} + 0.5*({a:1,b:0.9} - {a:1}) = {a:1.0, b:0.45}
    assert t.M["a"]["a"] == pytest.approx(1.0)
    assert t.M["a"]["b"] == pytest.approx(0.45)


def test_pr_td_update_arithmetic_reversed_chain():
    t = SuccessorPredecessorTables(pr_alpha=0.5, pr_gamma=0.9)
    t.update("a", "b")
    # N[b] starts {b:1}; target = onehot(b) + 0.9*N[a] = {b:1, a:0.9}
    assert t.N["b"]["b"] == pytest.approx(1.0)
    assert t.N["b"]["a"] == pytest.approx(0.45)


def test_bonus_decays_with_visitation():
    t = SuccessorPredecessorTables()
    novel = t.bonus("never_seen")
    for _ in range(50):
        t.update("hub", "spoke")
        t.update("spoke", "hub")
    visited = t.bonus("hub")
    assert novel == pytest.approx(1.0)   # unit-norm identity rows
    assert visited < novel


def test_bonus_modes():
    t = SuccessorPredecessorTables()
    for _ in range(20):
        t.update("a", "b")
    full = t.bonus("b", mode="full")
    sr = t.bonus("b", mode="sr_only")
    pr = t.bonus("b", mode="pr_only")
    none = t.bonus("b", mode="none")
    assert none == 0.0
    assert full > 0 and sr > 0 and pr > 0
    with pytest.raises(ValueError):
        t.bonus("b", mode="bogus")


# ---------------------------------------------------------------------
# Abstractions
# ---------------------------------------------------------------------


def test_milestone_abstraction_counts_milestones():
    env = AxiomForgeEnv(make_v1_0_config())
    obs, _ = env.reset()
    assert milestone_abstraction(obs) == (0, 1, 1, 0)
    # walk to M (two steps right) and read it -> milestone count 1
    obs, *_ = env.step(3)
    obs, *_ = env.step(3)
    obs, *_ = env.step(4)
    assert milestone_abstraction(obs) == (0, 1, 3, 1)


def test_make_abstraction_resolves_names():
    assert make_abstraction("milestone", encode_state) is milestone_abstraction
    assert make_abstraction("full", encode_state) is encode_state
    with pytest.raises(ValueError):
        make_abstraction("bogus", encode_state)


# ---------------------------------------------------------------------
# Reduction to vanilla Q-learning (--mode none)
# ---------------------------------------------------------------------


def spie_args(**overrides):
    base = dict(
        episodes=40, alpha=0.1, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.998,
        mode="none", abstraction="milestone",
        beta=0.5, beta_decay=0.999, beta_end=0.0,
        sr_alpha=0.1, sr_gamma=0.95,
        shaping=False, demo_sweeps=0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_mode_none_is_bit_identical_to_vanilla_q_learning():
    df_spie, q_spie, _, _ = spie_train_seed(0, spie_args())
    df_q, q_q = q_train_seed(
        seed=0, num_episodes=40, alpha=0.1, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.998,
        shaping=False, demo_sweeps=0,
    )
    assert list(df_spie["episode_return"]) == list(df_q["episode_return"])
    assert list(df_spie["steps"]) == list(df_q["steps"])
    assert len(q_spie) == len(q_q)
    for state, values in q_q.items():
        assert np.allclose(q_spie[state], values)


def test_mode_full_changes_training_but_logs_extrinsic_return():
    df, q_table, tables, milestones = spie_train_seed(
        0, spie_args(mode="full", episodes=30))
    # SR/PR actually learned something
    assert len(tables.M) > 1 and len(tables.N) > 1
    # episode_return logs EXTRINSIC return: with step_penalty -0.01 and
    # max_steps 250, no episode's return can exceed the success ceiling -
    # and unsuccessful exploration episodes must be near -2.5, NOT
    # inflated by intrinsic bonus.
    no_success = df[~df["success"]]
    assert (no_success["episode_return"] < 0).all()
    assert set(milestones.keys()) == set(MILESTONE_NAMES)


def test_spie_rows_match_canonical_schema():
    df_spie, *_ = spie_train_seed(0, spie_args(episodes=3))
    df_q, _ = q_train_seed(
        seed=0, num_episodes=3, alpha=0.1, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.998,
        shaping=False, demo_sweeps=0,
    )
    assert list(df_spie.columns) == list(df_q.columns)


def test_demo_seeded_spie_solves_v1_0():
    # demo seeding + SPIE must not break the verified baseline behavior
    df, q_table, *_ = spie_train_seed(
        0, spie_args(mode="full", episodes=60, demo_sweeps=50,
                     epsilon_start=0.1, epsilon_decay=0.99))
    from scripts.axiom_forge_baselines_common import evaluate_greedy
    assert evaluate_greedy(q_table, episodes=5) == 1.0
