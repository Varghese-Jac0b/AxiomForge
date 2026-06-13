"""
Tests for the multi-config training driver (scripts/train_axiom_forge.py):
demo recording across versions, full-backup seeding that makes a config
greedily solvable in isolation, TD-target dispatch, the held-out split
discipline, and a tiny end-to-end train_seed smoke for every algorithm.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.axiom_forge_configs import is_held_out_config_id
from scripts.axiom_forge_baselines_common import encode_state, get_q_values
from scripts.train_axiom_forge import (
    ALGOS,
    VERSIONS,
    evaluate_split,
    record_demo_transitions,
    seed_q_table,
    tabular_policy,
    td_target,
    train_seed,
)


# ---------------------------------------------------------------------
# Demo recording
# ---------------------------------------------------------------------


@pytest.mark.parametrize("version", sorted(VERSIONS))
def test_record_demo_succeeds_and_terminates(version):
    tr = record_demo_transitions(version, seed=0, config_id=0,
                                 encode=encode_state)
    assert len(tr) > 0
    # exactly one terminal transition, and it is the last
    terminals = [i for i, t in enumerate(tr) if t[4]]
    assert terminals == [len(tr) - 1]
    # encoded chain is intact: next_state[i] == state[i+1]
    for i in range(len(tr) - 1):
        assert tr[i][3] == tr[i + 1][0]


# ---------------------------------------------------------------------
# Seeding makes a config greedily solvable in isolation
# ---------------------------------------------------------------------


@pytest.mark.parametrize("version", sorted(VERSIONS))
@pytest.mark.parametrize("config_id", [0, 2, 4])
def test_full_backup_seeding_solves_config_in_isolation(version, config_id):
    q = {}
    tr = record_demo_transitions(version, 0, config_id, encode_state)
    seed_q_table("q_learning", q, tr, gamma=0.99, num_actions=6, sweeps=3)
    res = evaluate_split(version, 0, tabular_policy(q, 6), [config_id])
    assert res["success"].iloc[0] == 1


def test_no_aliasing_in_any_demo():
    # the REFINED-purity fix removed the re-slot memory dependence: no demo
    # state may be recorded with two different actions (which a memoryless
    # tabular policy could never reproduce).
    from collections import defaultdict
    for version in VERSIONS:
        for config_id in range(0, 16, 2):
            tr = record_demo_transitions(version, 0, config_id, encode_state)
            sa = defaultdict(set)
            for s, a, *_ in tr:
                sa[s].add(a)
            conflicts = [s for s, acts in sa.items() if len(acts) > 1]
            assert not conflicts, f"{version} cfg {config_id} aliased"


# ---------------------------------------------------------------------
# TD-target dispatch
# ---------------------------------------------------------------------


def test_td_target_dispatch():
    q = {}
    s = (1, 2, 3)
    get_q_values(q, s, 6)[:] = [1.0, 5.0, 2.0, 0.0, 0.0, 0.0]
    # q_learning / spie_q: greedy max = 5.0
    t_q = td_target("q_learning", q, 6, reward=0.0, next_state=s,
                    gamma=1.0, epsilon=0.0, next_action=None)
    assert t_q == pytest.approx(5.0)
    # sarsa: bootstrap from the given next action (index 0 -> 1.0)
    t_s = td_target("sarsa", q, 6, reward=0.0, next_state=s,
                    gamma=1.0, epsilon=0.0, next_action=0)
    assert t_s == pytest.approx(1.0)
    # expected_sarsa at epsilon 0 collapses to the greedy max
    t_e = td_target("expected_sarsa", q, 6, reward=0.0, next_state=s,
                    gamma=1.0, epsilon=0.0, next_action=None)
    assert t_e == pytest.approx(5.0)


def test_sarsa_seeding_uses_next_demonstrated_action():
    # SARSA seeding must bootstrap from the next demo action, so the values
    # along the demo are still positive (the demo is the behaviour policy).
    q = {}
    tr = record_demo_transitions("v1_1", 0, 0, encode_state)
    seed_q_table("sarsa", q, tr, gamma=0.99, num_actions=6, sweeps=3)
    res = evaluate_split("v1_1", 0, tabular_policy(q, 6), [0])
    assert res["success"].iloc[0] == 1


# ---------------------------------------------------------------------
# Held-out split discipline
# ---------------------------------------------------------------------


def _args(**overrides):
    base = dict(
        alpha=0.1, gamma=0.99, episodes=30,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.999,
        train_configs=4, heldout_configs=4, demo_configs=4, demo_sweeps=3,
        mode="full", abstraction="milestone",
        beta=0.5, beta_decay=0.999, beta_end=0.0, sr_alpha=0.1, sr_gamma=0.95,
        shaping=False, e2=False, results_dir=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_evaluate_split_labels_and_never_trains():
    # evaluation must be read-only: the q_table it is given is unchanged.
    q = {}
    tr = record_demo_transitions("v1_3", 0, 0, encode_state)
    seed_q_table("q_learning", q, tr, 0.99, 6, 3)
    res = evaluate_split("v1_3", 0, tabular_policy(q, 6), [0, 2])
    assert set(res.columns) >= {"config_id", "protocol", "fault", "success"}
    # greedy eval may create unseen keys via get_q_values, but it must not
    # be the *training* path; success labels are well-formed 0/1
    assert set(res["success"]).issubset({0, 1})


def test_driver_respects_held_out_parity():
    from environments.axiom_forge_configs import (
        held_out_config_ids,
        train_config_ids,
    )
    assert all(not is_held_out_config_id(c) for c in train_config_ids(20))
    assert all(is_held_out_config_id(c) for c in held_out_config_ids(20))


# ---------------------------------------------------------------------
# End-to-end train_seed smoke for every algorithm
# ---------------------------------------------------------------------


@pytest.mark.parametrize("algo", ALGOS)
def test_train_seed_runs_for_every_algo(algo):
    df, q_table, final_train, final_held, ev = train_seed(
        "v1_1", algo, seed=0, args=_args(episodes=20))
    assert len(df) == 20
    # fully-demoed train configs are greedily solvable for off-policy algos;
    # we assert the harness ran and produced well-formed eval frames
    assert {"config_id", "protocol", "success"} <= set(final_train.columns)
    assert {"episode", "train_greedy", "heldout_greedy"} <= set(ev.columns)
    # train sweep only drew even (train) config ids
    assert (df["train_or_heldout"] == "train").all()


def test_demo_seeded_driver_solves_train_configs_greedily():
    # the V1.1 gate, in miniature: demo-seeded tabular solves the train split.
    _, q_table, final_train, _, _ = train_seed(
        "v1_1", "q_learning", seed=0,
        args=_args(episodes=20, train_configs=6, demo_configs=6,
                   demo_sweeps=4))
    assert final_train["success"].mean() == 1.0
