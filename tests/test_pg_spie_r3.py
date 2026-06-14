"""
Tests for R3 trap-aware PER: the observable is_trap plumbing, the hard-cap-at-
batch-median priority rule (and that naive PER does NOT cap), the diagnostics,
and the two new variants.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.dqn_axiom_forge import DQNAgent, PrioritizedReplayBuffer
from scripts.train_axiom_forge import ALL_ALGOS, DEEP_ALGOS, train_seed_deep

OBS = 4


def _per(trap_aware):
    return PrioritizedReplayBuffer(capacity=64, obs_dim=OBS, demo_capacity=0,
                                   trap_aware=trap_aware)


def _add(buf, trap):
    z = np.zeros(OBS, dtype=np.float32)
    return buf.add(z, 0, 0.0, z, False, trap=trap)


# ---------------------------------------------------------------------
# is_trap plumbing
# ---------------------------------------------------------------------


def test_buffer_records_trap_flag():
    buf = _per(True)
    i0 = _add(buf, trap=False)
    i1 = _add(buf, trap=True)
    assert buf.is_trap[i0] == False and buf.is_trap[i1] == True


def test_agent_store_passes_trap():
    agent = DQNAgent(OBS, 6, per=True, trap_aware=True, seed=0)
    z = np.zeros(OBS, dtype=np.float32)
    agent.store(z, 0, 0.0, z, False, trap=True)
    assert isinstance(agent.buffer, PrioritizedReplayBuffer)
    assert bool(agent.buffer.is_trap[0]) is True


# ---------------------------------------------------------------------
# The hard-cap-at-batch-median rule
# ---------------------------------------------------------------------


def test_trap_aware_caps_trap_priority_at_batch_median():
    buf = _per(trap_aware=True)
    idx = [_add(buf, trap=False) for _ in range(4)] + [_add(buf, trap=True)]
    indices = np.array(idx)
    td = np.array([0.1, 0.2, 0.3, 0.4, 100.0])     # trap has the huge TD error
    buf.update_priorities(indices, td)
    prios = (np.abs(td) + buf.priority_eps) ** buf.alpha
    median = float(np.median(prios))
    # the trap transition is capped at the batch median ...
    assert buf.tree.get(idx[4]) <= median + 1e-9
    assert buf.stat_capped >= 1
    # ... while a non-trap transition keeps its (uncapped) priority
    assert buf.tree.get(idx[3]) == pytest.approx(prios[3])


def test_naive_per_does_not_cap():
    buf = _per(trap_aware=False)
    idx = [_add(buf, trap=False) for _ in range(4)] + [_add(buf, trap=True)]
    indices = np.array(idx)
    td = np.array([0.1, 0.2, 0.3, 0.4, 100.0])
    buf.update_priorities(indices, td)
    prios = (np.abs(td) + buf.priority_eps) ** buf.alpha
    # the high-TD trap transition keeps its full (large) priority
    assert buf.tree.get(idx[4]) == pytest.approx(prios[4])
    assert buf.stat_capped == 0


def test_trap_insertion_priority_capped_under_trap_aware():
    buf = _per(trap_aware=True)
    # drive the running median below max via an update
    idx = [_add(buf, trap=False) for _ in range(3)]
    buf.update_priorities(np.array(idx), np.array([0.1, 0.1, 0.1]))
    med = buf._median_priority
    buf.max_priority = 10.0                          # force max >> median
    t = _add(buf, trap=True)
    assert buf.tree.get(t) <= med + 1e-9             # entered capped, not at max


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------


def test_per_stats_shape_and_ratio():
    buf = _per(trap_aware=True)
    idx = [_add(buf, trap=(i == 0)) for i in range(5)]
    rng = np.random.default_rng(0)
    for _ in range(3):
        buf.sample(8, rng)
    buf.update_priorities(np.array(idx), np.ones(5))
    s = buf.per_stats()
    assert {"mean_trap_priority", "mean_nontrap_priority", "trap_replay_ratio",
            "n_capped_trap", "cap_median_value"} <= set(s)
    assert 0.0 <= s["trap_replay_ratio"] <= 1.0


# ---------------------------------------------------------------------
# Variant registration
# ---------------------------------------------------------------------


def test_r3_variants_registered():
    naive = DEEP_ALGOS["pg_spie_r2_noisy_dueling_ddqn_per"]
    taper = DEEP_ALGOS["pg_spie_r2_taper_noisy_dueling_ddqn_per"]
    for sw in (naive, taper):
        assert sw["per"] and sw["spie"] and sw.get("pg") and sw.get("pp")
    assert naive.get("taper", False) is False
    assert taper.get("taper") is True
    assert "pg_spie_r2_taper_noisy_dueling_ddqn_per" in ALL_ALGOS
    # naive == taper minus the taper knob
    assert {k: v for k, v in taper.items() if k != "taper"} == naive


# ---------------------------------------------------------------------
# Integration: trap-aware variant trains and exposes stats
# ---------------------------------------------------------------------


def _deep_args(**ov):
    base = dict(
        episodes=2, gamma=0.99, lr=5e-4, batch_size=16, buffer_size=2000,
        target_update=50, train_freq=1, pretrain_steps=0, demo_configs=0,
        demo_sweeps=3, demo_fraction=0.25, margin=0.8, margin_weight=1.0,
        train_configs=2, heldout_configs=2, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay=0.999, mode="full", abstraction="milestone", beta=0.5,
        beta_decay=0.999, beta_end=0.0, sr_alpha=0.1, sr_gamma=0.95, kappa=4.0,
        device="cpu", shaping=False, e2=True)
    base.update(ov)
    return argparse.Namespace(**base)


def test_trap_aware_variant_trains_and_reports_stats():
    df, agent, *_ = train_seed_deep(
        "v1_4", "pg_spie_r2_taper_noisy_dueling_ddqn_per", 0, _deep_args())
    assert len(df) == 2
    assert isinstance(agent.buffer, PrioritizedReplayBuffer)
    assert agent.buffer.trap_aware is True
    s = agent.buffer.per_stats()
    assert s["trap_sampled"] + s["nontrap_sampled"] > 0
