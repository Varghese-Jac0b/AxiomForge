"""
Tests for R2 = PG-SPIE + proxy-coupled penalty. Covers the pure gate->intrinsic
mapping (gated_intrinsic) including the kappa-scaled proxy penalty, the new
variant's switches, and the per-episode sink invariants (every proxy step
penalized; net intrinsic = allowed - penalty; R1 unchanged = no penalty).
"""

import argparse
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_axiom_forge import (
    ALL_ALGOS,
    DEEP_ALGOS,
    gated_intrinsic,
    train_seed_deep,
)


def _deep_args(**ov):
    base = dict(
        episodes=3, gamma=0.99, lr=5e-4,
        batch_size=16, buffer_size=2000, target_update=50, train_freq=1,
        pretrain_steps=0, demo_configs=0, demo_sweeps=3,
        demo_fraction=0.25, margin=0.8, margin_weight=1.0,
        train_configs=2, heldout_configs=2,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.999,
        mode="full", abstraction="milestone",
        beta=0.5, beta_decay=0.999, beta_end=0.0,
        sr_alpha=0.1, sr_gamma=0.95, kappa=1.0,
        device="cpu", shaping=False, e2=True,
    )
    base.update(ov)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------
# Pure gate -> intrinsic mapping
# ---------------------------------------------------------------------


def test_advance_keeps_bonus():
    assert gated_intrinsic("advance", 0.8, proxy_penalty_enabled=True,
                           kappa=2.0) == (0.8, 0.8, 0.0, 0.0)


def test_neutral_suppresses():
    assert gated_intrinsic("neutral", 0.8, proxy_penalty_enabled=True,
                           kappa=2.0) == (0.0, 0.0, 0.8, 0.0)


def test_proxy_r1_suppresses_when_penalty_off():
    # R1 behaviour: proxy is merely suppressed (no penalty)
    assert gated_intrinsic("proxy", 0.8, proxy_penalty_enabled=False,
                           kappa=2.0) == (0.0, 0.0, 0.8, 0.0)


def test_proxy_r2_applies_negative_penalty():
    final, allowed, suppressed, penalty = gated_intrinsic(
        "proxy", 0.8, proxy_penalty_enabled=True, kappa=2.0)
    assert (allowed, suppressed) == (0.0, 0.0)
    assert penalty == pytest.approx(1.6)
    assert final == pytest.approx(-1.6)        # repulsion added to r_train


def test_kappa_scales_penalty_linearly():
    p1 = gated_intrinsic("proxy", 0.8, proxy_penalty_enabled=True, kappa=1.0)[3]
    p2 = gated_intrinsic("proxy", 0.8, proxy_penalty_enabled=True, kappa=2.0)[3]
    assert p2 == pytest.approx(2 * p1)


# ---------------------------------------------------------------------
# Variant registration
# ---------------------------------------------------------------------


def test_r2_variant_registered():
    sw = DEEP_ALGOS["pg_spie_r2_noisy_dueling_ddqn"]
    assert sw["spie"] and sw.get("pg") and sw.get("pp")
    assert sw["noisy"] and sw["dueling"] and sw["per"] is False
    assert "pg_spie_r2_noisy_dueling_ddqn" in ALL_ALGOS
    # identical to R1 except the new pp knob
    r1 = DEEP_ALGOS["pg_spie_noisy_dueling_ddqn"]
    assert {k: v for k, v in sw.items() if k != "pp"} == r1


# ---------------------------------------------------------------------
# Per-episode sink invariants
# ---------------------------------------------------------------------


def test_r2_sink_invariants():
    sink: list = []
    args = _deep_args(episodes=4, kappa=1.0)
    df, *_ = train_seed_deep("v1_4", "pg_spie_r2_noisy_dueling_ddqn", 0, args,
                             metrics_sink=sink)
    assert len(sink) == 4
    for row in sink:
        # under R2 every proxy step is penalized
        assert row["proxy_penalty_count"] == row["gate_proxy"]
        assert row["proxy_penalty_total"] >= 0.0
        # net intrinsic added to reward = allowed bonus - proxy penalty
        assert row["intrinsic_contribution"] == pytest.approx(
            row["allowed_intrinsic"] - row["proxy_penalty_total"])


def test_r1_variant_has_no_penalty():
    sink: list = []
    args = _deep_args(episodes=3)
    train_seed_deep("v1_4", "pg_spie_noisy_dueling_ddqn", 0, args,
                    metrics_sink=sink)
    for row in sink:
        assert row["proxy_penalty_total"] == 0.0
        assert row["proxy_penalty_count"] == 0
        # R1: net intrinsic equals the allowed bonus (no penalty term)
        assert row["intrinsic_contribution"] == pytest.approx(
            row["allowed_intrinsic"])


def test_r2_deterministic():
    a = _deep_args(episodes=3, kappa=1.5)
    d1, *_ = train_seed_deep("v1_4", "pg_spie_r2_noisy_dueling_ddqn", 0, a)
    d2, *_ = train_seed_deep("v1_4", "pg_spie_r2_noisy_dueling_ddqn", 0, a)
    assert d1["episode_return"].tolist() == d2["episode_return"].tolist()
    assert d1["steps"].tolist() == d2["steps"].tolist()
