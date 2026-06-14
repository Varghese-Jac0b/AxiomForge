"""
Tests for R1 Protocol-Gated SPIE (the `protocol_gate` decision + the gated
intrinsic in train_seed_deep). Confirms: the gate is observable-only and
correctly classifies advance / neutral / proxy with proxy precedence; the new
variant is registered; the gate suppresses (zeros) intrinsic on non-advance
steps; per-episode gate tallies are consistent; and it is deterministic.
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
    protocol_gate,
    train_seed_deep,
)


def make_obs(*, attempt=0, claim=0, manifest=0, archive=0, diagnostics=0,
             sample=0, purity=0, analyzed=0, layer=0, row=1, col=1):
    return {
        "agent": [layer, row, col],
        "work_order": [manifest, 0, 0, 0, 0],
        "knowledge_bits": [archive, diagnostics, 0, 0, 0, 0, 0, 0, 0],
        "sample_state": [sample, purity, 0, 0, 0, analyzed],
        "proxy_attempt_count": attempt,
        "proxy_claim_count": claim,
    }


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
        sr_alpha=0.1, sr_gamma=0.95,
        device="cpu", shaping=False, e2=True,
    )
    base.update(ov)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------
# protocol_gate decision (observable-only)
# ---------------------------------------------------------------------


def test_gate_advance_on_milestone_increase():
    prev = make_obs(manifest=0)
    obs = make_obs(manifest=1)            # +1 milestone, no proxy change
    assert protocol_gate(prev, obs) == "advance"


def test_gate_neutral_when_nothing_changes():
    prev = make_obs(manifest=1)
    obs = make_obs(manifest=1, col=2)     # moved, no milestone, no proxy
    assert protocol_gate(prev, obs) == "neutral"


@pytest.mark.parametrize("field", ["attempt", "claim"])
def test_gate_proxy_on_counter_increase(field):
    prev = make_obs()
    obs = make_obs(**{field: 1})
    assert protocol_gate(prev, obs) == "proxy"


def test_proxy_takes_precedence_over_advance():
    # even if a milestone also increased, a proxy increment suppresses
    prev = make_obs(manifest=0, attempt=0)
    obs = make_obs(manifest=1, attempt=1)
    assert protocol_gate(prev, obs) == "proxy"


def test_gate_is_observable_only():
    # no true_score / hidden_context anywhere in the inputs
    prev, obs = make_obs(), make_obs(manifest=1)
    assert "true_score" not in obs and "debug" not in obs
    assert protocol_gate(prev, obs) in {"advance", "neutral", "proxy"}


# ---------------------------------------------------------------------
# Variant registration
# ---------------------------------------------------------------------


def test_pg_variant_registered():
    assert "pg_spie_noisy_dueling_ddqn" in DEEP_ALGOS
    assert "pg_spie_noisy_dueling_ddqn" in ALL_ALGOS
    sw = DEEP_ALGOS["pg_spie_noisy_dueling_ddqn"]
    assert sw["spie"] and sw.get("pg") and sw["noisy"] and sw["dueling"]
    assert sw["per"] is False        # PER stays OFF at R1
    # identical to naive SPIE except the pg knob
    naive = DEEP_ALGOS["spie_noisy_dueling_ddqn"]
    assert {k: v for k, v in sw.items() if k != "pg"} == naive


# ---------------------------------------------------------------------
# Gated training: tallies + suppression are consistent
# ---------------------------------------------------------------------


def test_gate_tallies_and_suppression():
    sink: list = []
    args = _deep_args(episodes=3)
    df, *_ = train_seed_deep("v1_4", "pg_spie_noisy_dueling_ddqn", 0, args,
                             metrics_sink=sink)
    assert len(sink) == 3
    for row, (_, ep) in zip(sink, df.iterrows()):
        # the gate is evaluated on every SPIE step == every env step
        total_gate = row["gate_advance"] + row["gate_neutral"] + row["gate_proxy"]
        assert total_gate == int(ep["steps"])
        # ep_intrinsic accumulates ONLY the allowed bonus
        assert row["intrinsic_contribution"] == pytest.approx(
            row["allowed_intrinsic"])
        assert row["allowed_intrinsic"] >= 0.0
        assert row["suppressed_intrinsic"] >= 0.0
    # in the no-demo V1.4 regime the agent spends most steps NOT advancing,
    # so the gate must have suppressed some intrinsic somewhere
    assert sum(r["suppressed_intrinsic"] for r in sink) > 0.0


def test_naive_spie_has_zero_gate_tallies():
    sink: list = []
    args = _deep_args(episodes=2)
    train_seed_deep("v1_4", "spie_noisy_dueling_ddqn", 0, args, metrics_sink=sink)
    for row in sink:
        assert (row["gate_advance"], row["gate_neutral"],
                row["gate_proxy"]) == (0, 0, 0)
        # ungated: intrinsic flows through unblocked, suppression is zero
        assert row["suppressed_intrinsic"] == 0.0


def test_pg_training_is_deterministic():
    a = _deep_args(episodes=3)
    d1, *_ = train_seed_deep("v1_4", "pg_spie_noisy_dueling_ddqn", 0, a)
    d2, *_ = train_seed_deep("v1_4", "pg_spie_noisy_dueling_ddqn", 0, a)
    assert d1["episode_return"].tolist() == d2["episode_return"].tolist()
    assert d1["steps"].tolist() == d2["steps"].tolist()
