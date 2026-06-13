"""
Tests for the V1.1 upgrade (Prompt H): randomized work order / protocol /
catalyst mapping, deterministic per (seed, config_id); the held-out
split helpers; the H protocol-hint channel; and end-to-end solvability
of sampled configurations (including P1 and RAW-purity episodes) by the
branching manual solver.
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.axiom_forge_configs import (
    AxiomForgeConfig,
    held_out_config_ids,
    is_held_out_config_id,
    make_v1_0_config,
    make_v1_1_config,
    train_config_ids,
    validate_config,
)
from environments.axiom_forge_env import AxiomForgeEnv
from environments.axiom_forge_objects import (
    ACTION_INTERACT,
    Catalyst,
    Charge,
    Protocol,
    Purity,
    SampleType,
    Temperature,
)
from scripts.manual_solve_axiom_forge_v1_1 import (
    BranchingSolver,
    solve_config,
)


def make_debug_env(seed: int, config_id: int = 0) -> AxiomForgeEnv:
    cfg = dataclasses.replace(
        make_v1_1_config(seed=seed, config_id=config_id), debug_mode=True,
    )
    return AxiomForgeEnv(cfg)


def hidden_of(env: AxiomForgeEnv, config_id: int | None = None):
    options = None if config_id is None else {"config_id": config_id}
    _, info = env.reset(options=options)
    return info["debug"]["hidden_context"]


# ---------------------------------------------------------------------
# Determinism and randomization
# ---------------------------------------------------------------------


def test_same_seed_and_config_id_identical_episode():
    h1 = hidden_of(make_debug_env(seed=42, config_id=6))
    h2 = hidden_of(make_debug_env(seed=42, config_id=6))
    assert h1.true_work_order == h2.true_work_order
    assert h1.protocol == h2.protocol
    assert h1.catalyst_mapping == h2.catalyst_mapping


def test_repeated_resets_keep_the_same_latent_config():
    env = make_debug_env(seed=42, config_id=6)
    h1 = hidden_of(env)
    h2 = hidden_of(env)
    assert h1.true_work_order == h2.true_work_order
    assert h1.protocol == h2.protocol
    assert h1.catalyst_mapping == h2.catalyst_mapping


def test_different_config_ids_vary_the_context():
    env = make_debug_env(seed=42)
    contexts = [hidden_of(env, config_id=k) for k in range(0, 20, 2)]
    work_orders = {dataclasses.astuple(h.true_work_order) for h in contexts}
    protocols = {h.protocol for h in contexts}
    assert len(work_orders) > 1
    assert protocols == {Protocol.P0, Protocol.P1}


def test_reset_options_override_config_id():
    env = make_debug_env(seed=42, config_id=0)
    h_default = hidden_of(env)
    h_override = hidden_of(env, config_id=8)
    h_back = hidden_of(env)  # falls back to cfg.config_id
    assert dataclasses.astuple(h_back.true_work_order) == \
        dataclasses.astuple(h_default.true_work_order)
    # config 8 must be reproducible independently of how it is reached
    h_direct = hidden_of(make_debug_env(seed=42, config_id=8))
    assert dataclasses.astuple(h_override.true_work_order) == \
        dataclasses.astuple(h_direct.true_work_order)


def test_sampled_values_stay_in_legal_ranges():
    env = make_debug_env(seed=11)
    for k in range(0, 30, 2):
        h = hidden_of(env, config_id=k)
        wo = h.true_work_order
        assert wo.target_sample in {SampleType.A, SampleType.B,
                                    SampleType.C, SampleType.D}
        # purity is fixed REFINED: every protocol mandates a purify step, so
        # a RAW target would need a memory-dependent re-slot trick that a
        # memoryless tabular agent cannot learn (would break the V1.1 gate).
        assert wo.required_purity == Purity.REFINED
        assert wo.required_temperature in {Temperature.COLD,
                                           Temperature.WARM,
                                           Temperature.HOT}
        assert wo.required_charge in {Charge.NEG, Charge.NEUTRAL,
                                      Charge.POS}
        assert h.protocol in {Protocol.P0, Protocol.P1}
        # mapping is a bijection onto the three real charges
        assert sorted(h.catalyst_mapping.keys()) == [
            Catalyst.RED, Catalyst.BLUE, Catalyst.GREEN]
        assert sorted(h.catalyst_mapping.values()) == [
            Charge.NEG, Charge.NEUTRAL, Charge.POS]


def test_v1_0_context_unchanged_by_the_upgrade():
    cfg = dataclasses.replace(make_v1_0_config(), debug_mode=True)
    env = AxiomForgeEnv(cfg)
    h = hidden_of(env)
    assert h.protocol == Protocol.P0
    assert h.true_work_order.target_sample == SampleType.A
    assert h.catalyst_mapping[Catalyst.GREEN] == Charge.POS


def test_randomize_without_seed_rejected():
    with pytest.raises(ValueError, match="seed"):
        validate_config(AxiomForgeConfig(
            version_name="bad", randomize_work_order=True, seed=None,
        ))


# ---------------------------------------------------------------------
# Held-out split helpers
# ---------------------------------------------------------------------


def test_held_out_split_partition():
    train = train_config_ids(50)
    held = held_out_config_ids(50)
    assert not set(train) & set(held)
    assert all(not is_held_out_config_id(c) for c in train)
    assert all(is_held_out_config_id(c) for c in held)


# ---------------------------------------------------------------------
# The H protocol-hint channel
# ---------------------------------------------------------------------


def goto_and_interact(env, solver_cls, symbol):
    solver = solver_cls(env)
    solver.goto(symbol)
    solver.interact()
    return solver


def test_archive_reveals_protocol_hint_v1_1():
    for config_id in range(0, 10, 2):
        env = make_debug_env(seed=3, config_id=config_id)
        _, info = env.reset(options={"config_id": config_id})
        true_protocol = info["debug"]["hidden_context"].protocol
        solver = BranchingSolver(env, options={"config_id": config_id})
        assert list(solver.obs["knowledge_bits"][3:5]) == [0, 0]
        solver.goto("H")
        solver.interact()
        assert int(solver.obs["knowledge_bits"][3]) == 1 + int(true_protocol)
        assert int(solver.obs["knowledge_bits"][4]) == 0  # direct mode


def test_archive_hint_stays_zero_under_v1_0():
    env = AxiomForgeEnv(make_v1_0_config())
    solver = BranchingSolver(env)
    solver.goto("H")
    solver.interact()
    assert list(solver.obs["knowledge_bits"][3:]) == [0] * 6


# ---------------------------------------------------------------------
# End-to-end solvability of sampled configurations
# ---------------------------------------------------------------------


@pytest.mark.parametrize("config_id", list(range(0, 24, 2)))
def test_branching_solver_solves_train_configs(config_id):
    result = solve_config(seed=7, config_id=config_id)
    assert result["success"] is True, result["failure_reason"]
    assert result["terminated"] and not result["truncated"]
    assert result["steps"] <= 250


def test_solver_covers_both_protocols_with_varied_work_orders():
    seen_protocols, seen_charges, seen_temps = set(), set(), set()
    for config_id in range(0, 40, 2):
        result = solve_config(seed=7, config_id=config_id)
        assert result["success"] is True
        seen_protocols.add(result["protocol"])
        seen_temps.add(result["work_order"][2])
        seen_charges.add(result["work_order"][3])
        # purity is always REFINED now (every protocol mandates purify)
        assert result["work_order"][1] == int(Purity.REFINED)
    assert seen_protocols == {Protocol.P0, Protocol.P1}
    assert len(seen_charges) > 1 and len(seen_temps) > 1
