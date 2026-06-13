"""
Tests for the V1.2 upgrade (Prompt I): catalyst probe, sample probe,
once-per-target probe rewards, P2 protocol, missing_sample_probe
grading, and probe-station inertness before V1.2.
"""

import dataclasses
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.axiom_forge_configs import (
    make_v1_0_config,
    make_v1_1_config,
    make_v1_2_config,
)
from environments.axiom_forge_env import AxiomForgeEnv
from environments.axiom_forge_objects import (
    Catalyst,
    Protocol,
    STEP_SAMPLE_PROBE,
)
from scripts.manual_solve_axiom_forge_v1_1 import BranchingSolver
from scripts.manual_solve_axiom_forge_v1_2 import (
    ProbeFirstSolver,
    solve_config,
)


def debug_env(seed=3, config_id=0):
    cfg = dataclasses.replace(
        make_v1_2_config(seed=seed, config_id=config_id), debug_mode=True)
    return AxiomForgeEnv(cfg)


def fresh_solver(env, config_id=0):
    return ProbeFirstSolver(env, options={"config_id": config_id})


# ---------------------------------------------------------------------
# Catalyst probe
# ---------------------------------------------------------------------


def test_catalyst_probe_truthful_and_paid_once_per_color():
    env = debug_env()
    solver = fresh_solver(env)
    mapping_truth = env._hidden.catalyst_mapping
    cfg = env.cfg

    solver.goto("P")
    for color in (Catalyst.RED, Catalyst.BLUE, Catalyst.GREEN):
        solver.cycle_selector_to(1, int(color))
        reward, _, _ = solver.interact()
        # first probe of each color pays useful_probe_reward (+ step penalty)
        assert reward == pytest.approx(
            cfg.step_penalty + cfg.useful_probe_reward)
        slot = 2 * (int(color) - 1)
        assert int(solver.obs["probe_results"][slot]) == 1
        assert int(solver.obs["probe_results"][slot + 1]) == int(
            mapping_truth[color])

    # re-probing an already-known color pays nothing extra
    reward, _, _ = solver.interact()
    assert reward == pytest.approx(cfg.step_penalty)


def test_probe_results_zero_before_probing():
    env = debug_env()
    obs, _ = env.reset(options={"config_id": 0})
    assert list(obs["probe_results"]) == [0] * 7


# ---------------------------------------------------------------------
# Sample probe
# ---------------------------------------------------------------------


def test_sample_probe_writes_true_token_and_history():
    env = debug_env()
    solver = fresh_solver(env)
    cfg = env.cfg
    solver.goto("M"); solver.interact()
    target = int(solver.obs["work_order"][1])
    solver.goto("R")
    solver.cycle_selector_to(0, target)
    solver.interact()
    solver.goto("P")
    reward, _, _ = solver.interact()
    assert reward == pytest.approx(cfg.step_penalty + cfg.useful_probe_reward)
    assert int(solver.obs["probe_results"][6]) == target
    assert int(solver.obs["knowledge_bits"][2]) == 1  # sample_probe_used
    assert env._machine_history[-1] == STEP_SAMPLE_PROBE
    # second probe of the sample: history appends again, no second payment
    reward, _, _ = solver.interact()
    assert reward == pytest.approx(cfg.step_penalty)


# ---------------------------------------------------------------------
# missing_sample_probe grading / P2
# ---------------------------------------------------------------------


def find_p2_config(seed=7, limit=60):
    for config_id in range(0, limit, 2):
        env = debug_env(seed=seed, config_id=config_id)
        env.reset(options={"config_id": config_id})
        if env._hidden.protocol == Protocol.P2:
            return config_id
    raise RuntimeError("no P2 config found in range")


def test_p2_fails_without_sample_probe():
    config_id = find_p2_config()
    env = AxiomForgeEnv(make_v1_2_config(seed=7, config_id=config_id))
    # run the V1.1 solver, which never probes: must FAIL on P2 with
    # missing_sample_probe and wrong_order both firing.
    solver = BranchingSolver(env, options={"config_id": config_id})
    # V1.1 solver asserts protocol from hint; P2 hint = 3 works fine.
    result = solver.solve()
    assert result["success"] is False
    assert result["failure_reason"]["missing_sample_probe"] is True
    assert result["failure_reason"]["wrong_order"] is True


def test_p2_succeeds_with_probe_first_solver():
    config_id = find_p2_config()
    result = solve_config(seed=7, config_id=config_id)
    assert result["success"] is True, result["failure_reason"]
    assert result["protocol"] == Protocol.P2


@pytest.mark.parametrize("config_id", list(range(0, 24, 2)))
def test_probe_first_solver_solves_all_v1_2_train_configs(config_id):
    result = solve_config(seed=7, config_id=config_id)
    assert result["success"] is True, result["failure_reason"]
    assert result["steps"] <= 250


def test_v1_2_samples_all_three_protocols():
    protocols = set()
    for config_id in range(0, 40, 2):
        env = debug_env(seed=7, config_id=config_id)
        env.reset(options={"config_id": config_id})
        protocols.add(env._hidden.protocol)
    assert protocols == {Protocol.P0, Protocol.P1, Protocol.P2}


# ---------------------------------------------------------------------
# Inertness before V1.2
# ---------------------------------------------------------------------


@pytest.mark.parametrize("make_cfg", [
    lambda: make_v1_0_config(),
    lambda: make_v1_1_config(seed=3),
])
def test_probe_station_inert_before_v1_2(make_cfg):
    env = AxiomForgeEnv(make_cfg())
    solver = BranchingSolver(env)
    solver.goto("P")
    cfg = env.cfg
    reward, _, _ = solver.interact()
    assert reward == pytest.approx(cfg.step_penalty + cfg.invalid_penalty)
    reward, _, _ = solver._step(5)  # ACTION_CYCLE
    assert reward == pytest.approx(cfg.step_penalty + cfg.invalid_penalty)
    assert list(solver.obs["probe_results"]) == [0] * 7
