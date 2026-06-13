"""
Tests for the V1.3 upgrade (Prompt J): fault sampling, the three fault
behaviors (each detectable by a scripted experiment), decon, the D
reveal channel (direct + differential), the three-dial calibrated
report with its scoring matrix, and safety_failed grading.
"""

import dataclasses
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.axiom_forge_configs import make_v1_3_config
from environments.axiom_forge_env import AxiomForgeEnv, REPORT_SELECTOR_SIZE
from environments.axiom_forge_objects import (
    ACTION_CYCLE,
    AnalyzerToken,
    Fault,
    Protocol,
    Temperature,
)
from scripts.manual_solve_axiom_forge_v1_3 import (
    FaultAwareSolver,
    solve_config,
)

SEED = 7


def debug_cfg(config_id, **overrides):
    cfg = make_v1_3_config(seed=SEED, config_id=config_id)
    return dataclasses.replace(cfg, debug_mode=True, **overrides)


def env_with_fault(fault: Fault, **overrides):
    """Find a train config whose sampled fault is `fault`; return the
    debug env + config_id."""
    for config_id in range(0, 120, 2):
        env = AxiomForgeEnv(debug_cfg(config_id, **overrides))
        env.reset(options={"config_id": config_id})
        if env._hidden.fault == fault:
            return env, config_id
    raise RuntimeError(f"no config with fault {fault}")


# ---------------------------------------------------------------------
# Fault sampling
# ---------------------------------------------------------------------


def test_fault_sampling_deterministic_and_covers_pool():
    faults_first = {}
    for config_id in range(0, 60, 2):
        env = AxiomForgeEnv(debug_cfg(config_id))
        env.reset(options={"config_id": config_id})
        faults_first[config_id] = env._hidden.fault
    for config_id, fault in faults_first.items():
        env = AxiomForgeEnv(debug_cfg(config_id))
        env.reset(options={"config_id": config_id})
        assert env._hidden.fault == fault
    assert set(faults_first.values()) == {
        Fault.NONE, Fault.HEATER_SWAP, Fault.PURIFIER_LEAK,
        Fault.ANALYZER_BIAS,
    }


# ---------------------------------------------------------------------
# Scripted fault-detection experiments (behavioral, no D hint needed)
# ---------------------------------------------------------------------


def place_any_sample(solver):
    solver.goto("R")
    solver.interact()


def test_heater_swap_detectable_by_thermal_experiment():
    env, config_id = env_with_fault(Fault.HEATER_SWAP)
    solver = FaultAwareSolver(env, options={"config_id": config_id})
    place_any_sample(solver)
    solver.goto("E"); solver.interact()
    solver.goto("T")
    solver.cycle_selector_to(2, int(Temperature.COLD))
    solver.interact()
    # the swapped heater applied HOT where COLD was selected
    assert int(solver.obs["sample_state"][2]) == int(Temperature.HOT)
    # WARM is unaffected by the swap
    solver.cycle_selector_to(2, int(Temperature.WARM))
    solver.interact()
    assert int(solver.obs["sample_state"][2]) == int(Temperature.WARM)


def test_purifier_leak_detectable_and_decon_clears():
    env, config_id = env_with_fault(Fault.PURIFIER_LEAK)
    solver = FaultAwareSolver(env, options={"config_id": config_id})
    place_any_sample(solver)
    solver.goto("E"); solver.interact()
    solver.goto("U"); solver.interact()
    assert int(solver.obs["sample_state"][4]) == 1   # contaminated, visible
    solver.goto("V"); solver.interact()
    assert int(solver.obs["sample_state"][4]) == 0   # decon cleared it
    assert env._machine_history[-1] == "decon"


def test_purifier_clean_when_fault_is_other():
    env, config_id = env_with_fault(Fault.HEATER_SWAP)
    solver = FaultAwareSolver(env, options={"config_id": config_id})
    place_any_sample(solver)
    solver.goto("E"); solver.interact()
    solver.goto("U"); solver.interact()
    assert int(solver.obs["sample_state"][4]) == 0


def test_analyzer_bias_lies_but_probe_stays_truthful():
    env, config_id = env_with_fault(Fault.ANALYZER_BIAS)
    truth = env._hidden
    solver = FaultAwareSolver(env, options={"config_id": config_id})
    # sample probe first: ALWAYS truthful
    solver.goto("R")
    solver.cycle_selector_to(0, int(truth.true_work_order.target_sample))
    solver.interact()
    solver.goto("P"); solver.interact()
    assert int(solver.obs["probe_results"][6]) == int(
        truth.true_work_order.target_sample)
    # analyze a NON-matching sample: truthful token would be RED, the
    # biased analyzer shows GREEN (the lie that tempts wrong submissions)
    solver.goto("E"); solver.interact()
    solver.goto("N"); solver.interact()
    assert int(solver.obs["last_analyzer_token"]) == int(AnalyzerToken.GREEN)


def test_analyzer_truthful_when_no_bias_fault():
    env, config_id = env_with_fault(Fault.NONE)
    solver = FaultAwareSolver(env, options={"config_id": config_id})
    place_any_sample(solver)
    solver.goto("E"); solver.interact()
    solver.goto("N"); solver.interact()
    assert int(solver.obs["last_analyzer_token"]) in {
        int(AnalyzerToken.GREEN), int(AnalyzerToken.RED)}


# ---------------------------------------------------------------------
# D reveal channel
# ---------------------------------------------------------------------


def test_diagnostics_reveals_exact_fault_direct_mode():
    for fault in Fault:
        env, config_id = env_with_fault(fault)
        solver = FaultAwareSolver(env, options={"config_id": config_id})
        assert list(solver.obs["knowledge_bits"][5:8]) == [0, 0, 0]
        solver.goto("D"); solver.interact()
        assert int(solver.obs["knowledge_bits"][5]) == 1 + int(fault)
        assert int(solver.obs["knowledge_bits"][6]) == 0
        assert int(solver.obs["knowledge_bits"][7]) == 1  # safety normal


def test_diagnostics_differential_mode_two_candidates():
    env, config_id = env_with_fault(
        Fault.HEATER_SWAP,
        enable_differential_diagnosis=True,
        enable_direct_diagnostics=False,
    )
    true_fault = env._hidden.fault
    solver = FaultAwareSolver(env, options={"config_id": config_id})
    solver.goto("D"); solver.interact()
    a = int(solver.obs["knowledge_bits"][5])
    b = int(solver.obs["knowledge_bits"][6])
    assert a > 0 and b > 0 and a < b          # sorted, two candidates
    assert (1 + int(true_fault)) in {a, b}    # truth is in the set


# ---------------------------------------------------------------------
# Three-dial report: 32-state cycle + scoring matrix
# ---------------------------------------------------------------------


def test_report_dial_sweeps_32_states_and_exposes_confidence():
    env = AxiomForgeEnv(debug_cfg(0))
    solver = FaultAwareSolver(env, options={"config_id": 0})
    solver.goto("E"); solver.interact()   # G lives on the Forge layer
    solver.goto("G")
    seen = set()
    for _ in range(2 * REPORT_SELECTOR_SIZE):
        state = (int(solver.obs["report_state"]),
                 int(solver.obs["knowledge_bits"][8]))
        seen.add(state)
        solver._step(ACTION_CYCLE)
    assert len(seen) == 2 * REPORT_SELECTOR_SIZE
    # full wrap returns to (0, low)
    assert int(solver.obs["report_state"]) == 0
    assert int(solver.obs["knowledge_bits"][8]) == 0


def correct_combined(result_protocol, result_fault):
    return int(result_protocol) * len(Fault) + int(result_fault)


def run_report_case(config_id, cycles):
    env = AxiomForgeEnv(make_v1_3_config(seed=SEED, config_id=config_id))
    solver = FaultAwareSolver(env, options={"config_id": config_id})
    return solver.solve(report_cycles=cycles)


def test_confidence_scoring_matrix():
    # pick a config and compute its correct (protocol, fault) dial index
    probe = solve_config(seed=SEED, config_id=0)
    assert probe["success"] is True
    base = correct_combined(probe["protocol"], probe["fault"])
    step = -0.01  # cfg.step_penalty on the submission step

    high_correct = run_report_case(0, REPORT_SELECTOR_SIZE + base)
    assert high_correct["success"] is True
    assert high_correct["final_reward"] == pytest.approx(step + 10.0 + 2.0)

    low_correct = run_report_case(0, base)
    assert low_correct["success"] is True
    assert low_correct["final_reward"] == pytest.approx(step + 10.0 + 1.0)

    wrong = (base + 1) % REPORT_SELECTOR_SIZE
    low_wrong = run_report_case(0, wrong)
    assert low_wrong["success"] is False
    assert low_wrong["final_reward"] == pytest.approx(step - 2.0 - 0.5)
    assert low_wrong["failure_reason"]["overconfident_wrong_report"] is False

    high_wrong = run_report_case(0, REPORT_SELECTOR_SIZE + wrong)
    assert high_wrong["success"] is False
    assert high_wrong["final_reward"] == pytest.approx(step - 2.0 - 3.0)
    assert high_wrong["failure_reason"]["overconfident_wrong_report"] is True


# ---------------------------------------------------------------------
# safety_failed grading
# ---------------------------------------------------------------------


def test_contaminated_submission_fails_safety():
    env, config_id = env_with_fault(Fault.PURIFIER_LEAK)
    cfg = env.cfg
    solver = FaultAwareSolver(env, options={"config_id": config_id})
    solver.goto("M"); solver.interact()
    target = int(solver.obs["work_order"][1])
    solver.goto("R")
    solver.cycle_selector_to(0, target)
    solver.interact()
    solver.goto("E"); solver.interact()
    solver.goto("U"); solver.interact()           # contaminates
    solver.goto("G"); solver.interact()           # submit dirty
    assert solver.last_info["success"] is False
    assert solver.last_info["failure_reason"]["safety_failed"] is True


# ---------------------------------------------------------------------
# End-to-end: the fault-aware solver beats every fault
# ---------------------------------------------------------------------


@pytest.mark.parametrize("config_id", list(range(0, 24, 2)))
def test_fault_aware_solver_solves_v1_3_train_configs(config_id):
    result = solve_config(seed=SEED, config_id=config_id)
    assert result["success"] is True, result["failure_reason"]
    assert result["steps"] <= 250


def test_solver_coverage_spans_all_faults():
    seen = set()
    for config_id in range(0, 60, 2):
        result = solve_config(seed=SEED, config_id=config_id)
        assert result["success"] is True
        seen.add(result["fault"])
    assert seen == {Fault.NONE, Fault.HEATER_SWAP, Fault.PURIFIER_LEAK,
                    Fault.ANALYZER_BIAS}
