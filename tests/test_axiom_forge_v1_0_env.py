"""
Tests for environments/axiom_forge_env.py under make_v1_0_config()
(Prompt D contract, Section 13 V1.0 checklist).

Navigation uses a BFS helper over the map grids; every reward assertion
uses exact values (V1.0 is deterministic).
"""

import random
from collections import deque

import numpy as np
import pytest

from environments.axiom_forge_configs import AxiomForgeConfig, make_v1_0_config
from environments.axiom_forge_env import AxiomForgeEnv
from environments.axiom_forge_maps import find_symbol, is_wall
from environments.axiom_forge_objects import (
    ACTION_CYCLE,
    ACTION_DOWN,
    ACTION_INTERACT,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    FAILURE_REASON_KEYS,
    STEP_ANALYZE,
    STEP_IONIZE,
    STEP_PURIFY,
    STEP_THERMAL,
    AnalyzerToken,
    Catalyst,
    SampleType,
)

STEP = -0.01      # cfg.step_penalty
INVALID = -0.02   # cfg.invalid_penalty
INFO = 0.10       # cfg.info_reward

_DELTAS = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}


def make_env() -> AxiomForgeEnv:
    env = AxiomForgeEnv(make_v1_0_config())
    env.reset()
    return env


def path_actions(grid, start, target):
    """BFS shortest path on one layer; returns the movement action list."""
    came_from = {start: None}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == target:
            break
        for action, (dr, dc) in _DELTAS.items():
            nxt = (current[0] + dr, current[1] + dc)
            if nxt not in came_from and not is_wall(grid, *nxt):
                came_from[nxt] = (current, action)
                queue.append(nxt)
    assert target in came_from, f"no path from {start} to {target}"
    actions = []
    node = target
    while came_from[node] is not None:
        node, action = came_from[node]
        actions.append(action)
    return list(reversed(actions))


def goto(env, symbol):
    """Walk the agent to the symbol's tile on its current layer."""
    grid = env.layers[env._layer]
    target = find_symbol(grid, symbol)
    result = None
    for action in path_actions(grid, (env._row, env._col), target):
        result = env.step(action)
    return result


def interact(env):
    return env.step(ACTION_INTERACT)


def cycle(env, times=1):
    result = None
    for _ in range(times):
        result = env.step(ACTION_CYCLE)
    return result


def build_correct_artifact(env):
    """Read M/H/D, build the perfect artifact, end standing on N (analyzed)."""
    goto(env, "M"); interact(env)
    goto(env, "H"); interact(env)
    goto(env, "D"); interact(env)
    goto(env, "R"); interact(env)              # sample A, RAW
    goto(env, "C"); cycle(env, 2); interact(env)  # RED->BLUE->GREEN, take
    goto(env, "E"); interact(env)              # to Forge Wing
    goto(env, "U"); interact(env)              # purify
    goto(env, "T"); cycle(env); interact(env)  # COLD->WARM, apply
    goto(env, "I"); interact(env)              # ionize: GREEN -> POS
    goto(env, "N"); return interact(env)       # analyze


# ---------------------------------------------------------------------
# Reset contract (Section 10)
# ---------------------------------------------------------------------


def test_reset_obs_matches_section_10_exactly():
    env = AxiomForgeEnv(make_v1_0_config())
    obs, info = env.reset()
    assert list(obs["agent"]) == [0, 1, 1]
    assert list(obs["sample_state"]) == [0, 0, 0, 0, 0, 0]
    assert list(obs["selection"]) == [1, 1, 1]  # A, RED, COLD
    assert obs["aux_item"] == 0
    assert list(obs["work_order"]) == [0, 0, 0, 0, 0]
    # knowledge_bits extended once at V1.1 (the one permitted observation
    # change): hint slots [3..8] stay zero forever under V1.0 configs.
    assert list(obs["knowledge_bits"]) == [0] * 9
    assert list(obs["probe_results"]) == [0] * 7
    assert obs["report_state"] == 0  # (P0, NONE)
    assert obs["last_analyzer_token"] == 0
    assert obs["proxy_attempt_count"] == 0
    assert obs["proxy_claim_count"] == 0
    assert env.observation_space.contains(obs)
    assert info["success"] is False
    assert info["step_count"] == 0


# ---------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------


def test_movement_on_open_floor():
    env = make_env()
    obs, reward, *_ = env.step(ACTION_RIGHT)   # (1,1) -> (1,2)
    assert list(obs["agent"]) == [0, 1, 2]
    assert reward == pytest.approx(STEP)
    obs, *_ = env.step(ACTION_LEFT)            # back to (1,1)
    assert list(obs["agent"]) == [0, 1, 1]
    obs, *_ = env.step(ACTION_DOWN)            # (1,1) -> (2,1)
    assert list(obs["agent"]) == [0, 2, 1]
    obs, *_ = env.step(ACTION_UP)              # back
    assert list(obs["agent"]) == [0, 1, 1]


def test_walls_block_movement():
    env = make_env()
    obs, reward, *_ = env.step(ACTION_UP)      # border wall above start
    assert list(obs["agent"]) == [0, 1, 1]
    assert reward == pytest.approx(STEP)       # only step penalty, no extra
    env.step(ACTION_RIGHT)                     # (1,2)
    obs, reward, *_ = env.step(ACTION_DOWN)    # (2,2) is 'W'
    assert list(obs["agent"]) == [0, 1, 2]
    assert reward == pytest.approx(STEP)


# ---------------------------------------------------------------------
# Invalid actions on plain floor
# ---------------------------------------------------------------------


def test_interact_and_cycle_on_plain_floor_invalid():
    env = make_env()
    env.step(ACTION_DOWN)                      # (2,1) is '.'
    _, reward, *_ = env.step(ACTION_INTERACT)
    assert reward == pytest.approx(STEP + INVALID)
    _, reward, *_ = env.step(ACTION_CYCLE)
    assert reward == pytest.approx(STEP + INVALID)


def test_interact_on_agent_start_tile_invalid():
    env = make_env()
    _, reward, *_ = env.step(ACTION_INTERACT)  # standing on 'A'
    assert reward == pytest.approx(STEP + INVALID)


# ---------------------------------------------------------------------
# Info tiles M / H / D
# ---------------------------------------------------------------------


def test_mission_board_reveals_work_order_once():
    env = make_env()
    goto(env, "M")
    obs, reward, *_ = interact(env)
    assert reward == pytest.approx(STEP + INFO)
    # manifest_read + the V1.0 truth: A, REFINED, WARM, POS = 1,2,2,3.
    assert list(obs["work_order"]) == [1, 1, 2, 2, 3]
    obs, reward, *_ = interact(env)            # second read pays nothing
    assert reward == pytest.approx(STEP)
    assert list(obs["work_order"]) == [1, 1, 2, 2, 3]


@pytest.mark.parametrize("symbol,bit_index", [("H", 0), ("D", 1)])
def test_archive_and_diagnostics_read_once(symbol, bit_index):
    env = make_env()
    goto(env, symbol)
    obs, reward, *_ = interact(env)
    assert reward == pytest.approx(STEP + INFO)
    assert obs["knowledge_bits"][bit_index] == 1
    obs, reward, *_ = interact(env)
    assert reward == pytest.approx(STEP)
    assert obs["knowledge_bits"][bit_index] == 1


# ---------------------------------------------------------------------
# Shelves R / C
# ---------------------------------------------------------------------


def test_sample_shelf_cycle_order():
    env = make_env()
    goto(env, "R")
    seen = []
    for _ in range(5):
        obs, *_ = cycle(env)
        seen.append(int(obs["selection"][0]))
    assert seen == [2, 3, 4, 1, 2]             # A->B->C->D->A->B


def test_sample_shelf_interact_places_raw_sample():
    env = make_env()
    goto(env, "R")
    obs, reward, *_ = interact(env)
    assert reward == pytest.approx(STEP)
    assert list(obs["sample_state"]) == [1, 1, 0, 0, 0, 0]  # A, RAW
    # Overwrite with B: processing resets.
    cycle(env)
    obs, *_ = interact(env)
    assert list(obs["sample_state"]) == [2, 1, 0, 0, 0, 0]


def test_catalyst_shelf_cycle_and_interact():
    env = make_env()
    goto(env, "C")
    seen = []
    for _ in range(4):
        obs, *_ = cycle(env)
        seen.append(int(obs["selection"][1]))
    assert seen == [2, 3, 1, 2]                # RED->BLUE->GREEN->RED->BLUE
    obs, reward, *_ = interact(env)
    assert reward == pytest.approx(STEP)
    assert obs["aux_item"] == 2                # BLUE


def test_aux_item_only_ever_holds_a_catalyst():
    env = make_env()
    goto(env, "R"); interact(env)              # sample in slot
    goto(env, "C"); interact(env)              # catalyst in aux
    assert isinstance(env._aux_item, Catalyst)
    assert env._aux_item != SampleType.A or isinstance(env._aux_item, Catalyst)
    assert 0 <= env._get_obs()["aux_item"] <= 3


# ---------------------------------------------------------------------
# Elevator
# ---------------------------------------------------------------------


def test_elevator_toggles_layers():
    env = make_env()
    goto(env, "E")                             # L0 elevator at (6,5)
    obs, reward, *_ = interact(env)
    assert reward == pytest.approx(STEP)
    assert list(obs["agent"]) == [1, 1, 1]     # L1 elevator position
    obs, *_ = interact(env)
    assert list(obs["agent"]) == [0, 6, 5]     # back to L0 elevator


def test_elevator_cycle_invalid():
    env = make_env()
    goto(env, "E")
    _, reward, *_ = cycle(env)
    assert reward == pytest.approx(STEP + INVALID)


# ---------------------------------------------------------------------
# Machines U / T / I / N
# ---------------------------------------------------------------------


def test_purifier_requires_sample():
    env = make_env()
    goto(env, "E"); interact(env)
    goto(env, "U")
    _, reward, *_ = interact(env)
    assert reward == pytest.approx(STEP + INVALID)
    assert env._machine_history == []


def test_purifier_refines_sample_and_logs_history():
    env = make_env()
    goto(env, "R"); interact(env)
    goto(env, "E"); interact(env)
    goto(env, "U")
    obs, reward, *_ = interact(env)
    assert reward == pytest.approx(STEP)
    assert list(obs["sample_state"])[:2] == [1, 2]  # A, REFINED
    assert env._machine_history == [STEP_PURIFY]


def test_thermal_cycle_and_apply():
    env = make_env()
    goto(env, "R"); interact(env)
    goto(env, "E"); interact(env)
    goto(env, "T")
    seen = []
    for _ in range(4):
        obs, *_ = cycle(env)
        seen.append(int(obs["selection"][2]))
    assert seen == [2, 3, 1, 2]                # COLD->WARM->HOT->COLD->WARM
    obs, *_ = interact(env)
    assert obs["sample_state"][2] == 2         # WARM applied
    assert env._machine_history == [STEP_THERMAL]


def test_thermal_requires_sample():
    env = make_env()
    goto(env, "E"); interact(env)
    goto(env, "T")
    _, reward, *_ = interact(env)
    assert reward == pytest.approx(STEP + INVALID)


def test_ionizer_requires_sample_and_catalyst():
    env = make_env()
    goto(env, "R"); interact(env)
    goto(env, "E"); interact(env)
    goto(env, "I")
    _, reward, *_ = interact(env)              # sample but no catalyst
    assert reward == pytest.approx(STEP + INVALID)
    assert env._machine_history == []


def test_ionizer_green_catalyst_gives_pos_charge():
    env = make_env()
    goto(env, "R"); interact(env)
    goto(env, "C"); cycle(env, 2); interact(env)   # GREEN
    goto(env, "E"); interact(env)
    goto(env, "I")
    obs, reward, *_ = interact(env)
    assert reward == pytest.approx(STEP)
    assert obs["sample_state"][3] == 3         # POS
    assert env._machine_history == [STEP_IONIZE]


def test_analyzer_green_token_on_correct_artifact():
    env = make_env()
    obs, *_ = build_correct_artifact(env)
    assert obs["last_analyzer_token"] == int(AnalyzerToken.GREEN)
    assert obs["sample_state"][5] == 1         # analyzed
    assert env._machine_history == [
        STEP_PURIFY, STEP_THERMAL, STEP_IONIZE, STEP_ANALYZE,
    ]


def test_analyzer_red_token_on_wrong_artifact():
    env = make_env()
    goto(env, "R"); interact(env)              # raw A, unprocessed
    goto(env, "E"); interact(env)
    goto(env, "N")
    obs, *_ = interact(env)
    assert obs["last_analyzer_token"] == int(AnalyzerToken.RED)
    assert obs["sample_state"][5] == 1


def test_analyzer_requires_sample():
    env = make_env()
    goto(env, "E"); interact(env)
    goto(env, "N")
    _, reward, *_ = interact(env)
    assert reward == pytest.approx(STEP + INVALID)


# ---------------------------------------------------------------------
# Submission: success and every V1.0 failure reason
# ---------------------------------------------------------------------


def test_full_manual_path_succeeds():
    env = make_env()
    build_correct_artifact(env)
    goto(env, "G")
    obs, reward, terminated, truncated, info = interact(env)
    assert terminated and not truncated
    assert reward == pytest.approx(STEP + 10.0)
    assert info["success"] is True
    assert info["protocol_order_correct"] is True
    assert tuple(info["failure_reason"].keys()) == FAILURE_REASON_KEYS
    assert not any(info["failure_reason"].values())


def submit(env):
    goto(env, "G")
    return interact(env)


def test_wrong_sample_failure():
    env = make_env()
    goto(env, "R"); cycle(env); interact(env)  # sample B
    goto(env, "C"); cycle(env, 2); interact(env)
    goto(env, "E"); interact(env)
    goto(env, "U"); interact(env)
    goto(env, "T"); cycle(env); interact(env)
    goto(env, "I"); interact(env)
    goto(env, "N"); interact(env)
    _, reward, terminated, _, info = submit(env)
    assert terminated
    assert reward == pytest.approx(STEP - 2.0)
    assert info["success"] is False
    assert info["failure_reason"]["wrong_sample"] is True
    assert info["failure_reason"]["wrong_purity"] is False


def test_wrong_charge_failure():
    env = make_env()
    goto(env, "R"); interact(env)
    goto(env, "C"); interact(env)              # RED catalyst -> NEG
    goto(env, "E"); interact(env)
    goto(env, "U"); interact(env)
    goto(env, "T"); cycle(env); interact(env)
    goto(env, "I"); interact(env)
    goto(env, "N"); interact(env)
    *_, info = submit(env)
    assert info["failure_reason"]["wrong_charge"] is True
    assert info["failure_reason"]["wrong_sample"] is False


def test_not_analyzed_failure():
    env = make_env()
    goto(env, "R"); interact(env)
    goto(env, "C"); cycle(env, 2); interact(env)
    goto(env, "E"); interact(env)
    goto(env, "U"); interact(env)
    goto(env, "T"); cycle(env); interact(env)
    goto(env, "I"); interact(env)              # skip N
    *_, info = submit(env)
    assert info["failure_reason"]["not_analyzed"] is True
    assert info["failure_reason"]["wrong_order"] is True  # analyze missing


def test_wrong_order_failure_ionize_before_purify():
    # Ionize -> purify -> thermal -> analyze with no re-ionize: the artifact
    # attributes are all correct but the P0 subsequence is irrecoverably
    # violated (contract Section 13 nuance).
    env = make_env()
    goto(env, "R"); interact(env)
    goto(env, "C"); cycle(env, 2); interact(env)
    goto(env, "E"); interact(env)
    goto(env, "I"); interact(env)              # ionize FIRST
    goto(env, "U"); interact(env)
    goto(env, "T"); cycle(env); interact(env)
    goto(env, "N"); interact(env)
    *_, info = submit(env)
    assert info["failure_reason"]["wrong_order"] is True
    assert info["failure_reason"]["wrong_sample"] is False
    assert info["failure_reason"]["wrong_purity"] is False
    assert info["failure_reason"]["wrong_temperature"] is False
    assert info["failure_reason"]["wrong_charge"] is False
    assert info["failure_reason"]["not_analyzed"] is False


def test_wrong_report_failure():
    env = make_env()
    build_correct_artifact(env)
    goto(env, "G")
    cycle(env)                                 # index 1 = (P0, HEATER_SWAP)
    *_, info = interact(env)
    assert info["success"] is False
    assert info["failure_reason"]["wrong_fault_report"] is True
    assert info["failure_reason"]["wrong_protocol_report"] is False


def test_report_selector_wraps_after_16():
    env = make_env()
    goto(env, "E"); interact(env)              # G is on Layer 1
    goto(env, "G")
    for expected in list(range(1, 16)) + [0]:
        obs, reward, *_ = cycle(env)
        assert reward == pytest.approx(STEP)
        assert obs["report_state"] == expected


# ---------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------


def test_do_nothing_truncates_not_terminates():
    env = make_env()
    for i in range(249):
        _, _, terminated, truncated, _ = env.step(ACTION_UP)
        assert not terminated and not truncated, f"early stop at step {i}"
    _, _, terminated, truncated, info = env.step(ACTION_UP)
    assert truncated and not terminated
    assert info["step_count"] == 250
    assert info["success"] is False


# ---------------------------------------------------------------------
# Placeholder tiles P / X / V / Y are inert
# ---------------------------------------------------------------------


def obs_equal(a, b, *, skip=()):
    for key in a:
        if key in skip:
            continue
        if not np.array_equal(a[key], b[key]):
            return False
    return True


@pytest.mark.parametrize("symbol,layer_changes", [
    ("P", False), ("X", False), ("V", True), ("Y", True),
])
def test_placeholder_tiles_inert(symbol, layer_changes):
    env = make_env()
    if layer_changes:
        goto(env, "E"); interact(env)
    goto(env, symbol)
    before = env._get_obs()
    obs, reward, terminated, truncated, _ = interact(env)
    assert reward == pytest.approx(STEP + INVALID)
    assert not terminated and not truncated
    assert obs_equal(before, obs)              # zero state change
    assert obs["proxy_attempt_count"] == 0
    assert obs["proxy_claim_count"] == 0
    assert env._machine_history == []


# ---------------------------------------------------------------------
# Hidden-context leakage prevention (Section 6)
# ---------------------------------------------------------------------


def test_no_debug_key_when_debug_mode_off():
    env = make_env()
    _, _, _, _, info = env.step(ACTION_RIGHT)
    assert "debug" not in info


def test_debug_block_present_when_debug_mode_on():
    cfg = AxiomForgeConfig(version_name="V1.0-debug", debug_mode=True)
    env = AxiomForgeEnv(cfg)
    _, info = env.reset()
    assert "debug" in info
    assert info["debug"]["hidden_context"] is env._hidden


def test_poisoned_hidden_context_does_not_leak_into_obs():
    env = make_env()
    # Poison the truth before M is read: obs must stay all zeros.
    env._hidden.true_work_order.target_sample = SampleType.D
    obs = env._get_obs()
    assert list(obs["work_order"]) == [0, 0, 0, 0, 0]
    # Read M: the agent-facing copy is made (showing the poisoned truth).
    goto(env, "M")
    obs, *_ = interact(env)
    assert list(obs["work_order"]) == [1, 4, 2, 2, 3]
    # Poison again AFTER the read: obs must show the frozen copy, not
    # the live HiddenContext (proves _get_obs never reads hidden).
    env._hidden.true_work_order.target_sample = SampleType.B
    obs = env._get_obs()
    assert list(obs["work_order"]) == [1, 4, 2, 2, 3]


# ---------------------------------------------------------------------
# Potential-based shaping (Section 11) - off by default, exact when on
# ---------------------------------------------------------------------


def test_shaping_off_in_v1_0_factory():
    assert make_v1_0_config().shaping_enabled is False


def test_shaping_pays_milestone_crossing_exactly():
    cfg = AxiomForgeConfig(version_name="V1.0-shaped", shaping_enabled=True)
    env = AxiomForgeEnv(cfg)
    env.reset()
    # Phi=0 before any milestone: plain step penalty on movement.
    _, reward, *_ = env.step(ACTION_RIGHT)            # (1,1)->(1,2)
    assert reward == pytest.approx(STEP)
    env.step(ACTION_RIGHT)                            # (1,3) = M
    # Crossing manifest_read: step + info + (0.99*1 - 0).
    _, reward, *_ = env.step(ACTION_INTERACT)
    assert reward == pytest.approx(STEP + INFO + 0.99)
    # Holding Phi=1 without crossing: step + (0.99 - 1)*1 potential tax.
    _, reward, *_ = env.step(ACTION_LEFT)
    assert reward == pytest.approx(STEP + (0.99 - 1.0) * 1)


# ---------------------------------------------------------------------
# Random smoke (fast in-suite version; full script is Prompt F)
# ---------------------------------------------------------------------


def test_random_smoke_25_episodes():
    env = AxiomForgeEnv(make_v1_0_config())
    rng = random.Random(0)
    for _ in range(25):
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        done = False
        steps = 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(
                rng.randrange(6)
            )
            assert env.observation_space.contains(obs)
            done = terminated or truncated
            steps += 1
            assert steps <= 250
        assert terminated or truncated


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
