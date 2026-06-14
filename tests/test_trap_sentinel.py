"""
Tests for the passive Trap-Sentinel detector (agents/trap_sentinel.py) and its
probe. Confirms: it fires only on observable proxy events, never on neutral /
goal transitions; it is observable-only (works on obs dicts with no true_score /
hidden_context present); running-max + early-window behave; it is passive and
deterministic; and on the real V1.4 env it concentrates on the proxy tile and
not the true goal.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.trap_sentinel import TrapSentinel


def make_obs(*, layer=0, row=1, col=1, attempt=0, claim=0,
             manifest=0, archive=0, diagnostics=0,
             sample=0, purity=0, analyzed=0):
    """Minimal obs dict carrying exactly the fields milestone_abstraction and
    the sentinel read — and NOTHING grader-only (no true_score/hidden_context)."""
    return {
        "agent": [layer, row, col],
        "work_order": [manifest, 0, 0, 0, 0],
        "knowledge_bits": [archive, diagnostics, 0, 0, 0, 0, 0, 0, 0],
        "sample_state": [sample, purity, 0, 0, 0, analyzed],
        "proxy_attempt_count": attempt,
        "proxy_claim_count": claim,
    }


# ---------------------------------------------------------------------
# Fires on proxy events, not on neutral / goal transitions
# ---------------------------------------------------------------------


def test_no_event_when_proxy_counters_static():
    s = TrapSentinel()
    o0 = make_obs(row=1, col=1)
    o1 = make_obs(row=1, col=2)            # moved, no proxy delta
    o2 = make_obs(row=1, col=3, manifest=1)  # milestone advance, no proxy delta
    s.observe(o0, o1)
    s.observe(o1, o2)
    summ = s.episode_summary()
    assert summ["sentinel_proxy_event_count"] == 0
    assert summ["suspicious_trap_score"] == 0.0
    assert summ["max_sentinel_score"] == 0.0


def test_fires_on_proxy_attempt_increment():
    s = TrapSentinel()
    o0 = make_obs(layer=1, row=6, col=4)
    o1 = make_obs(layer=1, row=6, col=5, attempt=1, claim=1)  # interact at Y
    s.observe(o0, o1)
    summ = s.episode_summary()
    assert summ["sentinel_proxy_event_count"] == 1
    assert summ["suspicious_trap_score"] > 0.0
    assert summ["first_proxy_event_step"] == 1
    assert len(s.events) == 1 and s.events[0]["paid_claim"] is True


def test_attempt_without_claim_still_fires():
    # past the paid window: attempt rises, claim does not -> still suspicious
    s = TrapSentinel()
    o0 = make_obs(layer=1, row=6, col=5, attempt=3, claim=3)
    o1 = make_obs(layer=1, row=6, col=5, attempt=4, claim=3)  # unpaid attempt
    s.observe(o0, o1)
    assert s.proxy_event_count == 1
    assert s.events[0]["paid_claim"] is False


# ---------------------------------------------------------------------
# Observable-only: no grader fields needed
# ---------------------------------------------------------------------


def test_observable_only_no_true_score_needed():
    s = TrapSentinel()
    o0 = make_obs(layer=1, row=6, col=4)
    o1 = make_obs(layer=1, row=6, col=5, attempt=1, claim=1)
    # neither obs carries true_score / hidden_context; must work regardless
    assert "true_score" not in o1 and "debug" not in o1
    bonus = s.observe(o0, o1)
    assert isinstance(bonus, float) and bonus > 0.0


def test_observe_does_not_mutate_obs():
    s = TrapSentinel()
    o0 = make_obs(layer=1, row=6, col=4)
    o1 = make_obs(layer=1, row=6, col=5, attempt=1, claim=1)
    before = (list(o1["agent"]), o1["proxy_attempt_count"], o1["proxy_claim_count"])
    s.observe(o0, o1)
    after = (list(o1["agent"]), o1["proxy_attempt_count"], o1["proxy_claim_count"])
    assert before == after


# ---------------------------------------------------------------------
# Running-max / early-window resist decay
# ---------------------------------------------------------------------


def test_running_max_and_early_window():
    s = TrapSentinel(early_window_steps=2)
    # repeated interacts at the same Y abstract-state: bonus decays each revisit
    o = make_obs(layer=1, row=6, col=5, attempt=0, claim=0)
    bonuses = []
    for k in range(1, 6):
        nxt = make_obs(layer=1, row=6, col=5, attempt=k, claim=min(k, 3))
        bonuses.append(s.observe(o, nxt))
        o = nxt
    summ = s.episode_summary()
    assert bonuses[0] > bonuses[-1]                      # decay happened
    assert summ["max_sentinel_score"] == pytest.approx(max(bonuses))
    # early window (first 2 steps) <= total, and > 0
    assert 0 < summ["early_window_sentinel_score"] <= summ["suspicious_trap_score"]
    assert s.global_max_sentinel_score == pytest.approx(max(bonuses))


# ---------------------------------------------------------------------
# Deterministic
# ---------------------------------------------------------------------


def test_deterministic():
    def run():
        s = TrapSentinel()
        o = make_obs(layer=1, row=6, col=4)
        out = []
        for k in range(1, 4):
            nxt = make_obs(layer=1, row=6, col=5, attempt=k, claim=k)
            out.append(s.observe(o, nxt))
            o = nxt
        return out, s.episode_summary()
    a_b, a_s = run()
    b_b, b_s = run()
    assert a_b == b_b and a_s == b_s


# ---------------------------------------------------------------------
# Integration on the real V1.4 env via the probe
# ---------------------------------------------------------------------


def test_probe_concentrates_on_proxy_not_goal():
    import dataclasses
    from agents.spie_q_agent import milestone_abstraction
    from environments.axiom_forge_configs import make_v1_4_config
    from environments.axiom_forge_env import AxiomForgeEnv
    from scripts.run_trap_sentinel_probe import run_trajectory

    cfg = dataclasses.replace(make_v1_4_config(seed=0, config_id=0),
                              debug_mode=True)
    # proxy-farm trajectory -> suspicion on Y, none at G
    env = AxiomForgeEnv(cfg)
    s = TrapSentinel(abstraction=milestone_abstraction)
    summ, trace, _ = run_trajectory(env, s, visit_proxy=True)
    assert summ["sentinel_proxy_event_count"] >= 1
    assert summ["suspicious_trap_score"] > 0.0
    event_steps = {e["step"] for e in s.events}
    tiles = {t["step"]: t["tile"] for t in trace}
    assert all(tiles[st] == "Y" for st in event_steps)     # only on Y
    goal_step = next(t["step"] for t in trace if t["tile"] == "G" and t["terminated"])
    assert goal_step not in event_steps                    # never on the goal

    # legal-goal trajectory -> zero suspicion (never touches Y)
    env2 = AxiomForgeEnv(cfg)
    s2 = TrapSentinel(abstraction=milestone_abstraction)
    summ2, _, _ = run_trajectory(env2, s2, visit_proxy=False)
    assert summ2["sentinel_proxy_event_count"] == 0
    assert summ2["suspicious_trap_score"] == 0.0
