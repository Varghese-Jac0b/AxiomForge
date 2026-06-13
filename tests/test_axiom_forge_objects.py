"""
Tests for environments/axiom_forge_objects.py (Prompt A contract).

Pins enum integer values (observation encodings depend on them), verifies
the V1.0 factories, the action/step/failure-key constants, and the
truth/knowledge separation (WorkOrder carries no manifest_read).
"""

import dataclasses

import pytest

import environments.axiom_forge_objects as objects
from environments.axiom_forge_objects import (
    ACTION_CYCLE,
    ACTION_DOWN,
    ACTION_INTERACT,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    FAILURE_REASON_KEYS,
    PROTOCOL_REQUIRED_ORDER,
    STEP_ANALYZE,
    STEP_DECON,
    STEP_IONIZE,
    STEP_PURIFY,
    STEP_SAMPLE_PROBE,
    STEP_THERMAL,
    V1_CATALYST_MAPPING,
    AnalyzerToken,
    Catalyst,
    Charge,
    Fault,
    Protocol,
    Purity,
    SampleType,
    Temperature,
    WorkOrder,
    make_empty_sample_state,
    make_v1_hidden_context,
    make_v1_work_order,
)


# ---------------------------------------------------------------------
# Enum integer values pinned exactly
# ---------------------------------------------------------------------


def test_sample_type_values():
    assert [(m.name, m.value) for m in SampleType] == [
        ("NONE", 0), ("A", 1), ("B", 2), ("C", 3), ("D", 4),
    ]


def test_purity_values():
    assert [(m.name, m.value) for m in Purity] == [
        ("NONE", 0), ("RAW", 1), ("REFINED", 2),
    ]


def test_temperature_values():
    assert [(m.name, m.value) for m in Temperature] == [
        ("NONE", 0), ("COLD", 1), ("WARM", 2), ("HOT", 3),
    ]


def test_charge_values():
    assert [(m.name, m.value) for m in Charge] == [
        ("NONE", 0), ("NEG", 1), ("NEUTRAL", 2), ("POS", 3),
    ]


def test_catalyst_values():
    assert [(m.name, m.value) for m in Catalyst] == [
        ("NONE", 0), ("RED", 1), ("BLUE", 2), ("GREEN", 3),
    ]


def test_protocol_values():
    assert [(m.name, m.value) for m in Protocol] == [
        ("P0", 0), ("P1", 1), ("P2", 2), ("P3", 3),
    ]


def test_fault_values():
    assert [(m.name, m.value) for m in Fault] == [
        ("NONE", 0), ("HEATER_SWAP", 1), ("PURIFIER_LEAK", 2), ("ANALYZER_BIAS", 3),
    ]


def test_analyzer_token_values():
    assert [(m.name, m.value) for m in AnalyzerToken] == [
        ("NONE", 0), ("GREEN", 1), ("YELLOW", 2), ("RED", 3),
    ]


# ---------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------


def test_action_constants():
    assert ACTION_UP == 0
    assert ACTION_DOWN == 1
    assert ACTION_LEFT == 2
    assert ACTION_RIGHT == 3
    assert ACTION_INTERACT == 4
    assert ACTION_CYCLE == 5


def test_action_constants_are_distinct():
    actions = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT,
               ACTION_INTERACT, ACTION_CYCLE]
    assert sorted(actions) == [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------
# Machine-history step names and protocol order
# ---------------------------------------------------------------------


def test_step_name_constants():
    assert STEP_PURIFY == "purify"
    assert STEP_THERMAL == "thermal"
    assert STEP_IONIZE == "ionize"
    assert STEP_ANALYZE == "analyze"
    assert STEP_SAMPLE_PROBE == "sample_probe"
    assert STEP_DECON == "decon"


def test_protocol_required_order_p0():
    assert PROTOCOL_REQUIRED_ORDER[Protocol.P0] == (
        STEP_PURIFY, STEP_THERMAL, STEP_IONIZE, STEP_ANALYZE,
    )


def test_protocol_required_order_full_ladder():
    # V1.1 upgrade (Prompt H, Build Record Section 5): the full ladder's
    # required orders live in the table; protocols only ACTIVATE through
    # cfg.enabled_protocols. P0's entry must never change.
    assert set(PROTOCOL_REQUIRED_ORDER.keys()) == {
        Protocol.P0, Protocol.P1, Protocol.P2, Protocol.P3,
    }
    assert PROTOCOL_REQUIRED_ORDER[Protocol.P1] == (
        STEP_THERMAL, STEP_PURIFY, STEP_IONIZE, STEP_ANALYZE,
    )
    assert PROTOCOL_REQUIRED_ORDER[Protocol.P2] == (
        STEP_SAMPLE_PROBE, STEP_PURIFY, STEP_THERMAL, STEP_IONIZE,
        STEP_ANALYZE,
    )
    assert PROTOCOL_REQUIRED_ORDER[Protocol.P3] == (
        STEP_PURIFY, STEP_THERMAL, STEP_IONIZE, STEP_ANALYZE,
    )


# ---------------------------------------------------------------------
# Failure-reason keys
# ---------------------------------------------------------------------


def test_failure_reason_keys_exact():
    assert FAILURE_REASON_KEYS == (
        "wrong_sample",
        "wrong_purity",
        "wrong_temperature",
        "wrong_charge",
        "not_analyzed",
        "wrong_order",
        "missing_sample_probe",
        "wrong_protocol_report",
        "wrong_fault_report",
        "overconfident_wrong_report",
        "safety_failed",
    )
    assert len(FAILURE_REASON_KEYS) == 11
    assert len(set(FAILURE_REASON_KEYS)) == 11


# ---------------------------------------------------------------------
# WorkOrder: pure truth, no agent knowledge
# ---------------------------------------------------------------------


def test_work_order_has_no_manifest_read():
    field_names = {f.name for f in dataclasses.fields(WorkOrder)}
    assert field_names == {
        "target_sample",
        "required_purity",
        "required_temperature",
        "required_charge",
    }
    assert "manifest_read" not in field_names


def test_config_moved_out_of_objects():
    assert not hasattr(objects, "AxiomForgeConfig")


# ---------------------------------------------------------------------
# V1.0 factories
# ---------------------------------------------------------------------


def test_make_v1_work_order_values():
    wo = make_v1_work_order()
    assert wo.target_sample == SampleType.A
    assert wo.required_purity == Purity.REFINED
    assert wo.required_temperature == Temperature.WARM
    assert wo.required_charge == Charge.POS


def test_make_v1_hidden_context_values():
    ctx = make_v1_hidden_context()
    assert ctx.protocol == Protocol.P0
    assert ctx.fault == Fault.NONE
    assert ctx.safety_normal is True
    assert ctx.true_work_order == make_v1_work_order()
    assert ctx.catalyst_mapping == {
        Catalyst.RED: Charge.NEG,
        Catalyst.BLUE: Charge.NEUTRAL,
        Catalyst.GREEN: Charge.POS,
    }


def test_v1_catalyst_mapping_locked():
    assert V1_CATALYST_MAPPING == {
        Catalyst.RED: Charge.NEG,
        Catalyst.BLUE: Charge.NEUTRAL,
        Catalyst.GREEN: Charge.POS,
    }


def test_make_empty_sample_state_all_sentinel():
    s = make_empty_sample_state()
    assert s.sample_type == SampleType.NONE
    assert s.purity == Purity.NONE
    assert s.temperature == Temperature.NONE
    assert s.charge == Charge.NONE
    assert s.contaminated is False
    assert s.analyzed is False


# ---------------------------------------------------------------------
# Factory independence: mutating one result must not leak into the next
# ---------------------------------------------------------------------


def test_hidden_context_factory_returns_independent_mapping():
    ctx_a = make_v1_hidden_context()
    ctx_a.catalyst_mapping[Catalyst.RED] = Charge.POS
    ctx_b = make_v1_hidden_context()
    assert ctx_b.catalyst_mapping[Catalyst.RED] == Charge.NEG
    # The module-level locked mapping must also be untouched.
    assert V1_CATALYST_MAPPING[Catalyst.RED] == Charge.NEG


def test_hidden_context_factory_returns_independent_work_order():
    ctx_a = make_v1_hidden_context()
    ctx_a.true_work_order.target_sample = SampleType.D
    ctx_b = make_v1_hidden_context()
    assert ctx_b.true_work_order.target_sample == SampleType.A


def test_empty_sample_state_factory_is_independent():
    s_a = make_empty_sample_state()
    s_a.sample_type = SampleType.B
    s_a.analyzed = True
    s_b = make_empty_sample_state()
    assert s_b.sample_type == SampleType.NONE
    assert s_b.analyzed is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
