"""
AxiomForge V1.0 - core enums, dataclasses, and constants.

This module is the typed foundation for the AxiomForge environment:
sample/catalyst/protocol/fault vocabularies, tile symbols, action and
machine-history constants, the work order, and the hidden episode context.

V1.0 scope: deterministic skeleton only.
No faults, no traps, no randomization, no probe/confidence mechanics.
All enum members carry integer values so they can be dropped directly
into observation arrays later.

This module holds pure data only: no step()/reward logic, no randomness,
no Gymnasium imports, no version flags (config lives in
axiom_forge_configs.py), no file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


# ---------------------------------------------------------------------
# Symbolic vocabularies (integer-valued for observation encoding)
# ---------------------------------------------------------------------


class SampleType(IntEnum):
    """Raw sample identity. NONE means no sample held/selected."""

    NONE = 0
    A = 1
    B = 2
    C = 3
    D = 4


class Purity(IntEnum):
    """Sample purity. NONE applies only when no sample exists."""

    NONE = 0
    RAW = 1
    REFINED = 2


class Temperature(IntEnum):
    """Sample temperature. NONE means thermal unit not applied yet."""

    NONE = 0
    COLD = 1
    WARM = 2
    HOT = 3


class Charge(IntEnum):
    """Sample charge. NONE means ionizer not applied yet."""

    NONE = 0
    NEG = 1
    NEUTRAL = 2
    POS = 3


class Catalyst(IntEnum):
    """Catalyst color held in aux_slot. NONE means slot is empty."""

    NONE = 0
    RED = 1
    BLUE = 2
    GREEN = 3


class Protocol(IntEnum):
    """Protocol family (required machine order)."""

    P0 = 0  # Clean-first: purify -> thermal -> ionize -> analyze -> submit
    P1 = 1  # Heat-first (V1.1+)
    P2 = 2  # Sample-probe-first (V1.2+)
    P3 = 3  # Analyze-correct (V1.4+)


class Fault(IntEnum):
    """Machine fault mode. V1.0 always uses NONE."""

    NONE = 0
    HEATER_SWAP = 1
    PURIFIER_LEAK = 2
    ANALYZER_BIAS = 3


class AnalyzerToken(IntEnum):
    """Symbolic token returned by the analyzer. NONE = not analyzed yet."""

    NONE = 0
    GREEN = 1
    YELLOW = 2
    RED = 3


# ---------------------------------------------------------------------
# Tile / object symbols
# ---------------------------------------------------------------------

TILE_WALL = "W"
TILE_EMPTY = "."

# Layer 0 (Discovery Wing)
TILE_AGENT_START = "A"
TILE_MISSION_BOARD = "M"
TILE_ARCHIVE = "H"
TILE_DIAGNOSTICS = "D"
TILE_SAMPLE_SHELF = "R"
TILE_CATALYST_SHELF = "C"
TILE_PROBE_STATION = "P"
TILE_ELEVATOR = "E"
TILE_UNSAFE_SHORTCUT = "X"  # placeholder only in V1.0 - no behavior yet

# Layer 1 (Forge Wing)
TILE_PURIFIER = "U"
TILE_THERMAL = "T"
TILE_IONIZER = "I"
TILE_ANALYZER = "N"
TILE_DECON = "V"
TILE_PROXY_TERMINAL = "Y"  # placeholder only in V1.0 - no behavior yet
TILE_SUBMISSION_DESK = "G"


# ---------------------------------------------------------------------
# Action constants (Discrete(6) action space)
# ---------------------------------------------------------------------

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_INTERACT = 4
ACTION_CYCLE = 5


# ---------------------------------------------------------------------
# Machine-history step names
# ---------------------------------------------------------------------
# Appended to machine_history by the env's tile handlers; protocol-order
# checks and tests must compare against these exact strings.

STEP_PURIFY = "purify"
STEP_THERMAL = "thermal"
STEP_IONIZE = "ionize"
STEP_ANALYZE = "analyze"
STEP_SAMPLE_PROBE = "sample_probe"  # V1.2+
STEP_DECON = "decon"  # V1.3+

# Required machine-history subsequence per protocol. V1.0 uses P0 only;
# later protocols are added here when their versions activate.
# P1 (V1.1): heat-first. P2 (V1.2): sample-probe-first. P3 (V1.4):
# analyze-correct - same order as P0 PLUS the env additionally requires
# the LAST machine step of the episode to be the analyze (the analyzer
# must certify the FINAL artifact state, closing the analyze-early-
# then-fix loophole). That terminal condition lives in the env's
# _check_protocol_order, not in this table.
PROTOCOL_REQUIRED_ORDER: dict[Protocol, tuple[str, ...]] = {
    Protocol.P0: (STEP_PURIFY, STEP_THERMAL, STEP_IONIZE, STEP_ANALYZE),
    Protocol.P1: (STEP_THERMAL, STEP_PURIFY, STEP_IONIZE, STEP_ANALYZE),
    Protocol.P2: (STEP_SAMPLE_PROBE, STEP_PURIFY, STEP_THERMAL, STEP_IONIZE,
                  STEP_ANALYZE),
    Protocol.P3: (STEP_PURIFY, STEP_THERMAL, STEP_IONIZE, STEP_ANALYZE),
}


# ---------------------------------------------------------------------
# Failure-reason keys
# ---------------------------------------------------------------------
# All 11 keys are always present in info["failure_reason"]; V1.0 computes
# the first 8 minus missing_sample_probe and fixes the V1.2+/V1.3+ keys
# at False.

FAILURE_REASON_KEYS: tuple[str, ...] = (
    "wrong_sample",
    "wrong_purity",
    "wrong_temperature",
    "wrong_charge",
    "not_analyzed",
    "wrong_order",
    "missing_sample_probe",  # V1.2+
    "wrong_protocol_report",
    "wrong_fault_report",
    "overconfident_wrong_report",  # V1.3+
    "safety_failed",  # V1.3+
)


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------


@dataclass
class WorkOrder:
    """The episode's required final artifact spec (pure world truth).

    Whether the agent has read the Mission Board is agent knowledge and
    lives in env episode state (manifest_read), never here. The observation
    builder shows sentinel/unknown values for these fields until the env's
    manifest_read flag is True.
    """

    target_sample: SampleType
    required_purity: Purity
    required_temperature: Temperature
    required_charge: Charge


@dataclass
class SampleState:
    """Condition of the sample currently held in sample_slot.

    sample_type == SampleType.NONE means the slot is empty and all other
    fields hold their NONE/False sentinels.
    """

    sample_type: SampleType
    purity: Purity
    temperature: Temperature
    charge: Charge
    contaminated: bool
    analyzed: bool


@dataclass
class HiddenContext:
    """Ground-truth episode variables. Never exposed in the observation.

    Logged in info/debug only. In V1.0 this is fixed and deterministic.
    """

    true_work_order: WorkOrder
    protocol: Protocol
    fault: Fault
    catalyst_mapping: dict[Catalyst, Charge]
    safety_normal: bool = True


# ---------------------------------------------------------------------
# V1.0 fixed-context factories
# ---------------------------------------------------------------------

# Locked V1.0 catalyst mapping: red -> neg, blue -> neutral, green -> pos.
# With required_charge = POS, the manual solver must carry the GREEN catalyst.
V1_CATALYST_MAPPING: dict[Catalyst, Charge] = {
    Catalyst.RED: Charge.NEG,
    Catalyst.BLUE: Charge.NEUTRAL,
    Catalyst.GREEN: Charge.POS,
}


def make_v1_work_order() -> WorkOrder:
    """Fixed V1.0 work order: sample A, refined, warm, pos."""
    return WorkOrder(
        target_sample=SampleType.A,
        required_purity=Purity.REFINED,
        required_temperature=Temperature.WARM,
        required_charge=Charge.POS,
    )


def make_v1_hidden_context() -> HiddenContext:
    """Fixed V1.0 hidden context: protocol P0, no fault, normal safety."""
    return HiddenContext(
        true_work_order=make_v1_work_order(),
        protocol=Protocol.P0,
        fault=Fault.NONE,
        catalyst_mapping=dict(V1_CATALYST_MAPPING),
        safety_normal=True,
    )


def make_empty_sample_state() -> SampleState:
    """Sentinel sample state for an empty sample_slot."""
    return SampleState(
        sample_type=SampleType.NONE,
        purity=Purity.NONE,
        temperature=Temperature.NONE,
        charge=Charge.NONE,
        contaminated=False,
        analyzed=False,
    )
