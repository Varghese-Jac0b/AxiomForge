"""
AxiomForge V1.3 manual solver: fault-aware, probe-first, calibrated.

Extends the V1.2 plan with fault compensation driven ONLY by the D hint
and observable sample state:

- HEATER_SWAP : select swap(required temp) so the swapped heater applies
                the required one (WARM unchanged).
- PURIFIER_LEAK: after processing, if the sample is observably
                contaminated (sample_state[4]), decon at V before
                analyzing.
- ANALYZER_BIAS: ignore the analyzer token entirely (it lies); the plan
                is correct by construction and the probe is truthful.

Report: three dials - (protocol, fault, confidence=HIGH) via the
32-state combined selector.

Run from the repo root:
    python scripts/manual_solve_axiom_forge_v1_3.py --seed 7 --config-ids 0 2 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environments.axiom_forge_configs import make_v1_3_config
from environments.axiom_forge_env import AxiomForgeEnv, REPORT_SELECTOR_SIZE
from environments.axiom_forge_objects import (
    ACTION_CYCLE,
    Fault,
    Protocol,
    Purity,
    Temperature,
)
from scripts.manual_solve_axiom_forge_v1_2 import ProbeFirstSolver

_SWAP = {
    int(Temperature.COLD): int(Temperature.HOT),
    int(Temperature.HOT): int(Temperature.COLD),
    int(Temperature.WARM): int(Temperature.WARM),
}


class FaultAwareSolver(ProbeFirstSolver):
    """V1.3 solver: V1.2 probe-first plan + fault compensation +
    calibrated three-dial report."""

    @property
    def fault(self) -> Fault:
        hint = int(self.obs["knowledge_bits"][5])
        assert hint > 0, "fault hint read before D?"
        return Fault(hint - 1)

    def solve(self, report_cycles: int | None = None) -> dict:
        """Run the full plan. `report_cycles` overrides the number of dial
        cycles at G (tests use it to probe the confidence scoring matrix);
        None = calibrated (protocol, fault, HIGH) report."""
        # 1. Clues (D now reveals the fault + safety mode).
        self.goto("M"); self.interact()
        self.goto("H"); self.interact()
        self.goto("D"); self.interact()
        target, purity, temperature, charge = self.work_order
        protocol = self.protocol
        fault = self.fault

        # 2. Full catalyst probing, then sample + sample probe (V1.2 plan).
        charge_to_color = self.probe_all_catalysts()
        self.goto("R")
        self.cycle_selector_to(0, target)
        self.interact()
        self.goto("P"); self.interact()
        self.goto("C")
        self.cycle_selector_to(1, charge_to_color[charge])
        self.interact()
        self.goto("E"); self.interact()

        # 3. Process in protocol order, compensating the heater fault.
        select_temp = (_SWAP[temperature] if fault == Fault.HEATER_SWAP
                       else temperature)

        def apply_thermal():
            self.goto("T")
            self.cycle_selector_to(2, select_temp)
            self.interact()
            assert int(self.obs["sample_state"][2]) == temperature

        if protocol == Protocol.P1:
            apply_thermal()
            self.goto("U"); self.interact()
        else:
            self.goto("U"); self.interact()
            apply_thermal()

        if purity == int(Purity.RAW):
            self.goto("E"); self.interact()
            self.goto("R"); self.interact()
            self.goto("E"); self.interact()
            apply_thermal()

        # 4. Decon if (and only if) the sample is observably contaminated.
        if int(self.obs["sample_state"][4]) == 1:
            self.goto("V"); self.interact()
            assert int(self.obs["sample_state"][4]) == 0

        # 5. Ionize, analyze (token untrusted under bias), report HIGH.
        self.goto("I"); self.interact()
        assert int(self.obs["sample_state"][3]) == charge
        self.goto("N"); self.interact()
        self.goto("G")
        combined_target = (REPORT_SELECTOR_SIZE          # confidence HIGH
                           + int(protocol) * len(Fault) + int(fault))
        if report_cycles is not None:
            combined_target = report_cycles
        for _ in range(combined_target):
            self._step(ACTION_CYCLE)
        reward, terminated, truncated = self.interact()

        return {
            "final_reward": reward,
            "success": self.last_info["success"],
            "failure_reason": self.last_info["failure_reason"],
            "steps": self.total_steps,
            "return": self.total_return,
            "terminated": terminated,
            "truncated": truncated,
            "protocol": protocol,
            "fault": fault,
            "work_order": (target, purity, temperature, charge),
        }


def solve_config(seed: int, config_id: int) -> dict:
    env = AxiomForgeEnv(make_v1_3_config(seed=seed, config_id=config_id))
    solver = FaultAwareSolver(env, options={"config_id": config_id})
    return solver.solve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fault-aware V1.3 manual solver.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config-ids", type=int, nargs="+",
                        default=[0, 2, 4, 6, 8, 10])
    args = parser.parse_args()

    for config_id in args.config_ids:
        result = solve_config(args.seed, config_id)
        assert result["success"] is True, (
            f"config {config_id} failed: {result['failure_reason']}")
        print(
            f"config_id {config_id:3d} | {result['protocol'].name} "
            f"| fault {result['fault'].name:13s} | wo {result['work_order']} "
            f"| steps {result['steps']:3d} | return {result['return']:.2f}"
        )
    print("V1.3 fault-aware solver: all configs solved.")


if __name__ == "__main__":
    main()
