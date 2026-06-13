"""
AxiomForge V1.2 manual solver: probe-first plan (Prompt I gate).

Extends the V1.1 branching solver with the probe station:

1. Read M, H, D.
2. Catalyst-probe all three colors at P (empty sample slot) - the
   mapping is now DISCOVERED, not trial-ionized, so no elevator
   round-trips are ever needed for the catalyst.
3. Place the target sample at R, then SAMPLE-probe it at P (satisfies
   P2's probe-first machine order; harmless extra step under P0/P1).
4. Take the correct catalyst (from probe results), process in the
   protocol's order, handle RAW re-slot, ionize once, analyze, report,
   submit.

Run from the repo root:
    python scripts/manual_solve_axiom_forge_v1_2.py --seed 7 --config-ids 0 2 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environments.axiom_forge_configs import make_v1_2_config
from environments.axiom_forge_env import AxiomForgeEnv
from environments.axiom_forge_objects import (
    ACTION_CYCLE,
    Catalyst,
    Fault,
    Protocol,
    Purity,
)
from scripts.manual_solve_axiom_forge_v1_1 import BranchingSolver


class ProbeFirstSolver(BranchingSolver):
    """V1.2 solver: full catalyst probing + sample probe, then process."""

    def probe_all_catalysts(self) -> dict[int, int]:
        """At P with an empty slot, probe RED/BLUE/GREEN; return
        {charge_value: catalyst_value} from the observation."""
        self.goto("P")
        for color in (Catalyst.RED, Catalyst.BLUE, Catalyst.GREEN):
            self.cycle_selector_to(1, int(color))
            self.interact()
        probe = self.obs["probe_results"]
        mapping: dict[int, int] = {}
        for color in (Catalyst.RED, Catalyst.BLUE, Catalyst.GREEN):
            slot = 2 * (int(color) - 1)
            assert int(probe[slot]) == 1, "probe did not record this color"
            mapping[int(probe[slot + 1])] = int(color)
        return mapping

    def solve(self) -> dict:
        # 1. Clues.
        self.goto("M"); self.interact()
        self.goto("H"); self.interact()
        self.goto("D"); self.interact()
        target, purity, temperature, charge = self.work_order
        protocol = self.protocol

        # 2. Discover the full catalyst mapping (empty slot -> catalyst
        #    probes), BEFORE any sample exists.
        charge_to_color = self.probe_all_catalysts()

        # 3. Place the target sample, then sample-probe it (P2 step one).
        self.goto("R")
        self.cycle_selector_to(0, target)
        self.interact()
        self.goto("P"); self.interact()   # sample probe
        assert int(self.obs["probe_results"][6]) == target

        # 4. Take the correct catalyst - known, not guessed.
        self.goto("C")
        self.cycle_selector_to(1, charge_to_color[charge])
        self.interact()
        self.goto("E"); self.interact()

        # 5. Purify/thermal in the protocol's order (P2 = clean-first
        #    after the probe; P1 = heat-first).
        def apply_thermal():
            self.goto("T")
            self.cycle_selector_to(2, temperature)
            self.interact()

        if protocol == Protocol.P1:
            apply_thermal()
            self.goto("U"); self.interact()
        else:                              # P0 and P2: purify then thermal
            self.goto("U"); self.interact()
            apply_thermal()

        # 6. RAW re-slot trick (sample_probe_used is an episode flag, so
        #    the fresh sample does not need re-probing).
        if purity == int(Purity.RAW):
            self.goto("E"); self.interact()
            self.goto("R"); self.interact()
            self.goto("E"); self.interact()
            apply_thermal()

        # 7. Ionize once (correct catalyst), analyze, report, submit.
        self.goto("I"); self.interact()
        assert int(self.obs["sample_state"][3]) == charge
        self.goto("N"); self.interact()
        self.goto("G")
        for _ in range(int(protocol) * len(Fault)):
            self._step(ACTION_CYCLE)
        reward, terminated, truncated = self.interact()

        return {
            "success": self.last_info["success"],
            "failure_reason": self.last_info["failure_reason"],
            "steps": self.total_steps,
            "return": self.total_return,
            "terminated": terminated,
            "truncated": truncated,
            "protocol": protocol,
            "work_order": (target, purity, temperature, charge),
        }


def solve_config(seed: int, config_id: int) -> dict:
    env = AxiomForgeEnv(make_v1_2_config(seed=seed, config_id=config_id))
    solver = ProbeFirstSolver(env, options={"config_id": config_id})
    return solver.solve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the probe-first V1.2 manual solver.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config-ids", type=int, nargs="+",
                        default=[0, 2, 4, 6, 8])
    args = parser.parse_args()

    for config_id in args.config_ids:
        result = solve_config(args.seed, config_id)
        assert result["success"] is True, (
            f"config {config_id} failed: {result['failure_reason']}")
        print(
            f"config_id {config_id:3d} | protocol {result['protocol'].name} "
            f"| wo {result['work_order']} | steps {result['steps']:3d} "
            f"| return {result['return']:.2f} | success True"
        )
    print("V1.2 probe-first solver: all configs solved.")


if __name__ == "__main__":
    main()
