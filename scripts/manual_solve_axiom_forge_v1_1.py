"""
AxiomForge V1.1 manual solver (Prompt H / Build Record Section 6).

Unlike the fixed V1.0 plan, this solver BRANCHES on what the episode
reveals through observations only:

- work order (target / purity / temperature / charge) from the M view;
- protocol from the H hint (knowledge_bits[3], direct mode);
- catalyst by OBSERVED-CHARGE trial: the mapping is randomized and not
  observable in V1.1 (probes arrive in V1.2), but the sample's charge IS
  observable after ionizing, so the solver ionizes, checks, and swaps
  catalysts (<= 3 trials, deterministic order RED -> BLUE -> GREEN);
- required_purity == RAW uses the re-slot trick: machine history is
  episode-level, so purify once (satisfying the order), then re-place a
  fresh RAW sample and finish processing on it.

Rules unchanged from V1.0: decisions come only from observations and the
static maps; info is consulted ONLY in assertions.

Run from the repo root:
    python scripts/manual_solve_axiom_forge_v1_1.py --seed 7 --config-ids 0 2 4
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environments.axiom_forge_configs import make_v1_1_config
from environments.axiom_forge_env import AxiomForgeEnv
from environments.axiom_forge_maps import find_symbol, is_wall, make_layers
from environments.axiom_forge_objects import (
    ACTION_CYCLE,
    ACTION_DOWN,
    ACTION_INTERACT,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    Catalyst,
    Fault,
    Protocol,
    Purity,
)

_DELTAS = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}

# Deterministic catalyst trial order for the observed-charge search.
_CATALYST_TRIALS = (Catalyst.RED, Catalyst.BLUE, Catalyst.GREEN)


class BranchingSolver:
    """Observation-driven V1.1 solver: BFS navigation + revealed-state
    branching. Reusable by later-version solvers and tests."""

    def __init__(self, env: AxiomForgeEnv, options: dict | None = None):
        self.env = env
        self.grids = make_layers()
        self.obs, _ = env.reset(options=options)
        self.total_steps = 0
        self.total_return = 0.0
        self.last_info: dict = {}

    # ---------------- primitives ----------------

    def _step(self, action: int):
        self.obs, reward, terminated, truncated, self.last_info = \
            self.env.step(action)
        self.total_steps += 1
        self.total_return += reward
        return reward, terminated, truncated

    def goto(self, symbol: str) -> None:
        layer = int(self.obs["agent"][0])
        grid = self.grids[layer]
        start = (int(self.obs["agent"][1]), int(self.obs["agent"][2]))
        target = find_symbol(grid, symbol)
        came_from: dict = {start: None}
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
        if target not in came_from:
            raise RuntimeError(f"no path to {symbol!r}")
        actions: list[int] = []
        node = target
        while came_from[node] is not None:
            node, action = came_from[node]
            actions.append(action)
        for action in reversed(actions):
            self._step(action)

    def interact(self):
        return self._step(ACTION_INTERACT)

    def cycle_selector_to(self, selection_index: int, value: int) -> None:
        """Cycle the current tile's selector until selection[i] == value."""
        for _ in range(5):
            if int(self.obs["selection"][selection_index]) == value:
                return
            self._step(ACTION_CYCLE)
        raise RuntimeError("selector never reached the requested value")

    # ---------------- revealed state ----------------

    @property
    def work_order(self) -> tuple[int, int, int, int]:
        wo = self.obs["work_order"]
        assert int(wo[0]) == 1, "work order read before M?"
        return int(wo[1]), int(wo[2]), int(wo[3]), int(wo[4])

    @property
    def protocol(self) -> Protocol:
        hint = int(self.obs["knowledge_bits"][3])
        assert hint > 0, "protocol hint read before H?"
        return Protocol(hint - 1)

    # ---------------- the plan ----------------

    def solve(self) -> dict:
        # 1. Read every clue tile first.
        self.goto("M"); self.interact()
        self.goto("H"); self.interact()
        self.goto("D"); self.interact()
        target, purity, temperature, charge = self.work_order
        protocol = self.protocol

        # 2. Pick the target sample and a first trial catalyst.
        self.goto("R")
        self.cycle_selector_to(0, target)
        self.interact()
        self.goto("C")
        self.cycle_selector_to(1, int(_CATALYST_TRIALS[0]))
        self.interact()
        self.goto("E"); self.interact()

        # 3. Purify/thermal in the protocol's required order.
        def apply_thermal():
            self.goto("T")
            self.cycle_selector_to(2, temperature)
            self.interact()

        if protocol == Protocol.P1:          # heat-first
            apply_thermal()
            self.goto("U"); self.interact()
        else:                                # P0: clean-first
            self.goto("U"); self.interact()
            apply_thermal()

        # 4. RAW work orders: re-slot a fresh sample (history keeps the
        #    purify step), then re-apply temperature to the new sample.
        if purity == int(Purity.RAW):
            self.goto("E"); self.interact()
            self.goto("R"); self.interact()   # dial still on target
            self.goto("E"); self.interact()
            apply_thermal()

        # 5. Ionize; verify the OBSERVED charge; swap catalysts if needed.
        self.goto("I"); self.interact()
        trial = 0
        while int(self.obs["sample_state"][3]) != charge:
            trial += 1
            assert trial < len(_CATALYST_TRIALS), "no catalyst matched"
            self.goto("E"); self.interact()
            self.goto("C")
            self.cycle_selector_to(1, int(_CATALYST_TRIALS[trial]))
            self.interact()
            self.goto("E"); self.interact()
            self.goto("I"); self.interact()

        # 6. Analyze, set the report dial to (protocol, NONE), submit.
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
    """Build a V1.1 env for (seed, config_id) and run the solver once."""
    env = AxiomForgeEnv(make_v1_1_config(seed=seed, config_id=config_id))
    solver = BranchingSolver(env, options={"config_id": config_id})
    return solver.solve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the branching V1.1 manual solver.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config-ids", type=int, nargs="+",
                        default=[0, 2, 4, 6, 8])
    args = parser.parse_args()

    for config_id in args.config_ids:
        result = solve_config(args.seed, config_id)
        assert result["success"] is True, (
            f"config {config_id} failed: {result['failure_reason']}")
        assert result["terminated"] and not result["truncated"]
        print(
            f"config_id {config_id:3d} | protocol {result['protocol'].name} "
            f"| wo {result['work_order']} | steps {result['steps']:3d} "
            f"| return {result['return']:.2f} | success True"
        )
    print("V1.1 branching solver: all configs solved.")


if __name__ == "__main__":
    main()
