"""
AxiomForge V1.0 manual solver (Prompt E / contract Section 12).

Executes the exact V1.0 intended plan:
    M, H, D, R interact, C cycle x2 interact, E, U, T cycle x1 interact,
    I, N, G interact
with BFS goto(tile) navigation per layer.

Rules: every decision (position, navigation target) comes only from the
observation and the static map layouts; info is consulted ONLY in the
final assertions. Prints the total step count, which feeds the
max_steps measurement rule.

Run from the repo root:
    python scripts/manual_solve_axiom_forge_v1_0.py
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environments.axiom_forge_configs import make_v1_0_config
from environments.axiom_forge_env import AxiomForgeEnv
from environments.axiom_forge_maps import find_symbol, is_wall, make_layers
from environments.axiom_forge_objects import (
    ACTION_CYCLE,
    ACTION_DOWN,
    ACTION_INTERACT,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    AnalyzerToken,
)

_DELTAS = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}


class ManualSolver:
    """Scripted navigator: observation-driven position, static-map BFS."""

    def __init__(self, env: AxiomForgeEnv):
        self.env = env
        self.grids = make_layers()  # static layout data, not env internals
        self.obs, _ = env.reset()
        self.total_steps = 0
        self.total_return = 0.0
        self.last_result = None

    def _position(self) -> tuple[int, int, int]:
        layer, row, col = (int(v) for v in self.obs["agent"])
        return layer, row, col

    def _step(self, action: int):
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        self.total_return += reward
        self.last_result = (reward, terminated, truncated, info)
        return self.last_result

    def goto(self, symbol: str) -> None:
        """BFS shortest path to the symbol's tile on the current layer."""
        layer, row, col = self._position()
        grid = self.grids[layer]
        start = (row, col)
        target = find_symbol(grid, symbol)
        came_from: dict[tuple[int, int], tuple[tuple[int, int], int] | None]
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
        if target not in came_from:
            raise RuntimeError(f"No path from {start} to {symbol!r} at {target}")
        actions: list[int] = []
        node = target
        while came_from[node] is not None:
            node, action = came_from[node]
            actions.append(action)
        for action in reversed(actions):
            self._step(action)

    def interact(self):
        return self._step(ACTION_INTERACT)

    def cycle(self, times: int = 1):
        result = None
        for _ in range(times):
            result = self._step(ACTION_CYCLE)
        return result


def main() -> None:
    env = AxiomForgeEnv(make_v1_0_config())
    solver = ManualSolver(env)

    solver.goto("M"); solver.interact()            # read work order
    solver.goto("H"); solver.interact()            # protocol clue (P0)
    solver.goto("D"); solver.interact()            # fault clue (NONE)
    solver.goto("R"); solver.interact()            # dial already A -> slot
    solver.goto("C"); solver.cycle(2); solver.interact()  # RED->BLUE->GREEN, take
    solver.goto("E"); solver.interact()            # elevator to Forge Wing
    solver.goto("U"); solver.interact()            # purify
    solver.goto("T"); solver.cycle(1); solver.interact()  # COLD->WARM, apply
    solver.goto("I"); solver.interact()            # ionize: GREEN -> POS
    solver.goto("N"); solver.interact()            # analyze

    # Observation-based sanity check before submitting (not info).
    token = int(solver.obs["last_analyzer_token"])
    assert token == int(AnalyzerToken.GREEN), (
        f"analyzer token {token}, expected GREEN ({int(AnalyzerToken.GREEN)})"
    )

    solver.goto("G")
    reward, terminated, truncated, info = solver.interact()  # selector already (P0, NONE)

    # info used ONLY in assertions, per the contract.
    assert info["success"] is True, f"failure_reason: {info['failure_reason']}"
    assert terminated and not truncated
    assert not any(info["failure_reason"].values())
    assert info["protocol_order_correct"] is True

    print("V1.0 manual solver: success = True")
    print(f"total steps        = {solver.total_steps}")
    print(f"total return       = {solver.total_return:.2f}")
    print(f"final step reward  = {reward:.2f}")
    print(f"max_steps headroom = {env.cfg.max_steps - solver.total_steps} "
          f"(max_steps={env.cfg.max_steps})")


if __name__ == "__main__":
    main()
