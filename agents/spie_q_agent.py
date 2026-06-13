"""
SPIE-Q building blocks: sparse successor/predecessor representations,
the intrinsic bonus, and the state abstractions.

Design: docs/spie_q_design_note.md (frozen ROADMAP decision, 2026-06-13).
The Q-learning core itself lives in the training script and is identical
to the verified canonical baseline; this module holds only the
exploration machinery, so `--mode none` reduces SPIE-Q to vanilla
Q-learning through the same code path.

Pure tabular, no torch, no env imports - obs dicts in, numbers out.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------
# State abstractions (the open design question; configurable, ablated)
# ---------------------------------------------------------------------

# milestone_count mirrors the env's six PBRS milestones, computed from
# the OBSERVATION only: manifest read, archive read, diagnostics read,
# sample present, refined, analyzed.
_REFINED = 2  # Purity.REFINED


def milestone_abstraction(obs: dict) -> tuple[int, ...]:
    """(layer, row, col, milestone_count) - <= 1,400 abstract states."""
    milestones = (
        int(obs["work_order"][0])            # manifest_read flag
        + int(obs["knowledge_bits"][0])      # archive_read
        + int(obs["knowledge_bits"][1])      # diagnostics_read
        + int(int(obs["sample_state"][0]) > 0)
        + int(int(obs["sample_state"][1]) == _REFINED)
        + int(obs["sample_state"][5])        # analyzed
    )
    return (
        int(obs["agent"][0]), int(obs["agent"][1]), int(obs["agent"][2]),
        milestones,
    )


def make_abstraction(name: str, full_encoder):
    """Resolve an abstraction by name. `full_encoder` is the canonical
    tabular Q-key function (the no-abstraction ablation)."""
    if name == "milestone":
        return milestone_abstraction
    if name == "full":
        return full_encoder
    raise ValueError(f"unknown abstraction {name!r}")


# ---------------------------------------------------------------------
# Successor / predecessor representations
# ---------------------------------------------------------------------


class SuccessorPredecessorTables:
    """Sparse tabular SR (M) and PR (N) with TD updates.

    Rows are dicts and initialise to the identity row {s: 1.0}, so a
    never-updated state has unit norm and maximal bonus; occupancy mass
    accumulated by the TD updates grows the norm and decays the bonus.
    """

    def __init__(self, *, sr_alpha: float = 0.1, sr_gamma: float = 0.95,
                 pr_alpha: float = 0.1, pr_gamma: float = 0.95):
        self.sr_alpha = sr_alpha
        self.sr_gamma = sr_gamma
        self.pr_alpha = pr_alpha
        self.pr_gamma = pr_gamma
        self.M: dict = {}   # SR rows: expected discounted FUTURE occupancy
        self.N: dict = {}   # PR rows: expected discounted PAST occupancy

    @staticmethod
    def _row(table: dict, state) -> dict:
        row = table.get(state)
        if row is None:
            row = {state: 1.0}
            table[state] = row
        return row

    @staticmethod
    def _td_row_update(row: dict, own_state, next_row: dict,
                       alpha: float, gamma: float) -> None:
        """row <- row + alpha * (onehot(own) + gamma * next_row - row)."""
        keys = set(row) | set(next_row) | {own_state}
        for key in keys:
            target = (1.0 if key == own_state else 0.0) \
                + gamma * next_row.get(key, 0.0)
            row[key] = row.get(key, 0.0) \
                + alpha * (target - row.get(key, 0.0))

    def update(self, state, next_state) -> None:
        """One transition s -> s': SR forward update, PR reversed update."""
        # SR: M[s] toward onehot(s) + gamma * M[s']
        self._td_row_update(
            self._row(self.M, state), state, self._row(self.M, next_state),
            self.sr_alpha, self.sr_gamma,
        )
        # PR: N[s'] toward onehot(s') + gamma * N[s]  (time-reversed chain)
        self._td_row_update(
            self._row(self.N, next_state), next_state,
            self._row(self.N, state), self.pr_alpha, self.pr_gamma,
        )

    @staticmethod
    def _norm(row: dict) -> float:
        return math.sqrt(sum(v * v for v in row.values()))

    def bonus(self, state, *, mode: str = "full",
              w_sr: float = 0.5, w_pr: float = 0.5) -> float:
        """Intrinsic bonus for arriving in `state`:
        w_sr/||M[state]|| (prospective novelty, Machado-style) +
        w_pr/||N[state]|| (retrospective bottleneck pressure)."""
        if mode == "none":
            return 0.0
        sr_term = w_sr / max(self._norm(self._row(self.M, state)), 1e-9)
        pr_term = w_pr / max(self._norm(self._row(self.N, state)), 1e-9)
        if mode == "sr_only":
            return sr_term / max(w_sr, 1e-9) * (w_sr + w_pr)
        if mode == "pr_only":
            return pr_term / max(w_pr, 1e-9) * (w_sr + w_pr)
        if mode == "full":
            return sr_term + pr_term
        raise ValueError(f"unknown SPIE mode {mode!r}")
