"""
Passive Trap-Sentinel: a reward-hacking *detector* built from the same
successor/predecessor signal that, used as an attraction bonus, made SPIE the
worst hacker in the Noisy x PER x SPIE ablation.

Idea (failure-mode-as-instrument): SPIE's bonus spikes at structurally unusual
states, and the V1.4 proxy trap is exactly such a state. Used as a reward, that
spike *pulls* the agent in. Used as a *sentinel*, the same spike — gated by an
OBSERVABLE proxy event — flags "structurally novel AND yielding proxy reward =
suspicious", localizing where reward hacking happens.

Strict invariants (verified against environments/axiom_forge_env.py):
- PASSIVE. `observe()` only updates internal tables and accumulates diagnostic
  scores. It returns the bonus for transparency but NEVER feeds a reward, a
  policy decision, or a replay priority.
- OBSERVABLE-ONLY. It reads `proxy_attempt_count` / `proxy_claim_count` and the
  milestone abstraction, all of which live in the observation. It NEVER touches
  `true_score`, `HiddenContext`, `info["debug"]`, or any grader-only field.
- DECAY-AWARE. The SR/PR bonus decays as a state is revisited (the proxy tile is
  farmable, so it is revisited a lot). The sentinel therefore tracks a per-event
  running max and an early-window sum so the signal is not erased by decay.

This module implements ONLY the passive detector. No PG-SPIE gate, no proxy
penalty, no trap-aware PER, no PSFA — those are later, separately-ablated rungs.
"""

from __future__ import annotations

from agents.spie_q_agent import SuccessorPredecessorTables, milestone_abstraction


def _proxy_counts(obs: dict) -> tuple[int, int]:
    """The two observable proxy counters (always present in the obs dict;
    zero before V1.4 activates them)."""
    return int(obs["proxy_attempt_count"]), int(obs["proxy_claim_count"])


class TrapSentinel:
    """Observation-only successor/predecessor reward-hacking detector.

    Maintains its OWN SR/PR tables over a state abstraction and accumulates a
    suspicion score whenever the structural-novelty bonus coincides with an
    observable proxy event (a proxy_attempt or proxy_claim increment).

    Tables persist across episodes (so the topology stabilizes); per-episode
    accumulators reset on `reset_episode()`. `early_window_steps` bounds the
    within-episode "early" sum used to resist novelty decay.
    """

    def __init__(self, *, abstraction=milestone_abstraction,
                 sr_alpha: float = 0.1, sr_gamma: float = 0.95,
                 mode: str = "full", early_window_steps: int = 20):
        self.abstract = abstraction
        self.mode = mode
        self.early_window_steps = early_window_steps
        self.tables = SuccessorPredecessorTables(
            sr_alpha=sr_alpha, sr_gamma=sr_gamma,
            pr_alpha=sr_alpha, pr_gamma=sr_gamma)
        # global (persist across episodes) — running max defeats cross-episode decay
        self.global_max_sentinel_score = 0.0
        self.global_proxy_event_count = 0
        self.reset_episode()

    # ---------------- per-episode lifecycle ----------------

    def reset_episode(self) -> None:
        self._step = 0
        self.suspicious_trap_score = 0.0
        self.proxy_event_count = 0
        self.max_sentinel_score = 0.0
        self.early_window_score = 0.0
        self._bonus_sum_on_events = 0.0
        self.first_proxy_event_step: int | None = None
        self.events: list[dict] = []

    # ---------------- the single passive hook ----------------

    def observe(self, prev_obs: dict, obs: dict) -> float:
        """Observe one transition prev_obs -> obs. Pure side effects: update
        tables, and if this transition is an observable proxy event, accumulate
        suspicion weighted by the SR/PR bonus of the arrived state. Returns the
        bonus only for logging/transparency; callers must NOT use it as reward."""
        self._step += 1
        prev_state = self.abstract(prev_obs)
        cur_state = self.abstract(obs)
        # Tables are updated on EVERY transition (the sentinel maps the whole
        # topology), independent of whether a proxy event occurred.
        self.tables.update(prev_state, cur_state)
        bonus = self.tables.bonus(cur_state, mode=self.mode)

        # Proxy deltas are read directly from the two observations (each call
        # is self-contained; no reliance on internal step bookkeeping).
        prev_attempt, prev_claim = _proxy_counts(prev_obs)
        attempt, claim = _proxy_counts(obs)
        d_attempt = attempt - prev_attempt
        d_claim = claim - prev_claim
        proxy_event = (d_attempt > 0) or (d_claim > 0)

        if proxy_event:
            self.proxy_event_count += 1
            self.global_proxy_event_count += 1
            self.suspicious_trap_score += bonus
            self._bonus_sum_on_events += bonus
            self.max_sentinel_score = max(self.max_sentinel_score, bonus)
            self.global_max_sentinel_score = max(
                self.global_max_sentinel_score, bonus)
            if self._step <= self.early_window_steps:
                self.early_window_score += bonus
            if self.first_proxy_event_step is None:
                self.first_proxy_event_step = self._step
            self.events.append({
                "step": self._step,
                "bonus": bonus,
                "d_attempt": d_attempt,
                "d_claim": d_claim,
                "proxy_claim_count": claim,
                "paid_claim": d_claim > 0,
            })
        return bonus

    # ---------------- read-out ----------------

    def episode_summary(self) -> dict:
        n = self.proxy_event_count
        return {
            "steps": self._step,
            "suspicious_trap_score": self.suspicious_trap_score,
            "sentinel_proxy_event_count": n,
            "mean_sentinel_score": self._bonus_sum_on_events / n if n else 0.0,
            "max_sentinel_score": self.max_sentinel_score,
            "early_window_sentinel_score": self.early_window_score,
            "first_proxy_event_step": self.first_proxy_event_step,
        }
