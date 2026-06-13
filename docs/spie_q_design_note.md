# SPIE-Q Design Note

*Successor-Predecessor Intrinsic Exploration Q-learning — AxiomForge implementation
(per ROADMAP "FROZEN DECISION", 2026-06-13; Yu & Burgess, NeurIPS 2023, arXiv:2305.15277).*

## Question this algorithm answers
Can structure-aware intrinsic exploration cross the V1.0 **exploration wall without
demonstrations**? Every existing baseline (tabular + DQN family) needs demo seeding.
Either outcome is a finding (first demo-free solve, or "bidirectional intrinsic
exploration crosses spatial but not causal-chain bottlenecks").

## Components

**State abstraction** (the open research question — configurable, ablated):
- `milestone` (default): `s̃ = (layer, row, col, milestone_count)` where
  milestone_count ∈ [0,6] counts the six observable milestones (manifest read,
  archive read, diagnostics read, sample present, refined, analyzed) — the same
  milestones as the env's PBRS potential. ≤ 2·10·10·7 = 1,400 abstract states.
- `full`: the complete tabular Q-key (the "no abstraction" ablation; SR rows are
  sparse dicts, so this stays tractable but is expected to dilute the signal).

**Successor representation (SR)** — sparse dict-of-dicts `M`, rows init `{s̃: 1.0}`,
on-policy TD update per transition `s̃ → s̃'`:
```
M[s̃] ← M[s̃] + α_sr · ( 1{s̃} + γ_sr · M[s̃'] − M[s̃] )
```

**Predecessor representation (PR)** — the SR of the time-reversed chain:
```
N[s̃'] ← N[s̃'] + α_pr · ( 1{s̃'} + γ_pr · N[s̃] − N[s̃'] )
```

**Intrinsic reward** (SR/PR-norm adaptation of SPIE):
```
r_int(s̃') = w_sr / ‖M[s̃']‖₂  +  w_pr / ‖N[s̃']‖₂        (w_sr = w_pr = 0.5)
```
Novel states have unit-norm rows → bonus ≈ 1; occupancy mass grows the norm →
bonus decays. The prospective term (Machado-style SR-norm) rewards states with
little expected future occupancy; the retrospective term rewards states few paths
lead INTO — bottleneck pressure (the elevator, the protocol gates).

**Q-learning core** — unchanged from the verified canonical script: same Q-key
(`encode_state`), ε-greedy, termination-only bootstrap, optional demo seeding.
Training target uses `r_total = r_ext + β_t · r_int`; **β anneals multiplicatively
per episode toward 0**.

## Locked discipline (from the ROADMAP)
- Intrinsic reward is **training-only**: `episode_return` in the CSV logs the
  **extrinsic** return; greedy evaluation runs β = 0 on the clean config — same
  rule as PBRS shaping.
- `done` for bootstrapping is termination-only (truncation bootstraps).
- Modes for ablation: `full` (SR+PR), `sr_only`, `pr_only`, `none` (≡ vanilla
  Q-learning through the identical code path).

## Documented deviation
The exact intrinsic-reward combination in the SPIE paper (their Eqs.) has not yet
been reproduced from the paper; this implementation uses the SR/PR-norm form above.
The paper-reproduction gate on their bottleneck gridworld remains an open follow-up
(recorded as a risk, per the ROADMAP).

## Ablation matrix (required)
| Arm | Command flags |
|---|---|
| baseline Q-learning | existing `train_q_learning_axiom_forge.py` |
| SPIE-Q | `--mode full` |
| SPIE-Q w/o intrinsic reward | `--mode none` (bit-identical to baseline) |
| SPIE-Q w/o abstraction | `--abstraction full` |
| SR-only / PR-only | `--mode sr_only` / `--mode pr_only` |

Headline run: `--demo-sweeps 0 --mode full` (the no-demonstration wall attempt).
Report: episodes-to-90% (if ever), greedy-eval success, **milestone-discovery
episodes** (first episode each milestone is reached — shows WHERE on the causal
chain exploration stalls), unique abstract states visited.
