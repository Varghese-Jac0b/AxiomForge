# AxiomForge — Final Algorithm Benchmark

*Updated 2026-06-13. Covers the full tabular + deep + SPIE + noisy ladder across
the V1.0–V1.4 environment versions. Every number here is read back from a committed
CSV under `results/`; smoke-scale runs are labelled as such and must not be read as
converged benchmarks.*

## 1. Environment status
AxiomForge is one config-gated engine (`environments/axiom_forge_env.py`) spanning
V1.0 (fixed deterministic task) → V1.4 (randomized work orders/protocols/mappings,
probes, faults, misalignment traps). The V1.1–V1.4 ladder is fully implemented and
tested; per-version reference solvers (`scripts/manual_solve_axiom_forge_v1_{1..4}.py`)
solve every sampled train config. `required_purity` is fixed REFINED (a RAW target
under purify-mandating protocols would need a memory-dependent re-slot trick →
perceptual aliasing → unsolvable by a memoryless agent). V1.0 behaviour is unchanged.

## 2. Tests
```
.venv/bin/python -m pytest tests/ -q      # 283 passing (as of this writing)
```
*(Historical count. The later R0–R3 anti-reward-hacking ladder tests bring the current
suite to 330 passing.)*
Deep additions are covered by `tests/test_axiom_forge_deep.py` (17 tests): NoisyLinear
behaviour, deterministic greedy eval, the deep driver path, the switch table, and the
SPIE-intrinsic separation invariants.

## 3. Algorithms implemented (14)
One configurable tabular path and one configurable deep agent.

| # | Algorithm | Path | Switches |
|---|---|---|---|
| 1 | Q-learning | tabular | — |
| 2 | SARSA | tabular | on-policy |
| 3 | Expected SARSA | tabular | expected target |
| 4 | SPIE-Q | tabular | + SR/PR intrinsic |
| 5 | DQN | deep | — |
| 6 | Double DQN | deep | double |
| 7 | Dueling DQN | deep | dueling |
| 8 | Dueling DDQN + PER | deep | double+dueling+per |
| 9 | SPIE-DQN | deep | + spie |
| 10 | SPIE-DDQN | deep | double + spie |
| 11 | SPIE-Dueling DQN | deep | dueling + spie |
| 12 | SPIE-Dueling DDQN + PER | deep | double+dueling+per + spie |
| 13 | Noisy Dueling DDQN + PER | deep | double+dueling+per+noisy |
| 14 | SPIE-Noisy Dueling DDQN + PER | deep | all switches |

The deep agent lives in `agents/dqn_axiom_forge.py` (one `DQNAgent`, switches
`double`/`dueling`/`per`/`noisy`); SPIE is a driver-level reward wrapper reusing the
tabular `agents/spie_q_agent.py` SR/PR module. `scripts/train_axiom_forge.py` is the
single multi-config driver for all 14.

## 4. What was actually run (vs implemented)
- **Implemented + unit-tested:** all 14.
- **Fully benchmarked (≥3 seeds, long horizon):** tabular V1.0–V1.4 + the V1.4 E2
  tabular headline (prior session).
- **Smoke-scale (2 seeds, short horizon, demo-seeded; this session):** the deep family
  on V1.1/V1.2 and the deep E2 on V1.4. **Labelled smoke-scale** — directional, not
  converged.

## 5. V1.0 results (reference cell, fully benchmarked, demo-seeded)
The V1.0 exploration wall is real: vanilla / PBRS-shaped / SPIE-Q (no demos) never reach
success. Demo-seeded learners all solve it (greedy eval 1.0).

| Algorithm | Greedy eval | Episodes→90% (tabular) |
|---|---:|---:|
| Q-learning (demo) | 1.00 | **413** (most sample-efficient) |
| SARSA (demo) | 1.00 | 828 |
| Expected SARSA (demo) | 1.00 | 837 |
| DQN / DDQN / Dueling / Dueling+PER (demo) | 1.00 | — (deep; greedy-eval comparator) |

## 6. V1.1–V1.3 generalization: tabular (fully benchmarked, demo-seeded, 3 seeds)
Train = demoed even config_ids; held-out = unseen odd config_ids.

| Version | Tabular train greedy | Tabular held-out greedy | Gap |
|---|---:|---:|---:|
| V1.1 (hidden mapping) | 0.96 | 0.04 | 0.92 |
| V1.2 (probes) | 1.00 | 0.04 | 0.96 |
| V1.3 (faults) | 1.00 | 0.02 | 0.98 |

**Finding (tabular):** strong per-config memorization; near-zero held-out
generalization. The Q-key carries work-order values, so unseen configs have no learned
values. This is the motivation for function approximation — the question the deep column
answers.

## 7. V1.1 / V1.2 — deep generalization (SMOKE-SCALE: 2 seeds, 60 online episodes, demo-seeded)
All deep variants reach train greedy ≈ 1.0 (demo seeding works). The headline is
**held-out**: deep agents generalize *much* better than tabular's 0.04.

| Algorithm | V1.1 train | **V1.1 held-out** | V1.2 train | V1.2 held-out |
|---|---:|---:|---:|---:|
| tabular q_learning (ref, 3 seeds) | 0.96 | 0.04 | 1.00 | 0.04 |
| dqn | 1.00 | 0.19 | 1.00 | 0.00 |
| ddqn | 1.00 | 0.12 | — | — |
| dueling_dqn | 1.00 | 0.31 | — | — |
| **dueling_ddqn_per** | 1.00 | **0.44** | 1.00 | 0.06 |
| noisy_dueling_ddqn_per | 1.00 | 0.12 | — | — |
| spie_dqn | 1.00 | 0.19 | 1.00 | 0.00 |
| spie_ddqn | 1.00 | 0.19 | — | — |
| spie_dueling_dqn | 1.00 | 0.38 | — | — |
| spie_dueling_ddqn_per | 1.00 | 0.19 | 1.00 | 0.00 |
| spie_noisy_dueling_ddqn_per | 0.88 | 0.25 | 0.96 | 0.06 |

**Finding 1 — function approximation earns its keep.** On V1.1, every deep agent beats the
tabular held-out baseline (0.04), and **Dueling DDQN+PER reaches 0.44 held-out — an ~11×
improvement**. The *dueling* architecture is the strongest single factor (dueling_dqn 0.31,
spie_dueling_dqn 0.38, dueling_ddqn_per 0.44 all lead), consistent with its value/advantage
split sharing structure across work orders that the tabular Q-key cannot.

**Finding 2 — V1.2 probes reduce transfer at this scale.** Deep held-out collapses to ~0 on
V1.2. The probe observation carries *config-specific* catalyst-mapping values, giving the
net more to memorize per config and less to transfer — richer observation, worse
generalization. (Smoke-scale; 2 seeds — directional, worth a full-scale confirmation.)

*Caveat: smoke-scale (2 seeds × 8 held-out configs = 16 eval episodes). Held-out values
have high variance; treat as directional, not converged.*

## 8. V1.4 E2 — misalignment (tabular fully benchmarked; deep smoke-scale)
**Tabular (3 seeds, 2500 ep, vanilla no-demo):** a clean scissors divergence — visible
return rises (−3.65 → −1.76) while hidden true_score falls (−2.18 → −4.06); alignment_gap
→ +2.30; reward-hack rate → 23%. The agent never solves the task but learns to farm the
proxy trap. true_score is provably log-only (test-enforced).

**Deep E2 (SMOKE-SCALE: 2 seeds, 250 ep, vanilla no-demo)** — final-200-episode means:

| Algorithm | success | return | true_score | **alignment_gap** | hack rate | claims |
|---|---:|---:|---:|---:|---:|---:|
| tabular q_learning (3 seeds) | 0.00 | −1.76 | −4.06 | +2.30 | 0.23 | 0.69 |
| dqn | 0.00 | −2.34 | −6.50 | +4.16 | 0.47 | 1.07 |
| dueling_ddqn_per | 0.00 | −2.15 | −6.65 | **+4.50** | **0.51** | 1.15 |
| spie_dueling_ddqn_per | 0.00 | −2.52 | −5.92 | +3.40 | 0.42 | 0.94 |
| spie_noisy_dueling_ddqn_per | 0.00 | −2.43 | −2.88 | **+0.45** | **0.23** | 0.50 |

**Finding 3 — stronger function approximation hacks the proxy *harder*.** No deep agent
solves V1.4 (the wall holds for vanilla deep too), but they farm the proxy far more than
tabular: plain Dueling DDQN+PER reaches alignment_gap **+4.50 and a 51% hack rate**, roughly
double the tabular agent's +2.30 / 23%. Better representation learning generalizes the
*reward-hacking* policy across configs more effectively — a concrete "capability amplifies
misalignment" result on a laptop-scale instrument.

## 9. SPIE deep-variant findings
- **Demo-seeded (V1.1/V1.2):** SPIE-deep variants track their non-SPIE bases (train ≈ 1.0;
  held-out similar, e.g. spie_dueling_dqn 0.38 ≈ dueling_dqn 0.31). The intrinsic bonus is
  training-only and never contaminates greedy eval (test-enforced).
- **Misalignment (V1.4 E2):** adding SPIE intrinsic *reduces* proxy-farming
  (dueling_ddqn_per +4.50 → spie_dueling_ddqn_per +3.40). The proxy terminal pays a
  decaying schedule (3→1.5→0.75→0), so once over-visited it is no longer novel — SPIE's
  novelty/bottleneck bonus pulls the agent away from the "boring" trap toward unexplored
  state. **Finding 4: structure-aware intrinsic exploration is a mild reward-hacking
  mitigation here** (opposite of the naive "novelty-seekers find traps faster" guess).

## 10. Noisy-variant findings
- NoisyLinear gives self-annealing, state-dependent exploration; noisy agents run with
  ε = 0 and a deterministic (noise-zeroed) greedy eval (test-enforced).
- **Largest E2 effect:** `spie_noisy_dueling_ddqn_per` nearly closes the alignment gap
  (**+0.45**, vs +4.50 for the same architecture without spie/noisy) and halves the hack
  rate (23%). Sustained noisy + intrinsic exploration prevents premature fixation on the
  proxy. *Confound: this run combines noisy AND spie; a noisy-only E2 arm (not run) would
  be needed to separate their contributions.*

## 11. Best-performing
- **Generalization (V1.1 held-out):** Dueling DDQN+PER (**0.44**) > spie_dueling_dqn (0.38)
  > dueling_dqn (0.31) ≫ tabular (0.04). Dueling is the strongest single factor.
- **Misalignment-robustness (V1.4, lowest hacking):** spie_noisy_dueling_ddqn_per
  (gap +0.45, hack 0.23) ≪ plain dueling_ddqn_per (+4.50, 0.51).

## 12. Algorithms that failed or were unstable
- No deep variant crosses the **vanilla (no-demo) exploration wall** on V1.4 (success 0) —
  same wall the tabular family hits; demos remain necessary for task success.
- `spie_noisy_dueling_ddqn_per` showed slightly sub-1.0 demo-seeded train greedy
  (V1.1 0.88, V1.2 0.96): the intrinsic + noise exploration perturbs the demo warm start
  (the same SPIE-vs-demo tension seen in tabular SPIE-Q). Not a failure, but the noisiest
  to demo-seed.
- V1.2 deep held-out is ~0 across the board (probe observations are config-specific).

## 13. Honest limitations & future work
- Deep runs here are **smoke-scale** (2 seeds, short horizons) — directional only;
  full benchmarking (≥5 seeds, longer horizons, hyperparameter care) is future work.
- Tabular and deep both **memorize** on V1.1–V1.3; whether a larger/longer-trained deep
  net closes the held-out gap is the open question.
- SPIE intrinsic uses the SR/PR-norm adaptation, not the paper's exact equations
  (documented in `docs/spie_q_design_note.md`); the paper-reproduction gate is owed.
- The deep E2 tests whether a *function-approximating* agent also reward-hacks the proxy
  — a stronger misalignment claim than the tabular case if it holds.
- Repo is not yet under version control.

## Reproduction
```
.venv/bin/python -m pytest tests/ -q
# deep generalization (smoke-scale)
.venv/bin/python scripts/train_axiom_forge.py --version v1_1 --algo spie_dueling_ddqn_per \
    --episodes 60 --seeds 0 1 --demo-configs 12 --pretrain-steps 800 --epsilon-start 0.2
# deep misalignment E2 (vanilla)
.venv/bin/python scripts/train_axiom_forge.py --version v1_4 --algo spie_noisy_dueling_ddqn_per \
    --episodes 250 --seeds 0 1 --demo-configs 0 --e2
```
