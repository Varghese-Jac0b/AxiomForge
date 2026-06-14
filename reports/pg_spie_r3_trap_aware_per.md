# R3 — Trap-Aware PER (orthogonal extension)

**R3 is not a rescue step.** R2 (κ=4) is already a valid anti-hacking stopping
point — it removes proxy farming without PER. R3 is an *orthogonal extension*
asking:

> *"Can trap-aware PER recover/reintroduce replay benefits without re-igniting
> the proxy-trap amplification seen in the earlier PER/SPIE ablation?"*

The R0 ablation showed PER over-replays high-TD-error transitions, which in V1.4
*are* the proxy transitions (PER+SPIE was the worst hacker). R3 re-introduces PER
on top of R2, but makes it **trap-aware**: transitions flagged by the observable
proxy signal have their replay priority **hard-capped at the running batch
median**, so they cannot dominate replay.

Driver knob: `taper` on `pg_spie_r2_taper_noisy_dueling_ddqn_per`
([scripts/train_axiom_forge.py](../scripts/train_axiom_forge.py)) +
`PrioritizedReplayBuffer(trap_aware=True)`
([agents/dqn_axiom_forge.py](../agents/dqn_axiom_forge.py)). Runner:
[scripts/run_pg_spie_r3_trap_aware_per.py](../scripts/run_pg_spie_r3_trap_aware_per.py).

## The mechanism (one new knob over R2)

Each stored transition carries an **observable** trap flag (the same signal the
gate/sentinel use: `proxy_attempt_count` / `proxy_claim_count` increased this
step). In the PER buffer, under `trap_aware`:

- **at insertion:** a trap transition enters at `min(max_priority, batch_median)`
  instead of max priority;
- **on every `update_priorities`:** trap transitions are capped at the current
  batch median of `|TD|^α`.

Naive PER = the same but cap off (so re-ignition is visible). Nothing reads
`true_score`, `HiddenContext`, or any grader-only field; `true_score` is
reporting-only. No PSFA, no environment change.

## Variants (Noisy Dueling DDQN backbone, demo-seeded, κ=4)

| # | variant | PER | gate | penalty | trap-aware | key |
|---|---|:---:|:---:|:---:|:---:|---|
| 1 | `plain_noisy` | – | – | – | – | `noisy_dueling_ddqn` |
| 2 | `per_spie` | ✔ | – | – | – | `spie_noisy_dueling_ddqn_per` |
| 3 | `r2_k4_no_per` | – | ✔ | ✔ | – | `pg_spie_r2_noisy_dueling_ddqn` |
| 4 | `r2_k4_naive_per` | ✔ | ✔ | ✔ | – | `pg_spie_r2_noisy_dueling_ddqn_per` |
| 5 | `r2_k4_trap_aware_per` | ✔ | ✔ | ✔ | **✔** | `pg_spie_r2_taper_noisy_dueling_ddqn_per` |

(2) is the old PER+SPIE hacker re-run apples-to-apples; (3) is the R2 bar; (4)
tests whether PER re-ignites; (5) is R3.

## Success criteria

- R3 (5) **does not reintroduce** high trap_hits / hack_rate (stays near R2 (3)).
- R3 maintains or improves success vs R2.
- R3 alignment_gap stays near R2 levels.
- PER priority stats show trap transitions **deprioritized/capped** in (5)
  relative to (4).
- If (5) ≈ (4) (re-ignites anyway) → report PER as **unsafe even with sentinel
  correction**.

## Diagnostics logged

trap_hits, reward_hack_rate, alignment_gap, true_score (reporting-only), success,
visible return, **mean PER priority for trap vs non-trap transitions**,
**trap/non-trap replay ratio**, **# capped trap transitions**, and the cap
(median) value.

## How to reproduce

```bash
python scripts/run_pg_spie_r3_trap_aware_per.py --episodes 400 \
    --seeds 0 1 2 3 4 --kappa 4 --regimes demo_seeded
```

Results: `results/pg_spie_r3_trap_aware_per/`.

## Results

<!-- RESULTS:START -->

_Generated 2026-06-14 17:05 — V1.4, 400 episodes, seeds [0, 1, 2, 3, 4], kappa=4.0, Noisy Dueling DDQN. success = final-100-ep training success; align_gap = visible return − hidden true_score (reporting-only)._

### demo_seeded — demo-seeded

| variant | succ | trap_hits | trap_sd | hack | align_gap | gap_sd | true_score | trap_prio | nontrap_prio | trap_replay | n_capped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| plain_noisy | 0.546 | 0.247 | 0.389 | 0.083 | 0.932 | 1.967 | 4.196 | 0.000 | 0.000 | 0.000 | 0.000 |
| per_spie | 0.518 | 0.530 | 0.336 | 0.182 | 2.233 | 1.668 | 3.196 | 0.475 | 0.438 | 0.032 | 0.000 |
| r2_k4_no_per | 0.546 | 0.039 | 0.050 | 0.013 | -0.051 | 0.292 | 4.859 | 0.000 | 0.000 | 0.000 | 0.000 |
| r2_k4_naive_per | 0.524 | 0.165 | 0.292 | 0.056 | 0.512 | 1.407 | 4.208 | 0.413 | 0.395 | 0.012 | 0.000 |
| r2_k4_trap_aware_per | 0.518 | 0.052 | 0.067 | 0.018 | -0.135 | 0.374 | 4.542 | 0.203 | 0.390 | 0.005 | 11148.000 |

- **Does naive PER re-ignite?** naive_per vs R2: trap_hits 0.165 vs 0.039, hack 0.056 vs 0.013, align_gap 0.512 vs -0.051 (higher for naive_per => PER re-ignites hacking).
- **Does trap-aware PER prevent it?** trap_aware vs naive_per: trap_hits 0.052 vs 0.165, align_gap -0.135 vs 0.512.
- **Stays safe vs R2?** trap_aware vs R2: trap_hits 0.052 vs 0.039, success 0.518 vs 0.546.
- **Priority cap working?** trap/non-trap priority — naive_per 0.413/0.395 vs trap_aware 0.203/0.390; capped trap transitions (trap_aware) 11148.000.

<!-- RESULTS:END -->
