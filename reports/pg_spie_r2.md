# R2 — PG-SPIE + Proxy-Coupled Penalty (Trap-Sentinel Penalty)

The second rung. It keeps R1's observable protocol gate unchanged and adds **one
new knob**: on a proxy step the SPIE bonus is turned into an **active negative
response** instead of merely being suppressed. This is the "failure-as-instrument"
idea made into a training signal — the same successor/predecessor novelty that
made naive SPIE farm the proxy now *repels* the agent from it.

This rung adds **no** trap-aware PER (R3) and **no** PSFA. Knobs:
`pp` + `--kappa` on `pg_spie_r2_noisy_dueling_ddqn` in
[scripts/train_axiom_forge.py](../scripts/train_axiom_forge.py)
(`gated_intrinsic`). Runner: [scripts/run_pg_spie_r2.py](../scripts/run_pg_spie_r2.py).

## Why this rung

R1 was a partial success: in the no-demo regime the gate fully removed naive
SPIE's hacking, but in the demo-seeded regime passive suppression was not enough —
demos already route the agent through proxy-adjacent regions, so merely *not
rewarding* novelty there left the proxy farming untouched. R2 tests whether an
**active** repulsion closes that gap.

## The mechanism (one knob over R1)

Per transition, the R1 gate still returns `advance` / `neutral` / `proxy`. The
only change is the **proxy** branch (`gated_intrinsic`):

| decision | R1 (PG-SPIE) | R2 (this rung) |
|---|---|---|
| advance | keep bonus `+β·b` | keep bonus `+β·b` |
| neutral | suppress `0` | suppress `0` |
| **proxy** | suppress `0` | **penalize `−κ·β·b`** |

`κ` (`--kappa`, default 1.0) scales the penalty relative to the SPIE bonus `b`
the proxy state emits. Everything is **observable-only**: the proxy branch fires
on a `proxy_attempt_count` / `proxy_claim_count` increment; nothing reads
`true_score`, `HiddenContext`, or any grader-only field. `true_score` is for
reporting only. Setting `pp=False` reproduces R1 exactly.

## Variants compared (Noisy Dueling DDQN backbone, PER off)

| variant | SPIE | gate | penalty | key |
|---|:---:|:---:|:---:|---|
| `plain_noisy` | – | – | – | `noisy_dueling_ddqn` |
| `naive_spie` | ✔ | – | – | `spie_noisy_dueling_ddqn` |
| `pg_spie` (R1) | ✔ | ✔ | – | `pg_spie_noisy_dueling_ddqn` |
| `pg_spie_r2` | ✔ | ✔ | **✔** | `pg_spie_r2_noisy_dueling_ddqn` |

Two regimes (no-demo / demo-seeded), same budget, seeds, and eval protocol as R1.

## Success criteria

- **no-demo:** R2 should stay **at least as safe** as R1 (don't regress the
  alignment win).
- **demo-seeded:** R2 should **reduce** `trap_hits`, `reward_hack_rate`, and
  `alignment_gap` vs R1 and naive SPIE.
- If success collapses while alignment improves → report the trade-off honestly.
- If R2 just avoids everything (success ≈ 0 everywhere) → report it as
  **over-conservative**.

## How to reproduce

```bash
python scripts/run_pg_spie_r2.py --episodes 400 --seeds 0 1 2 --kappa 1.0
# smoke:
python scripts/run_pg_spie_r2.py --episodes 30 --seeds 0 \
    --train-configs 4 --heldout-configs 4 --regimes no_demo
```

Results: `results/pg_spie_r2/<regime>/<variant>/` plus `pg_spie_r2_runs.csv`
(one row per regime/variant/seed) and `pg_spie_r2_summary.csv`.

## Results

<!-- RESULTS:START -->

_Generated 2026-06-14 11:03 — V1.4, 400 episodes, seeds [0, 1, 2], kappa=1.0, backbone Noisy Dueling DDQN (PER off). success = final-100-episode training success; align_gap = visible return − hidden true_score (higher = more hacking, reporting-only)._

### no_demo — no-demo (misalignment/E2)

| variant | succ | trap_hits | hack_rate | align_gap | true_score | final_ret | net_intrinsic | gate_proxy | pen_count | pen_total | mean_pen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| plain_noisy | 0.000 | 0.000 | 0.000 | -1.692 | -0.989 | -2.789 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| naive_spie | 0.000 | 0.308 | 0.124 | -0.185 | -2.429 | -3.008 | 21.797 | 0.000 | 0.000 | 0.000 | 0.000 |
| pg_spie | 0.000 | 0.000 | 0.000 | -1.911 | -0.984 | -3.146 | 0.071 | 0.000 | 0.000 | 0.000 | 0.000 |
| pg_spie_r2 | 0.000 | 0.000 | 0.000 | -1.911 | -0.984 | -3.146 | 0.071 | 0.000 | 0.000 | 0.000 | 0.000 |

- **R2 vs R1** (trap_hits 0.000 vs 0.000; hack_rate 0.000 vs 0.000; align_gap -1.911 vs -1.911) — lower for R2 means the active penalty added value beyond R1's passive suppression.
- **R2 vs naive SPIE** (trap_hits 0.000 vs 0.308; align_gap -1.911 vs -0.185).
- **Cost check** (success R2 0.000 vs plain 0.000) — if success collapses while alignment improves, the penalty is over-conservative.

### demo_seeded — demo-seeded (--demo-configs 8)

| variant | succ | trap_hits | hack_rate | align_gap | true_score | final_ret | net_intrinsic | gate_proxy | pen_count | pen_total | mean_pen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| plain_noisy | 0.543 | 0.392 | 0.132 | 1.615 | 3.788 | 5.416 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| naive_spie | 0.540 | 0.555 | 0.188 | 2.366 | 3.456 | 6.346 | 14.730 | 0.000 | 0.000 | 0.000 | 0.000 |
| pg_spie | 0.540 | 0.583 | 0.199 | 2.483 | 3.262 | 5.823 | 0.868 | 2070.667 | 0.000 | 0.000 | 0.000 |
| pg_spie_r2 | 0.543 | 0.507 | 0.172 | 2.119 | 3.499 | 5.788 | 0.652 | 2098.000 | 2098.000 | 82.745 | 0.046 |

- **R2 vs R1** (trap_hits 0.507 vs 0.583; hack_rate 0.172 vs 0.199; align_gap 2.119 vs 2.483) — lower for R2 means the active penalty added value beyond R1's passive suppression.
- **R2 vs naive SPIE** (trap_hits 0.507 vs 0.555; align_gap 2.119 vs 2.366).
- **Cost check** (success R2 0.543 vs plain 0.543) — if success collapses while alignment improves, the penalty is over-conservative.

<!-- RESULTS:END -->
