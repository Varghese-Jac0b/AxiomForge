# R2 Consolidation — κ-sweep + 5-seed tightening

A consolidation of R2 (PG-SPIE + proxy-coupled penalty) before deciding on R3.
**No new mechanism** — this just sweeps the existing penalty coefficient `κ` and
adds seeds.

## Why

R2 at κ=1 gave a directionally good but **seed-noisy** result in the demo-seeded
regime (trap_hits 0.583→0.507, gap 2.483→2.119 vs R1, but 2/3 seeds helped and 1
reversed, with std ≈ trap 0.4 / gap 2.0). The applied penalty was also tiny
(`mean_pen ≈ 0.046`). Two questions:

1. **Does a larger κ sharpen the effect** (more deterrence), or tip into
   **over-conservatism** (success collapse)?
2. **Does the R2 < R1 improvement survive 5 seeds**, or was it noise?

## Scope: demo-seeded only (on purpose)

In the **no-demo** regime the PG gate already keeps the agent off the proxy
(`gate_proxy = 0`), so the R2 penalty **never fires** and R2 is bit-identical to
R1 for every κ. That regime was already a clean win and is κ-invariant, so the
sweep is run in **demo-seeded** only (where the proxy is actually farmed and the
penalty is active).

## Design

- Backbone: Noisy Dueling DDQN, PER off. Same budget / eval protocol as R1/R2.
- Seeds: **0–4** (5 seeds).
- κ ∈ **{1, 4, 8}**. With β₀=0.5 and a fresh-state SR/PR bonus ≈ 1.0, the
  first-claim penalty is ≈ 0.5 / 2.0 / 4.0 respectively — i.e. κ=1 is weaker than
  the +3.0 proxy payment, κ=4 partial, κ=8 makes claiming **net-negative**. A
  principled weak → partial → over-the-bribe sweep.
- References `plain_noisy` / `naive_spie` / `pg_spie` (R1) are κ-invariant and run
  once at 5 seeds.

Everything observable-only; `true_score` is reporting-only. Reuses
`run_one`/`train_seed_deep` from [run_pg_spie_r2.py](../scripts/run_pg_spie_r2.py);
earlier artifacts untouched (separate folder/report).

## What would justify R3

- A κ at which R2 **clearly** beats R1 and naive SPIE on trap_hits / hack_rate /
  align_gap **beyond the 5-seed std**, without collapsing success → R2 is solid,
  proceed to R3 (trap-aware PER) on firm ground.
- If even κ=8 only matches R1 within noise, or success collapses → R2's passive+
  active gating has hit its ceiling; R3's orthogonal PER lever is needed.

## How to reproduce

```bash
python scripts/run_pg_spie_r2_ksweep.py --kappas 1 4 8 --seeds 0 1 2 3 4
```

Results: `results/pg_spie_r2_ksweep/` (`ksweep_runs.csv`, `ksweep_summary.csv`,
per-variant seed CSVs).

## Results

<!-- RESULTS:START -->

_Generated 2026-06-14 15:06 — V1.4, 400 episodes, seeds [0, 1, 2, 3, 4], kappa sweep [1.0, 4.0, 8.0], Noisy Dueling DDQN (PER off). success = final-100-ep training success; align_gap = visible return − hidden true_score (reporting-only)._

### demo_seeded — demo-seeded (--demo-configs 8)

| variant | succ | succ_sd | trap_hits | trap_sd | hack | align_gap | gap_sd | true_score | pen_n | mean_pen |
|---|---|---|---|---|---|---|---|---|---|---|
| plain_noisy | 0.546 | 0.038 | 0.247 | 0.389 | 0.083 | 0.932 | 1.967 | 4.196 | 0.000 | 0.000 |
| naive_spie | 0.544 | 0.036 | 0.378 | 0.327 | 0.129 | 1.668 | 1.383 | 3.881 | 0.000 | 0.000 |
| pg_spie | 0.544 | 0.040 | 0.374 | 0.438 | 0.128 | 1.489 | 2.115 | 3.882 | 0.000 | 0.000 |
| pg_spie_r2_k1 | 0.546 | 0.038 | 0.308 | 0.385 | 0.104 | 1.188 | 1.899 | 4.090 | 1306.800 | 0.093 |
| pg_spie_r2_k4 | 0.546 | 0.038 | 0.039 | 0.050 | 0.013 | -0.051 | 0.292 | 4.859 | 192.600 | 0.654 |
| pg_spie_r2_k8 | 0.544 | 0.040 | 0.022 | 0.011 | 0.007 | -0.132 | 0.079 | 4.817 | 230.600 | 1.022 |

**kappa trend (R2 vs R1 reference):**

- kappa=1.0: trap_hits 0.308 (R1 0.374), align_gap 1.188 (R1 1.489), success 0.546 (plain 0.546), mean_pen 0.093.
- kappa=4.0: trap_hits 0.039 (R1 0.374), align_gap -0.051 (R1 1.489), success 0.546 (plain 0.546), mean_pen 0.654.
- kappa=8.0: trap_hits 0.022 (R1 0.374), align_gap -0.132 (R1 1.489), success 0.544 (plain 0.546), mean_pen 1.022.

Read: lower trap_hits/align_gap as kappa rises = the penalty is biting; a success drop toward 0 = over-conservative. Compare the deltas against the std columns to judge whether the effect now exceeds seed noise at 5 seeds.

<!-- RESULTS:END -->
