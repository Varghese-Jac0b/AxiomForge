# R1 — Protocol-Gated SPIE (PG-SPIE)

The first rung of the alignment-gated exploration ladder. It converts naive SPIE
from a blind novelty bonus into a **protocol-gated** one: the existing SPIE
intrinsic is kept **only** when observable protocol progress increases, and
suppressed otherwise. This rung adds **no penalty** (that is R2), **no
trap-aware PER** (R3), and **no PSFA**.

Driver knob: the `pg` switch on `pg_spie_noisy_dueling_ddqn` in
[scripts/train_axiom_forge.py](../scripts/train_axiom_forge.py)
(`protocol_gate` + the gated intrinsic block). Runner:
[scripts/run_pg_spie_r1.py](../scripts/run_pg_spie_r1.py).

## Why this rung

The Noisy × PER × SPIE ablation showed naive SPIE **amplifies** reward hacking:
its bonus is `1/‖M(s)‖`, which spikes at structurally unusual states — and the
V1.4 proxy region is exactly such a state — so novelty *pays the agent to farm
the proxy*. The passive Trap-Sentinel probe then confirmed the proxy signal is
cleanly observable. PG-SPIE acts on that: it lets SPIE explore **only** where the
agent is actually making protocol progress, and shuts the bonus off on the
neutral wandering and proxy-claim steps where naive SPIE does its damage.

## The gate (observable-only)

Per transition `s → s′`, `protocol_gate` returns:

| decision | condition (from the observation only) | SPIE bonus |
|---|---|---|
| **proxy** | `proxy_attempt_count` or `proxy_claim_count` increased | **suppressed** (0) |
| **advance** | milestone progress increased (`milestone_abstraction(obs)[3]`) | **allowed** |
| **neutral** | neither | **suppressed** (0) |

`proxy` takes precedence over `advance` (different tiles, so they cannot co-occur,
but the rule is explicit). Milestone progress is a **monotone** count of protocol
milestones (manifest / archive / diagnostics read + sample present / refined /
analyzed), so `advance` fires at most once per milestone — the gate cannot be
farmed. Nothing here reads `true_score`, `HiddenContext`, or any grader-only
field; `true_score` is used for **reporting only**.

## Variants compared (same Noisy Dueling DDQN backbone, PER off)

| variant | SPIE | gate | key |
|---|:---:|:---:|---|
| `plain_noisy` | – | – | `noisy_dueling_ddqn` |
| `naive_spie` | ✔ | – | `spie_noisy_dueling_ddqn` |
| `pg_spie` (R1) | ✔ | **✔** | `pg_spie_noisy_dueling_ddqn` |

Run on V1.4 in two regimes — **no-demo** (misalignment/E2) and **demo-seeded**
(`--demo-configs 8`) — with the same budget, seeds, and evaluation protocol.

## Success criteria

- PG-SPIE should **reduce** `trap_hits`, `reward_hack_rate`, and `alignment_gap`
  vs naive SPIE.
- In **no-demo**, it is acceptable for PG-SPIE to behave close to `plain_noisy`,
  as long as it avoids naive SPIE's hacking.
- In **demo-seeded**, check whether PG-SPIE preserves alignment while increasing
  useful protocol-progress exploration (more `gate_advance`, allowed bonus
  concentrated on progress steps).

## How to reproduce

```bash
# full comparison (both regimes, 3 seeds)
python scripts/run_pg_spie_r1.py --episodes 400 --seeds 0 1 2

# smoke
python scripts/run_pg_spie_r1.py --episodes 30 --seeds 0 \
    --train-configs 4 --heldout-configs 4 --regimes no_demo
```

Results: `results/pg_spie_r1/<regime>/<variant>/` plus `pg_spie_r1_runs.csv`
(one row per regime/variant/seed) and `pg_spie_r1_summary.csv`.

## Results

<!-- RESULTS:START -->

_Generated 2026-06-14 07:55 — V1.4, 400 episodes, seeds [0, 1, 2], backbone Noisy Dueling DDQN (PER off). success = final-100-episode training success; align_gap = visible return − hidden true_score (higher = more hacking, reporting-only)._

### no_demo — no-demo (misalignment/E2)

| variant | succ | trap_hits | hack_rate | align_gap | true_score | final_ret | intrinsic(allowed) | gate_adv | gate_proxy | gate_neutral | mean_allowed | mean_suppr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain_noisy | 0.000 | 0.000 | 0.000 | -1.692 | -0.989 | -2.789 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| naive_spie | 0.000 | 0.308 | 0.124 | -0.185 | -2.429 | -3.008 | 21.797 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| pg_spie | 0.000 | 0.000 | 0.000 | -1.911 | -0.984 | -3.146 | 0.071 | 342.667 | 0.000 | 110057.333 | 0.085 | 0.057 |

- **PG vs naive SPIE** (trap_hits 0.000 vs 0.308; align_gap -1.911 vs -0.185; hack_rate 0.000 vs 0.124) — lower for PG = the gate removed SPIE's pro-hacking pull.
- **PG vs plain noisy** (align_gap -1.911 vs -1.692; success 0.000 vs 0.000) — PG should match plain on alignment; any success/exploration gain is the gate's directed-exploration value.

### demo_seeded — demo-seeded (--demo-configs 8)

| variant | succ | trap_hits | hack_rate | align_gap | true_score | final_ret | intrinsic(allowed) | gate_adv | gate_proxy | gate_neutral | mean_allowed | mean_suppr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain_noisy | 0.543 | 0.392 | 0.132 | 1.615 | 3.788 | 5.416 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| naive_spie | 0.540 | 0.555 | 0.188 | 2.366 | 3.456 | 6.346 | 14.730 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| pg_spie | 0.540 | 0.583 | 0.199 | 2.483 | 3.262 | 5.823 | 0.868 | 2311.667 | 2070.667 | 67023.333 | 0.150 | 0.073 |

- **PG vs naive SPIE** (trap_hits 0.583 vs 0.555; align_gap 2.483 vs 2.366; hack_rate 0.199 vs 0.188) — lower for PG = the gate removed SPIE's pro-hacking pull.
- **PG vs plain noisy** (align_gap 2.483 vs 1.615; success 0.540 vs 0.543) — PG should match plain on alignment; any success/exploration gain is the gate's directed-exploration value.

<!-- RESULTS:END -->
