# Benchmark Summary (readable digest)

A short version of [axiom_forge_full_benchmark_report.md](axiom_forge_full_benchmark_report.md).
Full-scale = 5 seeds; 16 train / 12 held-out configs; train/held-out protocol unchanged.

## Generalization — held-out greedy success (mean)

| algo | V1.1 | V1.2 | V1.3 | V1.4 |
|---|---:|---:|---:|---:|
| tabular (all 4) | 0.02 | 0.02 | 0.02 | 0.00 |
| dqn | **0.40** | 0.08 | 0.00 | 0.00 |
| dueling_ddqn_per | 0.33 | 0.15 | 0.03 | 0.02 |
| noisy_dueling_ddqn_per | 0.35 | 0.13 | 0.03 | 0.02 |
| spie_dueling_ddqn_per | 0.37 | 0.10 | 0.03 | 0.02 |
| spie_noisy_dueling_ddqn_per | 0.37 | 0.07 | 0.02 | 0.00 |

Train greedy success is ≈ 0.98–1.00 for nearly all algorithms (demo seeding works);
`expected_sarsa` degrades to 0.40 on V1.4.

**Takeaways:** (1) tabular memorizes — held-out ≈ 0.02 everywhere; (2) deep generalizes on
V1.1 (dqn 0.40, ~20× tabular); (3) the deep advantage **decays monotonically** V1.1 → V1.4
as probes/faults/traps add config-specific structure.

## V1.4 E2 — misalignment (vanilla, final-200-episode means)
No agent solves the task (success 0). `alignment_gap = return − true_score`
(higher = more hacking; negative = does the right thing).

| algo | align_gap | hack rate |
|---|---:|---:|
| **noisy_dueling_ddqn_per** | **−0.90** | **0.06** |
| spie_noisy_dueling_ddqn_per | −0.63 | 0.11 |
| spie_dueling_ddqn_per | 0.69 | 0.24 |
| sarsa | 1.76 | 0.21 |
| q_learning | 2.94 | 0.26 |
| dqn | 2.84 | 0.37 |
| ddqn | 4.38 | 0.46 |
| spie_dqn | 4.39 | 0.51 |
| **spie_q (tabular)** | **6.79** | **0.52** |

**Takeaways:** (1) **noisy exploration is the best alignment-preserver** (negative gap);
(2) **SPIE intrinsic *alone* worsens hacking** (spie_q worst; spie_dqn > dqn) because
novelty pulls a weak agent toward the under-explored proxy; (3) more DQN capability tends
to amplify hacking among non-noisy variants.

## Rankings
- Best tabular: **spie_q** (most robust train; held-out is a 4-way tie at ~0).
- Best deep generalization: **dqn** on V1.1 (0.40).
- Best alignment-preserving: **noisy_dueling_ddqn_per** (gap −0.90, hack 0.06).
- Best SPIE: **spie_dueling_ddqn_per**.
- Most sample-efficient (canonical, V1.0): demo-seeded **q_learning** (~413 episodes).

## Negative findings
No vanilla agent solves V1.4; deep generalization collapses to tabular levels by V1.3/V1.4;
no tabular algorithm generalizes; E2 seed variance is high (gap std up to 8.4).

Plots: `results/full_benchmark/plots/` (alignment gap, hacking rate, return-vs-true_score)
and `figures/`.
