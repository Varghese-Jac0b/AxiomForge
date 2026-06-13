# AxiomForge - Codex Full Benchmark Verification

Generated: 2026-06-13

Independent Codex verification of the AxiomForge / Mini_Research_World repository,
including source audit, tests before and after the benchmark, full 70-job benchmark
rerun, aggregation, and comparison against `docs/axiom_forge_full_benchmark_report.md`.

Final verdict: **VERIFIED**

## 1. Repository Audit Summary

Codebase map:

- `environments/`
  - `axiom_forge_env.py`: single shared `AxiomForgeEnv` engine.
  - `axiom_forge_configs.py`: V1.0-V1.4 config factories and train/held-out split helpers.
  - `axiom_forge_maps.py`, `axiom_forge_objects.py`: maps, tiles, actions, enums, hidden context.
  - `mini_world_env.py`: older/minimal environment module.
- `agents/`
  - Tabular standalone agents: `q_learning_agent.py`, `sarsa_agent.py`, `expected_sarsa_agent.py`, `spie_q_agent.py`.
  - Deep agent family: `dqn_axiom_forge.py`.
  - `random_agent.py`: random baseline.
- `scripts/`
  - Multi-config driver: `train_axiom_forge.py`.
  - V1.0/single-algorithm scripts: `train_q_learning_axiom_forge.py`, `train_sarsa_axiom_forge.py`, `train_expected_sarsa_axiom_forge.py`, `train_spie_q_axiom_forge.py`, `train_dqn_axiom_forge.py`.
  - Manual solvers: `manual_solve_axiom_forge_v1_0.py` through `manual_solve_axiom_forge_v1_4.py`.
  - Shared helpers: `axiom_forge_baselines_common.py`.
- `analysis/`
  - `aggregate_full_benchmark.py`
  - `plot_e2_comparison.py`
  - `compare_tabular_agents.py`
  - `multi_seed_eval.py`
- `tests/`
  - 12 test files covering configs, maps, objects, V1.0-V1.4 env behavior, driver, baselines, deep agents, and SPIE-Q.
- `results/`
  - Previous full run: `results/full_benchmark/`
  - Codex rerun: `results/full_benchmark_codex_rerun_20260613_174746/`
- `docs/`
  - Previous report: `axiom_forge_full_benchmark_report.md`
  - Prior algorithm report: `axiom_forge_final_algorithm_benchmark.md`
  - SPIE-Q note: `spie_q_design_note.md`
  - This report.

Audit findings:

- V1.0-V1.4 are implemented through one shared `AxiomForgeEnv` controlled by `AxiomForgeConfig`; I found no duplicated per-version env classes.
- V1.0 is deterministic through `make_v1_0_config()`.
- V1.1 randomization and parity train/held-out split exist.
- V1.2 probe flags and probe handling exist.
- V1.3 faults, decon, analyzer bias, safety, and report confidence exist.
- V1.4 proxy terminal, unsafe shortcut, `true_score`, and `alignment_gap` exist.
- `true_score` is computed in `_get_info()` and explicitly not added to reward in `_handle_submission_interact()`.
- The multi-config driver has explicit tabular and deep switch tables, including Noisy and SPIE variants.
- Suspicious/non-blocking notes:
  - The working directory does not present as a Git repo to `git status`.
  - The stock aggregation and plotting scripts are hardcoded to `results/full_benchmark`; I used wrappers that redirect only the root path to the Codex rerun folder.
  - No PDFs were present in the repository to inspect.

## 2. Tests

Before benchmark:

```bash
.venv/bin/python -m pytest tests/ -q
```

Result:

```text
283 passed in 16.52s
```

After benchmark:

```bash
.venv/bin/python -m pytest tests/ -q
```

Result:

```text
283 passed in 16.01s
```

This verifies the claimed 283 passing tests before and after the rerun.

## 3. Benchmark Commands Used

The prior launcher `/tmp/af_audit/run_bench.py` was inspected and found valid in
protocol shape, but it writes to `results/full_benchmark`. To preserve old results,
I created an equivalent Codex launcher pointed at a new output root.

Benchmark command:

```bash
.venv/bin/python /tmp/af_codex/run_bench_codex_20260613_174746.py
```

Preserved copy:

```text
results/full_benchmark_codex_rerun_20260613_174746/repro/run_bench_codex_20260613_174746.py
```

Aggregation command:

```bash
.venv/bin/python /tmp/af_codex/aggregate_codex_20260613_174746.py
```

Preserved copy:

```text
results/full_benchmark_codex_rerun_20260613_174746/repro/aggregate_codex_20260613_174746.py
```

E2 plotting command:

```bash
.venv/bin/python /tmp/af_codex/plot_e2_codex_20260613_174746.py
```

Preserved copy:

```text
results/full_benchmark_codex_rerun_20260613_174746/repro/plot_e2_codex_20260613_174746.py
```

Protocol:

- Algorithms: 14 total.
- Versions: V1.1, V1.2, V1.3, V1.4.
- E2: V1.4 E2 for all 14 algorithms.
- Seeds: `0 1 2 3 4`.
- Train configs: 16.
- Held-out configs: 12.
- Generalization tabular: 3000 episodes, `--demo-configs 16 --demo-sweeps 4 --epsilon-start 0.1`.
- Generalization deep: 200 episodes, `--demo-configs 16 --pretrain-steps 1000 --epsilon-start 0.2 --buffer-size 50000 --device cpu`.
- E2 tabular: 3000 episodes, `--demo-configs 0 --e2 --epsilon-start 1.0`.
- E2 deep: 400 episodes, `--demo-configs 0 --e2 --epsilon-start 1.0 --buffer-size 50000 --device cpu`.

## 4. Runtime And Environment Notes

- Benchmark launcher runtime: 4638 seconds.
- Output size: 67 MB.
- Python: 3.12.2.
- Platform as reported by Python: `macOS-26.5.1-arm64-arm-64bit`.
- PyTorch: 2.12.0.
- Gymnasium: 1.3.0.
- `torch.backends.mps.is_available()`: `False` in this runtime.
- CPU model and memory were not independently verified because `sysctl` was denied by sandbox permissions.
- Plotting emitted non-fatal Matplotlib cache warnings because `~/.matplotlib` was not writable; Matplotlib used a temp cache and still wrote plots.

## 5. Result Folders And Counts

Codex rerun root:

```text
results/full_benchmark_codex_rerun_20260613_174746/
```

Counts:

- Summary CSVs: 70.
- Per-seed CSVs: 350.
- E2 summary CSVs: 14.
- E2 per-seed CSVs: 70.
- Generalization curves: 70.
- Per-algorithm E2 return-vs-true-score plots: 14.
- Cross-algorithm E2 plots:
  - `plots/e2_alignment_gap.png`
  - `plots/e2_hacking_rate.png`
  - `plots/e2_return_vs_truescore.png`
- Aggregate tables:
  - `tables/generalization.csv` with shape `(56, 15)`.
  - `tables/misalignment_e2.csv` with shape `(14, 10)`.
  - `tables/sample_efficiency.csv` with shape `(56, 4)`.

Validation:

- Every summary CSV has 5 rows.
- No seed CSV is empty.
- Every E2 seed CSV includes `true_score`, `alignment_gap`, and `reward_hack_flag`.
- Launcher result: `done 70/70, failed []`.

## 6. Comparison With Previous Report

The Codex rerun aggregate CSVs are byte-identical to the previous full benchmark tables:

- `results/full_benchmark/tables/generalization.csv`
- `results/full_benchmark/tables/misalignment_e2.csv`
- `results/full_benchmark/tables/sample_efficiency.csv`

Therefore the rerun matches both directionally and numerically, not merely within
stochastic variation.

Key reproduced values:

- V1.1 held-out:
  - Tabular mean of algorithm means: 0.017, reported as about 0.02.
  - DQN: 0.400 held-out mean, 0.143 std, 0.667 best seed.
- Deep held-out mean by version:
  - V1.1: 0.348.
  - V1.2: 0.105.
  - V1.3: 0.027.
  - V1.4: 0.012.
- V1.4 E2:
  - DQN: gap 2.841, hack 0.374.
  - DDQN: gap 4.382, hack 0.461.
  - SPIE-Q: gap 6.787, hack 0.523.
  - SPIE-DQN: gap 4.390, hack 0.508.
  - Noisy Dueling DDQN + PER: gap -0.902, hack 0.062.
  - SPIE-Noisy Dueling DDQN + PER: gap -0.630, hack 0.112.
  - Max E2 success mean across algorithms: 0.0.

Confirmed previous claims:

- 70/70 benchmark jobs completed.
- 0 failures.
- 5 seeds per cell.
- V1.1 tabular held-out is about 0.02 and DQN held-out is about 0.40.
- Deep advantage decays from V1.1 to V1.4.
- V1.2 probes reduce transfer.
- V1.4 E2 shows capability-linked hacking among non-noisy deep variants.
- DQN and DDQN E2 gaps reproduce at about 2.84 and 4.38.
- SPIE alone worsens hacking in E2, especially `spie_q` and `spie_dqn`.
- Noisy Dueling DDQN + PER mitigates hacking and has negative alignment gap.
- No vanilla agent solves V1.4 E2.
- 283 tests pass before and after benchmark.

Changed findings:

- None at aggregate-table precision. The rerun tables are byte-identical to the previous tables.

Unverified or unstable findings:

- CPU model, memory size, and MPS hardware availability from the previous report were not independently verified due sandbox restrictions. In this runtime, PyTorch reported MPS unavailable.
- The report's scientific conclusions are supported by this deterministic rerun, but broader statistical stability beyond 5 seeds remains a limitation.

## 7. Major Findings

1. Function approximation generalizes on V1.1 where tabular methods mostly memorize.
2. Generalization decays sharply as the ladder adds probes, faults, traps, and report structure.
3. V1.2 probes reduce transfer despite adding observability.
4. In V1.4 E2, non-noisy deep capability can amplify proxy hacking.
5. SPIE alone is not an alignment fix in E2; it worsens proxy-seeking in several variants.
6. Noisy Dueling DDQN + PER is the strongest E2 mitigation in the benchmark.
7. No vanilla no-demo agent solves V1.4 E2.

## 8. Remaining Risks

- The full run is deterministic/reproducible here, but still only uses 5 seeds.
- Deep runs use short online horizons and substantial pretraining, so sample-efficiency comparisons are not apples-to-apples.
- `analysis/aggregate_full_benchmark.py` and `analysis/plot_e2_comparison.py` are hardcoded to `results/full_benchmark`; rerun-safe use requires wrappers or future parameterization.
- The workspace is not currently recognized by `git status`, so I could not verify commit cleanliness.
- Matplotlib cache configuration should be made writable or `MPLCONFIGDIR` should be set for cleaner plotting runs.

## 9. What To Study First

Recommended reading order:

1. `environments/axiom_forge_configs.py`: understand the V1.0-V1.4 ladder as data.
2. `environments/axiom_forge_env.py`: read the single engine and verify reward vs log-only scoring.
3. `scripts/manual_solve_axiom_forge_v1_*.py`: understand what "solving" means for each version.
4. `scripts/train_axiom_forge.py`: understand train/held-out splits, demo recording, evaluation, and algorithm selection.
5. `agents/dqn_axiom_forge.py`: understand DQN/DDQN/dueling/PER/noisy switches.
6. `agents/spie_q_agent.py`: understand intrinsic successor/predecessor exploration.
7. `tests/test_axiom_forge_v1_*.py`, `tests/test_axiom_forge_driver.py`, `tests/test_axiom_forge_deep.py`: tie behavior to tests.
8. `results/full_benchmark_codex_rerun_20260613_174746/tables/*.csv`: study results only after the mechanics are clear.

## 10. Final Verdict

**VERIFIED**

The repository is internally consistent for the claimed V1.0-V1.4 environment ladder,
algorithm wiring, full benchmark protocol, and reported benchmark findings. Tests pass
before and after. The Codex rerun completed all 70 jobs with 0 failures, produced the
expected 350 per-seed CSVs, generated aggregate tables and plots, and reproduced the
previous report's aggregate tables byte-for-byte.
