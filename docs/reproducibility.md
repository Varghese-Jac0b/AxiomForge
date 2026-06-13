# Reproducibility

## Environment
- Python 3.12, PyTorch 2.12 (CPU), Gymnasium, NumPy, pandas, matplotlib.
- The full 5-seed benchmark was run on an Apple M4 Max (16 cores, 128 GB), CPU only,
  `OMP_NUM_THREADS=2` per process, 6-way process parallelism. Wall time ≈ 75 min.
- Install: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
  (exact pins in `requirements-lock.txt`).

## Commands

```bash
# 1. tests (must be green before trusting any run)
.venv/bin/python -m pytest tests/ -q                      # expect: 283 passed

# 2. full benchmark matrix (70 jobs) — ~75 min, 6-way parallel
.venv/bin/python scripts/run_full_benchmark.py            # writes results/full_benchmark/...

# 3. aggregate the per-seed CSVs into statistical tables (deterministic)
.venv/bin/python analysis/aggregate_full_benchmark.py     # results/full_benchmark/tables/

# 4. cross-algorithm E2 misalignment plots
.venv/bin/python analysis/plot_e2_comparison.py           # results/full_benchmark/plots/
```

A single benchmark cell can be run directly, e.g.:
```bash
.venv/bin/python scripts/train_axiom_forge.py --version v1_1 --algo dueling_ddqn_per \
    --episodes 200 --seeds 0 1 2 3 4 --train-configs 16 --heldout-configs 12 \
    --demo-configs 16 --pretrain-steps 1000 --epsilon-start 0.2 --buffer-size 50000 \
    --results-dir results/full_benchmark/V1.1/dueling_ddqn_per
```

## Seeds
All benchmark cells use seeds **`0 1 2 3 4`**. Latent episode configurations are
deterministic per `(seed, config_id)`; train ids are even, held-out ids are odd.

## Protocol parameters (full benchmark)
- Train / held-out configs: **16 / 12**.
- Generalization runs are **demo-seeded** (16 demo configs). Tabular: 3000 episodes,
  ε_start 0.1. Deep: 200 online episodes, 1000 offline pretrain steps, ε_start 0.2 (noisy
  agents use ε=0).
- V1.4 **E2** runs are **vanilla** (no demos), ε_start 1.0; tabular 3000 ep, deep 400 ep.

## Expected artifact counts (results/full_benchmark/)
| Artifact | Count |
|---|---|
| per-cell `summary.csv` (14 algos × V1.1–V1.4 + 14 E2) | 70 |
| per-seed CSVs (`seed_*.csv`) | 350 |
| `generalization_curve.png` | 70 |
| per-cell E2 `e2_return_vs_true_score.png` | 14 |
| aggregate tables (`tables/*.csv`) | 3 (generalization, sample_efficiency, misalignment_e2) |
| comparison plots (`plots/*.png`) | 3 |
| per-job training logs (`logs/*.log`) | 70 |

## Result folder structure
```
results/
  full_benchmark/
    V1.1/<algo>/        summary.csv, seed_0..4.csv, generalization_curve.png, heldout_by_*.csv
    V1.2/<algo>/  V1.3/<algo>/  V1.4/<algo>/
    V1.4_e2/<algo>/     + e2_return_vs_true_score.png
    tables/             generalization.csv, sample_efficiency.csv, misalignment_e2.csv
    plots/              e2_alignment_gap.png, e2_hacking_rate.png, e2_return_vs_truescore.png
    logs/               <job>.log, full_progress.log
  V1.0/ V1.1/ ... multi_seed/ plots/   # earlier single-config + smoke-scale runs
```

## Independent reproduction
The benchmark was independently re-run; the aggregate tables matched **byte-for-byte**.
The rerun lives locally under `results/full_benchmark_codex_rerun_<timestamp>/` (gitignored
to avoid committing a 67 MB exact duplicate) and is summarized in
`docs/axiom_forge_codex_full_benchmark_verification.md`. Re-running
`analysis/aggregate_full_benchmark.py` on `results/full_benchmark/` reproduces the
committed tables byte-for-byte (verified).

## Notes / caveats
- The multi-config sample-efficiency proxy (episodes to train-rolling-90%) is
  **pretrain-dominated** for deep agents (offline pretraining solves train before episode
  1); the canonical online sample-efficiency number is the single-config **V1.0** result.
- V1.0 is single-config and is **not** part of the multi-config train/held-out matrix; its
  reference results live under `results/V1.0/`.
