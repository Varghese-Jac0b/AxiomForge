# AxiomForge

**A compact causal-safety reinforcement learning benchmark for hidden-rule inference,
sparse rewards, proxy traps, and alignment-gap measurement.**

AxiomForge is a two-layer symbolic, Gymnasium-style "research lab" environment. An agent
must read clues, **infer a hidden work order** (which sample, purity, temperature, charge),
follow the episode's **required protocol**, compensate for **machine faults**, file a
**calibrated report**, and avoid **reward-hacking traps** — all under a sparse success
signal. One configuration-gated engine spans a difficulty ladder (V1.0 → V1.4), and a
suite of 14 value-based algorithms (tabular through Rainbow-style deep RL, plus a
successor/predecessor intrinsic-exploration variant) is benchmarked on it with a
train / held-out generalization protocol and a dedicated misalignment study.

> **Disclaimer.** This is a compact, laptop-scale research testbed for studying
> generalization and reward hacking in a controlled causal task — **not** a frontier-scale
> benchmark. Results below are from a verified 5-seed run and are meant to be illustrative
> and reproducible, not competitive SOTA numbers.

---

## Environment ladder

| Version | What it adds | Research question |
|---|---|---|
| **V1.0** | Deterministic causal-chain lab (fixed work order, protocol P0) | Can undirected exploration even reach the sparse success? (No — the "exploration wall.") |
| **V1.1** | Randomized work order / protocol (P0,P1) / **hidden** catalyst mapping; train vs held-out config split | Do agents generalize across latent configs, or memorize? |
| **V1.2** | Catalyst + sample **probes** (mapping becomes observable); protocol P2 | Does richer observation help or hurt transfer? |
| **V1.3** | **Faults** (heater swap, purifier leak, analyzer bias), decon, calibrated 3-dial report | Can agents diagnose and compensate for hidden faults? |
| **V1.4** | **Misalignment traps**: proxy-reward terminal, unsafe shortcut, hidden `true_score`, `alignment_gap` | Do agents reward-hack the proxy? Which methods preserve alignment? |

The `HiddenContext` (the grader's answer key) is **never** exposed in the observation;
`true_score` is a logged metric only and never enters the training reward. See
[docs/environment_ladder.md](docs/environment_ladder.md).

## Algorithms (14)

**Tabular** (shared encoded Q-key): Q-learning · SARSA · Expected SARSA · **SPIE-Q**
(successor/predecessor intrinsic exploration).

**Deep value-based** (one configurable `DQNAgent`; switches `double` / `dueling` / `per` /
`noisy`, plus a driver-level `spie` intrinsic wrapper): DQN · DDQN · Dueling DQN ·
Dueling DDQN + PER · Noisy Dueling DDQN + PER · SPIE-DQN · SPIE-DDQN · SPIE-Dueling DQN ·
SPIE-Dueling DDQN + PER · SPIE-Noisy Dueling DDQN + PER.

See [docs/algorithm_suite.md](docs/algorithm_suite.md).

## Key verified findings (full 5-seed benchmark)

- **Tabular methods memorize and do not generalize** — held-out greedy success ≈ 0.02
  across V1.1–V1.4 for all four tabular algorithms.
- **DQN generalizes far better on V1.1** — held-out 0.40 ± 0.14 (best seed 0.67), ~20× the
  tabular baseline. The deep advantage is real where the task is simplest.
- **The deep advantage decays from V1.1 → V1.4** (held-out ≈ 0.35 → 0.10 → 0.03 → 0.02):
  each added mechanism (probes, faults, traps) adds config-specific structure the flat
  network memorizes rather than transfers.
- **V1.4 E2 shows proxy reward-hacking and a measurable alignment gap** — no vanilla agent
  solves the task, but most learn to farm the proxy: visible return rises while the hidden
  `true_score` falls.
- **SPIE intrinsic exploration alone *worsens* proxy-seeking** in E2 (tabular SPIE-Q is the
  worst hacker), because novelty-seeking pulls a weak agent toward the under-explored trap.
- **Noisy Dueling DDQN + PER is the strongest alignment-preserving variant** in the
  verified run — a *negative* alignment gap (−0.90) and a 0.06 hack rate (it does the task
  instead of farming the proxy).
- **No no-demo (vanilla) agent solves V1.4 E2** — the exploration wall holds for every
  value-based method here.

Full numbers, tables, and plots: [docs/axiom_forge_full_benchmark_report.md](docs/axiom_forge_full_benchmark_report.md)
· independent reproduction: [docs/axiom_forge_codex_full_benchmark_verification.md](docs/axiom_forge_codex_full_benchmark_verification.md).

![V1.4 alignment gap over training](figures/e2_alignment_gap.png)

## Quickstart

```bash
# 1. create a virtual environment + install
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. run the test suite (283 tests)
.venv/bin/python -m pytest tests/ -q

# 3. run one small training command (tabular Q-learning on V1.1)
.venv/bin/python scripts/train_axiom_forge.py --version v1_1 --algo q_learning \
    --episodes 300 --seeds 0 --train-configs 8 --heldout-configs 4 \
    --demo-configs 8 --epsilon-start 0.1 --results-dir /tmp/quickstart

# 4. read the benchmark report
open docs/axiom_forge_full_benchmark_report.md
```

## Reproduction

```bash
.venv/bin/python -m pytest tests/ -q                       # 283 passed
.venv/bin/python scripts/run_full_benchmark.py             # the 70-job matrix (~75 min, 6-way parallel)
.venv/bin/python analysis/aggregate_full_benchmark.py      # tables -> results/full_benchmark/tables/
.venv/bin/python analysis/plot_e2_comparison.py            # plots  -> results/full_benchmark/plots/
```

Seeds: `0 1 2 3 4`. Full protocol, expected artifact counts, and folder layout:
[docs/reproducibility.md](docs/reproducibility.md).

## Results location

- `results/full_benchmark/` — the verified 5-seed benchmark: 70 per-cell `summary.csv`,
  350 per-seed CSVs, 70 generalization curves, 14 E2 plots, aggregate `tables/`, and
  comparison `plots/`.
- `results/V1.0/`, `results/V1.1…V1.4/`, `results/multi_seed/`, `results/plots/` — earlier
  single-config and smoke-scale runs backing the reference and smoke reports.

## Project status

- **V1.0–V1.4 complete** (one config-gated engine; 283 tests).
- **Full 5-seed benchmark verified** (70/70 jobs, 0 failures; tables reproduce
  byte-for-byte).
- **Next phase** (see [docs/future_roadmap.md](docs/future_roadmap.md)): code-study pass,
  PPO / policy-gradient agents, POMDP / DRQN recurrent agents, transformer-memory agents,
  counterfactual protocol tasks, SFT / imitation from solver traces, an LLM-agent wrapper,
  and a multi-agent (MARL) extension.

## Related work

AxiomForge draws on the spirit of **AI Safety Gridworlds** (specification gaming, safe
interruptibility), **DeepMind Alchemy** (latent structure inference), **causal RL**,
**intrinsic-motivation / exploration** (successor & predecessor representations,
NoisyNets), and **value-based deep RL** (DQN / Double / Dueling / PER / Rainbow). A formal
citation list is a TODO for the first tagged release.

## Limitations

AxiomForge is a deliberately small, controlled testbed; read the numbers with these caveats:

- **Laptop-scale and value-based only.** The suite is entirely value-based (tabular →
  Rainbow-style DQN + SPIE). No policy-gradient, recurrent, or memory agents yet — those
  are roadmap items, so the generalization/alignment rankings are specific to this family.
- **Baselines are demo-seeded by design.** V1.0 has an *exploration wall*: undirected
  exploration never reaches the sparse success over the ~61-step solution, so every baseline
  is seeded with the BFS solver trajectory. Reported "success" therefore measures
  generalization *from demonstrations*, not from-scratch discovery — and **no vanilla
  (no-demo) agent solves V1.0–V1.4**.
- **Single task family, config-level split.** One 10×10 two-layer task; the held-out split
  is over latent configurations, not over genuinely novel task structures.
- **Modest seed count, high E2 variance.** Headline numbers are 5 seeds; the V1.4 E2
  alignment gap has high seed variance (std up to ~8.4), so E2 rankings are indicative, not
  definitive.
- **Illustrative, not SOTA.** Greedy/deterministic evaluation on a compact benchmark —
  meant to be reproducible and attributable, not competitive state-of-the-art.

## Citation

If you use AxiomForge in your work, please cite this repository (a formal paper/citation is
planned for the first tagged release):

```bibtex
@software{axiomforge_2026,
  title  = {AxiomForge: A Compact Causal-Safety Reinforcement Learning Benchmark
            for Hidden-Rule Inference, Sparse Rewards, Proxy Traps, and
            Alignment-Gap Measurement},
  author = {AxiomForge contributors},
  year   = {2026},
  note   = {Version 1.4},
  url    = {https://github.com/<your-username>/axiomforge}
}
```

## License

MIT — see [LICENSE](LICENSE).
