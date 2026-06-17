# Mini Research World — Pre-AxiomForge Era (archive)

This folder contains **pre-AxiomForge / "Mini Research World"-era files** — the earlier phase
of this project, before it evolved into AxiomForge.

## What this is
Mini Research World was the original, simpler 8×8 two-layer symbolic gridworld RL task. It was
solvable by standard value-based agents (tabular SARSA reached ~97% success). Through later
research iterations the project grew into **AxiomForge** (the current main artifact: hidden
work-order inference, faults, calibrated reports, the V1.4 misalignment traps, and the R0→R3
anti-reward-hacking ladder).

## Why these files are here
- They are **preserved for provenance / project history**, not deleted.
- The **current AxiomForge benchmark and the R0→R3 anti-reward-hacking ladder do NOT depend on
  any file in this folder.** They are a self-contained legacy island with zero inbound imports
  from the active AxiomForge scripts, tests, or agents.
- They were **archived (moved), not removed**, so the active source tree
  (`environments/`, `agents/`, `analysis/`, `results/`) reads as AxiomForge-focused while the
  earlier phase remains available.

## Contents
- `environment/mini_world_env.py` — the original Mini Research World Gymnasium environment
  (`MiniResearchWorldEnv`).
- `agents/` — the legacy tabular / random command-line agents that trained on that environment:
  `sarsa_agent.py`, `q_learning_agent.py`, `expected_sarsa_agent.py`, `random_agent.py`.
- `analysis/` — `multi_seed_eval.py` (legacy confidence-band study) and
  `compare_tabular_agents.py` (legacy tabular comparison plots).
- `results/` — the matching legacy outputs: `sarsa_10k.csv`, `q_learning_10k.csv`,
  `expected_sarsa_10k.csv` (and their `*_qtable_10k.pkl` Q-tables), `random_baseline_1k.csv`,
  the `multi_seed/` confidence-band artifacts, and the `plots/` tabular-comparison figures.

## Important caveat
These are **historical artifacts**. The archived scripts preserve their original code,
including absolute imports such as `from environments.mini_world_env import ...`. Because the
files have moved out of the original `environments/` and `agents/` package locations, they may
**not be runnable in-place without import-path adjustments**. They are kept for reference and
provenance, not as in-tree runnable tools. The current AxiomForge tabular algorithms are
re-implemented independently under `scripts/train_*_axiom_forge.py`.
