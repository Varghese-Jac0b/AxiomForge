# Future Roadmap

AxiomForge V1.0–V1.4 (environment ladder + 14-algorithm value-based suite + verified
5-seed benchmark) is complete. This document lists the planned next directions. None of
these are implemented yet; they are research directions, not commitments.

## Agents / algorithms
- **PPO / policy-gradient agents.** Add an on-policy actor-critic column to test whether
  policy-gradient methods generalize differently from value-based ones on the V1.1–V1.4
  ladder (the suite is currently entirely value-based).
- **DRQN / recurrent agents.** A recurrent value agent (LSTM/GRU) to handle the genuine
  partial observability (hidden mapping pre-probe, episode-level machine history) that a
  memoryless agent cannot represent.
- **Transformer-DQN / memory agents.** A small attention-based value head over the
  observation/clue history, to study whether explicit memory closes the V1.3/V1.4
  generalization collapse.

## Environment extensions
- **POMDP AxiomForge.** A windowed-observation variant (local view instead of full grid)
  to make partial observability first-class and motivate the recurrent/memory agents.
- **Counterfactual protocol tasks.** Episodes that require reasoning about what *would*
  have satisfied an alternative work order / protocol — a sharper test of causal-structure
  inference than the current single-target task.

## Learning from demonstrations
- **SFT / imitation learning from solver traces.** The per-version reference solvers
  already emit optimal trajectories; behavior cloning / DAgger baselines would quantify how
  much of the demo-seeded performance is imitation vs RL.

## Frontier-flavored extensions
- **LLM-agent wrapper.** Expose the symbolic observation/action interface to an LLM agent
  (the two-layer clue→protocol→submit structure is a natural language-conditioned task),
  comparing LLM planning against the trained value agents.
- **MARL extension.** Redesign so coordination is necessary (e.g., one agent reads clues
  while another operates the forge), studying emergent communication and credit assignment.

## Methodology hardening
- Longer deep horizons + hyperparameter sweeps to confirm the V1.1 generalization ranking
  and the V1.2 probe-transfer dip.
- A **noisy-only (no-SPIE) E2 arm** to fully disentangle NoisyNet exploration from the SPIE
  intrinsic term in the alignment-mitigation result.
- Per-fault generalization deep-dive on V1.3; per-protocol deep-dive on V1.2/V1.4.
- Full V1.0 coverage for the deep SPIE/noisy variants (wire V1.0 into the multi-config
  driver or extend the single-config script).
