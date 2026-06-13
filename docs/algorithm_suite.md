# AxiomForge Algorithm Suite (14)

All 14 algorithms run through one multi-config driver,
`scripts/train_axiom_forge.py --algo <name>`, on the shared train/held-out protocol. No
algorithm has its own forked training loop.

## Tabular (4)
A single dict Q-table over the encoded observation key (`encode_state`); ε-greedy with
per-episode decay; demo seeding via reverse value sweeps over a reference-solver
trajectory; termination-only bootstrap.

| Algorithm | Update target |
|---|---|
| **Q-learning** | `r + γ·max_a Q(s',a)` (off-policy) |
| **SARSA** | `r + γ·Q(s',a')`, `a'` the action actually taken (on-policy) |
| **Expected SARSA** | `r + γ·Σ_a π(a\|s')·Q(s',a)` under the ε-greedy policy |
| **SPIE-Q** | Q-learning **+** a training-only intrinsic bonus `β·r_int` (see SPIE below) |

## Deep value-based (10)
One configurable `DQNAgent` (`agents/dqn_axiom_forge.py`) with **four boolean switches**;
the ten deep algorithms are the named switch combinations. SPIE is added at the **driver**
level (it modifies the stored reward, not the agent), so it is not a fifth class.

| Switch | Effect | Reference |
|---|---|---|
| `double` | Double-DQN target: `a* = argmax Q_online(s')`, bootstrap `Q_target(s',a*)` | van Hasselt 2016 |
| `dueling` | `Q = V(s) + A(s,a) − mean_a A` (separate value/advantage heads) | Wang 2016 |
| `per` | Proportional prioritized replay with IS-weight correction | Schaul 2016 |
| `noisy` | `NoisyLinear` factorized Gaussian noise → self-annealing exploration (ε=0; deterministic greedy eval via noise-zeroing) | Fortunato 2017 |
| `spie` (driver) | Adds `β·r_int` to the **stored** reward only | Yu & Burgess 2023 |

| Algorithm | double | dueling | per | noisy | spie |
|---|:---:|:---:|:---:|:---:|:---:|
| dqn | | | | | |
| ddqn | ✓ | | | | |
| dueling_dqn | | ✓ | | | |
| dueling_ddqn_per | ✓ | ✓ | ✓ | | |
| noisy_dueling_ddqn_per | ✓ | ✓ | ✓ | ✓ | |
| spie_dqn | | | | | ✓ |
| spie_ddqn | ✓ | | | | ✓ |
| spie_dueling_dqn | | ✓ | | | ✓ |
| spie_dueling_ddqn_per | ✓ | ✓ | ✓ | | ✓ |
| spie_noisy_dueling_ddqn_per | ✓ | ✓ | ✓ | ✓ | ✓ |

All deep agents share: DQfD-lite demonstration seeding (a protected demo replay region, a
guaranteed demo fraction / PER priority bonus, a large-margin loss); a dynamic one-hot
observation encoder that auto-adapts to the (frozen-shape) observation; termination-only
bootstrap.

## SPIE (Successor–Predecessor Intrinsic Exploration)
`agents/spie_q_agent.py` — sparse **successor** (SR, forward occupancy) and **predecessor**
(PR, backward occupancy) tables over an abstracted state, with an intrinsic bonus
`r_int = w_sr/‖M[s']‖ + w_pr/‖N[s']‖` (prospective novelty + retrospective bottleneck
pressure). The reused module serves both tabular SPIE-Q and the deep SPIE-* variants.
`--mode none` gives exactly zero bonus (bit-identical to the non-SPIE base path), and `β`
anneals to 0 so it is training-only. Design note: `docs/spie_q_design_note.md`.

## Clean-evaluation guarantees (test-enforced)
- Intrinsic reward never enters greedy evaluation; `episode_return` logs the **extrinsic**
  return.
- `true_score` is log-only; `alignment_gap` is metric-only.
- Noisy greedy eval is deterministic (noise zeroed).
