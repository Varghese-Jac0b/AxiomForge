# AxiomForge Anti-Reward-Hacking — Codex Verification Packet

> **Public-release note.** This packet originated in the source experiment
> worktree and is shipped here in the AxiomForge public repository, where the
> anti-hacking code, reports, and result CSVs/plots are committed to git. Two
> items below are source-side only and are **not** shipped publicly: the per-run
> `*.log` files (absolute paths) and the `archived_ablation_noisy_per_spi_*`
> immutability-backup folder. The CSV immutability check in §8 therefore applies
> to the source worktree; in this repo, integrity is checked directly against the
> committed CSVs (row counts, dedup, NaN, 400-row seeds — §8).

A self-contained audit map for an external verifier (Codex). It documents what
was implemented, what was run, where every artifact lives, and how to check the
claims. **No new experiments are needed to verify** — all results ship as
CSVs/plots/reports in this repository.

> **Provenance note.** In the source worktree these artifacts were present but
> mostly untracked (HEAD `ee1e387`); in this **public release** they are committed.
> Verify against the repository files.

> **Learner clarification (important).** R0–R3 are all **DQN-family, value-based**
> experiments. We did **not** switch to tabular Q-learning. The base learner is a
> **Noisy Dueling Double-DQN** (one configurable `DQNAgent`). The experiments
> change only the *exploration / replay / control* machinery around that fixed
> backbone: plain noisy → PER → SPIE → passive Trap-Sentinel → R1 PG-SPIE →
> R2 proxy-coupled penalty → R3 trap-aware PER.

---

## 1. Executive summary

- **What AxiomForge tests:** a symbolic two-layer "research lab" (V1.4) where an
  agent must infer a hidden work order, follow a protocol, and submit at the true
  goal `G`, while a **deliberate proxy-reward trap** tempts it.
- **Main reward-hacking location:** tile **`Y`** (`_handle_proxy_terminal`), a
  **farmable, non-terminal** trap paying `3.0 / 1.5 / 0.75` then 0. The true goal
  `G` (`_handle_submission_interact`) is the absorbing terminal. Observable proxy
  counters `proxy_attempt_count` / `proxy_claim_count` rise **only** at `Y`.
- **Main result:** **R2 (proxy-coupled penalty) at κ=4/8 nearly eliminates trap
  behavior with no success loss.** Demo-seeded trap_hits fall from 0.374 (R1) to
  **0.039 (κ=4) / 0.022 (κ=8)**; alignment_gap crosses negative (−0.05 / −0.13);
  success unchanged (~0.545); variance collapses; the pathological seed is tamed.
- **R3 (trap-aware PER):** **naive PER re-ignites** hacking on top of R2
  (trap_hits 0.039 → 0.165, gap −0.05 → +0.51), while **trap-aware PER prevents
  that** (0.052, gap −0.135). But R3 adds no benefit (success slightly lower), so
  **R2 without PER remains the recommended setting.**

---

## 2. Experiment ladder

| step | name | what it adds (knob[s] over predecessor) | regime(s) | seeds |
|---|---|---|---|---|
| R0 | Noisy × PER × SPIE ablation | sweeps PER & SPIE on/off | V1.4 no-demo (E2) | 3 |
| — | Passive Trap-Sentinel | read-only detector (no training change) | scripted V1.4 | n/a |
| R1 | Protocol-Gated SPIE | +`pg` (observable milestone gate on the SPIE bonus) | no-demo + demo-seeded | 3 |
| R2 | Proxy-coupled penalty | +`pp` (proxy step → `−κ·bonus`, `--kappa`) | no-demo + demo-seeded | 3 |
| R2 κ-sweep | consolidation | κ ∈ {1,4,8}, 5 seeds (no knob change) | demo-seeded | 5 |
| R3 | Trap-aware PER | **two one-knob sub-steps:** +`per` (naive-PER arm) → +`taper` (trap-aware cap) | demo-seeded | 5 |

R0–R2 are strictly one knob each. R3 is **two** one-knob sub-steps (naive PER,
then the `taper` cap), so the ladder stays single-variable *between adjacent
arms*; it is **not** a single knob jumping from R2-no-PER straight to trap-aware
PER (that pair differs by two keys, `per` + `taper`).

---

## 3. Exact file map

### Source files modified (existed at commit `ee1e387`)
- `agents/dqn_axiom_forge.py` — added: `last_td_abs_mean` (R0 probe); `is_trap`
  array in `ReplayBuffer`; `trap_aware` median-cap + priority diagnostics +
  `per_stats()` in `PrioritizedReplayBuffer`; `trap_aware` arg and `store(trap=)`
  in `DQNAgent`.
- `scripts/train_axiom_forge.py` — added: `protocol_gate()` (R1), `gated_intrinsic()`
  (R1/R2), the gate + proxy-penalty block in `train_seed_deep`, observable
  `is_trap` at `store`, `metrics_sink` gate/penalty fields, and the deep variants
  `noisy_dueling_ddqn`, `spie_noisy_dueling_ddqn`, `pg_spie_noisy_dueling_ddqn`,
  `pg_spie_r2_noisy_dueling_ddqn`, `pg_spie_r2_noisy_dueling_ddqn_per`,
  `pg_spie_r2_taper_noisy_dueling_ddqn_per`.
- `tests/test_axiom_forge_deep.py` — extended the naming-convention test for the
  `pg`/`per` dimensions.

### New agent / helper files
- `agents/trap_sentinel.py` — passive `TrapSentinel` detector (observable-only).

### Runner scripts (new)
- `scripts/run_noisy_per_spi_ablation.py` — R0 ablation.
- `scripts/run_trap_sentinel_probe.py` — passive sentinel probe + validation.
- `scripts/run_pg_spie_r1.py` — R1.
- `scripts/run_pg_spie_r2.py` — R2.
- `scripts/run_pg_spie_r2_ksweep.py` — R2 κ-sweep consolidation.
- `scripts/run_pg_spie_r3_trap_aware_per.py` — R3.
- `scripts/build_r0_r2_summary.py` — consolidated plots/tables generator.

### Tests (new; 47 anti-hacking tests)
- `tests/test_noisy_per_spi_ablation.py` (12) · `tests/test_trap_sentinel.py` (8)
  · `tests/test_pg_spie_r1.py` (10) · `tests/test_pg_spie_r2.py` (9)
  · `tests/test_pg_spie_r3.py` (8). Full suite = **330 passed**.

### Reports (new)
- `reports/noisy_per_spi_ablation.md` · `reports/trap_sentinel_probe.md` ·
  `reports/pg_spie_r1.md` · `reports/pg_spie_r2.md` · `reports/pg_spie_r2_ksweep.md`
  · `reports/axiomforge_r0_r2_antihacking_summary.md` ·
  `reports/pg_spie_r3_trap_aware_per.md` · `reports/codex_verification_packet.md`
  (this file).

### Results folders (per-experiment; each `<root>/<...>/seed_*.csv` are per-seed
episode logs of 400 rows, `signals_*.csv` per-episode gate/penalty/intrinsic
diagnostics, `*_runs.csv` one row per run, `*_summary.csv` one row per variant)
- `results/ablation_noisy_per_spi/` — 38 csv, 1 log
- `results/trap_sentinel_probe/` — 6 csv (validation + traces)
- `results/pg_spie_r1/` — 38 csv, 1 log
- `results/pg_spie_r2/` — 50 csv, 1 log
- `results/pg_spie_r2_ksweep/` — 62 csv, 1 log
- `results/r0_r2_summary/` — 3 csv tables + 3 png plots
- `results/pg_spie_r3_trap_aware_per/` — 52 csv, 1 log
- `results/archived_ablation_noisy_per_spi_20260614_062807/` — backup of the R0
  ablation (38 csv, 1 log) used as the immutability reference.

### Plots
- `results/r0_r2_summary/plots/{r0_ablation,demo_ladder,ksweep_trend}.png`.

---

## 4. Key reports and where they are

| report | covers |
|---|---|
| `reports/noisy_per_spi_ablation.md` | R0 Noisy × PER × SPIE ablation |
| `reports/trap_sentinel_probe.md` | passive sentinel + env verification (6 Qs) |
| `reports/pg_spie_r1.md` | R1 Protocol-Gated SPIE |
| `reports/pg_spie_r2.md` | R2 proxy-coupled penalty (κ=1) |
| `reports/pg_spie_r2_ksweep.md` | R2 κ-sweep (1/4/8, 5 seeds) |
| `reports/axiomforge_r0_r2_antihacking_summary.md` | consolidated R0→R2 freeze |
| `reports/pg_spie_r3_trap_aware_per.md` | R3 trap-aware PER |
| `reports/codex_verification_packet.md` | this audit map |

## 5. Key result folders and contents

| folder | runs.csv rows | summary | notes |
|---|---|---|---|
| `results/ablation_noisy_per_spi/` | 12 (4 var × 3) | `ablation_summary.csv` | V1.4 E2 no-demo |
| `results/trap_sentinel_probe/` | n/a | `validation_summary.csv` | proxy-farm & legal-goal traces |
| `results/pg_spie_r1/` | 18 (3 × 3 × 2 regimes) | `pg_spie_r1_summary.csv` | gate metrics |
| `results/pg_spie_r2/` | 24 (4 × 3 × 2) | `pg_spie_r2_summary.csv` | penalty metrics |
| `results/pg_spie_r2_ksweep/` | 30 (6 × 5, demo) | `ksweep_summary.csv` | κ∈{1,4,8} |
| `results/r0_r2_summary/` | n/a | `table_*.csv` + plots | derived figures/tables |
| `results/pg_spie_r3_trap_aware_per/` | 25 (5 × 5, demo) | `pg_spie_r3_summary.csv` | PER priority diagnostics |

---

## 6. Main result tables

### 6.1 R0 ablation — V1.4 no-demo (3 seeds) [`ablation_summary.csv`]
| variant | success | trap_hits | hack_rate | align_gap |
|---|---:|---:|---:|---:|
| noisy | 0.000 | 0.000 | 0.000 | −1.692 |
| +PER | 0.000 | 0.200 | 0.070 | −0.886 |
| +SPIE | 0.000 | 0.308 | 0.124 | −0.185 |
| +PER+SPIE | 0.000 | 0.470 | 0.208 | +0.435 |

### 6.2 R1 / R2 — no-demo (3 seeds) [`pg_spie_r1/2_summary.csv`]
| variant | success | trap_hits | hack_rate | align_gap |
|---|---:|---:|---:|---:|
| naive_spie | 0.000 | 0.308 | 0.124 | −0.185 |
| pg_spie (R1) | 0.000 | 0.000 | 0.000 | −1.911 |
| pg_spie_r2 (κ=1) | 0.000 | 0.000 | 0.000 | −1.911 |

### 6.3 R2 κ-sweep — demo-seeded (5 seeds) [`ksweep_summary.csv`]
| variant | success | trap_hits ± sd | hack | align_gap ± sd | true_score |
|---|---:|---:|---:|---:|---:|
| plain_noisy | 0.546 | 0.247 ± 0.39 | 0.083 | 0.932 ± 1.97 | 4.196 |
| naive_spie | 0.544 | 0.378 ± 0.33 | 0.129 | 1.668 ± 1.38 | 3.881 |
| pg_spie (R1) | 0.544 | 0.374 ± 0.44 | 0.128 | 1.489 ± 2.12 | 3.882 |
| R2 κ=1 | 0.546 | 0.308 ± 0.39 | 0.104 | 1.188 ± 1.90 | 4.090 |
| **R2 κ=4** | 0.546 | **0.039 ± 0.05** | 0.013 | **−0.051 ± 0.29** | 4.859 |
| **R2 κ=8** | 0.544 | **0.022 ± 0.01** | 0.007 | **−0.132 ± 0.08** | 4.817 |

### 6.4 R3 trap-aware PER — demo-seeded, κ=4 (5 seeds) [`pg_spie_r3_summary.csv`]
| variant | success | trap_hits | hack | align_gap | trap_prio / nontrap | trap_replay | n_capped |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain_noisy | 0.546 | 0.247 | 0.083 | 0.932 | – | – | 0 |
| per_spie (old hacker) | 0.518 | 0.530 | 0.182 | 2.233 | 0.475/0.438 | 0.032 | 0 |
| R2 κ=4 (no PER) | 0.546 | 0.039 | 0.013 | −0.051 | – | – | 0 |
| R2 κ=4 + naive PER | 0.524 | 0.165 | 0.056 | +0.512 | 0.413/0.395 | 0.012 | 0 |
| R2 κ=4 + trap-aware PER | 0.518 | 0.052 | 0.018 | −0.135 | **0.203**/0.390 | 0.005 | 11148 |

---

## 7. Hidden-leakage audit

The grader's answer key is never read in training. Verifiable facts in code:

| item | location | usage |
|---|---|---|
| `true_score` | `axiom_forge_env.py::_compute_true_score`, returned only in `_get_info` under `enable_true_score` | reporting/eval only; **not** in `obs`; env comment (line ~498): "must NEVER touch this reward" |
| `HiddenContext` | `_get_info` → `info["debug"]["hidden_context"]` only under `debug_mode` | eval/debug only; **not** in `obs` |
| eval read of hidden | `train_axiom_forge.py::evaluate_split` reads `hidden.protocol/fault` | **labels eval rows only** (per-protocol/fault breakdown); greedy eval, no reward/policy effect |
| `true_score` in plots | `plot_e2`, `--e2` flag | figure generation only |

Training-side signals are **all observable**:
- R1 gate `protocol_gate(prev_obs, obs)` uses `obs["proxy_attempt_count"]`,
  `obs["proxy_claim_count"]`, and `milestone_abstraction(obs)[3]`
  (manifest/archive/diagnostics read + sample present/refined/analyzed) — all
  fields of the observation dict.
- R2 penalty triggers on the same observable proxy increment.
- R3 `is_trap` (in `train_seed_deep`, the `store` call) is computed **only** from
  `proxy_attempt_count` / `proxy_claim_count` increments; the trap-aware PER cap
  uses that flag. `TrapSentinel.observe` reads only obs fields.
- Greedy evaluation never applies any intrinsic/penalty/priority (training-only).

**Conclusion:** no `true_score` / `HiddenContext` / grader-only label enters any
training reward, gate, penalty, replay priority, or observation.

---

## 8. Integrity checks already performed

| check | result |
|---|---|
| Test suite (`.venv/bin/python -m pytest tests/ -q`) | **330 passed** |
| Row counts (`*_runs.csv`) | ablation **12**, R1 **18**, R2 **24**, κ-sweep **30**, R3 **25** — all as expected |
| Duplicate `(regime,variant,seed)` rows | **0** in every experiment |
| NaN in critical metrics (success/trap_hits/hack/gap/true_score) | **0** in every experiment |
| Per-seed `seed_*.csv` row counts | **all 400** (84 files across R1/R2/κ-sweep checked; 25 in R3) |
| Artifact folders present | all 8 (incl. backup) present |
| Prior artifacts unchanged | R0 ablation CSVs **byte-identical to backup** (see exact command below) |
| No overwrite across rungs | each rung wrote to its own `results/<name>/` dir |

**Exact immutability-check command** (the earlier packet quoted a hash without
its method, which is non-reproducible — use this instead). Path-independent
content hash; live and backup must print the **same** value:

```bash
hash() { find "$1" -name '*.csv' -exec shasum -a 256 {} \; \
         | awk '{print $1}' | LC_ALL=C sort | shasum -a 256 | awk '{print $1}'; }
hash results/ablation_noisy_per_spi
hash results/archived_ablation_noisy_per_spi_20260614_062807
# both => 273cd6e27419210a1bf1c36bd1cf576bea2cb1233c3a842e1843bb946528f492
```

Equivalent CSV/data-tree check (the backup folder additionally holds a
`noisy_per_spi_ablation.md` report snapshot, so exclude non-data files):

```bash
diff -rq results/ablation_noisy_per_spi \
         results/archived_ablation_noisy_per_spi_20260614_062807 \
         -x '*.md' -x '*.log'        # => no differences (CSV/data trees identical)
```

A *plain* `diff -rq` (without the excludes) lists exactly one extra file in the
backup — `noisy_per_spi_ablation.md` — and **no CSV differences**. The
previously-quoted `607e5447…` was a SHA-1-based variant; the *claim* — live CSVs
== backup CSVs — holds under any consistent method.

---

## 9. Exact commands (used / reproducible)

```bash
PY=.venv/bin/python    # Noisy Dueling DDQN backbone; CPU

# R0 ablation (V1.4 no-demo / E2)
$PY scripts/run_noisy_per_spi_ablation.py --episodes 400 --seeds 0 1 2

# Passive Trap-Sentinel probe (read-only)
$PY scripts/run_trap_sentinel_probe.py

# R1 Protocol-Gated SPIE
$PY scripts/run_pg_spie_r1.py --episodes 400 --seeds 0 1 2 --regimes no_demo demo_seeded

# R2 proxy-coupled penalty (kappa=1)
$PY scripts/run_pg_spie_r2.py --episodes 400 --seeds 0 1 2 --regimes no_demo demo_seeded --kappa 1.0

# R2 kappa-sweep (consolidation)
$PY scripts/run_pg_spie_r2_ksweep.py --kappas 1 4 8 --seeds 0 1 2 3 4 --regimes demo_seeded

# R3 trap-aware PER
$PY scripts/run_pg_spie_r3_trap_aware_per.py --episodes 400 --seeds 0 1 2 3 4 --kappa 4 --regimes demo_seeded

# Consolidated plots/tables (read-only over the CSVs)
$PY scripts/build_r0_r2_summary.py

# Test suite
$PY -m pytest tests/ -q
```

---

## 10. Instructions for Codex

Codex should verify the packet **from the working-tree files** (mostly untracked;
re-running is optional; the claims must be checkable from code + CSVs):

1. **Inspect the code changes.** Diff `agents/dqn_axiom_forge.py`,
   `scripts/train_axiom_forge.py`, `agents/trap_sentinel.py`, and the runner
   scripts. Confirm the mechanisms match the descriptions (gate, penalty,
   trap-aware median cap).
2. **Verify no hidden leakage.** Grep the training path for `true_score`,
   `HiddenContext`, `hidden_context`, `_hidden`; confirm every hit is in
   eval/reporting/plotting/docstrings, **not** in `train_seed_deep`'s
   reward/store/gate/penalty/PER path, and not in `TrapSentinel.observe`. Confirm
   the gate/penalty/`is_trap` read only `proxy_attempt_count`/`proxy_claim_count`
   and `milestone_abstraction`.
3. **Verify one-knob changes.** R1 adds only the `pg` gate; R2 adds only `pp`+κ
   (with `pp=False`, R2 ≡ R1 — see the identical no-demo rows). **R3 is split into
   two one-knob sub-steps** (it is *not* one knob directly from R2-no-PER): first
   `+per` (the `r2_k4_naive_per` arm = R2 + PER) tests re-ignition, then `+taper`
   (the `r2_k4_trap_aware_per` arm = naive-PER + the cap) tests the fix. So
   `trap_aware` differs from `naive_per` by exactly one key (`taper`), and
   `naive_per` differs from `r2_k4_no_per` by exactly one key (`per`). Confirm
   each adjacent pair in that chain differs by a single `DEEP_ALGOS` key.
4. **Verify row counts / integrity.** For each `*_runs.csv`: row count
   (12/18/24/30/25), 0 duplicate `(regime,variant,seed)`, 0 NaN in critical
   columns, and every `seed_*.csv` has 400 rows.
5. **Verify report ↔ CSV agreement.** Recompute the headline means from the
   per-run CSVs and confirm they match the numbers in §6 and in each report's
   filled Results block (e.g., R2 κ=4 trap_hits 0.039, gap −0.051; R3 naive_per
   trap_hits 0.165, trap_aware 0.052; cap `n_capped` ≈ 11148).
6. **Verify prior artifacts not overwritten.** Recompute the R0 ablation CSV
   content hash and confirm it equals the backup folder's hash.
7. **Verify claims are supported.** Confirm: SPIE/PER amplify hacking (R0
   monotone increase); sentinel fires on `Y` not `G` (probe validation);
   R1 zeroes hacking in no-demo; R2 κ≥4 zeroes it demo-seeded at flat success;
   naive PER re-ignites and trap-aware PER prevents it (R3).
8. **Report any** metric mismatch, hidden-field leakage, multi-knob change, row/
   integrity discrepancy, or unsupported claim.

---

## 11. Final recommended claim

> In AxiomForge, naive structural novelty (SPIE) and PER can **amplify** proxy
> reward hacking. A failed novelty signal can be **repurposed as an observable
> trap sentinel**. **Protocol gating** and **proxy-coupled reversal** reduce
> proxy hacking sharply, with **R2 κ=4/8 nearly eliminating trap behavior without
> reducing success**. **Trap-aware PER prevents replay-based re-amplification**,
> but **R2 without PER remains the recommended setting.** This is a controlled,
> single-environment, value-based (Noisy Dueling DDQN) demonstration — not a
> general alignment solution.

---

## 12. External verification log (Codex)

An independent read-only Codex audit returned **PARTIALLY VERIFIED**: PASS on
deep-DQN-family, ladder description, hidden-leakage, tests (**330 passed**),
CSV/data integrity (row counts 12/18/24/30/25, 0 dups, 0 NaN, 400-row seed
files), report↔CSV agreement, and prior-artifact immutability (live == backup).
It raised three documentation issues, all addressed in this revision:

1. **"committed" was inaccurate** → corrected to "present in the working tree"
   with an explicit provenance note (artifacts are mostly untracked; HEAD
   `ee1e387`). §1, §10.
2. **Backup hash quoted without a method** (`607e5447…`, a SHA-1 variant; Codex's
   SHA-256 differed) → replaced with an **exact, path-independent SHA-256
   command** and value (`273cd6e2…`), plus the `diff -rq` equivalent. §8.
3. **R3 "one-knob" wording** → clarified that R3 is **two** one-knob sub-steps
   (`+per`, then `+taper`); the single-variable property holds between adjacent
   arms, not from R2-no-PER directly to trap-aware PER. §2, §10.

A **second** Codex revalidation re-confirmed all of the above (PASS on
provenance wording, R3 switch diffs `+pg`/`+pp`/`+per`/`+taper`, the pinned hash
`273cd6e2…`, and 330 passing tests) and flagged one more doc nit:

4. **`diff -rq` claim was imprecise** — the backup folder also contains a
   `noisy_per_spi_ablation.md` report snapshot, so a *plain* `diff -rq` lists that
   one extra file (no CSV differences). Fixed in §8 to use a CSV/data-only diff
   (`-x '*.md' -x '*.log'` ⇒ no differences). §8.

No result, metric, or code claim was changed — these are documentation
corrections only. The substantive findings (R0 amplification; sentinel
detection; R1 no-demo elimination; R2 κ=4/8 headline; R3 re-ignition vs
trap-aware suppression) were all independently confirmed across both audits.
