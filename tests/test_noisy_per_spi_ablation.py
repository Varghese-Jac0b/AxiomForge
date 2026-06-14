"""
Tests for the Noisy x PER x SPIE ablation
(scripts/run_noisy_per_spi_ablation.py + the train_seed_deep metrics_sink
and DQNAgent.last_td_abs_mean instrumentation that back it).

Covers: the 2x2 variant switch table, PER on/off buffer behaviour, SPIE on/off
intrinsic logging, result logging via metrics_sink, the agent TD-error probe,
that instrumentation is non-invasive, and seed reproducibility.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.dqn_axiom_forge import (
    DQNAgent,
    PrioritizedReplayBuffer,
    make_obs_encoder,
)
from environments.axiom_forge_configs import make_v1_4_config
from environments.axiom_forge_env import AxiomForgeEnv
from scripts.train_axiom_forge import ALL_ALGOS, DEEP_ALGOS, train_seed_deep
from scripts.run_noisy_per_spi_ablation import (
    VARIANTS,
    assert_backbone_is_fixed,
    summarize,
)


def _deep_args(**ov):
    """Tiny deep-training Namespace (mirrors the runner's build_args fields)."""
    base = dict(
        episodes=3, gamma=0.99, lr=5e-4,
        batch_size=16, buffer_size=2000, target_update=50, train_freq=1,
        pretrain_steps=0, demo_configs=0, demo_sweeps=3,
        demo_fraction=0.25, margin=0.8, margin_weight=1.0,
        train_configs=2, heldout_configs=2,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.999,
        mode="full", abstraction="milestone",
        beta=0.5, beta_decay=0.999, beta_end=0.0,
        sr_alpha=0.1, sr_gamma=0.95,
        device="cpu", shaping=False, e2=True,
    )
    base.update(ov)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------
# 1. Variant switch table (config switches)
# ---------------------------------------------------------------------


def test_new_variants_registered():
    # the two no-PER noisy siblings added for this ablation must exist
    for key in ("noisy_dueling_ddqn", "spie_noisy_dueling_ddqn"):
        assert key in DEEP_ALGOS
        assert key in ALL_ALGOS


def test_variants_are_fixed_backbone_2x2():
    # backbone fixed (double+dueling+noisy), and exactly the 2x2 of (per, spie)
    assert_backbone_is_fixed()
    combos = {(DEEP_ALGOS[k]["per"], DEEP_ALGOS[k]["spie"]) for _, k in VARIANTS}
    assert combos == {(False, False), (True, False), (False, True), (True, True)}
    for _, key in VARIANTS:
        sw = DEEP_ALGOS[key]
        assert sw["double"] and sw["dueling"] and sw["noisy"]


def test_only_per_and_spie_differ_between_paired_variants():
    # v1 vs v2 differ ONLY in per; v1 vs v3 differ ONLY in spie
    base = DEEP_ALGOS["noisy_dueling_ddqn"]
    per_on = DEEP_ALGOS["noisy_dueling_ddqn_per"]
    spie_on = DEEP_ALGOS["spie_noisy_dueling_ddqn"]
    assert {k: v for k, v in per_on.items() if k != "per"} == \
        {k: v for k, v in base.items() if k != "per"}
    assert per_on["per"] != base["per"]
    assert {k: v for k, v in spie_on.items() if k != "spie"} == \
        {k: v for k, v in base.items() if k != "spie"}
    assert spie_on["spie"] != base["spie"]


# ---------------------------------------------------------------------
# 2. PER on/off behaviour (buffer type)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("use_per", [True, False])
def test_per_switch_selects_buffer(use_per):
    env = AxiomForgeEnv(make_v1_4_config(seed=0, config_id=0))
    _, obs_dim = make_obs_encoder(env.observation_space)
    agent = DQNAgent(obs_dim, env.action_space.n, double=True, dueling=True,
                     noisy=True, per=use_per, device="cpu", seed=0)
    assert isinstance(agent.buffer, PrioritizedReplayBuffer) is use_per


# ---------------------------------------------------------------------
# 3. Agent TD-error probe (last_td_abs_mean)
# ---------------------------------------------------------------------


def test_agent_records_td_abs_mean_after_learn():
    env = AxiomForgeEnv(make_v1_4_config(seed=0, config_id=0))
    encode, obs_dim = make_obs_encoder(env.observation_space)
    agent = DQNAgent(obs_dim, env.action_space.n, double=True, dueling=True,
                     noisy=True, per=True, batch_size=8, device="cpu", seed=0)
    assert agent.last_td_abs_mean is None          # nothing learned yet
    obs, _ = env.reset(options={"config_id": 0})
    state = encode(obs)
    for _ in range(20):                            # fill the buffer a bit
        a = agent.act(state, epsilon=1.0)
        nobs, r, term, trunc, _ = env.step(a)
        agent.store(state, a, r, encode(nobs), term)
        state = encode(nobs)
        if term or trunc:
            obs, _ = env.reset(options={"config_id": 0})
            state = encode(obs)
    loss = agent.learn()
    assert loss is not None
    assert isinstance(agent.last_td_abs_mean, float)
    assert agent.last_td_abs_mean >= 0.0


# ---------------------------------------------------------------------
# 4. SPIE on/off intrinsic logging + result logging via metrics_sink
# ---------------------------------------------------------------------


def test_metrics_sink_logs_one_row_per_episode_no_spie():
    sink: list = []
    args = _deep_args(episodes=3)
    df, *_ = train_seed_deep("v1_4", "noisy_dueling_ddqn", 0, args,
                             metrics_sink=sink)
    assert len(sink) == args.episodes == len(df)
    for row in sink:
        assert set(row) == {"episode", "intrinsic_contribution",
                            "td_abs_mean", "learn_calls",
                            "gate_advance", "gate_neutral", "gate_proxy",
                            "allowed_intrinsic", "suppressed_intrinsic",
                            "proxy_penalty_total", "proxy_penalty_count"}
        # non-gated variant: gate tallies are all zero
        assert (row["gate_advance"], row["gate_neutral"],
                row["gate_proxy"]) == (0, 0, 0)
    # SPIE OFF -> intrinsic is exactly zero every episode
    assert all(r["intrinsic_contribution"] == 0.0 for r in sink)


def test_spie_variant_logs_positive_intrinsic():
    sink: list = []
    args = _deep_args(episodes=3)
    train_seed_deep("v1_4", "spie_noisy_dueling_ddqn", 0, args,
                    metrics_sink=sink)
    total_intrinsic = sum(r["intrinsic_contribution"] for r in sink)
    assert total_intrinsic > 0.0                   # SPIE ON contributes bonus


def test_metrics_sink_records_td_when_learning():
    sink: list = []
    # train_freq=1 + a buffer that fills quickly -> learning happens
    args = _deep_args(episodes=3, train_freq=1)
    train_seed_deep("v1_4", "noisy_dueling_ddqn_per", 0, args,
                    metrics_sink=sink)
    assert sum(r["learn_calls"] for r in sink) > 0
    recorded = [r["td_abs_mean"] for r in sink if r["td_abs_mean"] is not None]
    assert recorded and all(v >= 0.0 for v in recorded)


# ---------------------------------------------------------------------
# 5. Instrumentation is non-invasive + seed reproducibility
# ---------------------------------------------------------------------


def test_metrics_sink_is_non_invasive():
    # passing a sink must NOT change the training trajectory
    a = _deep_args(episodes=3)
    df_no_sink, *_ = train_seed_deep("v1_4", "noisy_dueling_ddqn", 0, a)
    df_sink, *_ = train_seed_deep("v1_4", "noisy_dueling_ddqn", 0, a,
                                  metrics_sink=[])
    assert df_no_sink["episode_return"].tolist() == \
        df_sink["episode_return"].tolist()
    assert df_no_sink["success"].tolist() == df_sink["success"].tolist()


def test_same_seed_is_reproducible():
    a = _deep_args(episodes=3)
    df1, *_ = train_seed_deep("v1_4", "noisy_dueling_ddqn", 0, a)
    df2, *_ = train_seed_deep("v1_4", "noisy_dueling_ddqn", 0, a)
    assert df1["episode_return"].tolist() == df2["episode_return"].tolist()
    assert df1["steps"].tolist() == df2["steps"].tolist()


# ---------------------------------------------------------------------
# 6. Summary aggregation shape
# ---------------------------------------------------------------------


def test_summarize_one_row_per_variant_in_order():
    # two seeds x four variants -> 4 summary rows in canonical order
    rows = []
    for display, key in VARIANTS:
        sw = DEEP_ALGOS[key]
        for seed in (0, 1):
            rows.append(dict(
                variant=display, algo_key=key, seed=seed,
                use_per=sw["per"], use_spie=sw["spie"], episodes=3,
                success_rate=0.1, train_greedy_success=0.0,
                heldout_greedy_success=0.0, final_correct_submissions=0,
                mean_final_reward=-1.0, mean_episode_return=-1.0,
                mean_episode_length=50.0, mean_trap_hits=1.0,
                total_trap_hits=3, reward_hack_rate=0.2,
                safety_violation_rate=0.0, mean_true_score=-2.0,
                mean_alignment_gap=1.5, mean_intrinsic_contribution=0.0,
                mean_td_abs=0.3, learn_steps=10))
    summary = summarize(pd.DataFrame(rows))
    assert list(summary["variant"]) == [d for d, _ in VARIANTS]
    assert (summary["n_seeds"] == 2).all()
    assert "success_rate_mean" in summary.columns
    assert "mean_alignment_gap_mean" in summary.columns
