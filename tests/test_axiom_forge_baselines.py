"""
Tests for the V1.0 baseline-ladder additions:
SARSA, Expected SARSA, and the DQN family (DQN / DDQN / Dueling / PER).

The verified Q-learning baseline and its 107-test contract suite are
untouched; these tests cover only the new shared helpers, the tabular
variants' update rules and demo seeding, and the deep-RL building blocks
(observation encoder, networks, replay buffers, agent learning step).
"""

import numpy as np
import pytest
import torch

from environments.axiom_forge_configs import make_v1_0_config
from environments.axiom_forge_env import AxiomForgeEnv
from scripts.axiom_forge_baselines_common import (
    encode_state,
    evaluate_greedy,
    generate_demo_transitions_encoded,
    get_q_values,
)
from scripts.train_expected_sarsa_axiom_forge import expected_q_value
from scripts.train_q_learning_axiom_forge import seed_q_table_from_demo
from scripts.train_sarsa_axiom_forge import seed_q_table_from_demo_sarsa
from agents.dqn_axiom_forge import (
    DQNAgent,
    DuelingQNetwork,
    PrioritizedReplayBuffer,
    QNetwork,
    ReplayBuffer,
    SumTree,
    make_obs_encoder,
)

NUM_ACTIONS = 6
GAMMA = 0.99


@pytest.fixture(scope="module")
def env():
    return AxiomForgeEnv(make_v1_0_config())


@pytest.fixture(scope="module")
def demo_tabular():
    return generate_demo_transitions_encoded(make_v1_0_config(), encode_state)


# ---------------------------------------------------------------------
# Shared demo rollout
# ---------------------------------------------------------------------


class TestDemoTransitionsEncoded:
    def test_demo_succeeds_in_61_steps(self, demo_tabular):
        assert len(demo_tabular) == 61
        *_, terminated, truncated = demo_tabular[-1]
        assert terminated is True and truncated is False

    def test_only_final_transition_terminates(self, demo_tabular):
        assert not any(t[4] or t[5] for t in demo_tabular[:-1])

    def test_demo_return_matches_manual_solver(self, demo_tabular):
        total = sum(t[2] for t in demo_tabular)
        assert total == pytest.approx(9.69)

    def test_encoder_parameterization(self):
        # With a vector encoder the endpoint states are float vectors.
        cfg = make_v1_0_config()
        env = AxiomForgeEnv(cfg)
        encode, dim = make_obs_encoder(env.observation_space)
        transitions = generate_demo_transitions_encoded(cfg, encode)
        assert transitions[0][0].shape == (dim,)
        assert transitions[0][0].dtype == np.float32


# ---------------------------------------------------------------------
# Tabular variants: demo seeding alone must make greedy play succeed
# ---------------------------------------------------------------------


class TestTabularDemoSeeding:
    def test_sarsa_seeding_reaches_greedy_success(self, demo_tabular):
        q_table = {}
        seed_q_table_from_demo_sarsa(q_table, demo_tabular, GAMMA,
                                     NUM_ACTIONS, sweeps=50)
        assert evaluate_greedy(q_table, episodes=5) == 1.0

    def test_expected_sarsa_uses_q_learning_seeding(self, demo_tabular):
        # Greedy-limit seeding shared with the official baseline.
        q_table = {}
        # seed_q_table_from_demo expects (s, a, r, s', done) 5-tuples.
        five_tuples = [(s, a, r, s2, term or trunc)
                       for s, a, r, s2, term, trunc in demo_tabular]
        seed_q_table_from_demo(q_table, five_tuples, GAMMA, NUM_ACTIONS,
                               sweeps=50)
        assert evaluate_greedy(q_table, episodes=5) == 1.0

    def test_sarsa_seeding_values_positive_along_demo(self, demo_tabular):
        q_table = {}
        seed_q_table_from_demo_sarsa(q_table, demo_tabular, GAMMA,
                                     NUM_ACTIONS, sweeps=50)
        first_state, first_action = demo_tabular[0][0], demo_tabular[0][1]
        q_values = get_q_values(q_table, first_state, NUM_ACTIONS)
        assert q_values[first_action] > 0
        assert q_values.argmax() == first_action


class TestExpectedQValue:
    def test_uniform_at_epsilon_one(self):
        q_table = {"s": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])}
        expected = expected_q_value(q_table, "s", NUM_ACTIONS, epsilon=1.0)
        assert expected == pytest.approx(3.5)

    def test_greedy_at_epsilon_zero(self):
        q_table = {"s": np.array([1.0, 2.0, 9.0, 4.0, 5.0, 6.0])}
        expected = expected_q_value(q_table, "s", NUM_ACTIONS, epsilon=0.0)
        assert expected == pytest.approx(9.0)

    def test_tie_split_keeps_probabilities_normalized(self):
        q_table = {"s": np.array([7.0, 7.0, 0.0, 0.0, 0.0, 0.0])}
        expected = expected_q_value(q_table, "s", NUM_ACTIONS, epsilon=0.0)
        assert expected == pytest.approx(7.0)


# ---------------------------------------------------------------------
# Observation encoder
# ---------------------------------------------------------------------


class TestObsEncoder:
    def test_dim_counts_onehot_plus_scaled(self, env):
        _, dim = make_obs_encoder(env.observation_space)
        expected = 0
        for key, space in env.observation_space.spaces.items():
            if key in ("proxy_attempt_count", "proxy_claim_count"):
                expected += 1
            elif hasattr(space, "nvec"):
                expected += int(sum(space.nvec))
            else:
                expected += int(space.n)
        assert dim == expected

    def test_onehot_structure_at_reset(self, env):
        encode, dim = make_obs_encoder(env.observation_space)
        obs, _ = env.reset()
        vec = encode(obs)
        assert vec.shape == (dim,)
        # V1.0 reset: proxy counters are zero, so the vector is exactly the
        # one-hot blocks - one hot bit per discrete component (30 total).
        n_components = 0
        for key, space in env.observation_space.spaces.items():
            if key in ("proxy_attempt_count", "proxy_claim_count"):
                continue
            n_components += len(space.nvec) if hasattr(space, "nvec") else 1
        assert vec.sum() == pytest.approx(n_components)
        assert set(np.unique(vec)).issubset({0.0, 1.0})

    def test_encoding_changes_with_state(self, env):
        encode, _ = make_obs_encoder(env.observation_space)
        obs, _ = env.reset()
        before = encode(obs)
        obs, *_ = env.step(1)  # move down
        after = encode(obs)
        assert not np.array_equal(before, after)


# ---------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------


class TestNetworks:
    def test_qnetwork_output_shape(self):
        net = QNetwork(32, NUM_ACTIONS)
        out = net(torch.randn(7, 32))
        assert out.shape == (7, NUM_ACTIONS)
        assert torch.isfinite(out).all()

    def test_dueling_output_shape(self):
        net = DuelingQNetwork(32, NUM_ACTIONS)
        out = net(torch.randn(7, 32))
        assert out.shape == (7, NUM_ACTIONS)
        assert torch.isfinite(out).all()

    def test_dueling_advantage_mean_correction(self):
        """Q - V must have zero mean over actions by construction."""
        torch.manual_seed(0)
        net = DuelingQNetwork(32, NUM_ACTIONS)
        x = torch.randn(5, 32)
        with torch.no_grad():
            q = net(x)
            value = net.value_head(net.trunk(x))
        residual = (q - value).mean(dim=1)
        assert torch.allclose(residual, torch.zeros(5), atol=1e-5)


# ---------------------------------------------------------------------
# Replay buffers
# ---------------------------------------------------------------------


def _fill_demo(buffer, n, obs_dim, reward=99.0):
    for i in range(n):
        buffer.add_demo(np.full(obs_dim, i, dtype=np.float32), i % 6,
                        reward, np.zeros(obs_dim, dtype=np.float32), False)


class TestReplayBuffer:
    def test_demo_region_never_overwritten(self):
        obs_dim = 4
        buffer = ReplayBuffer(capacity=12, obs_dim=obs_dim, demo_capacity=4)
        _fill_demo(buffer, 4, obs_dim)
        for i in range(30):  # cycles the 8 agent slots several times
            buffer.add(np.zeros(obs_dim, dtype=np.float32), 0, -1.0,
                       np.zeros(obs_dim, dtype=np.float32), False)
        assert buffer.n_demo == 4
        assert (buffer.rewards[:4] == 99.0).all()
        assert buffer.is_demo[:4].all()
        assert not buffer.is_demo[4:].any()

    def test_demo_fraction_guaranteed(self):
        obs_dim = 4
        buffer = ReplayBuffer(capacity=100, obs_dim=obs_dim, demo_capacity=10)
        _fill_demo(buffer, 10, obs_dim)
        for _ in range(50):
            buffer.add(np.zeros(obs_dim, dtype=np.float32), 0, -1.0,
                       np.zeros(obs_dim, dtype=np.float32), False)
        rng = np.random.default_rng(0)
        batch, weights, indices = buffer.sample(32, rng, demo_fraction=0.25)
        assert batch["is_demo"].sum() == 8
        assert (weights == 1.0).all()

    def test_demo_only_sampling_before_agent_data(self):
        obs_dim = 4
        buffer = ReplayBuffer(capacity=100, obs_dim=obs_dim, demo_capacity=10)
        _fill_demo(buffer, 10, obs_dim)
        rng = np.random.default_rng(0)
        batch, _, _ = buffer.sample(32, rng, demo_fraction=0.25)
        assert batch["is_demo"].all()


class TestSumTreeAndPER:
    def test_sumtree_total_and_find(self):
        tree = SumTree(8)
        for i, priority in enumerate([1.0, 2.0, 3.0, 4.0]):
            tree.update(i, priority)
        assert tree.total() == pytest.approx(10.0)
        # mass in (3.0, 6.0] must land on leaf 2 (prefix 1+2 < mass <= 6)
        assert tree.find(4.5) == 2
        assert tree.find(0.5) == 0

    def test_per_priorities_bias_sampling(self):
        obs_dim = 4
        buffer = PrioritizedReplayBuffer(capacity=64, obs_dim=obs_dim)
        for _ in range(20):
            buffer.add(np.zeros(obs_dim, dtype=np.float32), 0, 0.0,
                       np.zeros(obs_dim, dtype=np.float32), False)
        # Give index 7 overwhelming priority.
        buffer.update_priorities(np.arange(20), np.full(20, 1e-6))
        buffer.update_priorities(np.array([7]), np.array([1000.0]))
        rng = np.random.default_rng(0)
        _, _, indices = buffer.sample(64, rng, beta=1.0)
        assert (indices == 7).mean() > 0.9

    def test_per_weights_bounded(self):
        obs_dim = 4
        buffer = PrioritizedReplayBuffer(capacity=64, obs_dim=obs_dim)
        for _ in range(20):
            buffer.add(np.zeros(obs_dim, dtype=np.float32), 0, 0.0,
                       np.zeros(obs_dim, dtype=np.float32), False)
        buffer.update_priorities(np.arange(20),
                                 np.linspace(0.1, 2.0, 20))
        rng = np.random.default_rng(1)
        _, weights, _ = buffer.sample(16, rng, beta=0.5)
        assert weights.max() == pytest.approx(1.0)
        assert (weights > 0).all()

    def test_per_demo_bonus_raises_priority(self):
        obs_dim = 4
        buffer = PrioritizedReplayBuffer(capacity=64, obs_dim=obs_dim,
                                         demo_capacity=1)
        _fill_demo(buffer, 1, obs_dim)
        buffer.add(np.zeros(obs_dim, dtype=np.float32), 0, 0.0,
                   np.zeros(obs_dim, dtype=np.float32), False)
        # Same TD error, but index 0 is a demo -> higher refreshed priority.
        buffer.update_priorities(np.array([0, 1]), np.array([0.5, 0.5]))
        assert buffer.tree.get(0) > buffer.tree.get(1)


# ---------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------


def _demo_filled_agent(demo_vec, obs_dim, **kwargs) -> DQNAgent:
    agent = DQNAgent(obs_dim, NUM_ACTIONS, demo_capacity=len(demo_vec),
                     batch_size=32, device="cpu", seed=0, **kwargs)
    for state, action, reward, next_state, terminated, _ in demo_vec:
        agent.store_demo(state, action, reward, next_state, terminated)
    return agent


@pytest.fixture(scope="module")
def demo_vec(env):
    encode, _ = make_obs_encoder(env.observation_space)
    return generate_demo_transitions_encoded(make_v1_0_config(), encode)


class TestDQNAgent:
    def test_act_returns_valid_action(self, env, demo_vec):
        _, obs_dim = make_obs_encoder(env.observation_space)
        agent = DQNAgent(obs_dim, NUM_ACTIONS, device="cpu", seed=0)
        action = agent.act(demo_vec[0][0], epsilon=0.0)
        assert 0 <= action < NUM_ACTIONS

    @pytest.mark.parametrize("double,dueling,per", [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, True),
    ])
    def test_learn_step_runs_for_every_variant(self, env, demo_vec,
                                               double, dueling, per):
        _, obs_dim = make_obs_encoder(env.observation_space)
        agent = _demo_filled_agent(demo_vec, obs_dim, double=double,
                                   dueling=dueling, per=per)
        loss = agent.learn()
        assert loss is not None and np.isfinite(loss)
        assert agent.learn_steps == 1

    def test_double_target_equals_vanilla_when_nets_identical(self, env,
                                                              demo_vec):
        _, obs_dim = make_obs_encoder(env.observation_space)
        vanilla = DQNAgent(obs_dim, NUM_ACTIONS, double=False, device="cpu",
                           seed=3)
        double = DQNAgent(obs_dim, NUM_ACTIONS, double=True, device="cpu",
                          seed=3)
        # Same seed -> identical init; target nets copy online at init, so
        # argmax_online == argmax_target and the two targets coincide.
        next_obs = torch.stack(
            [torch.as_tensor(t[3]) for t in demo_vec[:8]])
        rewards = torch.zeros(8)
        dones = torch.zeros(8)
        t_vanilla = vanilla.compute_targets(rewards, next_obs, dones)
        t_double = double.compute_targets(rewards, next_obs, dones)
        assert torch.allclose(t_vanilla, t_double)

    def test_double_target_diverges_after_online_update(self, env, demo_vec):
        _, obs_dim = make_obs_encoder(env.observation_space)
        agent = DQNAgent(obs_dim, NUM_ACTIONS, double=True, device="cpu",
                         seed=3)
        # Re-randomize the online net so it disagrees with the target net.
        torch.manual_seed(99)
        for parameter in agent.online_net.parameters():
            with torch.no_grad():
                parameter.copy_(torch.randn_like(parameter))
        next_obs = torch.stack(
            [torch.as_tensor(t[3]) for t in demo_vec[:32]])
        rewards = torch.zeros(32)
        dones = torch.zeros(32)
        double_targets = agent.compute_targets(rewards, next_obs, dones)
        agent.double = False
        vanilla_targets = agent.compute_targets(rewards, next_obs, dones)
        # Double bootstraps from the online argmax, which can only be <= max.
        assert (double_targets <= vanilla_targets + 1e-6).all()
        assert not torch.allclose(double_targets, vanilla_targets)

    def test_terminal_target_is_bare_reward(self, env, demo_vec):
        _, obs_dim = make_obs_encoder(env.observation_space)
        agent = DQNAgent(obs_dim, NUM_ACTIONS, device="cpu", seed=0)
        next_obs = torch.as_tensor(demo_vec[-1][3]).unsqueeze(0)
        rewards = torch.tensor([10.0])
        targets = agent.compute_targets(rewards, next_obs,
                                        dones=torch.tensor([1.0]))
        assert targets.item() == pytest.approx(10.0)

    def test_margin_pretraining_imitates_demo(self, env, demo_vec):
        """A few hundred DQfD pretraining steps must make the greedy policy
        reproduce the demonstrated actions at the demo states."""
        _, obs_dim = make_obs_encoder(env.observation_space)
        agent = _demo_filled_agent(demo_vec, obs_dim)
        for _ in range(300):
            agent.learn()
        matches = sum(
            agent.act(state, epsilon=0.0) == action
            for state, action, *_ in demo_vec
        )
        assert matches / len(demo_vec) > 0.9
