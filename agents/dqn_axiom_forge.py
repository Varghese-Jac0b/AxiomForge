"""
Deep Q-learning family for AxiomForge V1.0.

One agent class covers the whole ladder via three independent switches:

    double  - Double DQN target: a* = argmax_a Q_online(s', a),
              bootstrap from Q_target(s', a*) (van Hasselt et al. 2016).
    dueling - Dueling architecture: Q(s, a) = V(s) + A(s, a) - mean_a A
              (Wang et al. 2016).
    per     - Prioritized experience replay, proportional variant with
              importance-sampling weights (Schaul et al. 2016).

Demonstration support (DQfD-lite, Hester et al. 2018) is built in because
of the V1.0 exploration wall (Build Record Section 4): undirected
exploration never reaches the +10 success signal, so - exactly like the
official demo-seeded tabular baseline - the deep agents learn from the
manual-solver trajectory via (a) a protected demo region in the replay
buffer, (b) a guaranteed demo fraction per batch (uniform buffer) or a
priority bonus (PER), and (c) a large-margin supervised loss on demo
transitions.

Environment-contract notes:
- Observations are flattened with one-hot encoding for every discrete
  field except the two V1.4 proxy counters (Discrete(256)), which enter
  as single scaled scalars. The observation dict shape is frozen across
  versions, so this encoding never changes either.
- `done` for bootstrapping is TERMINATION ONLY: hitting max_steps is a
  time limit, not a terminal state (audit fix, Build Record Section 3).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

# Proxy counters are Discrete(256): one-hot would waste 512 inputs on
# V1.4-only diagnostics, so they enter as scaled scalars.
_SCALED_KEYS = ("proxy_attempt_count", "proxy_claim_count")


# ---------------------------------------------------------------------
# Observation flattening
# ---------------------------------------------------------------------


def make_obs_encoder(observation_space: spaces.Dict):
    """Build (encode_fn, input_dim) for the AxiomForge Dict observation.

    One-hot for every Discrete/MultiDiscrete field, except the proxy
    counters which become single floats in [0, 1].
    """
    specs: list[tuple[str, str, list[int]]] = []
    dim = 0
    for key, space in observation_space.spaces.items():
        if key in _SCALED_KEYS:
            specs.append((key, "scaled", [int(space.n)]))
            dim += 1
        elif isinstance(space, spaces.Discrete):
            specs.append((key, "onehot", [int(space.n)]))
            dim += int(space.n)
        elif isinstance(space, spaces.MultiDiscrete):
            sizes = [int(n) for n in space.nvec]
            specs.append((key, "multi", sizes))
            dim += sum(sizes)
        else:  # pragma: no cover - the V1 obs dict has no other space types
            raise TypeError(f"unsupported space for key {key!r}: {space}")

    def encode(obs: dict) -> np.ndarray:
        out = np.zeros(dim, dtype=np.float32)
        offset = 0
        for key, kind, sizes in specs:
            if kind == "scaled":
                out[offset] = float(obs[key]) / float(sizes[0] - 1)
                offset += 1
            elif kind == "onehot":
                out[offset + int(obs[key])] = 1.0
                offset += sizes[0]
            else:
                values = obs[key]
                for value, size in zip(values, sizes):
                    out[offset + int(value)] = 1.0
                    offset += size
        return out

    return encode, dim


# ---------------------------------------------------------------------
# Noisy networks (Fortunato et al. 2017, factorized Gaussian noise)
# ---------------------------------------------------------------------


class NoisyLinear(nn.Module):
    """Linear layer with learnable factorized Gaussian noise on weights and
    bias: y = (mu_w + sigma_w * eps_w) x + (mu_b + sigma_b * eps_b).

    Noise provides state-dependent, self-annealing exploration (the sigmas
    are learned and shrink as the agent gains confidence), so a noisy agent
    needs no epsilon schedule. Determinism for greedy evaluation is obtained
    by zeroing the noise (remove_noise), independent of train/eval module
    mode - so this never fights the existing target_net.eval() call.
    """

    def __init__(self, in_features: int, out_features: int,
                 sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.empty(out_features, in_features))
        self.register_buffer(
            "weight_epsilon", torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        self.weight_sigma.data.fill_(
            self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(
            self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """Resample the factorized exploration noise."""
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))
        self.bias_epsilon.copy_(eps_out)

    def remove_noise(self) -> None:
        """Zero the noise -> deterministic mean weights (greedy eval)."""
        self.weight_epsilon.zero_()
        self.bias_epsilon.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return nn.functional.linear(x, weight, bias)


def _linear(in_features: int, out_features: int, noisy: bool) -> nn.Module:
    return (NoisyLinear(in_features, out_features) if noisy
            else nn.Linear(in_features, out_features))


def reset_module_noise(module: nn.Module) -> None:
    for sub in module.modules():
        if isinstance(sub, NoisyLinear):
            sub.reset_noise()


def remove_module_noise(module: nn.Module) -> None:
    for sub in module.modules():
        if isinstance(sub, NoisyLinear):
            sub.remove_noise()


# ---------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------


class QNetwork(nn.Module):
    """Plain MLP Q-network: input -> 256 -> 256 -> |A|. With noisy=True the
    linear layers become NoisyLinear (NoisyNet exploration)."""

    def __init__(self, input_dim: int, num_actions: int,
                 hidden: tuple[int, ...] = (256, 256), noisy: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden:
            layers += [_linear(prev, width, noisy), nn.ReLU()]
            prev = width
        layers.append(_linear(prev, num_actions, noisy))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def reset_noise(self) -> None:
        reset_module_noise(self)

    def remove_noise(self) -> None:
        remove_module_noise(self)


class DuelingQNetwork(nn.Module):
    """Dueling architecture: shared trunk, separate V(s) and A(s, a) heads,
    combined as Q = V + A - mean_a(A) (the identifiability correction).
    With noisy=True the two value/advantage HEAD layers become NoisyLinear
    (the standard Rainbow placement: noise in the heads, not the trunk)."""

    def __init__(self, input_dim: int, num_actions: int,
                 trunk_width: int = 256, head_width: int = 128,
                 noisy: bool = False):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, trunk_width), nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            _linear(trunk_width, head_width, noisy), nn.ReLU(),
            _linear(head_width, 1, noisy),
        )
        self.advantage_head = nn.Sequential(
            _linear(trunk_width, head_width, noisy), nn.ReLU(),
            _linear(head_width, num_actions, noisy),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def reset_noise(self) -> None:
        reset_module_noise(self)

    def remove_noise(self) -> None:
        remove_module_noise(self)


# ---------------------------------------------------------------------
# Replay buffers
# ---------------------------------------------------------------------


class ReplayBuffer:
    """Uniform replay with a protected demonstration region.

    Slots [0, demo_capacity) hold demo transitions and are never
    overwritten; agent transitions cycle through the remaining slots.
    sample() guarantees `demo_fraction` of each batch comes from the demo
    region (sampling is with replacement, so small demos still fill
    batches during pretraining).
    """

    def __init__(self, capacity: int, obs_dim: int, demo_capacity: int = 0):
        assert 0 <= demo_capacity < capacity
        self.capacity = capacity
        self.demo_capacity = demo_capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)  # terminated only
        self.is_demo = np.zeros(capacity, dtype=bool)
        self.n_demo = 0
        self._agent_pos = demo_capacity
        self._agent_filled = 0

    def __len__(self) -> int:
        return self.n_demo + self._agent_filled

    def _write(self, index, state, action, reward, next_state, done, demo):
        self.obs[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_obs[index] = next_state
        self.dones[index] = float(done)
        self.is_demo[index] = demo
        return index

    def add_demo(self, state, action, reward, next_state, done) -> int:
        assert self.n_demo < self.demo_capacity, "demo region full"
        index = self._write(self.n_demo, state, action, reward, next_state,
                            done, True)
        self.n_demo += 1
        return index

    def add(self, state, action, reward, next_state, done) -> int:
        index = self._write(self._agent_pos, state, action, reward,
                            next_state, done, False)
        self._agent_pos += 1
        if self._agent_pos >= self.capacity:
            self._agent_pos = self.demo_capacity
        self._agent_filled = min(self._agent_filled + 1,
                                 self.capacity - self.demo_capacity)
        return index

    def sample_indices(self, batch_size: int, rng: np.random.Generator,
                       demo_fraction: float = 0.0) -> np.ndarray:
        n_demo_draw = 0
        if self.n_demo > 0:
            if self._agent_filled == 0:
                n_demo_draw = batch_size  # pretraining: demo-only buffer
            else:
                n_demo_draw = int(round(batch_size * demo_fraction))
        n_agent_draw = batch_size - n_demo_draw
        parts = []
        if n_demo_draw > 0:
            parts.append(rng.integers(0, self.n_demo, size=n_demo_draw))
        if n_agent_draw > 0:
            parts.append(self.demo_capacity
                         + rng.integers(0, self._agent_filled,
                                        size=n_agent_draw))
        return np.concatenate(parts)

    def sample(self, batch_size, rng, demo_fraction=0.0):
        indices = self.sample_indices(batch_size, rng, demo_fraction)
        weights = np.ones(batch_size, dtype=np.float32)
        return self._gather(indices), weights, indices

    def _gather(self, indices: np.ndarray) -> dict:
        return {
            "obs": self.obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "dones": self.dones[indices],
            "is_demo": self.is_demo[indices],
        }


class SumTree:
    """Array-backed sum tree for proportional prioritized sampling.
    Leaves live at [capacity, 2*capacity); internal node i sums its
    children 2i and 2i+1."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)

    def update(self, index: int, priority: float):
        node = index + self.capacity
        delta = priority - self.tree[node]
        while node >= 1:
            self.tree[node] += delta
            node //= 2

    def total(self) -> float:
        return float(self.tree[1])

    def get(self, index: int) -> float:
        return float(self.tree[index + self.capacity])

    def find(self, mass: float) -> int:
        """Descend to the leaf whose prefix-sum interval contains `mass`."""
        node = 1
        while node < self.capacity:
            left = 2 * node
            if mass <= self.tree[left]:
                node = left
            else:
                mass -= self.tree[left]
                node = left + 1
        return node - self.capacity


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional PER (Schaul et al. 2016) on top of the demo-protected
    buffer. New transitions get max priority; demo transitions receive a
    constant priority bonus when their TD errors are refreshed (DQfD's
    epsilon_d), which keeps demonstrations from fading out of replay."""

    def __init__(self, capacity, obs_dim, demo_capacity=0, *,
                 alpha=0.6, priority_eps=1e-3, demo_priority_bonus=0.3):
        super().__init__(capacity, obs_dim, demo_capacity)
        self.alpha = alpha
        self.priority_eps = priority_eps
        self.demo_priority_bonus = demo_priority_bonus
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def add_demo(self, *args) -> int:
        index = super().add_demo(*args)
        self.tree.update(index, self.max_priority)
        return index

    def add(self, *args) -> int:
        index = super().add(*args)
        self.tree.update(index, self.max_priority)
        return index

    def sample(self, batch_size, rng, demo_fraction=0.0, beta=0.4):
        # Stratified proportional sampling: one draw per equal-mass segment.
        # demo_fraction is unused here - priorities (with the demo bonus)
        # decide the demo share.
        total = self.tree.total()
        segment = total / batch_size
        indices = np.empty(batch_size, dtype=np.int64)
        for i in range(batch_size):
            mass = (i + rng.random()) * segment
            indices[i] = self.tree.find(min(mass, total - 1e-9))
        priorities = np.array([self.tree.get(i) for i in indices])
        probs = priorities / total
        size = len(self)
        weights = (size * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)
        return self._gather(indices), weights, indices

    def update_priorities(self, indices: np.ndarray, td_abs: np.ndarray):
        bonus = self.demo_priority_bonus * self.is_demo[indices]
        priorities = (np.abs(td_abs) + self.priority_eps + bonus) ** self.alpha
        for index, priority in zip(indices, priorities):
            self.tree.update(int(index), float(priority))
            self.max_priority = max(self.max_priority, float(priority))


# ---------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DQNAgent:
    """DQN / DDQN / Dueling (D)DQN / +PER agent with DQfD-lite demos."""

    def __init__(
        self, obs_dim: int, num_actions: int, *,
        double: bool = False, dueling: bool = False, per: bool = False,
        noisy: bool = False,
        gamma: float = 0.99, lr: float = 5e-4,
        buffer_capacity: int = 100_000, demo_capacity: int = 0,
        batch_size: int = 64, target_update_interval: int = 1000,
        demo_fraction: float = 0.25, margin: float = 0.8,
        margin_weight: float = 1.0,
        per_alpha: float = 0.6, per_beta_start: float = 0.4,
        per_beta_steps: int = 100_000,
        grad_clip: float = 10.0, device: str = "cpu",
        seed: int | None = None,
    ):
        self.num_actions = num_actions
        self.double = double
        self.dueling = dueling
        self.per = per
        self.noisy = noisy
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.demo_fraction = demo_fraction
        self.margin = margin
        self.margin_weight = margin_weight
        self.per_beta_start = per_beta_start
        self.per_beta_steps = per_beta_steps
        self.grad_clip = grad_clip
        self.device = resolve_device(device)
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        net_cls = DuelingQNetwork if dueling else QNetwork
        self.online_net = net_cls(obs_dim, num_actions, noisy=noisy).to(
            self.device)
        self.target_net = net_cls(obs_dim, num_actions, noisy=noisy).to(
            self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        if per:
            self.buffer: ReplayBuffer = PrioritizedReplayBuffer(
                buffer_capacity, obs_dim, demo_capacity, alpha=per_alpha,
            )
        else:
            self.buffer = ReplayBuffer(buffer_capacity, obs_dim,
                                       demo_capacity)
        self.learn_steps = 0

    # ---------------- acting ----------------

    def act(self, state_vec: np.ndarray, epsilon: float,
            deterministic: bool = False) -> int:
        """Epsilon-greedy action. For noisy agents, behaviour resamples the
        exploration noise each step; `deterministic=True` (greedy eval)
        zeroes the noise and skips the epsilon draw, giving a clean,
        reproducible greedy policy with no intrinsic stochasticity."""
        if not deterministic and self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.num_actions))
        if self.noisy:
            if deterministic:
                self.online_net.remove_noise()
            else:
                self.online_net.reset_noise()
        with torch.no_grad():
            state = torch.as_tensor(state_vec, device=self.device).unsqueeze(0)
            q_values = self.online_net(state).squeeze(0)
        return int(torch.argmax(q_values).item())

    # ---------------- storing ----------------

    def store(self, state, action, reward, next_state, terminated):
        self.buffer.add(state, action, reward, next_state, terminated)

    def store_demo(self, state, action, reward, next_state, terminated):
        self.buffer.add_demo(state, action, reward, next_state, terminated)

    # ---------------- learning ----------------

    def _per_beta(self) -> float:
        frac = min(1.0, self.learn_steps / self.per_beta_steps)
        return self.per_beta_start + frac * (1.0 - self.per_beta_start)

    def compute_targets(self, rewards: torch.Tensor, next_obs: torch.Tensor,
                        dones: torch.Tensor) -> torch.Tensor:
        """TD(0) targets. `dones` is termination only - truncation
        bootstraps (Build Record Section 3 audit fix)."""
        with torch.no_grad():
            if self.double:
                next_actions = self.online_net(next_obs).argmax(
                    dim=1, keepdim=True)
                next_q = self.target_net(next_obs).gather(
                    1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_obs).max(dim=1).values
            return rewards + self.gamma * next_q * (1.0 - dones)

    def learn(self) -> float | None:
        if len(self.buffer) == 0:
            return None
        if self.noisy:
            # Fresh exploration noise on both nets per gradient step.
            self.online_net.reset_noise()
            self.target_net.reset_noise()
        if self.per:
            batch, weights, indices = self.buffer.sample(
                self.batch_size, self.rng, beta=self._per_beta(),
            )
        else:
            batch, weights, indices = self.buffer.sample(
                self.batch_size, self.rng, demo_fraction=self.demo_fraction,
            )

        obs = torch.as_tensor(batch["obs"], device=self.device)
        actions = torch.as_tensor(batch["actions"], device=self.device)
        rewards = torch.as_tensor(batch["rewards"], device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], device=self.device)
        dones = torch.as_tensor(batch["dones"], device=self.device)
        is_demo = torch.as_tensor(batch["is_demo"], device=self.device)
        weights_t = torch.as_tensor(weights, device=self.device)

        targets = self.compute_targets(rewards, next_obs, dones)
        q_all = self.online_net(obs)
        q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        td_errors = targets - q_sa

        td_loss = (weights_t * nn.functional.smooth_l1_loss(
            q_sa, targets, reduction="none")).mean()

        # DQfD large-margin loss on demo transitions:
        # J_E = max_a [Q(s,a) + margin*1(a != a_E)] - Q(s, a_E).
        loss = td_loss
        if bool(is_demo.any()) and self.margin_weight > 0:
            margins = torch.full_like(q_all, self.margin)
            margins.scatter_(1, actions.unsqueeze(1), 0.0)
            margin_term = (q_all + margins).max(dim=1).values - q_sa
            margin_loss = (margin_term * is_demo.float()).sum() \
                / is_demo.float().sum()
            loss = loss + self.margin_weight * margin_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.per:
            self.buffer.update_priorities(
                indices, td_errors.detach().abs().cpu().numpy(),
            )

        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return float(loss.item())
