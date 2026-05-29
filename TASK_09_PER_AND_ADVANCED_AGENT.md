# TASK_09: PER + Advanced Agent Features (extends TASK_04)

## Context

TASK_04 implements uniform `ReplayBuffer` + basic `DQNAgent`.
This task adds three features from PLAN_AND_REVIEW.md:

1. **PER** (Prioritized Experience Replay) — from MOOO-RDQN comparison (Section 2.1)
2. **Save best checkpoint** + **auto-reset on divergence** (I02, I03)
3. **Noisy Networks** as optional alternative to epsilon-greedy (I04)

These go in `dqn/replay_buffer.py` and `dqn/agent.py`.

---

## 1. Add PrioritizedReplayBuffer to dqn/replay_buffer.py

```python
# Append to dqn/replay_buffer.py (after existing ReplayBuffer class)

import numpy as np

class SumTree:
    """Binary sum tree for O(log n) priority sampling."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)
        self.data_ptr = 0
        self.size = 0

    def add(self, priority, data_idx):
        tree_idx = data_ptr_to_tree(self.data_ptr, self.capacity)
        self.update(tree_idx, priority)
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx, priority):
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def get(self, value):
        """Find leaf index for given value."""
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx

    @property
    def total(self):
        return self.tree[0]


def data_ptr_to_tree(ptr, capacity):
    return ptr + capacity - 1


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay.
    priority = |TD_error| + epsilon_prio
    sample prob ∝ priority^alpha
    Importance sampling weights with beta annealing.
    """

    def __init__(self, capacity=100_000, state_dim=94,
                 alpha=0.6, beta_start=0.4, beta_end=1.0,
                 beta_steps=200_000, epsilon_prio=0.01):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_end = beta_end
        self.beta_increment = (beta_end - beta_start) / beta_steps
        self.epsilon_prio = epsilon_prio
        self.max_priority = 1.0

        self.tree = SumTree(capacity)
        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int32)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(self, s, a, r, s2, done):
        idx = self.ptr % self.capacity
        self.states[idx] = s
        self.actions[idx] = a
        self.rewards[idx] = r
        self.next_states[idx] = s2
        self.dones[idx] = float(done)
        # New transitions get max priority (so they're sampled at least once)
        tree_idx = data_ptr_to_tree(idx, self.capacity)
        self.tree.update(tree_idx, self.max_priority ** self.alpha)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = []
        tree_idxs = []
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)
            tree_idx = self.tree.get(value)
            data_idx = tree_idx - self.capacity + 1
            idxs.append(data_idx % self.size)
            tree_idxs.append(tree_idx)

        # Importance weights
        probs = self.tree.tree[tree_idxs] / self.tree.total
        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max()  # normalize

        self.beta = min(self.beta_end, self.beta + self.beta_increment)

        idxs = np.array(idxs)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
            weights.astype(np.float32),
            np.array(tree_idxs),
        )

    def update_priorities(self, tree_idxs, td_errors):
        priorities = (np.abs(td_errors) + self.epsilon_prio) ** self.alpha
        self.max_priority = max(self.max_priority, float(priorities.max()))
        for ti, p in zip(tree_idxs, priorities):
            self.tree.update(ti, p)

    def __len__(self):
        return self.size
```

---

## 2. Update DQNAgent to use PER (dqn/agent.py)

Add to `DQNAgent.__init__()`:
```python
# Replace uniform ReplayBuffer with PER
self.replay = PrioritizedReplayBuffer(
    capacity=buffer_size,
    state_dim=state_dim,
    alpha=0.6,
    beta_start=0.4,
    beta_end=1.0,
    beta_steps=total_steps or 200_000,
)
```

Replace `update()` method with PER-aware version:
```python
def update(self):
    if len(self.replay) < self.batch_size:
        return None

    s, a, r, s2, done, weights, tree_idxs = self.replay.sample(self.batch_size)
    s    = torch.FloatTensor(s).to(DEVICE)
    a    = torch.LongTensor(a).to(DEVICE)
    r    = torch.FloatTensor(r).to(DEVICE)
    s2   = torch.FloatTensor(s2).to(DEVICE)
    done = torch.FloatTensor(done).to(DEVICE)
    w    = torch.FloatTensor(weights).to(DEVICE)

    with torch.no_grad():
        next_actions = self.online(s2).argmax(dim=-1)
        next_q = self.target(s2).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q = r + (1.0 - done) * (self.gamma ** self.n_step) * next_q

    current_q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

    td_errors = (target_q - current_q).detach().cpu().numpy()
    self.replay.update_priorities(tree_idxs, td_errors)

    # Weighted Huber loss
    loss = (w * self.loss_fn(current_q, target_q)).mean()

    self.opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
    self.opt.step()
    self.scheduler.step()

    self.step_count += 1
    if self.step_count % self.target_update_freq == 0:
        self.target.load_state_dict(self.online.state_dict())

    self.eps = max(self.eps_end, self.eps - self.eps_decay)
    return float(loss.item())
```

Also update NStepBuffer to use PER's push:
```python
class NStepBuffer:
    def __init__(self, replay_buffer, n=3, gamma=0.99):
        # Works with both ReplayBuffer and PrioritizedReplayBuffer
        self.buf = replay_buffer
        ...
```

---

## 3. Save Best Checkpoint + Auto-reset (I02, I03)

Add to `DQNAgent`:

```python
def __init__(self, ..., checkpoint_path='results/best.pth'):
    ...
    self.checkpoint_path = checkpoint_path
    self._best_reward = -float('inf')
    self._recent_rewards = deque(maxlen=10)
    self._recent_losses = deque(maxlen=50)

def record_episode_reward(self, reward):
    """Call at end of each episode."""
    self._recent_rewards.append(reward)
    mean_r = np.mean(self._recent_rewards)
    if mean_r > self._best_reward and len(self._recent_rewards) == 10:
        self._best_reward = mean_r
        self.save(self.checkpoint_path)
        LOG.info(f'New best checkpoint: mean_reward={mean_r:.3f}')

def maybe_auto_reset(self, loss):
    """
    Auto-reset on divergence (I03).
    If training diverged (high loss while eps is low), reload best checkpoint.
    """
    if loss is not None:
        self._recent_losses.append(loss)

    if len(self._recent_losses) < 50:
        return False

    mean_loss = np.mean(self._recent_losses)
    # Divergence condition: high loss + not exploring (should be exploiting now)
    if mean_loss > 1.0 and self.eps < 0.1:
        import os
        if os.path.exists(self.checkpoint_path):
            LOG.warning(f'Divergence detected (loss={mean_loss:.3f}, eps={self.eps:.3f}). '
                        f'Reloading best checkpoint.')
            self.load(self.checkpoint_path)
            # Slightly bump epsilon to allow re-exploration
            self.eps = max(self.eps, 0.15)
            self._recent_losses.clear()
            return True
    return False
```

Usage in train.py:
```python
for ep in range(n_episodes):
    state = env.reset()
    ep_reward = 0.0
    for _ in range(max_steps):
        action = agent.select_action(state, ...)
        next_state, reward, done, info = env.step(action)
        agent.push(state, action, reward, next_state, done)
        loss = agent.update()
        agent.maybe_auto_reset(loss)   # ADD THIS
        state = next_state
        ep_reward += reward
        if done:
            break
    agent.record_episode_reward(ep_reward)  # ADD THIS
```

---

## 4. Optional: NoisyLinear Layer (I04)

Noisy Networks replace epsilon-greedy with learned stochastic exploration.
Implement as drop-in replacement for the advantage head in `DuelingDQN`:

```python
# Add to dqn/model.py

class NoisyLinear(nn.Module):
    """
    Factorized Noisy Linear layer (Fortunato et al. 2017).
    Replaces epsilon-greedy exploration.
    """

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_f = in_features
        self.out_f = out_features
        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_b = nn.Parameter(torch.empty(out_features))
        self.sigma_b = nn.Parameter(torch.empty(out_features))
        self.register_buffer('eps_w', torch.empty(out_features, in_features))
        self.register_buffer('eps_b', torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        mu_range = 1.0 / self.in_f ** 0.5
        self.mu_w.data.uniform_(-mu_range, mu_range)
        self.mu_b.data.uniform_(-mu_range, mu_range)
        self.sigma_w.data.fill_(self.sigma_init / self.in_f ** 0.5)
        self.sigma_b.data.fill_(self.sigma_init / self.out_f ** 0.5)

    def sample_noise(self):
        def f(x):
            return x.sign() * x.abs().sqrt()
        p = f(torch.randn(self.in_f))
        q = f(torch.randn(self.out_f))
        self.eps_w.copy_(q.outer(p))
        self.eps_b.copy_(q)

    def forward(self, x):
        if self.training:
            w = self.mu_w + self.sigma_w * self.eps_w
            b = self.mu_b + self.sigma_b * self.eps_b
        else:
            w = self.mu_w
            b = self.mu_b
        return nn.functional.linear(x, w, b)
```

To use NoisyLinear instead of epsilon-greedy, replace advantage head in DuelingDQN:
```python
# In DuelingDQN.__init__():
USE_NOISY = False   # set True to enable
if USE_NOISY:
    self.advantage = nn.Sequential(NoisyLinear(hidden, 128), nn.ReLU(), NoisyLinear(128, action_dim))
else:
    self.advantage = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, action_dim))
```

When using NoisyLinear: set `eps_start = eps_end = 0.0` in DQNAgent (no epsilon-greedy).
Call `model.online.advantage[0].sample_noise()` at start of each episode.

**Recommendation**: Start with epsilon-greedy (simpler, easier to debug). Switch to
NoisyLinear in Run 2 if epsilon-greedy is insufficient for exploration.

---

## Implementation order

1. Add `PrioritizedReplayBuffer` and `SumTree` to `dqn/replay_buffer.py`
2. Update `DQNAgent` in `dqn/agent.py` to use PER, add `record_episode_reward`
   and `maybe_auto_reset`
3. Update `train.py` (TASK_05) to call the new methods
4. `NoisyLinear` is optional — add to `dqn/model.py` but don't enable by default
