# dqn/replay_buffer.py
import numpy as np
from collections import deque


class ReplayBuffer:
    """Uniform experience replay buffer."""

    def __init__(self, capacity=100_000, state_dim=94):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int32)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)

    def push(self, s, a, r, s2, done):
        i = self.ptr % self.capacity
        self.states[i]      = s
        self.actions[i]     = a
        self.rewards[i]     = r
        self.next_states[i] = s2
        self.dones[i]       = float(done)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )

    def __len__(self):
        return self.size


class NStepBuffer:
    """
    N-step return buffer. Wraps any replay buffer.
    Computes: R = r_t + gamma*r_{t+1} + ... + gamma^{n-1}*r_{t+n-1}
    then pushes (s_t, a_t, R, s_{t+n}, done) into the backing buffer.
    """

    def __init__(self, replay_buffer, n=3, gamma=0.99):
        self.buf = replay_buffer
        self.n = n
        self.gamma = gamma
        self.queue = deque()

    def push(self, s, a, r, s2, done):
        self.queue.append((s, a, r, s2, done))
        if len(self.queue) < self.n and not done:
            return
        self._flush_front(done)
        if done:
            self.flush()

    def _flush_front(self, done):
        if not self.queue:
            return
        s0, a0 = self.queue[0][0], self.queue[0][1]
        G = 0.0
        s_n, done_n = self.queue[-1][3], self.queue[-1][4]
        for k, (_, _, rk, s2k, dk) in enumerate(self.queue):
            G += (self.gamma ** k) * rk
            if dk:
                s_n, done_n = s2k, True
                break
        self.buf.push(s0, a0, G, s_n, done_n)
        if not done:
            self.queue.popleft()

    def flush(self):
        """Drain remaining transitions at episode end."""
        while self.queue:
            self._flush_front(done=True)
            if self.queue:
                self.queue.popleft()

    def __len__(self):
        return len(self.buf)


# ---------------------------------------------------------------------------
# Prioritized Experience Replay
# ---------------------------------------------------------------------------

class SumTree:
    """Binary sum tree for O(log n) priority sampling."""

    def __init__(self, capacity):
        self.capacity = capacity
        # Tree has 2*capacity nodes; leaves are [capacity-1 .. 2*capacity-2]
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data_ptr = 0
        self.size = 0

    def _leaf_idx(self, data_idx):
        return data_idx + self.capacity - 1

    def add(self, priority):
        tree_idx = self._leaf_idx(self.data_ptr)
        self.update(tree_idx, priority)
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return tree_idx

    def update(self, tree_idx, priority):
        delta = float(priority) - self.tree[tree_idx]
        self.tree[tree_idx] = float(priority)
        idx = tree_idx
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def get(self, value):
        """Find leaf index (tree idx) for given cumulative value."""
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left] or self.tree[right] == 0:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx

    @property
    def total(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay.
    priority = (|TD_error| + epsilon_prio)^alpha
    Importance-sampling weights with beta annealing to 1.0.
    """

    def __init__(self, capacity=100_000, state_dim=94,
                 alpha=0.6, beta_start=0.4, beta_end=1.0,
                 beta_steps=200_000, epsilon_prio=0.01):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_end = beta_end
        self.beta_increment = (beta_end - beta_start) / max(beta_steps, 1)
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
        self._tree_idxs = np.zeros(capacity, dtype=np.int32)

    def push(self, s, a, r, s2, done):
        idx = self.ptr % self.capacity
        self.states[idx]      = s
        self.actions[idx]     = a
        self.rewards[idx]     = r
        self.next_states[idx] = s2
        self.dones[idx]       = float(done)
        # New transitions get max priority so they are sampled at least once
        priority = self.max_priority ** self.alpha
        self._tree_idxs[idx] = self.tree.add(priority)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.tree.total <= 0:
            idxs = np.random.randint(0, self.size, size=batch_size)
            weights = np.ones(batch_size, dtype=np.float32)
            tree_idxs = np.array([self.tree._leaf_idx(i) for i in idxs])
            return (self.states[idxs], self.actions[idxs], self.rewards[idxs],
                    self.next_states[idxs], self.dones[idxs], weights, tree_idxs)

        segment = self.tree.total / batch_size
        tree_idxs = []
        data_idxs = []

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)
            t_idx = self.tree.get(value)
            d_idx = t_idx - self.capacity + 1
            d_idx = max(0, min(d_idx, self.size - 1))
            tree_idxs.append(t_idx)
            data_idxs.append(d_idx)

        tree_idxs = np.array(tree_idxs, dtype=np.int32)
        data_idxs = np.array(data_idxs, dtype=np.int32)

        probs = self.tree.tree[tree_idxs] / (self.tree.total + 1e-8)
        probs = np.clip(probs, 1e-8, None)
        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(self.beta_end, self.beta + self.beta_increment)

        return (
            self.states[data_idxs],
            self.actions[data_idxs],
            self.rewards[data_idxs],
            self.next_states[data_idxs],
            self.dones[data_idxs],
            weights.astype(np.float32),
            tree_idxs,
        )

    def update_priorities(self, tree_idxs, td_errors):
        priorities = (np.abs(td_errors) + self.epsilon_prio) ** self.alpha
        self.max_priority = max(self.max_priority, float(priorities.max()))
        for ti, p in zip(tree_idxs, priorities):
            self.tree.update(int(ti), float(p))

    def __len__(self):
        return self.size
