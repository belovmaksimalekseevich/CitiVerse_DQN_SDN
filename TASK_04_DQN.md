# TASK_04: DQN Components — model.py, replay_buffer.py, agent.py

## Step 6 — dqn/model.py (Dueling DQN + LayerNorm)

```python
# dqn/model.py
import torch
import torch.nn as nn

class DuelingDQN(nn.Module):
    """
    Dueling DQN: separate value and advantage streams.
    LayerNorm instead of BatchNorm (works with batch_size=1 during inference).
    ACTION_DIM = N_SWITCHES * N_CONTROLLERS = 20 * 5 = 100
    STATE_DIM = 94
    """

    def __init__(self, state_dim=94, action_dim=100, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        # Value stream V(s)
        self.value = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Advantage stream A(s, a)
        self.advantage = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x, action_mask=None):
        features = self.shared(x)
        v = self.value(features)                       # (B, 1)
        a = self.advantage(features)                   # (B, action_dim)

        if action_mask is not None:
            # Set Q-values of invalid actions to large negative
            invalid = ~action_mask
            a = a.masked_fill(invalid, -1e9)

        # Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q
```

---

## Step 7 — dqn/replay_buffer.py (PER + NStepBuffer)

```python
# dqn/replay_buffer.py
import numpy as np
from collections import deque

class ReplayBuffer:
    """Uniform replay buffer. Single implementation (FIX: old code had two)."""

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
    N-step return buffer. Wraps ReplayBuffer.
    Computes n-step return: R = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1} + γ^n * Q(s_{t+n})
    n=3, gamma=0.99
    """

    def __init__(self, replay_buffer: ReplayBuffer, n=3, gamma=0.99):
        self.buf = replay_buffer
        self.n = n
        self.gamma = gamma
        self.queue = deque()  # (s, a, r, s2, done)

    def push(self, s, a, r, s2, done):
        self.queue.append((s, a, r, s2, done))
        if len(self.queue) < self.n and not done:
            return
        # Compute n-step return
        s0, a0 = self.queue[0][0], self.queue[0][1]
        G = 0.0
        for k, (_, _, rk, s2k, dk) in enumerate(self.queue):
            G += (self.gamma ** k) * rk
            if dk:
                s_n, done_n = s2k, True
                break
        else:
            s_n, done_n = self.queue[-1][3], self.queue[-1][4]
        self.buf.push(s0, a0, G, s_n, done_n)
        if done:
            self.queue.clear()
        else:
            self.queue.popleft()

    def flush(self):
        """Call at episode end to drain remaining transitions."""
        while self.queue:
            s0, a0 = self.queue[0][0], self.queue[0][1]
            G = 0.0
            for k, (_, _, rk, s2k, dk) in enumerate(self.queue):
                G += (self.gamma ** k) * rk
                if dk:
                    s_n, done_n = s2k, True
                    break
            else:
                s_n, done_n = self.queue[-1][3], self.queue[-1][4]
            self.buf.push(s0, a0, G, s_n, done_n)
            self.queue.popleft()
```

---

## Step 8 — dqn/agent.py (Double Dueling DQN)

```python
# dqn/agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn.model import DuelingDQN
from dqn.replay_buffer import ReplayBuffer, NStepBuffer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQNAgent:
    def __init__(
        self,
        state_dim=94,
        action_dim=100,        # N_SWITCHES * N_CONTROLLERS
        hidden=256,
        lr=3e-4,
        gamma=0.99,
        batch_size=256,
        buffer_size=100_000,
        target_update_freq=100,  # steps
        n_step=3,
        total_steps=None,        # for auto epsilon decay
        eps_start=1.0,
        eps_end=0.05,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.n_step = n_step

        self.online = DuelingDQN(state_dim, action_dim, hidden).to(DEVICE)
        self.target = DuelingDQN(state_dim, action_dim, hidden).to(DEVICE)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt = optim.Adam(self.online.parameters(), lr=lr)

        # Cosine LR annealing (FIX: was fixed LR)
        T_max = total_steps if total_steps else 200_000
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=T_max, eta_min=lr * 0.01)

        # Replay
        self.replay = ReplayBuffer(buffer_size, state_dim)
        self.n_step_buf = NStepBuffer(self.replay, n=n_step, gamma=gamma)

        # Epsilon decay (auto-calculated from total_steps)
        self.eps = eps_start
        self.eps_end = eps_end
        if total_steps:
            self.eps_decay = (eps_start - eps_end) / (total_steps * 0.8)
        else:
            self.eps_decay = 1e-5

        self.step_count = 0
        self.loss_fn = nn.HuberLoss()  # FIX: was MSELoss

    def select_action(self, state, action_mask=None, deterministic=False):
        """Epsilon-greedy with action masking."""
        if not deterministic and np.random.random() < self.eps:
            if action_mask is not None:
                valid = np.where(action_mask)[0]
                return int(np.random.choice(valid)) if len(valid) > 0 else 0
            return int(np.random.randint(self.action_dim))

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            mask = None
            if action_mask is not None:
                mask = torch.BoolTensor(action_mask).unsqueeze(0).to(DEVICE)
            q = self.online(s, action_mask=mask)
            return int(q.argmax(dim=-1).item())

    def push(self, s, a, r, s2, done):
        self.n_step_buf.push(s, a, r, s2, done)
        if done:
            self.n_step_buf.flush()

    def update(self):
        if len(self.replay) < self.batch_size:
            return None

        s, a, r, s2, done = self.replay.sample(self.batch_size)
        s    = torch.FloatTensor(s).to(DEVICE)
        a    = torch.LongTensor(a).to(DEVICE)
        r    = torch.FloatTensor(r).to(DEVICE)
        s2   = torch.FloatTensor(s2).to(DEVICE)
        done = torch.FloatTensor(done).to(DEVICE)

        # Double DQN target
        with torch.no_grad():
            next_actions = self.online(s2).argmax(dim=-1)
            next_q = self.target(s2).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = r + (1.0 - done) * (self.gamma ** self.n_step) * next_q

        current_q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(current_q, target_q)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        self.scheduler.step()

        self.step_count += 1

        # Soft target update
        if self.step_count % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        # Epsilon decay
        self.eps = max(self.eps_end, self.eps - self.eps_decay)

        return float(loss.item())

    def save(self, path):
        torch.save({
            'online': self.online.state_dict(),
            'target': self.target.state_dict(),
            'opt': self.opt.state_dict(),
            'step': self.step_count,
            'eps': self.eps,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.online.load_state_dict(ckpt['online'])
        self.target.load_state_dict(ckpt['target'])
        self.opt.load_state_dict(ckpt['opt'])
        self.step_count = ckpt['step']
        self.eps = ckpt['eps']
```

---

## Key Parameter Choices (justified)

| Parameter | Value | Why |
|---|---|---|
| gamma | 0.99 | Long-horizon ICD optimization, not myopic (FIX: was 0.952) |
| lr | 3e-4 | Standard Adam starting LR, cosine decay to 3e-6 |
| batch_size | 256 | Stable gradients; GPU can handle comfortably |
| n_step | 3 | Reduces variance vs 1-step, less bias than n=5+ |
| target_update | 100 steps | Frequent enough for fast convergence |
| HuberLoss | delta=1 | Robust to outlier rewards (overload spikes) |
| hidden | 256 | Sufficient for STATE_DIM=94, ACTION_DIM=100 |
| eps_decay | auto | Decays to 0.05 over 80% of total training steps |
