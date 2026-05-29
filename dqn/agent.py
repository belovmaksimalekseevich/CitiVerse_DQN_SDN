# dqn/agent.py
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from dqn.model import DuelingDQN
from dqn.replay_buffer import PrioritizedReplayBuffer, NStepBuffer

LOG = logging.getLogger(__name__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DQNAgent:
    def __init__(
        self,
        state_dim=94,
        action_dim=100,
        hidden=256,
        lr=3e-4,
        gamma=0.99,
        batch_size=256,
        buffer_size=100_000,
        target_update_freq=100,
        n_step=3,
        total_steps=None,
        eps_start=1.0,
        eps_end=0.05,
        checkpoint_path='results/best.pth',
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.n_step = n_step
        self.checkpoint_path = checkpoint_path

        self.online = DuelingDQN(state_dim, action_dim, hidden).to(DEVICE)
        self.target = DuelingDQN(state_dim, action_dim, hidden).to(DEVICE)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        T_max = total_steps if total_steps else 200_000
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=T_max, eta_min=lr * 0.01
        )

        # PER + N-step
        self.replay = PrioritizedReplayBuffer(
            capacity=buffer_size,
            state_dim=state_dim,
            alpha=0.6,
            beta_start=0.4,
            beta_end=1.0,
            beta_steps=T_max,
        )
        self.n_step_buf = NStepBuffer(self.replay, n=n_step, gamma=gamma)

        # Epsilon decay
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = (eps_start - eps_end) / max(T_max * 0.8, 1)

        self.step_count = 0
        # HuberLoss with reduction='none' for per-sample PER weighting
        self.loss_fn = nn.HuberLoss(reduction='none')

        # Tracking for save_best and auto_reset
        self._best_reward = -float('inf')
        self._recent_rewards = deque(maxlen=10)
        self._recent_losses = deque(maxlen=50)

    # ------------------------------------------------------------------
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

        s, a, r, s2, done, weights, tree_idxs = self.replay.sample(self.batch_size)
        s       = torch.FloatTensor(s).to(DEVICE)
        a       = torch.LongTensor(a).to(DEVICE)
        r       = torch.FloatTensor(r).to(DEVICE)
        s2      = torch.FloatTensor(s2).to(DEVICE)
        done    = torch.FloatTensor(done).to(DEVICE)
        w       = torch.FloatTensor(weights).to(DEVICE)

        with torch.no_grad():
            next_actions = self.online(s2).argmax(dim=-1)
            next_q = self.target(s2).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = r + (1.0 - done) * (self.gamma ** self.n_step) * next_q

        current_q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        td_errors = (target_q - current_q).detach().cpu().numpy()
        self.replay.update_priorities(tree_idxs, td_errors)

        # IS-weighted Huber loss
        elementwise_loss = self.loss_fn(current_q, target_q)
        loss = (w * elementwise_loss).mean()

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

    # ------------------------------------------------------------------
    def record_episode_reward(self, reward):
        """Call at end of each episode. Saves best checkpoint."""
        self._recent_rewards.append(reward)
        if len(self._recent_rewards) < 10:
            return
        mean_r = float(np.mean(self._recent_rewards))
        if mean_r > self._best_reward:
            self._best_reward = mean_r
            os.makedirs(os.path.dirname(self.checkpoint_path) or '.', exist_ok=True)
            self.save(self.checkpoint_path)
            LOG.info(f'New best checkpoint: mean_reward={mean_r:.3f}')

    def maybe_auto_reset(self, loss):
        """
        Auto-reset on divergence: if loss is high and eps is low,
        reload best checkpoint and bump epsilon slightly.
        Returns True if reset occurred.
        """
        if loss is not None:
            self._recent_losses.append(loss)
        if len(self._recent_losses) < 50:
            return False
        mean_loss = float(np.mean(self._recent_losses))
        if mean_loss > 1.0 and self.eps < 0.1 and os.path.exists(self.checkpoint_path):
            LOG.warning(
                f'Divergence detected (loss={mean_loss:.3f}, eps={self.eps:.3f}). '
                f'Reloading best checkpoint.'
            )
            self.load(self.checkpoint_path)
            self.eps = max(self.eps, 0.15)
            self._recent_losses.clear()
            return True
        return False

    # ------------------------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'online':    self.online.state_dict(),
            'target':    self.target.state_dict(),
            'opt':       self.opt.state_dict(),
            'step':      self.step_count,
            'eps':       self.eps,
            'best_r':    self._best_reward,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.online.load_state_dict(ckpt['online'])
        self.target.load_state_dict(ckpt['target'])
        self.opt.load_state_dict(ckpt['opt'])
        self.step_count = ckpt['step']
        self.eps = ckpt['eps']
        self._best_reward = ckpt.get('best_r', -float('inf'))
        LOG.info(f'Loaded checkpoint from {path} (step={self.step_count})')
