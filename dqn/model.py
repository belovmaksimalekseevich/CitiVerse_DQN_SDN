# dqn/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    """
    Dueling DQN: separate value and advantage streams.
    LayerNorm works with batch_size=1 during inference (unlike BatchNorm).
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
        self.value = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x, action_mask=None):
        features = self.shared(x)
        v = self.value(features)       # (B, 1)
        a = self.advantage(features)   # (B, action_dim)

        if action_mask is not None:
            a = a.masked_fill(~action_mask, -1e9)

        q = v + a - a.mean(dim=-1, keepdim=True)
        return q


class NoisyLinear(nn.Module):
    """
    Factorized Noisy Linear layer (Fortunato et al. 2017).
    Optional replacement for epsilon-greedy exploration.
    """

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_f = in_features
        self.out_f = out_features
        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_b = nn.Parameter(torch.empty(out_features))
        self.sigma_b = nn.Parameter(torch.empty(out_features))
        self.register_buffer('eps_w', torch.zeros(out_features, in_features))
        self.register_buffer('eps_b', torch.zeros(out_features))
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
        return F.linear(x, w, b)
