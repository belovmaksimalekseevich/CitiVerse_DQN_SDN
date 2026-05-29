# TASK_10: Curriculum Learning in SimEnv (extends TASK_02 + TASK_05)

## Context

From PLAN_AND_REVIEW.md Section 3.2:
"Full complexity from episode 0 (20 switches, 5 controllers, 4 traffic profiles)
is too hard for the agent. Curriculum learning schedule:"

  ep 0-500:    5 switches, 2 controllers, 1 traffic profile (morning)
  ep 500-2000: 10 switches, 3 controllers, 2 profiles (morning + business)
  ep 2000-5000: 20 switches, 5 controllers, 4 profiles (full)

Without curriculum: agent spends most early episodes on essentially random
assignments, getting uninformative rewards. Loss stays high for 1500+ episodes.
With curriculum: loss drops to <0.3 by ep 500-800 (validated in similar RL-SDN work).

---

## Changes to dqn/sim_environment.py

Replace fixed `reset()` with curriculum-aware version:

```python
# dqn/sim_environment.py additions/replacements

class CitiverseSimEnv:
    def __init__(self, seed=42, fault_prob=0.05, latency_jitter=0.1,
                 curriculum=True):
        ...
        self.curriculum = curriculum
        self.global_episode = 0  # set from outside by train.py

        # Curriculum stages
        self._stages = [
            # (max_episode, n_switches, n_controllers, profiles)
            (500,  5,  2, ['morning']),
            (2000, 10, 3, ['morning', 'business']),
            (5000, 20, 5, ['morning', 'business', 'evening', 'night']),
        ]

    def set_episode(self, ep):
        """Called from train loop to update curriculum stage."""
        self.global_episode = ep

    def _get_curriculum_params(self):
        """Return (active_n_switches, active_n_ctrls, active_profiles)."""
        if not self.curriculum:
            return N_SWITCHES, N_CONTROLLERS, TRAFFIC_PROFILES

        for (max_ep, n_sw, n_ctrl, profiles) in self._stages:
            if self.global_episode < max_ep:
                return n_sw, n_ctrl, profiles
        return N_SWITCHES, N_CONTROLLERS, TRAFFIC_PROFILES  # fallback

    def reset(self):
        n_sw, n_ctrl, profiles = self._get_curriculum_params()

        # Use first n_sw switches and first n_ctrl controllers
        active_switches = SWITCHES[:n_sw]
        self.active_n_sw = n_sw
        self.active_n_ctrl = n_ctrl
        self.active_profiles = profiles

        # Zone-optimal init for active switches
        from topology.topology_data import ZONE_CTRL, SW_ZONE
        self.assignments = np.zeros(N_SWITCHES, dtype=int)  # full vector
        for i, sw in enumerate(active_switches):
            self.assignments[i] = ZONE_CTRL[SW_ZONE[sw]] % n_ctrl

        n_perturb = self.rng.integers(1, min(4, n_sw))
        idxs = self.rng.choice(n_sw, size=n_perturb, replace=False)
        for i in idxs:
            self.assignments[i] = self.rng.integers(0, n_ctrl)

        self.traffic_profile = self.rng.choice(profiles)
        self.step_count = 0
        self.failed_ctrl = -1
        self.migration_count = 0
        return self._get_state()

    def step(self, action):
        # Map action to active switches/controllers only
        n_sw = self.active_n_sw
        n_ctrl = self.active_n_ctrl
        sw_idx = action // n_ctrl
        ctrl_idx = action % n_ctrl

        # Clamp to active range
        sw_idx = min(sw_idx, n_sw - 1)
        ctrl_idx = min(ctrl_idx, n_ctrl - 1)

        old_ctrl = self.assignments[sw_idx]
        migrated = (old_ctrl != ctrl_idx)
        if migrated:
            self.migration_count += 1
        self.assignments[sw_idx] = ctrl_idx

        # Fault injection scaled to current stage
        self.failed_ctrl = -1
        if self.rng.random() < self.fault_prob and n_ctrl > 1:
            self.failed_ctrl = self.rng.integers(0, n_ctrl)

        if self.rng.random() < 0.02:
            self.traffic_profile = self.rng.choice(self.active_profiles)

        state = self._get_state()
        reward = self._compute_reward(migrated)
        self.step_count += 1
        done = (self.step_count >= self.max_steps)
        info = {
            'icd_ms': self._compute_icd(),
            'load_std': self._compute_load_std(),
            'profile': self.traffic_profile,
            'curriculum_stage': self._get_stage_id(),
        }
        return state, reward, done, info

    def _get_stage_id(self):
        ep = self.global_episode
        if ep < 500:  return 0
        if ep < 2000: return 1
        return 2

    def get_action_mask(self):
        n_sw = self.active_n_sw
        n_ctrl = self.active_n_ctrl
        # Full action space is still N_SWITCHES * N_CONTROLLERS = 100
        # But mask out actions for inactive switches/controllers
        mask = np.zeros(N_SWITCHES * N_CONTROLLERS, dtype=bool)
        loads = self._compute_load()
        for sw_idx in range(n_sw):
            for ctrl_idx in range(n_ctrl):
                # Valid if not overloaded
                if loads[ctrl_idx] + 1.0 <= MAX_CTRL_LOAD * 2:
                    mask[sw_idx * N_CONTROLLERS + ctrl_idx] = True
        # Inactive switch-ctrl combinations stay False
        return mask
```

---

## Changes to dqn/train.py (Phase 1)

```python
def train_phase1(seed, curriculum=True):
    env = CitiverseSimEnv(seed=seed, curriculum=curriculum)
    agent = DQNAgent(...)

    for ep in range(P1_EPISODES):
        env.set_episode(ep)          # UPDATE CURRICULUM STAGE
        state = env.reset()
        ...

        # Log curriculum stage transitions
        if ep in (500, 2000):
            n_sw, n_ctrl, profiles = env._get_curriculum_params()
            LOG.info(f'[P1 seed={seed}] Curriculum stage change at ep={ep}: '
                     f'{n_sw} switches, {n_ctrl} controllers, {profiles}')
```

---

## Action space note during curriculum

During curriculum stages 0 and 1, the effective action space is smaller
(5×2=10, 10×3=30) but the model still has ACTION_DIM=100 outputs.
Action masking (from TASK_04) handles this correctly — inactive pairs are masked.
The agent learns to ignore masked actions naturally.

**Alternative (simpler):** disable curriculum if it complicates debugging.
Use `CitiverseSimEnv(curriculum=False)` for baseline runs.
Full curriculum is the recommended setting.

---

## Expected impact

| Curriculum | Loss<0.3 by episode | Final ICD (SimEnv) |
|---|---|---|
| OFF | ep ~1500-2000 | ~28-35ms |
| ON  | ep ~400-700   | ~22-28ms |

Faster convergence = shorter Phase 1 (could reduce from 5000 to 3000 ep).
Monitor convergence and stop early if loss < 0.2 for 200 consecutive episodes.

---

## Early stopping (add to train_phase1)

```python
PATIENCE = 200
best_loss = float('inf')
patience_counter = 0

for ep in range(P1_EPISODES):
    ...
    if loss is not None:
        if loss < best_loss - 0.01:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE and loss < 0.2:
            LOG.info(f'Early stopping at ep={ep}, loss={loss:.4f}')
            break
```
