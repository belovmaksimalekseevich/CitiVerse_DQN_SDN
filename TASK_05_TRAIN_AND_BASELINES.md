# TASK_05: train.py + baselines/run_baselines.py

## Step 9 — dqn/train.py

Two-phase training:
- **Phase 1**: CitiverseSimEnv × 5000 ep × 3 seeds (~4 min total)
- **Phase 2**: CitiverseRealEnv × 200 ep × 3 seeds per seed (~2-3h)

```python
# dqn/train.py
import os, json, time, logging, numpy as np, torch
from dqn.agent import DQNAgent
from dqn.sim_environment import CitiverseSimEnv
from topology.topology_data import N_SWITCHES, N_CONTROLLERS, STATE_DIM

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

ACTION_DIM   = N_SWITCHES * N_CONTROLLERS  # 100
SEEDS        = [42, 123, 456]
RESULTS_DIR  = 'results'

# Phase 1 hyperparams
P1_EPISODES     = 5000
P1_MAX_STEPS    = 200
P1_BUFFER       = 100_000
P1_BATCH        = 256
P1_HIDDEN       = 256
P1_LR           = 3e-4

# Phase 2 hyperparams (fine-tuning)
P2_EPISODES     = 200
P2_MAX_STEPS    = 500
P2_BATCH        = 128
P2_LR           = 1e-4  # lower LR for fine-tuning


def train_phase1(seed):
    """SimEnv pre-training. Returns trained agent checkpoint path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    total_steps = P1_EPISODES * P1_MAX_STEPS
    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden=P1_HIDDEN,
        lr=P1_LR,
        gamma=0.99,
        batch_size=P1_BATCH,
        buffer_size=P1_BUFFER,
        target_update_freq=100,
        n_step=3,
        total_steps=total_steps,
    )
    env = CitiverseSimEnv(seed=seed)

    ep_rewards, ep_icds = [], []
    global_step = 0

    for ep in range(P1_EPISODES):
        state = env.reset()
        ep_reward = 0.0
        ep_icd = []

        for _ in range(P1_MAX_STEPS):
            mask = env.get_action_mask()
            action = agent.select_action(state, action_mask=mask)
            next_state, reward, done, info = env.step(action)
            agent.push(state, action, reward, next_state, done)
            loss = agent.update()
            state = next_state
            ep_reward += reward
            ep_icd.append(info['icd_ms'])
            global_step += 1
            if done:
                break

        ep_rewards.append(ep_reward)
        ep_icds.append(np.mean(ep_icd))

        if (ep + 1) % 500 == 0:
            LOG.info(f'[P1 seed={seed}] ep={ep+1}/{P1_EPISODES} '
                     f'reward={np.mean(ep_rewards[-100:]):.3f} '
                     f'icd={np.mean(ep_icds[-100:]):.2f}ms '
                     f'eps={agent.eps:.3f}')

    ckpt_path = f'{RESULTS_DIR}/p1_seed{seed}.pth'
    agent.save(ckpt_path)
    np.save(f'{RESULTS_DIR}/p1_rewards_seed{seed}.npy', ep_rewards)
    np.save(f'{RESULTS_DIR}/p1_icds_seed{seed}.npy', ep_icds)
    LOG.info(f'[P1 seed={seed}] done. Saved {ckpt_path}')
    return ckpt_path, agent


def train_phase2(seed, p1_ckpt_path, net, ctrl_mgr):
    """Fine-tune on real Mininet env. Load P1 checkpoint."""
    from dqn.environment import CitiverseRealEnv
    total_steps = P2_EPISODES * P2_MAX_STEPS
    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden=P1_HIDDEN,
        lr=P2_LR,
        gamma=0.99,
        batch_size=P2_BATCH,
        buffer_size=50_000,
        target_update_freq=50,
        n_step=3,
        total_steps=total_steps,
        eps_start=0.3,  # start with lower eps (already pretrained)
        eps_end=0.02,
    )
    agent.load(p1_ckpt_path)

    env = CitiverseRealEnv(net, ctrl_mgr, seed=seed, max_steps=P2_MAX_STEPS)

    ep_rewards, ep_icds = [], []
    profiles = ['morning', 'business', 'evening', 'night']

    for ep in range(P2_EPISODES):
        # Cycle through traffic profiles
        profile = profiles[ep % len(profiles)]
        state = env.reset()
        env.set_traffic_profile(profile)
        ep_reward, ep_icd = 0.0, []

        for _ in range(P2_MAX_STEPS):
            mask = env.get_action_mask()
            action = agent.select_action(state, action_mask=mask)
            next_state, reward, done, info = env.step(action)
            agent.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            ep_reward += reward
            ep_icd.append(info['icd_ms'])
            if done:
                break

        ep_rewards.append(ep_reward)
        ep_icds.append(np.mean(ep_icd))

        if (ep + 1) % 20 == 0:
            LOG.info(f'[P2 seed={seed}] ep={ep+1}/{P2_EPISODES} '
                     f'reward={np.mean(ep_rewards[-20:]):.3f} '
                     f'icd={np.mean(ep_icds[-20:]):.2f}ms '
                     f'profile={profile}')

    ckpt_path = f'{RESULTS_DIR}/p2_seed{seed}.pth'
    agent.save(ckpt_path)
    np.save(f'{RESULTS_DIR}/p2_rewards_seed{seed}.npy', ep_rewards)
    np.save(f'{RESULTS_DIR}/p2_icds_seed{seed}.npy', ep_icds)
    env.close()
    return ckpt_path, agent


def evaluate_agent(agent, env, n_episodes=20):
    """Eval protocol: greedy policy (eps=0), separate from training."""
    icd_list, reward_list, load_std_list = [], [], []
    for ep in range(n_episodes):
        state = env.reset()
        ep_reward, ep_icd, ep_std = 0.0, [], []
        for _ in range(300):
            mask = env.get_action_mask()
            action = agent.select_action(state, action_mask=mask, deterministic=True)
            state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_icd.append(info['icd_ms'])
            ep_std.append(info.get('load_std', 0.0))
            if done:
                break
        icd_list.append(np.mean(ep_icd))
        reward_list.append(ep_reward)
        load_std_list.append(np.mean(ep_std))
    return {
        'icd_mean': np.mean(icd_list),
        'icd_std':  np.std(icd_list),
        'reward_mean': np.mean(reward_list),
        'load_std_mean': np.mean(load_std_list),
    }
```

---

## Step 10 — baselines/run_baselines.py

Fixed mislabeled baselines (FIX B13, B14).

```python
# baselines/run_baselines.py
import numpy as np
from sklearn.cluster import KMeans
from topology.topology_data import (
    SWITCHES, CONTROLLERS, SW_ZONE, ZONE_CTRL, CTRL_ZONE,
    LATENCY_MATRIX, N_SWITCHES, N_CONTROLLERS, MAX_CTRL_LOAD
)

# ------------------------------------------------------------------
# Baseline 1: AllToCtrl0 — all switches → controller 0
# (renamed from "CentralizedSDN" which was misleadingly labelled)
def baseline_all_to_ctrl0():
    return np.zeros(N_SWITCHES, dtype=int)

# ------------------------------------------------------------------
# Baseline 2: ZoneOptimal — assign each switch to its zone's controller
# (this is what old "CentralizedSDN" was actually doing — now correctly labelled)
def baseline_zone_optimal():
    return np.array([ZONE_CTRL[SW_ZONE[sw]] for sw in SWITCHES], dtype=int)

# ------------------------------------------------------------------
# Baseline 3: LoadBalanced — round-robin across controllers
def baseline_load_balanced():
    assignments = np.zeros(N_SWITCHES, dtype=int)
    loads = np.zeros(N_CONTROLLERS)
    for i in range(N_SWITCHES):
        c = int(np.argmin(loads))
        assignments[i] = c
        loads[c] += 1.0
    return assignments

# ------------------------------------------------------------------
# Baseline 4: KMeans — cluster switches by latency to controllers
def baseline_kmeans(n_clusters=5, seed=42):
    # Feature matrix: latency from each switch to each controller
    X = np.array(LATENCY_MATRIX, dtype=np.float32)  # (20, 5)
    # Use switch position relative to controllers as features
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    return labels.astype(int)

# ------------------------------------------------------------------
# Baseline 5: Random — uniform random assignment (correct labelling, was "StaticClustering")
def baseline_random(seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(0, N_CONTROLLERS, size=N_SWITCHES)

# ------------------------------------------------------------------
def compute_icd(assignments):
    total = sum(LATENCY_MATRIX[i][assignments[i]] for i in range(N_SWITCHES))
    return total / N_SWITCHES

def compute_load_std(assignments):
    loads = np.zeros(N_CONTROLLERS)
    for c in assignments:
        loads[c] += 1.0
    return float(np.std(loads))

def compute_zone_match(assignments):
    match = sum(
        1 for i, sw in enumerate(SWITCHES)
        if SW_ZONE[sw] == CTRL_ZONE.get(assignments[i], '')
    )
    return match / N_SWITCHES

def run_all_baselines():
    baselines = {
        'AllToCtrl0':    baseline_all_to_ctrl0(),
        'ZoneOptimal':   baseline_zone_optimal(),
        'LoadBalanced':  baseline_load_balanced(),
        'KMeans':        baseline_kmeans(),
        'Random':        baseline_random(),
    }
    results = {}
    for name, assignments in baselines.items():
        results[name] = {
            'icd_ms':       round(compute_icd(assignments), 3),
            'load_std':     round(compute_load_std(assignments), 3),
            'zone_match':   round(compute_zone_match(assignments), 3),
            'assignments':  assignments.tolist(),
        }
        print(f'{name:20s}  ICD={results[name]["icd_ms"]:.2f}ms  '
              f'load_std={results[name]["load_std"]:.2f}  '
              f'zone_match={results[name]["zone_match"]:.2f}')
    return results

if __name__ == '__main__':
    import json
    res = run_all_baselines()
    with open('results/baselines.json', 'w') as f:
        json.dump(res, f, indent=2)
    print('Saved results/baselines.json')
```

---

## Training Launch Commands (run as root)

```bash
# Terminal 1: start Mininet (Phase 2 only)
cd /home/maksim/dqn_simenv_mininet
source ../dqn_env/bin/activate
sudo mn --custom topology/citiverse_topo.py --topo CitiverseTopo --controller remote --switch ovsk,protocols=OpenFlow13

# Terminal 2: start 5 Ryu controllers (Phase 2 only)
source ../dqn_env/bin/activate
python3 -c "
from ryu_apps.multi_controller import MultiControllerManager
m = MultiControllerManager()
m.start_all()
import time; time.sleep(3600)
"

# Terminal 3: Run Phase 1 (SimEnv, no root needed, ~4 min)
source ../dqn_env/bin/activate
python3 -c "
from dqn.train import train_phase1, SEEDS
for seed in SEEDS:
    train_phase1(seed)
print('Phase 1 done for all seeds')
"

# Terminal 3 (after Phase 1): Run Phase 2 (~2-3h per seed)
sudo python3 -c "
from mininet.net import Mininet
from topology.citiverse_topo import CitiverseTopo
from ryu_apps.multi_controller import MultiControllerManager
from dqn.train import train_phase2, SEEDS
import os

ctrl_mgr = MultiControllerManager()
ctrl_mgr.start_all()
net = Mininet(topo=CitiverseTopo(), controller=None)
net.start()

for seed in SEEDS:
    ckpt = f'results/p1_seed{seed}.pth'
    train_phase2(seed, ckpt, net, ctrl_mgr)

net.stop()
ctrl_mgr.stop_all()
"
```
