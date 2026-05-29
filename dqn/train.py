# dqn/train.py
import os
import time
import logging
import numpy as np
from dqn.agent import DQNAgent
from dqn.sim_environment import CitiverseSimEnv
from topology.topology_data import N_SWITCHES, N_CONTROLLERS, STATE_DIM

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

ACTION_DIM  = N_SWITCHES * N_CONTROLLERS   # 100
SEEDS       = [42, 123, 456]
RESULTS_DIR = 'results'

# Phase 1
P1_EPISODES        = 3000
P1_MAX_STEPS       = 200
P1_BUFFER          = 100_000
P1_BATCH           = 256
P1_HIDDEN          = 256
P1_LR              = 3e-4

# Phase 2
P2_EPISODES        = 200
P2_MAX_STEPS        = 200
P2_BATCH           = 128
P2_LR              = 6e-5    # P1_LR / 5 — anti-forgetting
P2_MIX_RATIO       = 0.2     # fraction of SimEnv steps to mix
P2_MIX_EVERY       = 20      # mix every N episodes
P2_PREFILL         = 2000    # warm-start transitions from SimEnv


# ---------------------------------------------------------------------------
# Phase 1 — SimEnv pre-training
# ---------------------------------------------------------------------------

def train_phase1(seed):
    """SimEnv pre-training with curriculum. Returns (ckpt_path, agent)."""
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
        eps_start=1.0,
        eps_end=0.05,
        checkpoint_path=f'{RESULTS_DIR}/p1_best_seed{seed}.pth',
    )
    env = CitiverseSimEnv(seed=seed, curriculum=True)

    ep_rewards, ep_icds = [], []
    patience_counter = 0
    best_loss = float('inf')

    LOG.info(f'[P1 seed={seed}] Starting Phase 1 ({P1_EPISODES} episodes)')
    t0 = time.time()

    for ep in range(P1_EPISODES):
        env.set_episode(ep)
        state = env.reset()
        ep_reward, ep_icd = 0.0, []

        for _ in range(P1_MAX_STEPS):
            mask = env.get_action_mask()
            action = agent.select_action(state, action_mask=mask)
            next_state, reward, done, info = env.step(action)
            agent.push(state, action, reward, next_state, done)
            loss = agent.update()
            state = next_state
            ep_reward += reward
            ep_icd.append(info['icd_ms'])
            if done:
                break

        ep_rewards.append(ep_reward)
        ep_icds.append(float(np.mean(ep_icd)))
        agent.record_episode_reward(ep_reward)

        # Curriculum stage transition logging
        if ep in (300, 800):
            n_sw, n_ctrl, profiles = env._get_curriculum_params()
            LOG.info(f'[P1 seed={seed}] Curriculum stage at ep={ep}: '
                     f'{n_sw}sw {n_ctrl}ctrl {profiles}')

        # Early stopping DISABLED. The loss-based stop fired around ep~1000,
        # BEFORE curriculum Stage 3 (full 20sw/5ctrl topology) and while eps
        # was still ~0.77 (policy mostly random) -> training never learned the
        # real problem. Fixed P1_EPISODES bounds the runtime instead.
        _ = (loss, best_loss, patience_counter)

        if (ep + 1) % 500 == 0:
            elapsed = time.time() - t0
            LOG.info(f'[P1 seed={seed}] ep={ep+1}/{P1_EPISODES} '
                     f'reward={np.mean(ep_rewards[-100:]):.3f} '
                     f'icd={np.mean(ep_icds[-100:]):.2f}ms '
                     f'eps={agent.eps:.3f} t={elapsed:.0f}s')

    ckpt_path = f'{RESULTS_DIR}/p1_seed{seed}.pth'
    agent.save(ckpt_path)
    np.save(f'{RESULTS_DIR}/p1_rewards_seed{seed}.npy', np.array(ep_rewards))
    np.save(f'{RESULTS_DIR}/p1_icds_seed{seed}.npy', np.array(ep_icds))
    LOG.info(f'[P1 seed={seed}] Done in {time.time()-t0:.0f}s. Saved {ckpt_path}')
    return ckpt_path, agent


# ---------------------------------------------------------------------------
# Phase 2 — RealEnv fine-tuning with anti-forgetting
# ---------------------------------------------------------------------------

def train_phase2(seed, p1_ckpt_path, net, ctrl_mgr, monitor=None):
    """Fine-tune on real Mininet env with SimEnv mixing (anti-forgetting)."""
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
        eps_start=0.3,
        eps_end=0.02,
        checkpoint_path=f'{RESULTS_DIR}/p2_best_seed{seed}.pth',
    )
    agent.load(p1_ckpt_path)
    LOG.info(f'[P2 seed={seed}] Loaded P1 checkpoint: {p1_ckpt_path}')

    real_env = CitiverseRealEnv(net, ctrl_mgr, ctrl_monitor=monitor, seed=seed, max_steps=P2_MAX_STEPS)
    sim_env = CitiverseSimEnv(seed=seed + 1000, curriculum=False)

    ep_rewards, ep_icds = [], []
    profiles = ['morning', 'business', 'evening', 'night']

    # Warm-start: pre-fill replay with SimEnv transitions
    LOG.info(f'[P2 seed={seed}] Pre-filling {P2_PREFILL} SimEnv transitions...')
    _prefill_from_sim(agent, sim_env, P2_PREFILL)

    t0 = time.time()
    for ep in range(P2_EPISODES):
        profile = profiles[ep % len(profiles)]
        state = real_env.reset()
        real_env.set_traffic_profile(profile)
        sim_env.traffic_profile = profile

        ep_reward, ep_icd = 0.0, []

        for _ in range(P2_MAX_STEPS):
            mask = real_env.get_action_mask()
            action = agent.select_action(state, action_mask=mask)
            next_state, reward, done, info = real_env.step(action)
            agent.push(state, action, reward, next_state, done)
            loss = agent.update()
            agent.maybe_auto_reset(loss)
            state = next_state
            ep_reward += reward
            ep_icd.append(info['icd_ms'])
            if done:
                break

        # Anti-forgetting: mix SimEnv transitions every P2_MIX_EVERY episodes
        if (ep + 1) % P2_MIX_EVERY == 0:
            n_mix = int(P2_MAX_STEPS * P2_MIX_RATIO)
            _mix_sim_transitions(agent, sim_env, n_mix)
            LOG.debug(f'[P2 seed={seed}] Mixed {n_mix} SimEnv transitions at ep={ep+1}')

        agent.record_episode_reward(ep_reward)
        ep_rewards.append(ep_reward)
        ep_icds.append(float(np.mean(ep_icd)))

        if (ep + 1) % 20 == 0:
            sim_icd = _eval_sim_performance(agent, sim_env)
            LOG.info(f'[P2 seed={seed}] ep={ep+1}/{P2_EPISODES} '
                     f'real_icd={np.mean(ep_icds[-20:]):.2f}ms '
                     f'sim_icd={sim_icd:.2f}ms '
                     f'reward={np.mean(ep_rewards[-20:]):.3f} '
                     f'eps={agent.eps:.3f} profile={profile} '
                     f't={time.time()-t0:.0f}s')

    ckpt_path = f'{RESULTS_DIR}/p2_seed{seed}.pth'
    agent.save(ckpt_path)
    np.save(f'{RESULTS_DIR}/p2_rewards_seed{seed}.npy', np.array(ep_rewards))
    np.save(f'{RESULTS_DIR}/p2_icds_seed{seed}.npy', np.array(ep_icds))
    real_env.close()
    LOG.info(f'[P2 seed={seed}] Done. Saved {ckpt_path}')
    return ckpt_path, agent


def _prefill_from_sim(agent, sim_env, n_transitions):
    state = sim_env.reset()
    collected = 0
    while collected < n_transitions:
        mask = sim_env.get_action_mask()
        action = agent.select_action(state, action_mask=mask)
        next_state, reward, done, info = sim_env.step(action)
        agent.push(state, action, reward, next_state, done)
        state = next_state
        collected += 1
        if done:
            state = sim_env.reset()
    LOG.info(f'Pre-filled {collected} SimEnv transitions')


def _mix_sim_transitions(agent, sim_env, n_transitions):
    state = sim_env.reset()
    for _ in range(n_transitions):
        mask = sim_env.get_action_mask()
        action = agent.select_action(state, action_mask=mask, deterministic=True)
        next_state, reward, done, info = sim_env.step(action)
        agent.replay.push(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = sim_env.reset()


def _eval_sim_performance(agent, sim_env, n=5):
    """Quick eval on SimEnv to monitor forgetting."""
    icds = []
    for _ in range(n):
        state = sim_env.reset()
        for _ in range(100):
            action = agent.select_action(state, deterministic=True)
            state, _, done, info = sim_env.step(action)
            if done:
                icds.append(info['icd_ms'])
                break
    return float(np.mean(icds)) if icds else 0.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(agent, env, n_episodes=20):
    """Greedy policy eval. Returns overall + per-profile ICD metrics."""
    PROFILES = ['morning', 'business', 'evening', 'night']
    icd_list, reward_list, load_std_list = [], [], []
    profile_icds = {p: [] for p in PROFILES}
    profile_stds = {p: [] for p in PROFILES}

    for ep in range(n_episodes):
        profile = PROFILES[ep % len(PROFILES)]
        # Set profile if env supports it
        if hasattr(env, 'set_traffic_profile'):
            env.set_traffic_profile(profile)
        elif hasattr(env, 'traffic_profile'):
            env.traffic_profile = profile

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
        ep_mean_icd = float(np.mean(ep_icd))
        icd_list.append(ep_mean_icd)
        reward_list.append(ep_reward)
        load_std_list.append(float(np.mean(ep_std)))
        profile_icds[profile].append(ep_mean_icd)
        profile_stds[profile].append(float(np.mean(ep_std)))

    result = {
        'icd_mean':      float(np.mean(icd_list)),
        'icd_std':       float(np.std(icd_list)),
        'reward_mean':   float(np.mean(reward_list)),
        'load_std_mean': float(np.mean(load_std_list)),
        'icd_per_profile': {
            p: {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                'load_std': float(np.mean(profile_stds[p])) if profile_stds[p] else 0.0}
            for p, v in profile_icds.items() if v
        },
    }
    return result


# ---------------------------------------------------------------------------
# Failure-resilience evaluation (controller fault)
# ---------------------------------------------------------------------------

def evaluate_failure_resilience(agent, seed, dead_ctrl=0,
                                settle=80, react=80, n_per_profile=5):
    """Sustained controller-failure resilience under the queuing-aware ICD.
    The agent first settles to a good assignment, then controller `dead_ctrl`
    fails permanently; we measure post-failure ICD as the agent re-optimises
    online, versus a static zone-optimal "dumb failover" (dead controller's
    switches dumped onto its neighbour). Returns per-profile dict."""
    from dqn.sim_environment import CitiverseSimEnv
    from topology.topology_data import (
        SWITCHES, N_SWITCHES, N_CONTROLLERS, LATENCY_MATRIX, ZONE_LOAD_FACTORS,
        SW_ZONE, queuing_delay_ms, get_zone_optimal_array,
    )
    PROFILES = ['morning', 'business', 'evening', 'night']

    def static_failover_icd(profile):
        asg = get_zone_optimal_array().copy()
        for i in range(N_SWITCHES):
            if asg[i] == dead_ctrl:
                asg[i] = (dead_ctrl + 1) % N_CONTROLLERS
        zf = ZONE_LOAD_FACTORS[profile]
        L = [0.0] * N_CONTROLLERS
        for i in range(N_SWITCHES):
            L[int(asg[i])] += zf[SW_ZONE[SWITCHES[i]]]
        tot = sum(float(LATENCY_MATRIX[i][int(asg[i])]) + queuing_delay_ms(L[int(asg[i])])
                  for i in range(N_SWITCHES))
        return tot / N_SWITCHES

    env = CitiverseSimEnv(seed=seed + 7000, fault_prob=0.0, curriculum=False)
    out = {}
    for p in PROFILES:
        dqn_icds, static_icds = [], []
        for _ in range(n_per_profile):
            env.forced_failure = -1
            env.reset()
            env.traffic_profile = p
            env.active_profiles = [p]
            state = env._get_state()
            info = {'icd_ms': 0.0}
            for _ in range(settle):
                a = agent.select_action(state, action_mask=env.get_action_mask(),
                                        deterministic=True)
                state, _, _, info = env.step(a)
            env.forced_failure = dead_ctrl          # inject sustained failure
            for _ in range(react):
                a = agent.select_action(state, action_mask=env.get_action_mask(),
                                        deterministic=True)
                state, _, _, info = env.step(a)
            dqn_icds.append(info['icd_ms'])
            static_icds.append(static_failover_icd(p))
        dm, sm = float(np.mean(dqn_icds)), float(np.mean(static_icds))
        out[p] = {'dqn_icd': round(dm, 3), 'static_zo_icd': round(sm, 3),
                  'improvement_pct': round((sm - dm) / sm * 100, 1) if sm > 0 else 0.0}
    return out
