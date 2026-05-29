# TASK_11: Catastrophic Forgetting Prevention in Phase 2 (extends TASK_05)

## Context

From PLAN_AND_REVIEW.md Section 2.5 (Sim-to-Real Transfer):
"Catastrophic forgetting during fine-tuning: reduce LR by 5x for Phase 2.
Mixing transitions: every 20 episodes of Phase 2, add 20% SimEnv transitions
to replay buffer."

Problem: During Phase 2 (RealEnv fine-tuning), the agent adapts to real network
conditions but forgets the general policy learned in Phase 1. This causes:
- Sharp performance dip in first 20 Phase 2 episodes
- Instability if real ICD varies widely between episodes
- Risk of complete forgetting if real ICD distribution differs significantly from SimEnv

---

## Changes to dqn/train.py

### Modified train_phase2 with anti-forgetting:

```python
def train_phase2(seed, p1_ckpt_path, net, ctrl_mgr, mix_ratio=0.2, mix_every=20):
    """
    Fine-tune on real Mininet env.
    mix_ratio: fraction of SimEnv transitions to mix per update
    mix_every: mix SimEnv transitions every N episodes
    """
    from dqn.sim_environment import CitiverseSimEnv

    total_steps = P2_EPISODES * P2_MAX_STEPS
    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden=P1_HIDDEN,
        lr=P2_LR,           # P2_LR = P1_LR / 5 = 6e-5 (FIX: lower for fine-tuning)
        gamma=0.99,
        batch_size=P2_BATCH,
        buffer_size=50_000,
        target_update_freq=50,
        n_step=3,
        total_steps=total_steps,
        eps_start=0.3,
        eps_end=0.02,
        checkpoint_path=f'results/p2_best_seed{seed}.pth',
    )
    agent.load(p1_ckpt_path)
    LOG.info(f'Loaded Phase 1 checkpoint: {p1_ckpt_path}')

    real_env = CitiverseRealEnv(net, ctrl_mgr, seed=seed, max_steps=P2_MAX_STEPS)
    sim_env = CitiverseSimEnv(seed=seed + 1000)  # for mixing

    ep_rewards, ep_icds = [], []
    profiles = ['morning', 'business', 'evening', 'night']

    # Pre-fill replay with some SimEnv transitions (warm start)
    LOG.info('Pre-filling replay buffer with SimEnv transitions...')
    _prefill_from_sim(agent, sim_env, n_transitions=2000)

    for ep in range(P2_EPISODES):
        profile = profiles[ep % len(profiles)]
        state = real_env.reset()
        real_env.set_traffic_profile(profile)
        sim_env.traffic_profile = profile  # keep profiles in sync

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

        # Mix SimEnv transitions every mix_every episodes (anti-forgetting)
        if (ep + 1) % mix_every == 0:
            n_mix = int(P2_MAX_STEPS * mix_ratio)
            _mix_sim_transitions(agent, sim_env, n_mix)
            LOG.debug(f'[P2] Mixed {n_mix} SimEnv transitions at ep={ep+1}')

        agent.record_episode_reward(ep_reward)
        ep_rewards.append(ep_reward)
        ep_icds.append(np.mean(ep_icd))

        if (ep + 1) % 20 == 0:
            LOG.info(f'[P2 seed={seed}] ep={ep+1}/{P2_EPISODES} '
                     f'reward={np.mean(ep_rewards[-20:]):.3f} '
                     f'icd={np.mean(ep_icds[-20:]):.2f}ms '
                     f'eps={agent.eps:.3f} profile={profile}')

    ckpt_path = f'results/p2_seed{seed}.pth'
    agent.save(ckpt_path)
    np.save(f'results/p2_rewards_seed{seed}.npy', ep_rewards)
    np.save(f'results/p2_icds_seed{seed}.npy', ep_icds)
    real_env.close()
    return ckpt_path, agent


def _prefill_from_sim(agent, sim_env, n_transitions):
    """Warm-start Phase 2 replay buffer with SimEnv transitions."""
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
    LOG.info(f'Pre-filled {collected} SimEnv transitions into replay buffer')


def _mix_sim_transitions(agent, sim_env, n_transitions):
    """Add SimEnv transitions directly to agent's replay buffer."""
    state = sim_env.reset()
    for _ in range(n_transitions):
        mask = sim_env.get_action_mask()
        # Use greedy policy for mixing (no exploration noise from old policy)
        action = agent.select_action(state, action_mask=mask, deterministic=True)
        next_state, reward, done, info = sim_env.step(action)
        # Push directly to replay (bypass n_step buffer for mixing)
        agent.replay.push(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = sim_env.reset()
```

---

## LR Schedule for Phase 2

PLAN_AND_REVIEW.md: "reduce LR by 5x for Phase 2 fine-tuning"

```python
# In constants section of train.py:
P1_LR = 3e-4    # Phase 1
P2_LR = 6e-5    # Phase 2 = P1_LR / 5  (catastrophic forgetting prevention)
```

The cosine annealing scheduler in DQNAgent also restarts in Phase 2
(new optimizer is created when loading checkpoint — schedule resets automatically).

---

## Monitoring

Track these metrics to confirm anti-forgetting works:

```python
# At start of each Phase 2 episode, eval SimEnv performance (no gradient):
def _eval_sim_performance(agent, sim_env, n=5):
    """Check agent hasn't forgotten SimEnv policy."""
    icds = []
    for _ in range(n):
        state = sim_env.reset()
        for _ in range(100):
            action = agent.select_action(state, deterministic=True)
            state, _, done, info = sim_env.step(action)
            if done:
                icds.append(info['icd_ms'])
                break
    return np.mean(icds) if icds else 0.0
```

Log `sim_icd` alongside `real_icd` every 20 episodes in Phase 2.
If `sim_icd` increases sharply (>50% from Phase 1 final), increase `mix_ratio`.

---

## Expected behavior

| Phase 2 ep | Without anti-forgetting | With anti-forgetting |
|---|---|---|
| 0-20 | ICD drops, then spikes +40% | ICD drops steadily |
| 20-50 | Unstable oscillation | Stable improvement |
| 50-100 | May diverge or plateau | Converges to real ICD optimum |
| SimEnv ICD | Degrades 30-50% | Stays within 10-15% of Phase 1 |
