# TASK_12: Evaluation Protocol + Ablation Study + Success Criteria

## Context

From PLAN_AND_REVIEW.md B20, B22, B23, B29, Section 3.2, Section 9:
- B20: Eval = Train (no separate evaluation protocol)
- B22: No reproducibility (no seeds, no mean±std)
- B23: No confidence intervals or statistical significance tests
- B29: No ablation study — can't show which component matters
- I16: Continuous operation evaluation mode

---

## File: scripts/eval_protocol.py

```python
# scripts/eval_protocol.py
"""
Formal evaluation protocol for DQN and all baselines.
Produces: mean±std ICD per method per traffic profile.
Runs separately from training — no gradient updates.
"""
import os, json, numpy as np
from scipy import stats

SEEDS = [42, 123, 456]
N_EVAL_EPISODES = 20   # per seed per profile
PROFILES = ['morning', 'business', 'evening', 'night']
RESULTS_DIR = 'results'


def eval_dqn_agent(agent, env, n_episodes=N_EVAL_EPISODES, profile=None):
    """
    Greedy evaluation (eps=0, no gradient).
    Returns dict with icd_mean, icd_std, load_std_mean, zone_match_mean.
    """
    saved_eps = agent.eps
    agent.eps = 0.0  # greedy

    icd_list, load_std_list, zone_match_list = [], [], []

    for ep in range(n_episodes):
        if profile:
            env.set_traffic_profile(profile)
        state = env.reset()
        ep_icd = []

        for _ in range(300):  # fixed eval horizon
            mask = env.get_action_mask()
            action = agent.select_action(state, action_mask=mask, deterministic=True)
            state, reward, done, info = env.step(action)
            ep_icd.append(info['icd_ms'])
            if done:
                break

        icd_list.append(np.mean(ep_icd))
        load_std_list.append(info.get('load_std', 0.0))

    agent.eps = saved_eps  # restore

    return {
        'icd_mean':      float(np.mean(icd_list)),
        'icd_std':       float(np.std(icd_list)),
        'icd_median':    float(np.median(icd_list)),
        'load_std_mean': float(np.mean(load_std_list)),
        'n_episodes':    n_episodes,
    }


def eval_baseline(assignments_fn, sim_env_class, n_episodes=N_EVAL_EPISODES, profile=None):
    """Evaluate a static baseline policy (no learning)."""
    from topology.topology_data import N_SWITCHES, N_CONTROLLERS

    icd_list = []
    env = sim_env_class(seed=0)

    for ep in range(n_episodes):
        assignments = assignments_fn()
        if profile:
            env.traffic_profile = profile
        state = env.reset()
        env.assignments[:] = assignments

        ep_icd = []
        for _ in range(300):
            # Static policy: pick NO-OP action (keep assignments)
            # Find action that keeps current assignment for first switch
            action = assignments[0] * 1  # sw_0 stays at its ctrl (no-op equivalent)
            state, reward, done, info = env.step(action)
            ep_icd.append(info['icd_ms'])
            if done:
                break
        icd_list.append(np.mean(ep_icd))

    return {
        'icd_mean':   float(np.mean(icd_list)),
        'icd_std':    float(np.std(icd_list)),
        'n_episodes': n_episodes,
    }


def run_full_evaluation(agents_by_seed, env_class, net=None, ctrl_mgr=None):
    """
    Full evaluation: 3 seeds × 4 profiles × 20 episodes each.
    Produces Table 1 for paper.
    """
    from dqn.sim_environment import CitiverseSimEnv
    from baselines.run_baselines import (
        baseline_zone_optimal, baseline_load_balanced,
        baseline_all_to_ctrl0, baseline_random, baseline_kmeans
    )

    all_results = {}

    # DQN evaluation
    dqn_per_profile = {p: [] for p in PROFILES}
    for seed, agent in agents_by_seed.items():
        if net is not None:
            from dqn.environment import CitiverseRealEnv
            env = CitiverseRealEnv(net, ctrl_mgr, seed=seed+1000)
        else:
            env = CitiverseSimEnv(seed=seed+1000)

        for profile in PROFILES:
            metrics = eval_dqn_agent(agent, env, n_episodes=N_EVAL_EPISODES, profile=profile)
            dqn_per_profile[profile].append(metrics['icd_mean'])

        if hasattr(env, 'close'):
            env.close()

    all_results['DQN'] = {
        p: {
            'mean': float(np.mean(dqn_per_profile[p])),
            'std':  float(np.std(dqn_per_profile[p])),
        }
        for p in PROFILES
    }
    all_results['DQN']['overall'] = {
        'mean': float(np.mean([v for vals in dqn_per_profile.values() for v in vals])),
        'std':  float(np.std([v for vals in dqn_per_profile.values() for v in vals])),
    }

    # Baseline evaluation (SimEnv only — fast, no Mininet needed)
    baseline_fns = {
        'ZoneOptimal':  baseline_zone_optimal,
        'LoadBalanced': baseline_load_balanced,
        'AllToCtrl0':   baseline_all_to_ctrl0,
        'KMeans':       baseline_kmeans,
        'Random':       lambda: baseline_random(seed=42),
    }

    sim_env = CitiverseSimEnv(seed=0)
    for name, fn in baseline_fns.items():
        per_profile = {}
        for profile in PROFILES:
            metrics = eval_baseline(fn, CitiverseSimEnv, profile=profile)
            per_profile[profile] = {'mean': metrics['icd_mean'], 'std': metrics['icd_std']}
        all_results[name] = per_profile

    return all_results


def statistical_significance(dqn_icds, baseline_icds, alpha=0.05):
    """
    Welch's t-test: DQN ICD < Baseline ICD (one-sided).
    Returns (t_stat, p_value, significant).
    For paper: need p < 0.05.
    """
    t_stat, p_value = stats.ttest_ind(dqn_icds, baseline_icds,
                                       equal_var=False, alternative='less')
    return float(t_stat), float(p_value), bool(p_value < alpha)


def print_results_table(all_results):
    """Print Table 1 in paper format."""
    print('\n=== TABLE 1: Mean ICD (ms) ± std per Method per Traffic Profile ===')
    print(f'{"Method":<18}', end='')
    for p in PROFILES:
        print(f'  {p:>12}', end='')
    print(f'  {"Overall":>12}')
    print('-' * (18 + 13 * 5))

    for method, res in all_results.items():
        print(f'{method:<18}', end='')
        for p in PROFILES:
            pres = res.get(p, {})
            m = pres.get('mean', 0)
            s = pres.get('std', 0)
            print(f'  {m:>5.2f}±{s:<4.2f}', end='')
        ov = res.get('overall', res.get(PROFILES[0], {}))
        print(f'  {ov.get("mean",0):>5.2f}±{ov.get("std",0):<4.2f}')

    print()
```

---

## File: scripts/ablation.py

```python
# scripts/ablation.py
"""
Ablation study: identifies contribution of each architectural component.
All variants trained on SimEnv only (Phase 1) for speed.
"""
import numpy as np, logging
LOG = logging.getLogger(__name__)

ABLATION_EPISODES = 2000   # shorter than full training — enough to see differences
ABLATION_SEED = 42

VARIANTS = {
    'Full_DQN':        dict(curriculum=True,  n_step=3, dueling=True,  masking=True,  per=True),
    'No_Curriculum':   dict(curriculum=False, n_step=3, dueling=True,  masking=True,  per=True),
    'No_NStep':        dict(curriculum=True,  n_step=1, dueling=True,  masking=True,  per=True),
    'No_Dueling':      dict(curriculum=True,  n_step=3, dueling=False, masking=True,  per=True),
    'No_Masking':      dict(curriculum=True,  n_step=3, dueling=True,  masking=False, per=True),
    'No_PER':          dict(curriculum=True,  n_step=3, dueling=True,  masking=True,  per=False),
    'No_SimPretrain':  None,  # Phase 2 only, no Phase 1
}


def run_ablation():
    from dqn.sim_environment import CitiverseSimEnv
    from dqn.agent import DQNAgent
    from dqn.model import DuelingDQN
    from topology.topology_data import STATE_DIM, N_SWITCHES, N_CONTROLLERS
    import torch.nn as nn

    results = {}
    ACTION_DIM = N_SWITCHES * N_CONTROLLERS

    for variant_name, config in VARIANTS.items():
        if config is None:
            LOG.info(f'Ablation: {variant_name} — requires Phase 2, skip for now')
            results[variant_name] = {'note': 'Phase 2 only, run separately'}
            continue

        LOG.info(f'Running ablation: {variant_name}')

        env = CitiverseSimEnv(seed=ABLATION_SEED, curriculum=config['curriculum'])
        total_steps = ABLATION_EPISODES * 200

        # Model: optionally disable dueling
        if config['dueling']:
            from dqn.model import DuelingDQN
            # standard
        else:
            # Flat DQN (no dueling)
            import torch.nn as nn
            class FlatDQN(nn.Module):
                def __init__(self, s, a, h=256):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(s, h), nn.LayerNorm(h), nn.ReLU(),
                        nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(),
                        nn.Linear(h, a),
                    )
                def forward(self, x, action_mask=None):
                    q = self.net(x)
                    if action_mask is not None:
                        q = q.masked_fill(~action_mask, -1e9)
                    return q

        agent = DQNAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            lr=3e-4,
            gamma=0.99,
            batch_size=256,
            n_step=config['n_step'],
            total_steps=total_steps,
            use_per=config['per'],
        )

        ep_icds = []
        for ep in range(ABLATION_EPISODES):
            env.set_episode(ep)
            state = env.reset()
            ep_icd = []
            for _ in range(200):
                mask = env.get_action_mask() if config['masking'] else None
                action = agent.select_action(state, action_mask=mask)
                next_s, r, done, info = env.step(action)
                agent.push(state, action, r, next_s, done)
                agent.update()
                state = next_s
                ep_icd.append(info['icd_ms'])
                if done: break
            ep_icds.append(np.mean(ep_icd))
            agent.record_episode_reward(sum(ep_icd))

        # Report last 200 episodes (converged performance)
        final_icds = ep_icds[-200:]
        results[variant_name] = {
            'icd_mean': float(np.mean(final_icds)),
            'icd_std':  float(np.std(final_icds)),
            'conv_ep':  int(next((i for i, v in enumerate(ep_icds) if v < 35), ABLATION_EPISODES)),
        }
        LOG.info(f'{variant_name}: ICD={results[variant_name]["icd_mean"]:.2f}±'
                 f'{results[variant_name]["icd_std"]:.2f}ms, '
                 f'conv@ep={results[variant_name]["conv_ep"]}')

    import json
    with open('results/ablation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\n=== TABLE 2: Ablation Study ===')
    print(f'{"Variant":<25} {"ICD (ms)":<14} {"Conv ep"}')
    print('-' * 50)
    for name, res in results.items():
        if 'note' in res:
            print(f'{name:<25} {res["note"]}')
        else:
            print(f'{name:<25} {res["icd_mean"]:.2f}±{res["icd_std"]:.2f}     '
                  f'{res["conv_ep"]}')
    return results


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    run_ablation()
```

---

## Success / Failure Criteria (Section 9 of PLAN_AND_REVIEW.md)

```
PASS:  mean ICD (DQN) < mean ICD (ZoneOptimal) for dynamic traffic, p < 0.05
PASS:  DQN converges by ep 300 SimEnv (loss drops below 0.3)
PASS:  DQN > Random AND > AllToCtrl0 across all 4 traffic profiles
PASS:  3 seeds produce consistent results: ICD std < 15% of ICD mean
BONUS: DQN adapts within 5 episodes after traffic profile change
BONUS: DQN handles fault injection — ICD recovers within 3 episodes
FAIL:  Loss oscillates during exploitation → reload best checkpoint, reduce LR
FAIL:  DQN does NOT beat ZoneOptimal in static traffic → expected and acceptable
       (paper claim is about DYNAMIC traffic only)
```

```python
# scripts/check_success_criteria.py
def check_criteria(all_results, ablation_results=None):
    dqn = all_results.get('DQN', {}).get('overall', {})
    zo  = all_results.get('ZoneOptimal', {}).get('overall', {})

    print('=== SUCCESS CRITERIA CHECK ===')

    # PASS 1: DQN < ZoneOptimal
    dqn_m = dqn.get('mean', 999)
    zo_m  = zo.get('mean', 999)
    pct_improvement = (zo_m - dqn_m) / zo_m * 100
    status = 'PASS' if dqn_m < zo_m else 'FAIL'
    print(f'[{status}] ICD: DQN={dqn_m:.2f}ms, ZoneOptimal={zo_m:.2f}ms '
          f'({pct_improvement:+.1f}%)')

    # PASS 2: consistency across seeds
    dqn_std = dqn.get('std', 999)
    consistency_ok = dqn_std < dqn_m * 0.15
    print(f'[{"PASS" if consistency_ok else "FAIL"}] Seed consistency: '
          f'std={dqn_std:.2f}ms ({dqn_std/dqn_m*100:.1f}% of mean)')

    # Statistical test
    import numpy as np
    # Placeholder — replace with actual per-episode ICD arrays
    print('[INFO] Run statistical_significance() from eval_protocol.py with raw arrays')
```

---

## Integration in run_all.py

```python
# After Phase 2 training:
from scripts.eval_protocol import run_full_evaluation, print_results_table
from scripts.check_success_criteria import check_criteria

eval_results = run_full_evaluation(agents_by_seed, ...)
print_results_table(eval_results)
check_criteria(eval_results)

# Ablation (SimEnv only, can run in parallel with Phase 2):
from scripts.ablation import run_ablation
ablation_results = run_ablation()
```
