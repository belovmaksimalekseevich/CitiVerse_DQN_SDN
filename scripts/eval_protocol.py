# scripts/eval_protocol.py
"""
Formal evaluation protocol: 3 seeds × 4 profiles × 20 episodes.
Run after training is complete (no gradient updates here).
"""
import os
import json
import numpy as np
from scipy import stats

SEEDS        = [42, 123, 456]
N_EVAL_EPS   = 20
PROFILES     = ['morning', 'business', 'evening', 'night']
RESULTS_DIR  = 'results'


def eval_dqn_agent(agent, env, n_episodes=N_EVAL_EPS, profile=None):
    saved_eps = agent.eps
    agent.eps = 0.0

    icd_list, load_std_list = [], []
    for _ in range(n_episodes):
        if profile and hasattr(env, 'set_traffic_profile'):
            env.set_traffic_profile(profile)
        elif profile and hasattr(env, 'traffic_profile'):
            env.traffic_profile = profile
        state = env.reset()
        ep_icd = []
        for _ in range(300):
            mask = env.get_action_mask()
            action = agent.select_action(state, action_mask=mask, deterministic=True)
            state, _, done, info = env.step(action)
            ep_icd.append(info['icd_ms'])
            if done:
                break
        icd_list.append(float(np.mean(ep_icd)))
        load_std_list.append(float(info.get('load_std', 0.0)))

    agent.eps = saved_eps
    return {
        'icd_mean':      float(np.mean(icd_list)),
        'icd_std':       float(np.std(icd_list)),
        'icd_median':    float(np.median(icd_list)),
        'load_std_mean': float(np.mean(load_std_list)),
        'n_episodes':    n_episodes,
    }


def run_full_evaluation(agents_by_seed, net=None, ctrl_mgr=None):
    """
    agents_by_seed: {seed: DQNAgent}
    Returns nested dict: {method: {profile: {mean, std}, 'overall': {mean, std}}}
    """
    from dqn.sim_environment import CitiverseSimEnv
    from baselines.run_baselines import (
        baseline_zone_optimal, baseline_load_balanced,
        baseline_all_to_ctrl0, baseline_random, baseline_kmeans,
        compute_icd,
    )

    all_results = {}

    # DQN evaluation
    dqn_per_profile = {p: [] for p in PROFILES}
    for seed, agent in agents_by_seed.items():
        if net is not None:
            from dqn.environment import CitiverseRealEnv
            env = CitiverseRealEnv(net, ctrl_mgr, seed=seed + 1000)
        else:
            env = CitiverseSimEnv(seed=seed + 1000)

        for profile in PROFILES:
            m = eval_dqn_agent(agent, env, n_episodes=N_EVAL_EPS, profile=profile)
            dqn_per_profile[profile].append(m['icd_mean'])

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

    # Static baselines — analytical (fast, no env needed)
    baseline_fns = {
        'ZoneOptimal':  baseline_zone_optimal,
        'LoadBalanced': baseline_load_balanced,
        'AllToCtrl0':   baseline_all_to_ctrl0,
        'KMeans':       baseline_kmeans,
        'Random':       lambda: baseline_random(seed=42),
    }
    for name, fn in baseline_fns.items():
        icd = compute_icd(fn())
        all_results[name] = {
            p: {'mean': round(icd, 3), 'std': 0.0} for p in PROFILES
        }
        all_results[name]['overall'] = {'mean': round(icd, 3), 'std': 0.0}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f'{RESULTS_DIR}/eval_full.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'Saved eval_full.json')

    return all_results


def statistical_significance(dqn_icds, baseline_icds, alpha=0.05):
    """One-sided Welch t-test: DQN ICD < baseline ICD."""
    t_stat, p_value = stats.ttest_ind(
        dqn_icds, baseline_icds, equal_var=False, alternative='less'
    )
    return float(t_stat), float(p_value), bool(p_value < alpha)


def print_results_table(all_results):
    print('\n=== TABLE 1: Mean ICD (ms) ± std per Method per Traffic Profile ===')
    header = f'{"Method":<18}' + ''.join(f'  {p:>12}' for p in PROFILES) + f'  {"Overall":>12}'
    print(header)
    print('-' * len(header))
    for method, res in all_results.items():
        row = f'{method:<18}'
        for p in PROFILES:
            pres = res.get(p, {})
            m = pres.get('mean', 0); s = pres.get('std', 0)
            row += f'  {m:>5.2f}±{s:<4.2f}'
        ov = res.get('overall', {})
        row += f'  {ov.get("mean", 0):>5.2f}±{ov.get("std", 0):<4.2f}'
        print(row)
    print()


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/maksim/dqn_simenv_mininet')
    from dqn.agent import DQNAgent
    from topology.topology_data import STATE_DIM, N_SWITCHES, N_CONTROLLERS

    agents = {}
    for seed in SEEDS:
        ckpt = f'{RESULTS_DIR}/p2_best_seed{seed}.pth'
        if not os.path.exists(ckpt):
            ckpt = f'{RESULTS_DIR}/p2_seed{seed}.pth'
        if not os.path.exists(ckpt):
            print(f'No P2 checkpoint for seed={seed}, trying P1...')
            ckpt = f'{RESULTS_DIR}/p1_best_seed{seed}.pth'
        if os.path.exists(ckpt):
            agent = DQNAgent(
                state_dim=STATE_DIM,
                action_dim=N_SWITCHES * N_CONTROLLERS,
                total_steps=1,
            )
            agent.load(ckpt)
            agent.eps = 0.0
            agents[seed] = agent

    if not agents:
        print('No checkpoints found — run training first.')
        sys.exit(1)

    results = run_full_evaluation(agents)
    print_results_table(results)
