#!/usr/bin/env python3
"""Analytical FALLBACK summary (SimEnv only, no Mininet -> reliable).
Produces results/analytical_summary.json so there is always a result by morning,
independent of the measured Mininet stage. Uses the queuing-aware ICD + load_std.
"""
import sys, os, json
sys.path.insert(0, '/home/maksim/simenv_dqn_mininet_v2')
import numpy as np
from dqn.train import evaluate_agent
from dqn.sim_environment import CitiverseSimEnv
from dqn.agent import DQNAgent
from topology.topology_data import (
    STATE_DIM, N_SWITCHES, N_CONTROLLERS, SWITCHES, SW_ZONE,
    LATENCY_MATRIX, ZONE_LOAD_FACTORS, queuing_delay_ms,
)
from baselines.run_baselines import (
    baseline_zone_optimal, baseline_load_balanced, baseline_kmeans,
)

PROFILES = ['morning', 'business', 'evening', 'night']
SEEDS = [42, 123, 456]

def loads(asg, prof):
    zf = ZONE_LOAD_FACTORS[prof]; L = [0.0]*N_CONTROLLERS
    for i in range(N_SWITCHES):
        L[int(asg[i])] += zf[SW_ZONE[SWITCHES[i]]]
    return L

def analytic_icd(asg, prof):
    L = loads(asg, prof)
    return sum(float(LATENCY_MATRIX[i][int(asg[i])]) + queuing_delay_ms(L[int(asg[i])])
               for i in range(N_SWITCHES)) / N_SWITCHES

def main():
    base = {'ZoneOptimal': baseline_zone_optimal(),
            'LoadBalanced': baseline_load_balanced(),
            'KMeans': baseline_kmeans()}
    dqn_pp = {p: [] for p in PROFILES}
    dqn_ls = {p: [] for p in PROFILES}
    for s in SEEDS:
        ckpt = next((c for c in (f'results/p2_seed{s}.pth', f'results/p1_seed{s}.pth',
                                  f'results/p1_best_seed{s}.pth') if os.path.exists(c)), None)
        if not ckpt:
            continue
        ag = DQNAgent(state_dim=STATE_DIM, action_dim=N_SWITCHES*N_CONTROLLERS,
                      hidden=256, total_steps=100)
        ag.load(ckpt)
        env = CitiverseSimEnv(seed=s+5000, curriculum=False)
        res = evaluate_agent(ag, env, n_episodes=20)
        for p in PROFILES:
            d = res.get('icd_per_profile', {}).get(p)
            if d:
                dqn_pp[p].append(d['mean'])
                dqn_ls[p].append(d.get('load_std', 0.0))

    summary = {'note': 'ANALYTICAL fallback (queuing-aware M/M/1, SimEnv). '
                       'Headline result is results/measured_summary.json (live Mininet).',
               'per_profile': {}}
    for p in PROFILES:
        zo_icd = analytic_icd(base['ZoneOptimal'], p)
        entry = {'zone_optimal_icd': round(zo_icd, 2),
                 'load_balanced_icd': round(analytic_icd(base['LoadBalanced'], p), 2),
                 'kmeans_icd': round(analytic_icd(base['KMeans'], p), 2)}
        if dqn_pp[p]:
            dm = float(np.mean(dqn_pp[p]))
            entry['dqn_icd'] = round(dm, 2)
            entry['dqn_icd_std'] = round(float(np.std(dqn_pp[p])), 2)
            entry['improvement_vs_zo_pct'] = round((zo_icd - dm)/zo_icd*100, 1) if zo_icd > 0 else 0.0
            entry['dqn_load_std'] = round(float(np.mean(dqn_ls[p])), 2)
            entry['beats_zo'] = bool(dm < zo_icd)
        summary['per_profile'][p] = entry
    os.makedirs('results', exist_ok=True)
    json.dump(summary, open('results/analytical_summary.json', 'w'), indent=2)
    print('ANALYTICAL fallback ->', json.dumps(summary['per_profile'], indent=2))

if __name__ == '__main__':
    main()
