#!/usr/bin/env python3
"""Run each trained DQN (p1 checkpoints) deterministically on SimEnv per profile
and dump the settled switch->controller assignment. Output feeds measure_realnet.py
as the DQN assignments to MEASURE in Mininet.
Output: results/dqn_assignments.json  =  {profile: {"DQN_seed42": [...], ...}}
"""
import sys, os, json, argparse
sys.path.insert(0, '/home/maksim/simenv_dqn_mininet_v2')
import numpy as np
from dqn.sim_environment import CitiverseSimEnv
from dqn.agent import DQNAgent
from topology.topology_data import STATE_DIM, N_SWITCHES, N_CONTROLLERS

PROFILES = ['morning', 'business', 'evening', 'night']

def settle_assignment(agent, env, profile, steps=120):
    env.forced_failure = -1
    env.reset()
    env.traffic_profile = profile
    env.active_profiles = [profile]
    state = env._get_state()
    for _ in range(steps):
        a = agent.select_action(state, action_mask=env.get_action_mask(), deterministic=True)
        state, _, _, _ = env.step(a)
    return [int(x) for x in env.assignments.tolist()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='42,123,456')
    ap.add_argument('--out', default='results/dqn_assignments.json')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    out = {p: {} for p in PROFILES}
    for s in seeds:
        ckpt = None
        for cand in (f'results/p2_seed{s}.pth', f'results/p1_seed{s}.pth', f'results/p1_best_seed{s}.pth'):
            if os.path.exists(cand):
                ckpt = cand; break
        if ckpt is None:
            print(f'seed {s}: NO checkpoint, skip'); continue
        agent = DQNAgent(state_dim=STATE_DIM, action_dim=N_SWITCHES*N_CONTROLLERS,
                         hidden=256, total_steps=100)
        agent.load(ckpt)
        env = CitiverseSimEnv(seed=s+9000, curriculum=False)
        for p in PROFILES:
            asg = settle_assignment(agent, env, p)
            out[p][f'DQN_seed{s}'] = asg
        print(f'seed {s} ({ckpt}): done; morning asg = {out["morning"][f"DQN_seed{s}"]}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=2)
    print('saved', args.out)

if __name__ == '__main__':
    main()
