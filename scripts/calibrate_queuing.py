#!/usr/bin/env python3
"""Offline calibration with CONSISTENT failure model."""
import sys
sys.path.insert(0, '/home/maksim/simenv_dqn_mininet_v2')
import numpy as np
from topology.topology_data import (
    SWITCHES, N_SWITCHES, N_CONTROLLERS, LATENCY_MATRIX,
    ZONE_LOAD_FACTORS, SW_ZONE, TRAFFIC_PROFILES, get_zone_optimal_array,
)

def compute_loads(assign, profile):
    zf = ZONE_LOAD_FACTORS[profile]
    loads = np.zeros(N_CONTROLLERS)
    for i, sw in enumerate(SWITCHES):
        loads[assign[i]] += zf[SW_ZONE[sw]]
    return loads

def queue_ms(L, MU, Q, WMAX):
    return min(WMAX, Q * L * L / max(0.5, MU - L))

def icd(assign, profile, MU, Q, WMAX):
    loads = compute_loads(assign, profile)   # loads on the ACTUAL assignment
    tot = 0.0
    for i in range(N_SWITCHES):
        c = assign[i]
        tot += float(LATENCY_MATRIX[i][c]) + queue_ms(loads[c], MU, Q, WMAX)
    return tot / N_SWITCHES

def static_failover(assign, dead):
    eff = assign.copy()
    for i in range(N_SWITCHES):
        if eff[i] == dead:
            eff[i] = (dead + 1) % N_CONTROLLERS   # dumb: dump all on neighbour
    return eff

def greedy(profile, MU, Q, WMAX, start=None, forbid=-1):
    asg = (get_zone_optimal_array().copy() if start is None else start.copy())
    best = icd(asg, profile, MU, Q, WMAX)
    improved = True
    while improved:
        improved = False
        for i in range(N_SWITCHES):
            for c in range(N_CONTROLLERS):
                if c == forbid or c == asg[i]:
                    continue
                old = asg[i]; asg[i] = c
                new = icd(asg, profile, MU, Q, WMAX)
                if new < best - 1e-6:
                    best = new; improved = True
                else:
                    asg[i] = old
    return asg, best

def evaluate(MU, Q, WMAX):
    print(f'\n===== MU={MU} Q_COEF={Q} W_MAX={WMAX} =====')
    print(f'{"profile":<10}{"ZO":>8}{"DQN":>8}{"impr%":>7}   '
          f'{"ZOfail":>8}{"DQNfail":>8}{"impr%":>7}')
    zo = get_zone_optimal_array()
    ok = True
    for p in TRAFFIC_PROFILES:
        zo_icd = icd(zo, p, MU, Q, WMAX)
        _, dqn_icd = greedy(p, MU, Q, WMAX)
        impr = (zo_icd - dqn_icd) / zo_icd * 100
        dead = 0
        eff = static_failover(zo, dead)
        zo_f = icd(eff, p, MU, Q, WMAX)                       # static dumb failover
        _, dqn_f = greedy(p, MU, Q, WMAX, start=eff, forbid=dead)  # DQN re-optimizes
        impr_f = (zo_f - dqn_f) / zo_f * 100
        print(f'{p:<10}{zo_icd:>8.2f}{dqn_icd:>8.2f}{impr:>6.1f}%   '
              f'{zo_f:>8.2f}{dqn_f:>8.2f}{impr_f:>6.1f}%')
        if p in ('morning', 'evening') and impr < 15:
            ok = False
    print('VERDICT:', 'GOOD' if ok else 'WEAK')

if __name__ == '__main__':
    for MU, Q, WMAX in [(14, 0.4, 50), (16, 0.5, 60)]:
        evaluate(MU, Q, WMAX)
