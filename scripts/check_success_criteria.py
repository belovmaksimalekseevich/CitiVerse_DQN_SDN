# scripts/check_success_criteria.py
"""
Checks success/failure criteria from PLAN_AND_REVIEW.md Section 9.
Run after run_full_evaluation() produces eval_full.json.
"""
import os
import json
import numpy as np


def check_criteria(all_results=None, results_dir='results'):
    if all_results is None:
        path = f'{results_dir}/eval_full.json'
        if not os.path.exists(path):
            print(f'[INFO] {path} not found — run eval_protocol.py first')
            return
        with open(path) as f:
            all_results = json.load(f)

    dqn = all_results.get('DQN', {}).get('overall', {})
    zo  = all_results.get('ZoneOptimal', {}).get('overall', {})
    rnd = all_results.get('Random', {}).get('overall', {})
    c0  = all_results.get('AllToCtrl0', {}).get('overall', {})

    dqn_m = dqn.get('mean', 999)
    dqn_s = dqn.get('std', 999)
    zo_m  = zo.get('mean', 999)
    rnd_m = rnd.get('mean', 999)
    c0_m  = c0.get('mean', 999)

    pct = (zo_m - dqn_m) / zo_m * 100 if zo_m > 0 else 0.0
    consistency = dqn_s < dqn_m * 0.15 if dqn_m > 0 else False

    print('\n=== SUCCESS CRITERIA (DCCN 2026) ===')

    # PASS 1
    ok1 = dqn_m < zo_m
    print(f'[{"PASS" if ok1 else "FAIL"}] DQN ICD < ZoneOptimal: '
          f'{dqn_m:.2f}ms vs {zo_m:.2f}ms ({pct:+.1f}%)')

    # PASS 2
    ok2 = dqn_m < rnd_m and dqn_m < c0_m
    print(f'[{"PASS" if ok2 else "FAIL"}] DQN < Random AND < AllToCtrl0: '
          f'DQN={dqn_m:.2f} Random={rnd_m:.2f} AllToCtrl0={c0_m:.2f}')

    # PASS 3
    ok3 = consistency
    print(f'[{"PASS" if ok3 else "FAIL"}] Seed consistency (std < 15%): '
          f'std={dqn_s:.2f}ms ({dqn_s/dqn_m*100:.1f}% of mean)' if dqn_m > 0 else
          f'[FAIL] No DQN data')

    # Per-profile check
    print('\nPer-profile ICD:')
    for p in ['morning', 'business', 'evening', 'night']:
        dqn_p = all_results.get('DQN', {}).get(p, {}).get('mean', 999)
        zo_p  = all_results.get('ZoneOptimal', {}).get(p, {}).get('mean', 999)
        ok = dqn_p < zo_p
        print(f'  {p:<10} DQN={dqn_p:.2f}ms ZO={zo_p:.2f}ms '
              f'[{"PASS" if ok else "FAIL"}]')

    # Check P1 convergence
    print('\nP1 convergence (loss < 0.3 by ep 300):')
    for s in [42, 123, 456]:
        path = f'{results_dir}/p1_icds_seed{s}.npy'
        if os.path.exists(path):
            icds = np.load(path)
            early = np.mean(icds[200:300]) if len(icds) >= 300 else None
            if early:
                ok = early < 40.0
                print(f'  seed={s}: ICD@ep200-300={early:.2f}ms '
                      f'[{"PASS" if ok else "WARN"}]')
        else:
            print(f'  seed={s}: not found')

    print()
    all_pass = ok1 and ok2 and ok3
    print(f'Overall: {"ALL CRITERIA MET — ready to submit" if all_pass else "SOME CRITERIA NOT MET — review training"}')
    return all_pass


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/maksim/dqn_simenv_mininet')
    check_criteria()
