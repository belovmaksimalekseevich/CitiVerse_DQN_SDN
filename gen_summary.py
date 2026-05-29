import json, numpy as np, os
RESULTS = 'results'
SEEDS = [42, 123, 456]

p2 = {}
for s in SEEDS:
    with open(f'{RESULTS}/p2_eval_seed{s}.json') as f:
        p2[s] = json.load(f)

icds = [p2[s]['icd_mean'] for s in SEEDS]
stds = [p2[s]['icd_std']  for s in SEEDS]
loads = [p2[s]['load_std_mean'] for s in SEEDS]

dqn_mean = float(np.mean(icds))
dqn_std  = float(np.std(icds))
# Mininet ZoneOptimal from early measurement
zo_mininet = 3.3365
improvement_vs_zo_mininet = (zo_mininet - dqn_mean) / zo_mininet * 100

summary = {
    'dqn_icd_mean':           round(dqn_mean, 3),
    'dqn_icd_std':            round(dqn_std,  3),
    'dqn_icd_per_seed':       {str(s): round(icds[i],3) for i,s in enumerate(SEEDS)},
    'dqn_icd_std_per_seed':   {str(s): round(stds[i],3) for i,s in enumerate(SEEDS)},
    'dqn_load_std_mean':      round(float(np.mean(loads)), 3),
    'zone_optimal_icd_mininet': zo_mininet,
    'zone_optimal_icd_analytical': 3.0,
    'improvement_vs_zo_mininet_pct': round(improvement_vs_zo_mininet, 1),
    'beats_zone_optimal_mininet': dqn_mean < zo_mininet,
}

with open(f'{RESULTS}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
