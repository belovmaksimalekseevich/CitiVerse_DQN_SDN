#!/usr/bin/env python3
"""Build publication figures from the overnight result JSONs.
Robust: if measured_summary.json is missing/empty it still plots the analytical
fallback. Outputs PNGs (300 dpi) to results/figures/.

Inputs:
  results/measured_summary.json   {profile: {algo: {mean_ms,max_ms,p95_ms,ctrl_loads,ctrl_latency,...}}}
                                  (DQN appears as DQN_seed42/123/456 -> aggregated mean+/-std)
  results/analytical_summary.json {note, per_profile: {profile: {zone_optimal_icd, dqn_icd, ...}}}
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = 'results'
FIG = os.path.join(RES, 'figures')
os.makedirs(FIG, exist_ok=True)
PROFILES = ['morning', 'business', 'evening', 'night']
BASE_ORDER = ['ZoneOptimal', 'LoadBalanced', 'KMeans']
COLORS = {'ZoneOptimal': '#888888', 'LoadBalanced': '#4c72b0',
          'KMeans': '#55a868', 'DQN': '#c44e52'}


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f'  (skip {path}: {e})')
        return None


def _dqn_seed_vals(profile_dict, field):
    """Collect a numeric field across DQN_seed* entries -> list."""
    vals = []
    for k, v in profile_dict.items():
        if k.startswith('DQN') and isinstance(v, dict) and v.get(field) is not None:
            vals.append(float(v[field]))
    return vals


def plot_measured(data):
    if not data:
        print('measured: no data -> skip')
        return False
    profiles = [p for p in PROFILES if p in data] or list(data.keys())
    algos = BASE_ORDER + ['DQN']
    x = np.arange(len(profiles))
    w = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    for j, algo in enumerate(algos):
        means, errs = [], []
        for p in profiles:
            pd = data.get(p, {})
            if algo == 'DQN':
                vs = _dqn_seed_vals(pd, 'mean_ms')
                means.append(np.mean(vs) if vs else np.nan)
                errs.append(np.std(vs) if len(vs) > 1 else 0.0)
            else:
                v = pd.get(algo, {}).get('mean_ms')
                means.append(float(v) if v is not None else np.nan)
                errs.append(0.0)
        ax.bar(x + (j - 1.5) * w, means, w, yerr=errs, capsize=3,
               label=('DQN (n=3, mean±std)' if algo == 'DQN' else algo),
               color=COLORS[algo])
    ax.set_xticks(x); ax.set_xticklabels([p.capitalize() for p in profiles])
    ax.set_ylabel('Flow-setup latency, ms (measured on Mininet+Ryu)')
    ax.set_title('Measured flow-setup latency by traffic profile (lower is better)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, 'measured_latency_by_profile.png')
    fig.savefig(out, dpi=300); plt.close(fig)
    print('saved', out)

    # per-controller load balance for the morning profile (illustrative)
    p0 = profiles[0]
    pd = data.get(p0, {})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    series = []
    if 'ZoneOptimal' in pd and pd['ZoneOptimal'].get('ctrl_loads'):
        series.append(('ZoneOptimal', pd['ZoneOptimal']['ctrl_loads']))
    dqn_key = next((k for k in pd if k.startswith('DQN')), None)
    if dqn_key and pd[dqn_key].get('ctrl_loads'):
        series.append(('DQN', pd[dqn_key]['ctrl_loads']))
    if series:
        n = len(series[0][1]); cx = np.arange(n)
        for j, (nm, loads) in enumerate(series):
            ax.bar(cx + (j - 0.5) * 0.35, loads, 0.35, label=nm,
                   color=COLORS.get(nm, '#999'))
        ax.set_xticks(cx); ax.set_xticklabels([f'C{i}' for i in range(n)])
        ax.set_ylabel('Aggregate load on controller')
        ax.set_title(f'Controller load distribution ({p0})')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        out = os.path.join(FIG, 'measured_controller_loads.png')
        fig.savefig(out, dpi=300); plt.close(fig)
        print('saved', out)
    else:
        plt.close(fig)
    return True


def plot_analytical(data):
    if not data or 'per_profile' not in data:
        print('analytical: no data -> skip')
        return False
    pp = data['per_profile']
    profiles = [p for p in PROFILES if p in pp] or list(pp.keys())
    x = np.arange(len(profiles)); w = 0.2
    field = {'ZoneOptimal': 'zone_optimal_icd', 'LoadBalanced': 'load_balanced_icd',
             'KMeans': 'kmeans_icd', 'DQN': 'dqn_icd'}
    fig, ax = plt.subplots(figsize=(9, 5))
    for j, algo in enumerate(['ZoneOptimal', 'LoadBalanced', 'KMeans', 'DQN']):
        means, errs = [], []
        for p in profiles:
            e = pp.get(p, {})
            means.append(float(e.get(field[algo])) if e.get(field[algo]) is not None else np.nan)
            errs.append(float(e.get('dqn_icd_std', 0.0)) if algo == 'DQN' else 0.0)
        ax.bar(x + (j - 1.5) * w, means, w, yerr=errs, capsize=3,
               label=('DQN (n=3, mean±std)' if algo == 'DQN' else algo),
               color=COLORS[algo])
    ax.set_xticks(x); ax.set_xticklabels([p.capitalize() for p in profiles])
    ax.set_ylabel('Inter-controller delay ICD, ms (analytical M/M/1)')
    ax.set_title('Analytical ICD by traffic profile (stress model, lower is better)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, 'analytical_icd_by_profile.png')
    fig.savefig(out, dpi=300); plt.close(fig)
    print('saved', out)
    return True


def main():
    print('=== plot_results ===')
    m = _load(os.path.join(RES, 'measured_summary.json'))
    a = _load(os.path.join(RES, 'analytical_summary.json'))
    got_m = plot_measured(m)
    got_a = plot_analytical(a)
    if not (got_m or got_a):
        print('NOTHING plotted (no input JSONs found yet).')
    else:
        print('figures in', FIG)


if __name__ == '__main__':
    main()
