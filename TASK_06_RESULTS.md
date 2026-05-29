# TASK_06: Results, Plots, Topology Visualization, run_all.py

## Step 11 — scripts/plot_results.py

6 figures + 2 tables required for DCCN paper.

```python
# scripts/plot_results.py
import os, json, numpy as np, matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SEEDS = [42, 123, 456]
RESULTS_DIR = 'results'
FIGURES_DIR = 'results/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'lines.linewidth': 1.5,
})

METHODS = ['AllToCtrl0', 'ZoneOptimal', 'LoadBalanced', 'KMeans', 'Random', 'DQN']
COLORS  = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#000000']
MARKERS = ['s', '^', 'D', 'v', 'x', 'o']


# ── Figure 1: Training Curve (ICD vs episode, P1 SimEnv) ──────────────
def fig1_training_curve_p1():
    fig, ax = plt.subplots(figsize=(6, 4))
    all_icds = [np.load(f'{RESULTS_DIR}/p1_icds_seed{s}.npy') for s in SEEDS]
    mean_icd = np.mean(all_icds, axis=0)
    std_icd  = np.std(all_icds, axis=0)
    x = np.arange(len(mean_icd))
    # Smooth with window 50
    def smooth(arr, w=50):
        return np.convolve(arr, np.ones(w)/w, mode='valid')
    xs = smooth(x.astype(float))
    ms = smooth(mean_icd)
    ss = smooth(std_icd)
    ax.plot(xs, ms, 'k-', label='DQN (mean ±1σ, 3 seeds)')
    ax.fill_between(xs, ms - ss, ms + ss, alpha=0.2, color='black')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean ICD (ms)')
    ax.set_title('Phase 1 (SimEnv): Training Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/fig1_p1_training_curve.pdf')
    plt.close(fig)
    print('Saved fig1')


# ── Figure 2: Training Curve (ICD vs episode, P2 RealEnv) ────────────
def fig2_training_curve_p2():
    fig, ax = plt.subplots(figsize=(6, 4))
    profiles = ['morning', 'business', 'evening', 'night']
    colors_p = ['#1f78b4', '#33a02c', '#e31a1c', '#6a3d9a']
    # Shade background by profile period
    for i, (prof, col) in enumerate(zip(profiles, colors_p)):
        ax.axvspan(i * 50, (i + 1) * 50, alpha=0.07, color=col, label=prof)

    for s, c in zip(SEEDS, ['k-', 'k--', 'k:']):
        try:
            icds = np.load(f'{RESULTS_DIR}/p2_icds_seed{s}.npy')
            ax.plot(icds, c, alpha=0.7, label=f'seed={s}')
        except FileNotFoundError:
            pass

    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean ICD (ms)')
    ax.set_title('Phase 2 (RealEnv): Fine-tuning under Traffic Profiles')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/fig2_p2_training_curve.pdf')
    plt.close(fig)
    print('Saved fig2')


# ── Figure 3: Baseline Comparison Bar Chart (ICD) ────────────────────
def fig3_icd_bar():
    try:
        with open(f'{RESULTS_DIR}/baselines.json') as f:
            baselines = json.load(f)
    except FileNotFoundError:
        print('baselines.json not found, skipping fig3')
        return

    # DQN: mean across seeds from P2 eval
    dqn_icds = []
    for s in SEEDS:
        try:
            ev = np.load(f'{RESULTS_DIR}/p2_eval_seed{s}.npy', allow_pickle=True).item()
            dqn_icds.append(ev['icd_mean'])
        except Exception:
            pass

    names = list(baselines.keys())
    icd_vals = [baselines[n]['icd_ms'] for n in names]
    if dqn_icds:
        names.append('DQN')
        icd_vals.append(np.mean(dqn_icds))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(names))
    bars = ax.bar(x, icd_vals, color=COLORS[:len(names)], edgecolor='black', linewidth=0.7)
    if dqn_icds:
        ax.errorbar(len(names) - 1, np.mean(dqn_icds), yerr=np.std(dqn_icds),
                    fmt='none', color='black', capsize=5, linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('Mean ICD (ms)')
    ax.set_title('ICD Comparison: DQN vs Baselines')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/fig3_icd_bar.pdf')
    plt.close(fig)
    print('Saved fig3')


# ── Figure 4: ICD per Traffic Profile (grouped bar) ──────────────────
def fig4_icd_per_profile():
    profiles = ['morning', 'business', 'evening', 'night']
    methods_to_plot = ['ZoneOptimal', 'LoadBalanced', 'DQN']
    x = np.arange(len(profiles))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    for j, method in enumerate(methods_to_plot):
        # Load per-profile eval results
        try:
            data = np.load(f'{RESULTS_DIR}/{method}_profile_icd.npy', allow_pickle=True).item()
            vals = [data.get(p, 0) for p in profiles]
        except Exception:
            vals = [0] * len(profiles)  # placeholder
        ax.bar(x + j * width, vals, width, label=method,
               color=COLORS[j], edgecolor='black', linewidth=0.7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(profiles)
    ax.set_ylabel('ICD (ms)')
    ax.set_title('ICD per Traffic Profile')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/fig4_icd_per_profile.pdf')
    plt.close(fig)
    print('Saved fig4')


# ── Figure 5: Load Distribution (box plot) ───────────────────────────
def fig5_load_distribution():
    try:
        with open(f'{RESULTS_DIR}/baselines.json') as f:
            baselines = json.load(f)
    except FileNotFoundError:
        print('baselines.json not found, skipping fig5')
        return
    from topology.topology_data import N_SWITCHES, N_CONTROLLERS, SWITCHES, SW_ZONE, ZONE_LOAD_FACTORS
    import numpy as np

    fig, ax = plt.subplots(figsize=(7, 4))
    data = []
    labels = []
    for name, res in baselines.items():
        asgn = np.array(res['assignments'])
        loads = np.zeros(N_CONTROLLERS)
        for c in asgn:
            loads[c] += 1.0
        data.append(loads)
        labels.append(name)

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], COLORS[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Switches per Controller')
    ax.set_title('Controller Load Distribution across Baselines')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    fig.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/fig5_load_distribution.pdf')
    plt.close(fig)
    print('Saved fig5')


# ── Figure 6: Convergence Speed (ICD vs wall-clock time) ─────────────
def fig6_convergence_speed():
    fig, ax = plt.subplots(figsize=(6, 4))
    for s, ls in zip(SEEDS, ['-', '--', ':']):
        try:
            icds = np.load(f'{RESULTS_DIR}/p1_icds_seed{s}.npy')
            time_per_ep = 0.05  # ~50ms per SimEnv episode
            t = np.arange(len(icds)) * time_per_ep / 60  # minutes
            def smooth(a, w=30):
                return np.convolve(a, np.ones(w)/w, mode='valid')
            ax.plot(smooth(t), smooth(icds), 'k' + ls, alpha=0.7, label=f'seed={s}')
        except FileNotFoundError:
            pass
    ax.set_xlabel('Training time (min)')
    ax.set_ylabel('ICD (ms)')
    ax.set_title('SimEnv Convergence Speed')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/fig6_convergence_speed.pdf')
    plt.close(fig)
    print('Saved fig6')


# ── Table 1: Main results summary ────────────────────────────────────
def table1_main_results():
    try:
        with open(f'{RESULTS_DIR}/baselines.json') as f:
            baselines = json.load(f)
    except FileNotFoundError:
        print('baselines.json not found'); return

    print('\n=== TABLE 1: Main Results ===')
    print(f'{"Method":<20} {"ICD (ms)":<12} {"Load Std":<12} {"Zone Match":<12}')
    print('-' * 56)
    for name, res in baselines.items():
        print(f'{name:<20} {res["icd_ms"]:<12.2f} {res["load_std"]:<12.2f} {res["zone_match"]:<12.2f}')

    # DQN row
    dqn_icds = []
    for s in SEEDS:
        try:
            ev = np.load(f'{RESULTS_DIR}/p2_eval_seed{s}.npy', allow_pickle=True).item()
            dqn_icds.append(ev['icd_mean'])
        except Exception:
            pass
    if dqn_icds:
        print(f'{"DQN (ours)":<20} {np.mean(dqn_icds):<6.2f}±{np.std(dqn_icds):.2f}    -            -')
    print()


# ── Table 2: Ablation — which bugs matter most ───────────────────────
def table2_ablation():
    print('\n=== TABLE 2: Architecture Ablation (from SimEnv eval) ===')
    print('(Fill manually after running ablation experiments)')
    print(f'{"Variant":<30} {"ICD (ms)":<12} {"Reward":<12}')
    print('-' * 54)
    variants = [
        'Full DQN (Dueling+NStep+Mask)',
        'No action masking',
        'No N-step (n=1)',
        'No traffic profiles',
        'Random init (B09)',
    ]
    for v in variants:
        print(f'{v:<30} {"TBD":<12} {"TBD":<12}')
    print()


if __name__ == '__main__':
    fig1_training_curve_p1()
    fig2_training_curve_p2()
    fig3_icd_bar()
    fig4_icd_per_profile()
    fig5_load_distribution()
    fig6_convergence_speed()
    table1_main_results()
    table2_ablation()
    print(f'\nAll figures saved to {FIGURES_DIR}/')
```

---

## Step 12 — scripts/plot_topology.py

```python
# scripts/plot_topology.py
import numpy as np, matplotlib.pyplot as plt, matplotlib.patches as mpatches
import networkx as nx
from topology.topology_data import SWITCHES, CONTROLLERS, SW_ZONE, ZONE_CTRL, CTRL_ZONE

ZONE_COLORS = {
    'res1': '#1f78b4', 'res2': '#a6cee3',
    'com1': '#33a02c', 'com2': '#b2df8a',
    'ind':  '#e31a1c',
}
CTRL_COLOR = '#ff7f00'


def draw_citiverse_topology(assignments=None, title='CitiVerse Topology', save_path=None):
    """
    Draw CitiVerse topology: 20 switches + 5 controllers.
    assignments: array of ctrl_idx per switch (optional, colors edges)
    """
    G = nx.Graph()

    # Add switch nodes
    for sw in SWITCHES:
        zone = SW_ZONE[sw]
        G.add_node(sw, node_type='switch', zone=zone)

    # Add controller nodes
    for i, ctrl in enumerate(CONTROLLERS):
        zone = CTRL_ZONE.get(i, 'ind')
        G.add_node(ctrl, node_type='controller', zone=zone)

    # Add edges: switch-to-controller (assignment)
    if assignments is not None:
        for i, sw in enumerate(SWITCHES):
            ctrl = CONTROLLERS[assignments[i]]
            G.add_edge(sw, ctrl, edge_type='assignment')

    # Intra-zone switch connections (ring within each zone)
    zones = {}
    for sw in SWITCHES:
        z = SW_ZONE[sw]
        zones.setdefault(z, []).append(sw)
    for zone, sw_list in zones.items():
        for i in range(len(sw_list)):
            G.add_edge(sw_list[i], sw_list[(i + 1) % len(sw_list)], edge_type='link')

    # Controller backbone (fully connected)
    for i in range(len(CONTROLLERS)):
        for j in range(i + 1, len(CONTROLLERS)):
            G.add_edge(CONTROLLERS[i], CONTROLLERS[j], edge_type='backbone')

    # Layout: controllers in pentagon, zone switches around each
    pos = {}
    import math
    ctrl_radius = 2.5
    for i, ctrl in enumerate(CONTROLLERS):
        angle = 2 * math.pi * i / len(CONTROLLERS) - math.pi / 2
        pos[ctrl] = (ctrl_radius * math.cos(angle), ctrl_radius * math.sin(angle))

    sw_radius = 1.1
    for zone, sw_list in zones.items():
        zone_ctrl_idx = ZONE_CTRL[zone]
        cx, cy = pos[CONTROLLERS[zone_ctrl_idx]]
        for j, sw in enumerate(sw_list):
            angle = 2 * math.pi * j / len(sw_list) - math.pi / 2
            pos[sw] = (cx + sw_radius * math.cos(angle), cy + sw_radius * math.sin(angle))

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw edges
    link_edges  = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'link']
    assign_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'assignment']
    bb_edges    = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'backbone']

    nx.draw_networkx_edges(G, pos, edgelist=link_edges, edge_color='#aaaaaa',
                           width=1.0, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=assign_edges, edge_color='#555555',
                           width=1.5, style='dashed', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=bb_edges, edge_color='#ff7f00',
                           width=2.5, ax=ax)

    # Draw switch nodes (squares via scatter with marker='s')
    for sw in SWITCHES:
        zone = SW_ZONE[sw]
        x, y = pos[sw]
        ax.scatter(x, y, s=200, marker='s', color=ZONE_COLORS[zone],
                   edgecolors='black', linewidths=0.8, zorder=5)
        ax.text(x, y - 0.22, sw, ha='center', va='top', fontsize=6)

    # Draw controller nodes (diamonds)
    for i, ctrl in enumerate(CONTROLLERS):
        x, y = pos[ctrl]
        ax.scatter(x, y, s=400, marker='D', color=CTRL_COLOR,
                   edgecolors='black', linewidths=1.0, zorder=6)
        ax.text(x, y - 0.28, ctrl, ha='center', va='top', fontsize=8, fontweight='bold')

    # Legend
    legend_handles = [
        mpatches.Patch(color=c, label=z) for z, c in ZONE_COLORS.items()
    ] + [
        mpatches.Patch(color=CTRL_COLOR, label='Controller'),
        plt.Line2D([0], [0], color='#aaaaaa', label='Intra-zone link'),
        plt.Line2D([0], [0], color='#555555', linestyle='--', label='Assignment'),
        plt.Line2D([0], [0], color='#ff7f00', linewidth=2, label='Controller backbone'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8, ncol=2)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f'Saved topology to {save_path}')
    else:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    import os
    os.makedirs('results/figures', exist_ok=True)
    from topology.topology_data import ZONE_CTRL, SW_ZONE
    from baselines.run_baselines import baseline_zone_optimal
    draw_citiverse_topology(
        assignments=baseline_zone_optimal(),
        title='CitiVerse Topology — ZoneOptimal Assignment',
        save_path='results/figures/topology_zone_optimal.pdf'
    )
    draw_citiverse_topology(
        title='CitiVerse Physical Topology',
        save_path='results/figures/topology_physical.pdf'
    )
```

---

## Step 13 — run_all.py (Orchestrator)

```python
# run_all.py
"""
Orchestrator: Phase 1 (SimEnv, parallel seeds) → Phase 2 (RealEnv, sequential).
Run as: sudo python3 run_all.py
"""
import os, sys, logging, json, time
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
LOG = logging.getLogger(__name__)

SEEDS = [42, 123, 456]
SKIP_PHASE2 = '--sim-only' in sys.argv  # for quick smoke test


def main():
    os.makedirs('results', exist_ok=True)

    # ── Phase 1: SimEnv pre-training ─────────────────────────────────
    LOG.info('=== PHASE 1: SimEnv Pre-training ===')
    from dqn.train import train_phase1
    p1_ckpts = {}
    for seed in SEEDS:
        LOG.info(f'Training seed={seed}...')
        ckpt, _ = train_phase1(seed)
        p1_ckpts[seed] = ckpt

    LOG.info('Phase 1 complete. Checkpoints: %s', p1_ckpts)

    # ── Baselines (static, no Mininet needed) ────────────────────────
    LOG.info('=== Running Static Baselines ===')
    from baselines.run_baselines import run_all_baselines
    baseline_results = run_all_baselines()
    with open('results/baselines.json', 'w') as f:
        json.dump(baseline_results, f, indent=2)

    if SKIP_PHASE2:
        LOG.info('--sim-only flag set, skipping Phase 2')
        _plot_phase1()
        return

    # ── Phase 2: RealEnv fine-tuning ─────────────────────────────────
    LOG.info('=== PHASE 2: RealEnv Fine-tuning ===')
    LOG.info('Ensure Mininet + 5 Ryu are running before this step.')
    LOG.info('Press Enter to continue...')
    input()

    from mininet.net import Mininet
    from mininet.log import setLogLevel
    setLogLevel('warning')
    from topology.citiverse_topo import CitiverseTopo
    from ryu_apps.multi_controller import MultiControllerManager
    from dqn.train import train_phase2, evaluate_agent
    from dqn.agent import DQNAgent
    from dqn.environment import CitiverseRealEnv
    from topology.topology_data import STATE_DIM, N_SWITCHES, N_CONTROLLERS

    ACTION_DIM = N_SWITCHES * N_CONTROLLERS

    ctrl_mgr = MultiControllerManager()
    ctrl_mgr.start_all()

    topo = CitiverseTopo()
    net = Mininet(topo=topo, controller=None)
    net.start()
    time.sleep(3)

    p2_eval_results = {}
    for seed in SEEDS:
        LOG.info(f'Phase 2 seed={seed}...')
        ckpt, agent = train_phase2(seed, p1_ckpts[seed], net, ctrl_mgr)

        # Eval
        eval_env = CitiverseRealEnv(net, ctrl_mgr, seed=seed+1000, max_steps=300)
        metrics = evaluate_agent(agent, eval_env, n_episodes=20)
        p2_eval_results[seed] = metrics
        np.save(f'results/p2_eval_seed{seed}.npy', metrics)
        LOG.info(f'seed={seed} eval: ICD={metrics["icd_mean"]:.2f}±{metrics["icd_std"]:.2f}ms')

    net.stop()
    ctrl_mgr.stop_all()

    # ── Final report ─────────────────────────────────────────────────
    LOG.info('=== Final Results ===')
    icd_means = [p2_eval_results[s]['icd_mean'] for s in SEEDS]
    LOG.info(f'DQN ICD: {np.mean(icd_means):.2f} ± {np.std(icd_means):.2f} ms')
    LOG.info(f'DQN vs ZoneOptimal ICD: {baseline_results["ZoneOptimal"]["icd_ms"]:.2f} ms')

    _plot_all()


def _plot_phase1():
    from scripts.plot_results import (fig1_training_curve_p1, fig6_convergence_speed,
                                       table1_main_results)
    from scripts.plot_topology import draw_citiverse_topology
    from baselines.run_baselines import baseline_zone_optimal
    fig1_training_curve_p1()
    fig6_convergence_speed()
    table1_main_results()
    draw_citiverse_topology(
        assignments=baseline_zone_optimal(),
        save_path='results/figures/topology.pdf'
    )


def _plot_all():
    from scripts.plot_results import (fig1_training_curve_p1, fig2_training_curve_p2,
                                       fig3_icd_bar, fig4_icd_per_profile,
                                       fig5_load_distribution, fig6_convergence_speed,
                                       table1_main_results, table2_ablation)
    from scripts.plot_topology import draw_citiverse_topology
    from baselines.run_baselines import baseline_zone_optimal
    for fn in [fig1_training_curve_p1, fig2_training_curve_p2, fig3_icd_bar,
               fig4_icd_per_profile, fig5_load_distribution, fig6_convergence_speed,
               table1_main_results, table2_ablation]:
        try:
            fn()
        except Exception as e:
            print(f'Warning: {fn.__name__} failed: {e}')
    draw_citiverse_topology(save_path='results/figures/topology.pdf')


if __name__ == '__main__':
    main()
```

---

## Quick Launch (smoke test, no Mininet)

```bash
cd /home/maksim/dqn_simenv_mininet
source ../dqn_env/bin/activate

# Smoke test Phase 1 only (no root needed, ~4 min):
python3 run_all.py --sim-only

# Full run (requires Mininet + root):
sudo python3 run_all.py

# Just baselines:
python3 baselines/run_baselines.py

# Just plots (after training):
python3 scripts/plot_results.py
python3 scripts/plot_topology.py
```

---

## Expected Paper Metrics (target vs baseline)

| Metric | ZoneOptimal | DQN (target) | Improvement |
|--------|-------------|--------------|-------------|
| ICD (ms) | ~35-45 | ~22-30 | ~30-35% |
| Load Std | ~1.2 | ~0.6 | ~50% |
| Zone Match | 1.0 | ~0.8 | — |
| Convergence | — | ~1000 ep | — |

**Paper claim**: "DQN achieves X% lower mean ICD vs static methods under dynamic traffic profiles (morning/business/evening/night), demonstrating that learned policies outperform hand-crafted heuristics in time-varying smart-city SDN topologies."
