# scripts/plot_results.py
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def _smooth(arr, w=50):
    return np.convolve(arr, np.ones(w) / w, mode='valid')


def fig1_training_curve_p1():
    all_icds = []
    for s in SEEDS:
        path = f'{RESULTS_DIR}/p1_icds_seed{s}.npy'
        if os.path.exists(path):
            all_icds.append(np.load(path))
    if not all_icds:
        print('fig1: no p1_icds data yet, skipping')
        return
    min_len = min(len(a) for a in all_icds)
    all_icds = [a[:min_len] for a in all_icds]
    mean_icd = np.mean(all_icds, axis=0)
    std_icd  = np.std(all_icds, axis=0)
    x = np.arange(min_len).astype(float)
    w = min(50, min_len // 4)
    xs = _smooth(x, w); ms = _smooth(mean_icd, w); ss = _smooth(std_icd, w)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ms, 'k-', label='DQN mean ±1σ (3 seeds)')
    ax.fill_between(xs, ms - ss, ms + ss, alpha=0.2, color='black')
    ax.set_xlabel('Episode'); ax.set_ylabel('Mean ICD (ms)')
    ax.set_title('Phase 1 (SimEnv): Training Convergence')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/fig1_p1_training_curve.pdf')
    plt.close(fig)
    print('Saved fig1')


def fig2_training_curve_p2():
    profiles = ['morning', 'business', 'evening', 'night']
    colors_p = ['#1f78b4', '#33a02c', '#e31a1c', '#6a3d9a']
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (prof, col) in enumerate(zip(profiles, colors_p)):
        ax.axvspan(i * 50, (i + 1) * 50, alpha=0.07, color=col, label=prof)
    for s, ls in zip(SEEDS, ['k-', 'k--', 'k:']):
        path = f'{RESULTS_DIR}/p2_icds_seed{s}.npy'
        if os.path.exists(path):
            icds = np.load(path)
            ax.plot(icds, ls, alpha=0.7, label=f'seed={s}')
    ax.set_xlabel('Episode'); ax.set_ylabel('Mean ICD (ms)')
    ax.set_title('Phase 2 (RealEnv): Fine-tuning under Traffic Profiles')
    ax.legend(ncol=2, fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/fig2_p2_training_curve.pdf')
    plt.close(fig)
    print('Saved fig2')


def fig3_icd_bar():
    bl_path = f'{RESULTS_DIR}/baselines.json'
    if not os.path.exists(bl_path):
        print('fig3: baselines.json not found, skipping')
        return
    with open(bl_path) as f:
        baselines = json.load(f)

    dqn_icds = []
    for s in SEEDS:
        path = f'{RESULTS_DIR}/p2_eval_seed{s}.npy'
        if os.path.exists(path):
            ev = np.load(path, allow_pickle=True).item()
            dqn_icds.append(ev['icd_mean'])

    names = list(baselines.keys())
    icd_vals = [baselines[n]['icd_ms'] for n in names]
    colors = COLORS[:len(names)]
    if dqn_icds:
        names.append('DQN (ours)')
        icd_vals.append(float(np.mean(dqn_icds)))
        colors.append('#000000')

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(names))
    ax.bar(x, icd_vals, color=colors, edgecolor='black', linewidth=0.7)
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


def fig4_icd_per_profile():
    profiles = ['morning', 'business', 'evening', 'night']
    methods_to_plot = ['ZoneOptimal', 'LoadBalanced', 'DQN']
    x = np.arange(len(profiles))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, method in enumerate(methods_to_plot):
        path = f'{RESULTS_DIR}/{method}_profile_icd.npy'
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True).item()
            vals = [data.get(p, 0) for p in profiles]
        else:
            vals = [0.0] * len(profiles)
        ax.bar(x + j * width, vals, width, label=method,
               color=COLORS[j], edgecolor='black', linewidth=0.7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(profiles)
    ax.set_ylabel('ICD (ms)')
    ax.set_title('ICD per Traffic Profile')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/fig4_icd_per_profile.pdf')
    plt.close(fig)
    print('Saved fig4')


def fig5_load_distribution():
    bl_path = f'{RESULTS_DIR}/baselines.json'
    if not os.path.exists(bl_path):
        print('fig5: baselines.json not found, skipping')
        return
    with open(bl_path) as f:
        baselines = json.load(f)

    from topology.topology_data import N_CONTROLLERS
    data, labels = [], []
    for name, res in baselines.items():
        asgn = np.array(res['assignments'])
        loads = np.zeros(N_CONTROLLERS)
        for c in asgn:
            loads[c] += 1.0
        data.append(loads)
        labels.append(name)

    fig, ax = plt.subplots(figsize=(7, 4))
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


def fig6_convergence_speed():
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for s, ls in zip(SEEDS, ['-', '--', ':']):
        path = f'{RESULTS_DIR}/p1_icds_seed{s}.npy'
        if os.path.exists(path):
            icds = np.load(path)
            time_per_ep = 0.05
            t = np.arange(len(icds)) * time_per_ep / 60
            w = min(30, len(icds) // 4)
            ax.plot(_smooth(t, w), _smooth(icds, w), 'k' + ls, alpha=0.7, label=f'seed={s}')
            plotted = True
    if not plotted:
        print('fig6: no p1_icds data yet, skipping')
        plt.close(fig)
        return
    ax.set_xlabel('Training time (min)'); ax.set_ylabel('ICD (ms)')
    ax.set_title('SimEnv Convergence Speed')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/fig6_convergence_speed.pdf')
    plt.close(fig)
    print('Saved fig6')


def table1_main_results():
    bl_path = f'{RESULTS_DIR}/baselines.json'
    if not os.path.exists(bl_path):
        print('table1: baselines.json not found'); return
    with open(bl_path) as f:
        baselines = json.load(f)

    print('\n=== TABLE 1: Main Results ===')
    print(f'{"Method":<20} {"ICD (ms)":<14} {"Load Std":<12} {"Zone Match":<12}')
    print('-' * 58)
    for name, res in baselines.items():
        print(f'{name:<20} {res["icd_ms"]:<14.2f} {res["load_std"]:<12.3f} {res["zone_match"]:<12.3f}')

    dqn_icds = []
    for s in SEEDS:
        path = f'{RESULTS_DIR}/p2_eval_seed{s}.npy'
        if os.path.exists(path):
            ev = np.load(path, allow_pickle=True).item()
            dqn_icds.append(ev['icd_mean'])
    if dqn_icds:
        m, sd = np.mean(dqn_icds), np.std(dqn_icds)
        print(f'{"DQN (ours)":<20} {m:.2f}±{sd:.2f}         -            -')
    print()


def table2_ablation():
    abl_path = f'{RESULTS_DIR}/ablation.json'
    print('\n=== TABLE 2: Ablation Study ===')
    if os.path.exists(abl_path):
        with open(abl_path) as f:
            results = json.load(f)
        print(f'{"Variant":<25} {"ICD (ms)":<16} {"Conv ep"}')
        print('-' * 52)
        for name, res in results.items():
            if 'note' in res:
                print(f'{name:<25} {res["note"]}')
            else:
                print(f'{name:<25} {res["icd_mean"]:.2f}±{res["icd_std"]:.2f}          {res.get("conv_ep", "N/A")}')
    else:
        print('(Run scripts/ablation.py first)')
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
