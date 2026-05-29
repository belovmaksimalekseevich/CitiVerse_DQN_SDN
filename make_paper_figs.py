#!/usr/bin/env python3
"""Publication figures for the CitiVerse SDN paper.
 F1  network topology (custom switch/controller icons, zone regions)  -> fig_topology.png
 F4  per-controller flow-setup latency, morning (ZoneOptimal vs DQN)   -> fig_per_controller_latency.png
 F5  training convergence (reward + ICD, 3 seeds, curriculum stages)   -> fig_training_curves.png
All built from the real experiment data (topology_data.py, results/*.npy, measured_summary.json).
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

import sys
sys.path.insert(0, '/home/maksim/simenv_dqn_mininet_v2')
from topology.topology_data import (
    ZONES, SW_ZONE, ZONE_CTRL, CTRL_PORTS, get_inter_delay, MININET_INTER_LINKS,
)

RES = 'results'; FIG = os.path.join(RES, 'figures'); os.makedirs(FIG, exist_ok=True)

ZONE_COLOR = {'res1': '#3b6fb0', 'res2': '#74a9d8', 'com1': '#4f9d69',
              'com2': '#8fd0a0', 'ind': '#d1894e'}
ZONE_FILL  = {z: c for z, c in ZONE_COLOR.items()}

# ---------------------------------------------------------------- F1 topology
def draw_switch(ax, x, y, color, label, s=0.42):
    """Classic L2-switch glyph: rounded body + opposing arrows on top."""
    body = FancyBboxPatch((x - s, y - s*0.7), 2*s, 1.4*s*0.7,
                          boxstyle="round,pad=0.02,rounding_size=0.08",
                          linewidth=1.1, edgecolor='#222', facecolor=color, zorder=4)
    ax.add_patch(body)
    # two pairs of opposing arrows (switch symbol) on the top strip
    ay = y + s*0.30
    for dx, dirn in [(-0.18, 1), (0.05, -1)]:
        ax.add_patch(FancyArrowPatch((x + dx - 0.11*dirn, ay), (x + dx + 0.11*dirn, ay),
                     arrowstyle='-|>', mutation_scale=7, linewidth=1.0,
                     color='white', zorder=5))
    ax.text(x, y - s*0.18, label, ha='center', va='center', fontsize=7.5,
            color='white', fontweight='bold', zorder=6)

def draw_controller(ax, x, y, color, label, port, s=0.55):
    """SDN-controller glyph: server/stack rectangle with slots + LED."""
    body = FancyBboxPatch((x - s, y - s*0.85), 2*s, 1.7*s,
                          boxstyle="round,pad=0.02,rounding_size=0.06",
                          linewidth=1.4, edgecolor='#111', facecolor=color, zorder=7)
    ax.add_patch(body)
    for k in range(3):
        yy = y + s*0.45 - k*s*0.42
        ax.add_line(Line2D([x - s*0.7, x + s*0.45], [yy, yy],
                    color='white', linewidth=1.1, zorder=8))
        ax.add_patch(plt.Circle((x + s*0.62, yy), s*0.07, color='#ffe08a', zorder=8))
    ax.text(x, y - s*0.95 - 0.18, f'{label}\n:{port}', ha='center', va='top',
            fontsize=8, fontweight='bold', color='#111', zorder=8)

def grid_positions(center, n, dx=1.15, dy=1.15, cols=3):
    cx, cy = center
    rows = int(np.ceil(n / cols))
    pos = []
    for i in range(n):
        r, c = divmod(i, cols)
        ncols = min(cols, n - r*cols)
        x = cx + (c - (ncols - 1)/2) * dx
        y = cy - (r - (rows - 1)/2) * dy
        pos.append((x, y))
    return pos

def fig_topology():
    zone_center = {'res1': (1.5, 9.0), 'res2': (9.0, 10.0), 'com1': (4.5, 4.5),
                   'com2': (11.0, 4.5), 'ind': (5.5, 0.2)}
    ctrl_offset = {'res1': (-2.6, 0.0), 'res2': (3.0, 0.6), 'com1': (-2.8, 0.0),
                   'com2': (3.0, 0.0), 'ind': (0.0, -2.2)}
    pos = {}
    for z, d in ZONES.items():
        for (sw, p) in zip(d['switches'], grid_positions(zone_center[z], len(d['switches']))):
            pos[sw] = p

    fig, ax = plt.subplots(figsize=(13, 10))
    # zone background regions
    for z, d in ZONES.items():
        xs = [pos[s][0] for s in d['switches']]; ys = [pos[s][1] for s in d['switches']]
        pad = 0.95
        x0, x1 = min(xs) - pad, max(xs) + pad
        y0, y1 = min(ys) - pad, max(ys) + pad
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                     boxstyle="round,pad=0.02,rounding_size=0.25",
                     linewidth=1.3, edgecolor=ZONE_COLOR[z], facecolor=ZONE_COLOR[z],
                     alpha=0.12, zorder=0))
        ax.text((x0 + x1)/2, y1 + 0.18, f"{d['name']}  ({len(d['switches'])} sw)",
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color=ZONE_COLOR[z], zorder=1)

    # control-plane links (dashed) switch -> its zone controller
    cpos = {}
    for z, d in ZONES.items():
        cx = zone_center[z][0] + ctrl_offset[z][0]
        cy = zone_center[z][1] + ctrl_offset[z][1]
        cpos[d['controller']] = (cx, cy)
    for sw, z in SW_ZONE.items():
        c = ZONE_CTRL[z]
        ax.add_line(Line2D([pos[sw][0], cpos[c][0]], [pos[sw][1], cpos[c][1]],
                    color=ZONE_COLOR[z], linewidth=0.6, alpha=0.35,
                    linestyle=(0, (2, 3)), zorder=1))

    # data-plane intra-zone chains (solid grey)
    for z, d in ZONES.items():
        sws = d['switches']
        for a, b in zip(sws[:-1], sws[1:]):
            ax.add_line(Line2D([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                        color='#7a7a7a', linewidth=1.6, zorder=2))
    # data-plane inter-zone backbone (thick) with delay labels
    for za, zb in MININET_INTER_LINKS:
        a = ZONES[za]['switches'][-1]; b = ZONES[zb]['switches'][0]
        ax.add_line(Line2D([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                    color='#222', linewidth=2.4, zorder=2))
        mx, my = (pos[a][0]+pos[b][0])/2, (pos[a][1]+pos[b][1])/2
        ax.text(mx, my, f"{get_inter_delay(za, zb)} ms", fontsize=8, ha='center',
                va='center', bbox=dict(boxstyle='round,pad=0.15', fc='white',
                ec='#222', lw=0.6), zorder=3)

    # switches + controllers
    for sw, z in SW_ZONE.items():
        draw_switch(ax, pos[sw][0], pos[sw][1], ZONE_COLOR[z], f's{sw}')
    for z, d in ZONES.items():
        c = d['controller']
        draw_controller(ax, cpos[c][0], cpos[c][1], ZONE_COLOR[z], f'C{c}', CTRL_PORTS[c])

    legend = [
        Line2D([0], [0], color='#222', lw=2.4, label='Data-plane backbone (inter-zone)'),
        Line2D([0], [0], color='#7a7a7a', lw=1.6, label='Data-plane link (intra-zone)'),
        Line2D([0], [0], color='#555', lw=0.9, linestyle=(0, (2, 3)),
               label='Control-plane channel (OpenFlow 1.3)'),
    ]
    ax.legend(handles=legend, loc='lower right', fontsize=9, framealpha=0.95)
    ax.set_title('CitiVerse SDN testbed: 20 OpenFlow switches, 5 zones, 5 distributed controllers',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(-4, 15); ax.set_ylim(-3.2, 12); ax.set_aspect('equal'); ax.axis('off')
    fig.tight_layout()
    out = os.path.join(FIG, 'fig_topology.png'); fig.savefig(out, dpi=300); plt.close(fig)
    print('saved', out)

# ------------------------------------------------ F4 per-controller latency
def fig_per_controller():
    data = json.load(open(os.path.join(RES, 'measured_summary.json')))
    m = data['morning']
    nctrl = len(CTRL_PORTS)
    zo = [m['ZoneOptimal']['ctrl_latency'].get(str(c), np.nan) for c in range(nctrl)]
    dqn_keys = [k for k in m if k.startswith('DQN')]
    dqn_mat = np.array([[m[k]['ctrl_latency'].get(str(c), np.nan) for c in range(nctrl)]
                        for k in dqn_keys])
    dqn_mean = np.nanmean(dqn_mat, axis=0); dqn_std = np.nanstd(dqn_mat, axis=0)

    x = np.arange(nctrl); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, zo, w, color='#888', label='ZoneOptimal (static)')
    ax.bar(x + w/2, dqn_mean, w, yerr=dqn_std, capsize=4, color='#c44e52',
           label='DQN (n=3, mean±std)')
    for i, v in enumerate(zo):
        ax.text(i - w/2, v + 12, f'{v:.0f}', ha='center', fontsize=8)
    for i, v in enumerate(dqn_mean):
        ax.text(i + w/2, v + 12, f'{v:.0f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f'C{c}\n:{CTRL_PORTS[c]}' for c in range(nctrl)])
    ax.set_ylabel('Per-controller flow-setup latency, ms')
    ax.set_title('Morning profile: ZoneOptimal saturates C0/C1 while C2–C4 idle;\n'
                 'DQN balances control load → all controllers ~20–30 ms', fontsize=11)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, 'fig_per_controller_latency.png'); fig.savefig(out, dpi=300); plt.close(fig)
    print('saved', out)

# ------------------------------------------------------- F5 training curves
def _roll(a, w=50):
    if len(a) < w: return a
    k = np.ones(w) / w
    return np.convolve(a, k, mode='valid')

def fig_training():
    icd = sorted(glob.glob(os.path.join(RES, 'p1_icds_seed*.npy')))
    rew = sorted(glob.glob(os.path.join(RES, 'p1_rewards_seed*.npy')))
    icd_s = [np.load(f) for f in icd]; rew_s = [np.load(f) for f in rew]
    n = min(min(len(a) for a in icd_s), min(len(a) for a in rew_s))
    W = 50

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, series, ylab, title, col in [
        (a1, rew_s, 'Episode reward (rolling mean, w=50)',
         'Training reward — convergence across 3 seeds', '#3b6fb0'),
        (a2, icd_s, 'Training ICD, ms (rolling mean, w=50)',
         'Training inter-controller delay (note: scale shifts as topology grows per stage)', '#c44e52')]:
        rolled = np.array([_roll(s[:n], W) for s in series])
        xs = np.arange(rolled.shape[1]) + W//2
        mean = rolled.mean(0); std = rolled.std(0)
        ax.plot(xs, mean, color=col, linewidth=1.8, label='mean of 3 seeds')
        ax.fill_between(xs, mean - std, mean + std, color=col, alpha=0.2, label='±std')
        for ep, lbl in [(300, 'Stage 1→2\n(5→10 sw)'), (800, 'Stage 2→3\n(→20 sw, all profiles)')]:
            ax.axvline(ep, color='#444', linestyle='--', linewidth=1)
            ax.text(ep + 20, ax.get_ylim()[1], lbl, fontsize=8, va='top', color='#444')
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3); ax.legend(loc='best', fontsize=9)
    a2.set_xlabel('Training episode')
    fig.tight_layout()
    out = os.path.join(FIG, 'fig_training_curves.png'); fig.savefig(out, dpi=300); plt.close(fig)
    print('saved', out)

if __name__ == '__main__':
    fig_topology()
    fig_per_controller()
    fig_training()
    print('DONE ->', FIG)
