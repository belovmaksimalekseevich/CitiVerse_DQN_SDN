# TASK_18: Professional Topology Visualization (paper-quality figures)

## Context

TASK_06 has a basic NetworkX topology diagram.
This task replaces/extends it with netgraph which produces publication-quality
figures that look like standard network engineering diagrams.

Topology is heterogeneous:
  res1: 6 switches, res2: 5, com1: 4, com2: 3, ind: 2
  Load under ZoneOptimal: 6:5:4:3:2 — this visual imbalance should be visible.

---

## Install

```bash
pip install netgraph        # main library
pip install matplotlib      # already installed
pip install pillow          # for PNG icon support (optional)
```

---

## File: scripts/plot_topology.py (full replacement)

```python
# scripts/plot_topology.py
"""
Publication-quality CitiVerse topology visualization.
Produces two figures:
  1. Physical topology (switches + controllers + zones)
  2. Assignment overlay (colored edges showing DQN vs ZoneOptimal assignment)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import to_rgba
import networkx as nx
import os

# Try netgraph; fall back to networkx if not installed
try:
    from netgraph import Graph, EditableGraph
    HAS_NETGRAPH = True
except ImportError:
    HAS_NETGRAPH = False

from topology.topology_data import (
    SWITCHES, CONTROLLERS, SW_ZONE, ZONE_CTRL, CTRL_ZONE,
    ZONES, N_SWITCHES, N_CONTROLLERS, LATENCY_MATRIX,
    get_inter_delay, INTRA_DELAY_MS
)

# ── Visual constants ────────────────────────────────────────────────────
ZONE_COLORS = {
    'res1': '#2166ac',   # dark blue   — Residential-1
    'res2': '#74add1',   # light blue  — Residential-2
    'com1': '#1a9641',   # dark green  — Commercial-1
    'com2': '#a6d96a',   # light green — Commercial-2
    'ind':  '#d73027',   # red         — Industrial
}
CTRL_COLOR    = '#f4a582'   # orange — controllers
EDGE_INTRA    = '#aaaaaa'   # grey   — intra-zone links
EDGE_INTER    = '#555555'   # dark   — inter-zone links
EDGE_ASSIGN   = '#ff7f00'   # orange — assignment edges
EDGE_BACKBONE = '#d73027'   # red    — ctrl-ctrl backbone

ZONE_LABELS = {
    'res1': 'Residential-1\n(6 switches)',
    'res2': 'Residential-2\n(5 switches)',
    'com1': 'Commercial-1\n(4 switches)',
    'com2': 'Commercial-2\n(3 switches)',
    'ind':  'Industrial\n(2 switches)',
}


# ── Layout computation ──────────────────────────────────────────────────

def _compute_layout():
    """
    Pentagon layout: controllers at vertices of pentagon.
    Switches cluster around their default controller.
    Returns dict: node_id -> (x, y)
    """
    import math
    pos = {}
    ctrl_radius = 3.5
    sw_radius   = 1.3

    # Controllers: pentagon
    for i in range(N_CONTROLLERS):
        angle = 2 * math.pi * i / N_CONTROLLERS - math.pi / 2
        pos[f'ctrl_{i}'] = (ctrl_radius * math.cos(angle),
                             ctrl_radius * math.sin(angle))

    # Switches: cluster around their default controller
    for zone, data in ZONES.items():
        ctrl_idx = data['controller']
        cx, cy = pos[f'ctrl_{ctrl_idx}']
        sw_list = data['switches']
        n = len(sw_list)
        for j, sw in enumerate(sw_list):
            angle = 2 * math.pi * j / max(n, 1) - math.pi / 2
            # Spread: smaller zones get tighter clustering
            r = sw_radius * (0.7 + 0.3 * n / 6)
            pos[f'sw_{sw}'] = (cx + r * math.cos(angle),
                                cy + r * math.sin(angle))
    return pos


# ── Main figure (NetworkX + matplotlib) ────────────────────────────────

def draw_topology_networkx(assignments=None, title='CitiVerse SDN Topology',
                            save_path=None, show_labels=True, figsize=(12, 12)):
    """
    Draw topology with NetworkX + matplotlib.
    assignments: array shape (N_SWITCHES,) of ctrl indices, or None
    """
    G = nx.Graph()
    pos = _compute_layout()

    # Add nodes
    for sw in SWITCHES:
        G.add_node(f'sw_{sw}', ntype='switch', zone=SW_ZONE[sw])
    for i in range(N_CONTROLLERS):
        G.add_node(f'ctrl_{i}', ntype='controller', zone=CTRL_ZONE[i])

    # Intra-zone switch links (ring within each zone)
    for zone, data in ZONES.items():
        sw_list = data['switches']
        for k in range(len(sw_list)):
            a = f'sw_{sw_list[k]}'
            b = f'sw_{sw_list[(k + 1) % len(sw_list)]}'
            G.add_edge(a, b, etype='intra')

    # Inter-zone links (gateway switch of each zone to gateway of neighbor)
    inter_pairs = [
        ('res1', 'res2'), ('res1', 'com1'), ('res2', 'com2'), ('com1', 'ind'),
    ]
    for za, zb in inter_pairs:
        sw_a = f'sw_{ZONES[za]["switches"][0]}'  # gateway = first switch
        sw_b = f'sw_{ZONES[zb]["switches"][0]}'
        G.add_edge(sw_a, sw_b, etype='inter')

    # Controller backbone (fully connected)
    for i in range(N_CONTROLLERS):
        for j in range(i + 1, N_CONTROLLERS):
            G.add_edge(f'ctrl_{i}', f'ctrl_{j}', etype='backbone')

    # Assignment edges
    if assignments is not None:
        for sw_idx, sw in enumerate(SWITCHES):
            ctrl_idx = assignments[sw_idx] if hasattr(assignments, '__len__') else assignments[sw]
            G.add_edge(f'sw_{sw}', f'ctrl_{ctrl_idx}', etype='assign')

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw edge layers (order matters for visibility)
    for etype, color, width, style, alpha in [
        ('backbone', EDGE_BACKBONE, 3.0, 'solid', 0.8),
        ('inter',    EDGE_INTER,    1.5, 'dashed', 0.6),
        ('intra',    EDGE_INTRA,    1.0, 'solid', 0.5),
        ('assign',   EDGE_ASSIGN,   1.2, 'dotted', 0.9),
    ]:
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('etype') == etype]
        nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax,
                               edge_color=color, width=width,
                               style=style, alpha=alpha)

    # Draw switch nodes (squares)
    for sw in SWITCHES:
        nid = f'sw_{sw}'
        x, y = pos[nid]
        zone = SW_ZONE[sw]
        ax.scatter(x, y, s=280, marker='s',
                   color=ZONE_COLORS[zone], edgecolors='black',
                   linewidths=0.8, zorder=5)
        if show_labels:
            ax.text(x, y - 0.28, f's{sw}', ha='center', va='top',
                    fontsize=5.5, color='black')

    # Draw controller nodes (diamonds)
    for i in range(N_CONTROLLERS):
        nid = f'ctrl_{i}'
        x, y = pos[nid]
        zone = CTRL_ZONE[i]
        ax.scatter(x, y, s=600, marker='D',
                   color=CTRL_COLOR, edgecolors='black',
                   linewidths=1.2, zorder=6)
        ax.text(x, y, f'C{i}', ha='center', va='center',
                fontsize=8, fontweight='bold', color='black')
        ax.text(x, y - 0.4, zone, ha='center', va='top',
                fontsize=6.5, color='#555555')

    # Zone background patches (ellipses around each zone's switches)
    for zone, data in ZONES.items():
        sw_list = data['switches']
        xs = [pos[f'sw_{sw}'][0] for sw in sw_list]
        ys = [pos[f'sw_{sw}'][1] for sw in sw_list]
        cx_z = np.mean(xs)
        cy_z = np.mean(ys)
        # Ellipse to encompass the zone
        from matplotlib.patches import Ellipse
        w = (max(xs) - min(xs) + 1.2) if len(xs) > 1 else 1.2
        h = (max(ys) - min(ys) + 1.2) if len(ys) > 1 else 1.2
        ell = Ellipse((cx_z, cy_z), width=w, height=h,
                      facecolor=to_rgba(ZONE_COLORS[zone], 0.10),
                      edgecolor=ZONE_COLORS[zone], linewidth=1.5,
                      linestyle='--', zorder=1)
        ax.add_patch(ell)
        # Zone label
        ax.text(cx_z, cy_z + h/2 + 0.15, ZONE_LABELS[zone],
                ha='center', va='bottom', fontsize=8,
                color=ZONE_COLORS[zone], fontweight='bold')

    # Legend
    legend_handles = [
        mpatches.Patch(color=ZONE_COLORS[z], label=f'{z} ({len(ZONES[z]["switches"])} sw)')
        for z in ZONES
    ] + [
        mpatches.Patch(color=CTRL_COLOR, label='Controller'),
        mlines.Line2D([], [], color=EDGE_BACKBONE, linewidth=2.5, label='Controller backbone'),
        mlines.Line2D([], [], color=EDGE_INTER, linewidth=1.5,
                      linestyle='dashed', label='Inter-zone link'),
        mlines.Line2D([], [], color=EDGE_ASSIGN, linewidth=1.5,
                      linestyle='dotted', label='Assignment'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8,
              ncol=2, framealpha=0.9)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.close(fig)
    return fig


# ── netgraph version (if installed) ────────────────────────────────────

def draw_topology_netgraph(assignments=None, title='CitiVerse SDN Topology',
                            save_path=None, figsize=(12, 12)):
    """
    Higher-quality version using netgraph library.
    Produces cleaner node layouts and edge routing.
    """
    if not HAS_NETGRAPH:
        print('netgraph not installed — falling back to NetworkX version')
        return draw_topology_networkx(assignments, title, save_path, figsize=figsize)

    import math
    from netgraph import Graph

    edges = []
    edge_color = {}
    edge_width = {}
    edge_style = {}

    node_color = {}
    node_shape = {}
    node_size  = {}
    node_label = {}

    pos = _compute_layout()

    # Build node lists
    all_nodes = [f'sw_{sw}' for sw in SWITCHES] + [f'ctrl_{i}' for i in range(N_CONTROLLERS)]

    for sw in SWITCHES:
        nid = f'sw_{sw}'
        node_color[nid] = ZONE_COLORS[SW_ZONE[sw]]
        node_shape[nid] = 's'      # square
        node_size[nid]  = 3
        node_label[nid] = f's{sw}'

    for i in range(N_CONTROLLERS):
        nid = f'ctrl_{i}'
        node_color[nid] = CTRL_COLOR
        node_shape[nid] = 'd'      # diamond
        node_size[nid]  = 5
        node_label[nid] = f'C{i}'

    # Edges
    # Intra-zone
    for zone, data in ZONES.items():
        sw_list = data['switches']
        for k in range(len(sw_list)):
            a = f'sw_{sw_list[k]}'
            b = f'sw_{sw_list[(k+1) % len(sw_list)]}'
            e = (a, b)
            edges.append(e)
            edge_color[e] = EDGE_INTRA
            edge_width[e] = 1.0
            edge_style[e] = '-'

    # Inter-zone
    for za, zb in [('res1','res2'),('res1','com1'),('res2','com2'),('com1','ind')]:
        a = f'sw_{ZONES[za]["switches"][0]}'
        b = f'sw_{ZONES[zb]["switches"][0]}'
        e = (a, b)
        edges.append(e)
        edge_color[e] = EDGE_INTER
        edge_width[e] = 1.5
        edge_style[e] = '--'

    # Controller backbone
    for i in range(N_CONTROLLERS):
        for j in range(i+1, N_CONTROLLERS):
            e = (f'ctrl_{i}', f'ctrl_{j}')
            edges.append(e)
            edge_color[e] = EDGE_BACKBONE
            edge_width[e] = 2.5
            edge_style[e] = '-'

    # Assignments
    if assignments is not None:
        for sw_idx, sw in enumerate(SWITCHES):
            ci = assignments[sw_idx] if hasattr(assignments, '__getitem__') else 0
            e = (f'sw_{sw}', f'ctrl_{ci}')
            edges.append(e)
            edge_color[e] = EDGE_ASSIGN
            edge_width[e] = 1.2
            edge_style[e] = ':'

    fig, ax = plt.subplots(figsize=figsize)

    Graph(
        edges,
        node_layout=pos,
        node_color=node_color,
        node_shape=node_shape,
        node_size=node_size,
        node_labels=node_label,
        node_label_fontdict={'size': 6},
        edge_color=edge_color,
        edge_width=edge_width,
        ax=ax,
    )

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.close(fig)


# ── Convenience wrappers ────────────────────────────────────────────────

def draw_citiverse_topology(assignments=None, title='CitiVerse Topology',
                             save_path=None, use_netgraph=True):
    """Main entry point — uses netgraph if available, else NetworkX."""
    if use_netgraph and HAS_NETGRAPH:
        draw_topology_netgraph(assignments, title, save_path)
    else:
        draw_topology_networkx(assignments, title, save_path)


def draw_assignment_comparison(dqn_assignments, baseline_assignments,
                                baseline_name='ZoneOptimal',
                                save_path='results/figures/assignment_compare.pdf'):
    """Side-by-side: DQN assignment vs baseline assignment."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for ax, asgn, label in [
        (axes[0], baseline_assignments, f'{baseline_name} Assignment'),
        (axes[1], dqn_assignments,      'DQN Assignment'),
    ]:
        # Draw on this axis
        G = nx.Graph()
        pos = _compute_layout()
        for sw in SWITCHES:
            G.add_node(f'sw_{sw}')
        for i in range(N_CONTROLLERS):
            G.add_node(f'ctrl_{i}')

        # Intra-zone edges
        for zone, data in ZONES.items():
            sw_list = data['switches']
            for k in range(len(sw_list)):
                G.add_edge(f'sw_{sw_list[k]}', f'sw_{sw_list[(k+1)%len(sw_list)]}')

        # Assignment edges colored by zone-match
        for sw_idx, sw in enumerate(SWITCHES):
            ci = asgn[sw_idx]
            match = (SW_ZONE[sw] == CTRL_ZONE.get(ci, ''))
            color = '#2ca02c' if match else '#d62728'  # green=match, red=mismatch
            nx.draw_networkx_edges(
                G, pos, edgelist=[(f'sw_{sw}', f'ctrl_{ci}')],
                edge_color=color, width=1.5, style='dotted', alpha=0.8, ax=ax
            )

        # Switches
        for sw in SWITCHES:
            x, y = pos[f'sw_{sw}']
            ax.scatter(x, y, s=250, marker='s',
                       color=ZONE_COLORS[SW_ZONE[sw]], edgecolors='black',
                       linewidths=0.8, zorder=5)

        # Controllers
        for i in range(N_CONTROLLERS):
            x, y = pos[f'ctrl_{i}']
            n_assigned = sum(1 for a in asgn if a == i)
            # Size proportional to load
            ax.scatter(x, y, s=400 + n_assigned * 100, marker='D',
                       color=CTRL_COLOR, edgecolors='black',
                       linewidths=1.2, zorder=6)
            ax.text(x, y, f'C{i}\n({n_assigned}sw)', ha='center',
                    va='center', fontsize=7, fontweight='bold')

        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.axis('off')

    # Shared legend
    green_line = mlines.Line2D([], [], color='#2ca02c', linestyle=':', label='Zone-matched')
    red_line   = mlines.Line2D([], [], color='#d62728', linestyle=':', label='Cross-zone')
    fig.legend(handles=[green_line, red_line], loc='lower center',
               ncol=2, fontsize=10, framealpha=0.9)

    fig.suptitle('Controller Assignment: DQN vs Baseline', fontsize=14, fontweight='bold')
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close(fig)


# ── CLI entry point ─────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('results/figures', exist_ok=True)

    from baselines.run_baselines import baseline_zone_optimal, baseline_random
    import numpy as np

    zo_asgn  = baseline_zone_optimal()          # array shape (20,)
    rnd_asgn = baseline_random(seed=42)

    # Figure 1: Physical topology (no assignment)
    draw_citiverse_topology(
        title='CitiVerse Smart-City Topology (Physical)',
        save_path='results/figures/topology_physical.pdf',
    )

    # Figure 2: ZoneOptimal assignment
    draw_citiverse_topology(
        assignments=zo_asgn,
        title='CitiVerse Topology — ZoneOptimal Assignment',
        save_path='results/figures/topology_zone_optimal.pdf',
    )

    # Figure 3: DQN assignment (load from checkpoint if available)
    try:
        from dqn.agent import DQNAgent
        from topology.topology_data import STATE_DIM, N_SWITCHES, N_CONTROLLERS
        agent = DQNAgent(state_dim=STATE_DIM, action_dim=N_SWITCHES * N_CONTROLLERS)
        import torch
        ckpt_path = 'results/p2_best_seed42.pth'
        if os.path.exists(ckpt_path):
            agent.load(ckpt_path)
            from dqn.sim_environment import CitiverseSimEnv
            env = CitiverseSimEnv(seed=42)
            state = env.reset()
            dqn_asgn = env.assignments.copy()
            for _ in range(50):  # let agent settle
                action = agent.select_action(state, deterministic=True)
                state, _, done, _ = env.step(action)
                dqn_asgn = env.assignments.copy()
                if done: break
            draw_assignment_comparison(
                dqn_asgn, zo_asgn,
                save_path='results/figures/assignment_compare.pdf',
            )
        else:
            print('No DQN checkpoint found — skipping assignment comparison')
    except Exception as e:
        print(f'DQN viz skipped: {e}')

    print('Topology figures saved to results/figures/')
```

---

## Output figures

```
results/figures/topology_physical.pdf       — clean topology, no assignment
results/figures/topology_zone_optimal.pdf   — ZoneOptimal (baseline for paper Fig 1)
results/figures/assignment_compare.pdf      — DQN vs ZoneOptimal side-by-side
```

## Notes on heterogeneous zone sizes

The zone ellipses in the visualization intentionally vary in size to reflect
the actual switch counts (6, 5, 4, 3, 2). In the assignment_compare figure,
controller nodes are scaled proportionally to load (number of assigned switches).
This visually communicates the load imbalance problem the DQN is solving.

In the paper caption: "Fig. 1: CitiVerse topology with 5 heterogeneous zones
(6/5/4/3/2 switches per zone). Controller node size proportional to load."
