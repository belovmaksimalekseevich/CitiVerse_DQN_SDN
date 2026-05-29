# scripts/plot_topology.py
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from topology.topology_data import SWITCHES, CONTROLLERS, SW_ZONE, ZONE_CTRL, CTRL_ZONE

ZONE_COLORS = {
    'res1': '#1f78b4',
    'res2': '#a6cee3',
    'com1': '#33a02c',
    'com2': '#b2df8a',
    'ind':  '#e31a1c',
}
CTRL_COLOR = '#ff7f00'


def draw_citiverse_topology(assignments=None, title='CitiVerse Topology', save_path=None):
    """
    Draw CitiVerse topology: 20 switches + 5 controllers.
    assignments: array of ctrl_idx per switch (optional, colours assignment edges).
    """
    G = nx.Graph()

    for sw in SWITCHES:
        G.add_node(sw, node_type='switch', zone=SW_ZONE[sw])
    for i, ctrl in enumerate(CONTROLLERS):
        G.add_node(ctrl, node_type='controller', zone=CTRL_ZONE.get(i, 'ind'))

    if assignments is not None:
        for i, sw in enumerate(SWITCHES):
            ctrl = CONTROLLERS[int(assignments[i])]
            G.add_edge(sw, ctrl, edge_type='assignment')

    zones = {}
    for sw in SWITCHES:
        zones.setdefault(SW_ZONE[sw], []).append(sw)
    for zone_sw_list in zones.values():
        for i in range(len(zone_sw_list)):
            G.add_edge(zone_sw_list[i], zone_sw_list[(i + 1) % len(zone_sw_list)],
                       edge_type='link')

    for i in range(len(CONTROLLERS)):
        for j in range(i + 1, len(CONTROLLERS)):
            G.add_edge(CONTROLLERS[i], CONTROLLERS[j], edge_type='backbone')

    pos = {}
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

    link_edges   = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'link']
    assign_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'assignment']
    bb_edges     = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'backbone']

    nx.draw_networkx_edges(G, pos, edgelist=link_edges,
                           edge_color='#aaaaaa', width=1.0, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=assign_edges,
                           edge_color='#555555', width=1.5, style='dashed', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=bb_edges,
                           edge_color='#ff7f00', width=2.5, ax=ax)

    for sw in SWITCHES:
        x, y = pos[sw]
        ax.scatter(x, y, s=200, marker='s',
                   color=ZONE_COLORS[SW_ZONE[sw]],
                   edgecolors='black', linewidths=0.8, zorder=5)
        ax.text(x, y - 0.22, sw, ha='center', va='top', fontsize=6)

    for i, ctrl in enumerate(CONTROLLERS):
        x, y = pos[ctrl]
        ax.scatter(x, y, s=400, marker='D',
                   color=CTRL_COLOR, edgecolors='black', linewidths=1.0, zorder=6)
        ax.text(x, y - 0.28, ctrl, ha='center', va='top',
                fontsize=8, fontweight='bold')

    legend_handles = [
        mpatches.Patch(color=c, label=z) for z, c in ZONE_COLORS.items()
    ] + [
        mpatches.Patch(color=CTRL_COLOR, label='Controller'),
        plt.Line2D([0], [0], color='#aaaaaa', label='Intra-zone link'),
        plt.Line2D([0], [0], color='#555555', linestyle='--', label='Assignment'),
        plt.Line2D([0], [0], color='#ff7f00', linewidth=2, label='Ctrl backbone'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8, ncol=2)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f'Saved topology to {save_path}')
    else:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs('results/figures', exist_ok=True)
    from baselines.run_baselines import baseline_zone_optimal
    draw_citiverse_topology(
        assignments=baseline_zone_optimal(),
        title='CitiVerse Topology — ZoneOptimal Assignment',
        save_path='results/figures/topology_zone_optimal.pdf',
    )
    draw_citiverse_topology(
        title='CitiVerse Physical Topology',
        save_path='results/figures/topology_physical.pdf',
    )
