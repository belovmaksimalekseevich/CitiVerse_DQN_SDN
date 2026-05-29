#!/usr/bin/env python3
"""Topology figure v2 — proper 3D Cisco-style switch/server icons (no 'Lego')."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import sys
sys.path.insert(0, '/home/maksim/simenv_dqn_mininet_v2')
from topology.topology_data import (
    ZONES, SW_ZONE, ZONE_CTRL, CTRL_PORTS, get_inter_delay, MININET_INTER_LINKS,
)

FIG = 'results/figures'; os.makedirs(FIG, exist_ok=True)
ZC = {'res1': '#3b6fb0', 'res2': '#5a91cc', 'com1': '#3f9d69', 'com2': '#5cb87f',
      'ind': '#d1894e'}

def lighten(hexc, f):
    c = np.array([int(hexc[i:i+2], 16) for i in (1, 3, 5)]) / 255
    c = c + (1 - c) * f if f > 0 else c * (1 + f)
    return tuple(np.clip(c, 0, 1))

def switch_icon(ax, x, y, color, label, w=0.46, h=0.27, dx=0.20, dy=0.14):
    """Isometric L2-switch slab + double-headed arrows on top (the switch symbol)."""
    front = lighten(color, 0.0); top = lighten(color, 0.32); side = lighten(color, -0.28)
    # right side face
    ax.add_patch(Polygon([(x+w, y-h), (x+w, y+h), (x+w+dx, y+h+dy), (x+w+dx, y-h+dy)],
                 closed=True, facecolor=side, edgecolor='#1b1b1b', linewidth=0.9, zorder=4))
    # top face
    ax.add_patch(Polygon([(x-w, y+h), (x+w, y+h), (x+w+dx, y+h+dy), (x-w+dx, y+h+dy)],
                 closed=True, facecolor=top, edgecolor='#1b1b1b', linewidth=0.9, zorder=4))
    # front face
    ax.add_patch(Polygon([(x-w, y-h), (x+w, y-h), (x+w, y+h), (x-w, y+h)],
                 closed=True, facecolor=front, edgecolor='#1b1b1b', linewidth=1.0, zorder=5))
    # two double-headed arrows on the top face (switch glyph)
    cy = y + h + dy*0.55
    for off in (-0.14, 0.06):
        ax.add_patch(FancyArrowPatch((x + off + dx*0.4 - 0.12, cy - 0.015),
                     (x + off + dx*0.4 + 0.12, cy + 0.015),
                     arrowstyle='<|-|>', mutation_scale=6, linewidth=1.0,
                     color='white', zorder=6))
    ax.text(x, y - 0.02, label, ha='center', va='center', fontsize=7.5,
            color='white', fontweight='bold', zorder=7)

def server_icon(ax, x, y, color, label, port, w=0.40, h=0.62, dx=0.20, dy=0.14):
    """Isometric server/controller tower with rack slots + LEDs."""
    front = lighten(color, -0.05); top = lighten(color, 0.30); side = lighten(color, -0.30)
    ax.add_patch(Polygon([(x+w, y-h), (x+w, y+h), (x+w+dx, y+h+dy), (x+w+dx, y-h+dy)],
                 closed=True, facecolor=side, edgecolor='#111', linewidth=0.9, zorder=7))
    ax.add_patch(Polygon([(x-w, y+h), (x+w, y+h), (x+w+dx, y+h+dy), (x-w+dx, y+h+dy)],
                 closed=True, facecolor=top, edgecolor='#111', linewidth=0.9, zorder=7))
    ax.add_patch(Polygon([(x-w, y-h), (x+w, y-h), (x+w, y+h), (x-w, y+h)],
                 closed=True, facecolor=front, edgecolor='#111', linewidth=1.1, zorder=8))
    for k in range(3):
        yy = y + h*0.55 - k*h*0.42
        ax.add_line(Line2D([x - w*0.72, x + w*0.45], [yy, yy], color='white',
                    linewidth=1.2, zorder=9))
        ax.add_patch(Circle((x + w*0.66, yy), 0.035, color='#ffe08a',
                     ec='#111', linewidth=0.4, zorder=9))
    ax.text(x + dx*0.5, y - h - 0.16, f'{label}  :{port}', ha='center', va='top',
            fontsize=8.5, fontweight='bold', color='#111', zorder=9)

def grid_positions(center, n, dxs=1.25, dys=1.30, cols=3):
    cx, cy = center; rows = int(np.ceil(n/cols)); pos = []
    for i in range(n):
        r, c = divmod(i, cols); ncols = min(cols, n - r*cols)
        pos.append((cx + (c-(ncols-1)/2)*dxs, cy - (r-(rows-1)/2)*dys))
    return pos

zone_center = {'res1': (1.5, 9.2), 'res2': (9.2, 10.2), 'com1': (4.7, 4.6),
               'com2': (11.4, 4.6), 'ind': (5.7, 0.2)}
ctrl_off = {'res1': (-2.9, 0.0), 'res2': (3.3, 0.6), 'com1': (-3.1, 0.0),
            'com2': (3.3, 0.0), 'ind': (0.0, -2.4)}
pos = {}
for z, d in ZONES.items():
    for sw, p in zip(d['switches'], grid_positions(zone_center[z], len(d['switches']))):
        pos[sw] = p

fig, ax = plt.subplots(figsize=(13.5, 10.5))
# zone regions
for z, d in ZONES.items():
    xs = [pos[s][0] for s in d['switches']]; ys = [pos[s][1] for s in d['switches']]
    x0, x1 = min(xs)-1.05, max(xs)+1.2; y0, y1 = min(ys)-0.95, max(ys)+1.15
    ax.add_patch(FancyBboxPatch((x0, y0), x1-x0, y1-y0,
                 boxstyle="round,pad=0.02,rounding_size=0.3", linewidth=1.4,
                 edgecolor=ZC[z], facecolor=ZC[z], alpha=0.10, zorder=0))
    ax.text((x0+x1)/2, y1+0.16, f"{d['name']}  ({len(d['switches'])} sw)",
            ha='center', va='bottom', fontsize=11.5, fontweight='bold',
            color=ZC[z], zorder=1)
# controller positions
cpos = {d['controller']: (zone_center[z][0]+ctrl_off[z][0], zone_center[z][1]+ctrl_off[z][1])
        for z, d in ZONES.items()}
# control-plane (dashed)
for sw, z in SW_ZONE.items():
    c = ZONE_CTRL[z]
    ax.add_line(Line2D([pos[sw][0], cpos[c][0]], [pos[sw][1], cpos[c][1]],
                color=ZC[z], linewidth=0.6, alpha=0.30, linestyle=(0, (2, 3)), zorder=1))
# intra-zone data links
for z, d in ZONES.items():
    for a, b in zip(d['switches'][:-1], d['switches'][1:]):
        ax.add_line(Line2D([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                    color='#6f6f6f', linewidth=1.7, zorder=2))
# inter-zone backbone
for za, zb in MININET_INTER_LINKS:
    a = ZONES[za]['switches'][-1]; b = ZONES[zb]['switches'][0]
    ax.add_line(Line2D([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color='#1b1b1b', linewidth=2.6, zorder=2))
    mx, my = (pos[a][0]+pos[b][0])/2, (pos[a][1]+pos[b][1])/2
    ax.text(mx, my, f"{get_inter_delay(za, zb)} ms", fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='#1b1b1b', lw=0.6), zorder=3)
# icons
for sw, z in SW_ZONE.items():
    switch_icon(ax, pos[sw][0], pos[sw][1], ZC[z], f's{sw}')
for z, d in ZONES.items():
    c = d['controller']
    server_icon(ax, cpos[c][0], cpos[c][1], ZC[z], f'C{c}', CTRL_PORTS[c])

legend = [Line2D([0],[0], color='#1b1b1b', lw=2.6, label='Data-plane backbone (inter-zone)'),
          Line2D([0],[0], color='#6f6f6f', lw=1.7, label='Data-plane link (intra-zone)'),
          Line2D([0],[0], color='#555', lw=0.9, linestyle=(0,(2,3)),
                 label='Control channel (OpenFlow 1.3)')]
ax.legend(handles=legend, loc='lower right', fontsize=9.5, framealpha=0.96)
ax.set_title('CitiVerse SDN testbed: 20 OpenFlow switches, 5 zones, 5 distributed controllers',
             fontsize=13.5, fontweight='bold')
ax.set_xlim(-4.5, 15.5); ax.set_ylim(-3.6, 12.4); ax.set_aspect('equal'); ax.axis('off')
fig.tight_layout()
out = os.path.join(FIG, 'fig_topology_v2.png'); fig.savefig(out, dpi=300); plt.close(fig)
print('saved', out)
