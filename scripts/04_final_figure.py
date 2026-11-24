"""Final network figure: the 2022 channel network (G6), with channels that lie on a loop and channels that are bridges.

Run from the top of this folder:  python scripts/04_final_figure.py
Reads data_processed/timeseries/graphs/all_graphs.pkl and data_processed/timeseries/2022-10-21/mask.tif.
Writes figure/ID5080_MangroveMazeRunners_fig.pdf: A2 landscape, network in vector over the raster water mask, fonts embedded.
"""
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rasterio
from matplotlib.lines import Line2D

TEAM, PROJECT = 'Mangrove Maze Runners', 'P0X'   # set the project number before submitting
GRAPH, DATE = 'G6', '2022-10-21'
LOOP, BRIDGE = '#0072B2', '#E69F00'        # Okabe-Ito blue and orange: colourblind-safe, distinct in greyscale

plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'DejaVu Sans'})

d = pickle.load(open('data_processed/timeseries/graphs/all_graphs.pkl', 'rb'))
G = d['G'][GRAPH]
inlets, outlets = set(d['inlets'][GRAPH]), set(d['outlets'][GRAPH])
with rasterio.open(f'data_processed/timeseries/{DATE}/mask.tif') as m:
    water = m.read(1).astype(bool)
    T = m.transform
H, W = water.shape

# bridges: subdivide parallel links so two channels around one island count as a loop
S = nx.Graph()
first = {}
for u, v, k in G.edges(keys=True):
    pair = frozenset((u, v))
    if pair in first:
        S.add_edge(u, ('mid', k)); S.add_edge(('mid', k), v)
    else:
        first[pair] = k; S.add_edge(u, v)
bridge_pairs = {frozenset(e) for e in nx.bridges(S)}
is_bridge = {k: frozenset((u, v)) in bridge_pairs and first.get(frozenset((u, v))) == k and G.number_of_edges(u, v) == 1
             for u, v, k in G.edges(keys=True)}
share = np.mean(list(is_bridge.values()))
loops = G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G)

km = lambda c, r: ((c + 0.5) * T.a / 1e3, (H - r - 0.5) * -T.e / 1e3)
fig, ax = plt.subplots(figsize=(23.4, 16.5))          # A2 landscape
fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.04)
ax.imshow(np.where(water[..., None], np.array([.85, .89, .93]), np.array([1, 1, 1])),
          extent=[0, W * T.a / 1e3, 0, H * -T.e / 1e3], interpolation='nearest', zorder=0)
widths = np.array([a['width'] for *_, a in G.edges(data=True)])
rank = (widths.argsort().argsort() + 1) / len(widths)
for (u, v, k, a), rk in zip(G.edges(keys=True, data=True), rank):
    r, c = np.unravel_index(np.asarray(a['idx']), (H, W))
    x, y = km(c, r)
    if is_bridge[k]:
        ax.plot(x, y, color=BRIDGE, lw=2.0 + 3 * rk, solid_capstyle='round', zorder=2)
    else:
        ax.plot(x, y, color=LOOP, lw=2.5 + 4 * rk, solid_capstyle='round', zorder=3)
for n, a in G.nodes(data=True):
    x, y = (a['pos'][0] - T.c) / 1e3, (a['pos'][1] - (T.f + H * T.e)) / 1e3
    if n in inlets:
        ax.plot(x, y, marker='^', ms=13, color='black', zorder=4)
    elif n in outlets:
        ax.plot(x, y, marker='s', ms=11, color='black', mfc='white', mew=2, zorder=4)

ax.set_xlim(0, W * T.a / 1e3); ax.set_ylim(0, H * -T.e / 1e3); ax.set_aspect('equal'); ax.axis('off')
ax.plot([3, 13], [3, 3], color='black', lw=4); ax.text(8, 3.8, '10 km', ha='center', fontsize=24)

fig.text(0.03, 0.955, f'Most Sundarbans channels lie on loops: only {share:.0%} are bridges', fontsize=44, weight='bold')
fig.text(0.03, 0.935, f'Tidal channel network, {DATE} (Sentinel-2): {G.number_of_nodes()} junctions, {G.number_of_edges()} channels, '
         f'{loops} independent loops.\nIn a river-like tree, every channel would be a bridge.', fontsize=24, color='0.25', va='top')
ax.legend(handles=[Line2D([], [], color=LOOP, lw=7, label='channel on a loop: water has another route'),
                   Line2D([], [], color=BRIDGE, lw=5, label='bridge: blocking it cuts the network in two'),
                   Line2D([], [], marker='^', ls='', ms=16, color='black', label='inlet (channel leaves the image inland)'),
                   Line2D([], [], marker='s', ls='', ms=14, color='black', mfc='white', mew=2, label='outlet (channel meets the sea)'),
                   Line2D([], [], color=(.85, .89, .93), lw=14, label='water')],
          loc='lower right', fontsize=22, frameon=True, framealpha=0.95, borderpad=1.0)
fig.text(0.97, 0.012, f'ID5080 · {PROJECT} · {TEAM}', ha='right', fontsize=16, color='0.35')

out = sys.argv[1] if len(sys.argv) > 1 else 'figure/ID5080_MangroveMazeRunners_fig.pdf'
fig.savefig(out)
print(f'{out}: {G.number_of_edges()} channels, bridge share {share:.3f}, {loops} loops')
