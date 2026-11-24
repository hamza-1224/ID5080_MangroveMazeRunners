"""Builds G1-G10 and the random-tree baselines B1-B10 from data_raw/images, computes the network parameters
for each graph and for 1,000 random trees per year (seeded), and writes everything to data_processed/timeseries/.
Converted from the notebook main.ipynb. Takes about 3 minutes.

Run from the top of this folder:  python scripts/01_build_graphs_and_metrics.py
"""
import sys
sys.path.insert(0, 'common')
import matplotlib
matplotlib.use('Agg')
# # Sundarbans channel networks 2016–2026: G1…G10 and the random-tree baseline
# 
# For each image in `data_raw/images/` (October–November, 2016–2026, no 2018), this notebook:
# 
# 1. extracts the cleaned RivGraph network → **G1 … G10** (undirected node–edge graphs; a directed copy is kept too),
# 2. builds the **baseline model B1 … B10**: a random tree on the same year's nodes (outlets = roots, inlets = leaves, junctions get a random parent, max degree 3),
# 3. saves all graphs to `data_processed/timeseries/graphs/`,
# 4. plots the parameters over the years for G and for the baseline (1,000 random trees per year).
# 
# **Same rules every year:**
# * All images are resampled onto **one common grid**: the overlap of the ten footprints, which shift by up to ~3 km between dates.
# * **One shoreline for all years**, made from the water mask that at least 5 of the 10 images agree on. Outlets therefore don't move just because of tide stage.
# * The same water threshold (blue − green > 0), shoreline closing and Sundarbans pruning (`sundarbans_rg.py`).
# 
# Kernel: *Python (rivgraph)*.

import glob, re, os, pickle, warnings
from datetime import datetime
import numpy as np, pandas as pd, networkx as nx, matplotlib.pyplot as plt, geopandas as gpd
from matplotlib.lines import Line2D
import sundarbans_rg as srg
import tree_nulls as tn
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})

OUT = 'data_processed/timeseries'
os.makedirs(f'{OUT}/graphs', exist_ok=True)

def image_date(path):
    name = re.sub(r'\s+', ' ', os.path.basename(path).rsplit('.', 1)[0].replace('Novemb er', 'November'))
    d, m, y = name.split()
    return datetime.strptime(f'{re.sub(r"[^0-9]", "", d)} {m} {y}', '%d %B %Y')

files = sorted(glob.glob('data_raw/images/*.tif*'), key=image_date)
index = pd.DataFrame({'graph': [f'G{k}' for k in range(1, len(files) + 1)],
                      'baseline': [f'B{k}' for k in range(1, len(files) + 1)],
                      'date': [image_date(f).date() for f in files], 'file': files})
index

# ## 1. Common grid and shoreline

grid = srg.common_grid(files)
shore, _ = srg.common_shoreline(files, grid, f'{OUT}/shoreline_common.gpkg')
T = grid['transform']; H, W = grid['height'], grid['width']
EXT = [0, W * T.a / 1e3, 0, H * -T.e / 1e3]
print(f"common grid: {W} x {H} px at {T.a:.1f} x {-T.e:.1f} m = {EXT[1]:.1f} x {EXT[3]:.1f} km, {grid['crs']}")

# ## 2. Extract G1 … G10 and build the baselines B1 … B10

KIND_COL = {'inlet': '#2ca02c', 'outlet': '#ff7f0e', 'junction': '#444444'}
nets, G_und, G_dir, B_und, B_dir = {}, {}, {}, {}, {}
for k, row in index.iterrows():
    gname, bname = row['graph'], row['baseline']
    net = srg.extract_network(row['file'], f"{OUT}/{row['date']}", grid=grid, shoreline_path=f'{OUT}/shoreline_common.gpkg')
    nets[gname] = net
    G_und[gname], G_dir[gname] = net['G'], net['D']
    rng = np.random.default_rng(k)                      # fixed seed -> the saved baseline is reproducible
    B_dir[bname], B_und[bname] = tn.random_parent_tree(net['G'], net['rn'].nodes['inlets'], net['rn'].nodes['outlets'], rng)

counts = pd.DataFrame({g: {'date': index.set_index('graph').loc[g, 'date'],
                           'water fraction': nets[g]['water'].mean(), 'raw links': nets[g]['n_raw_links'],
                           'nodes': G.number_of_nodes(), 'links': G.number_of_edges(),
                           'inlets': len(nets[g]['rn'].nodes['inlets']), 'outlets': len(nets[g]['rn'].nodes['outlets']),
                           'baseline links': B_und[g.replace('G', 'B')].number_of_edges(),
                           'direction loops (RivGraph)': len(list(nx.simple_cycles(nx.DiGraph(G_dir[g]))))}
                       for g, G in G_und.items()}).T
counts

# ### Save the graphs
# 
# | file | contents |
# |---|---|
# | `data_processed/timeseries/graphs/G{k}_{date}.graphml` | undirected node–edge graph. Nodes: `kind` (inlet/outlet/junction), `x`, `y` (UTM 45N, m). Links: `length_m`, `width_m` |
# | `data_processed/timeseries/graphs/B{k}_{date}.graphml` | baseline random tree on the same nodes |
# | `data_processed/timeseries/graphs/all_graphs.pkl` | everything as networkx objects: `G` and `B`, undirected and directed (`D` is upstream → downstream for G, child → parent for B) |
# | `data_processed/timeseries/graphs/index.csv` | G/B number ↔ date ↔ image file |
# 
# Read back with `nx.read_graphml(path)` or `pickle.load(open('data_processed/timeseries/graphs/all_graphs.pkl', 'rb'))`.

def for_graphml(g):
    h = nx.MultiGraph()
    for n, d in g.nodes(data=True):
        h.add_node(n, kind=d['kind'], x=float(d['pos'][0]), y=float(d['pos'][1]))
    for u, v, k, d in g.edges(keys=True, data=True):
        h.add_edge(u, v, key=k, **({'length_m': float(d['length']), 'width_m': float(d['width'])} if 'length' in d else {}))
    return h

for _, row in index.iterrows():
    nx.write_graphml(for_graphml(G_und[row['graph']]), f"{OUT}/graphs/{row['graph']}_{row['date']}.graphml")
    nx.write_graphml(for_graphml(B_und[row['baseline']]), f"{OUT}/graphs/{row['baseline']}_{row['date']}.graphml")
with open(f'{OUT}/graphs/all_graphs.pkl', 'wb') as fh:
    pickle.dump({'index': index, 'G': G_und, 'G_directed': G_dir, 'B': B_und, 'B_directed': B_dir,
                 'inlets': {g: list(nets[g]['rn'].nodes['inlets']) for g in G_und},
                 'outlets': {g: list(nets[g]['rn'].nodes['outlets']) for g in G_und}}, fh)
index.to_csv(f'{OUT}/graphs/index.csv', index=False)
sorted(os.listdir(f'{OUT}/graphs'))

# ## 3. The graphs: G1 … G10 (node–edge, no weights or direction)

def km_pos(g):
    return {n: ((d['pos'][0] - T.c) / 1e3, (d['pos'][1] - (T.f + H * T.e)) / 1e3) for n, d in g.nodes(data=True)}

def draw_graph(g, ax, title, edge_color='#333'):
    pos = km_pos(g)
    nx.draw_networkx_edges(nx.Graph(g), pos, ax=ax, edge_color=edge_color, width=0.8)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=[14 if g.nodes[n]['kind'] != 'junction' else 4 for n in g],
                           node_color=[KIND_COL[g.nodes[n]['kind']] for n in g], linewidths=0)
    ax.set_xlim(EXT[:2]); ax.set_ylim(EXT[2:]); ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title, fontsize=10)

legend = [Line2D([], [], marker='o', ls='', color=c, label=k) for k, c in KIND_COL.items()]
fig, axs = plt.subplots(2, 5, figsize=(25, 9.5))
for ax, (_, row) in zip(axs.ravel(), index.iterrows()):
    g = G_und[row['graph']]
    draw_graph(g, ax, f"{row['graph']}  ({row['date']})\n{g.number_of_nodes()} nodes, {g.number_of_edges()} links, "
                      f"{g.number_of_edges() - g.number_of_nodes() + nx.number_connected_components(g)} cycles")
fig.legend(handles=legend, loc='lower center', ncol=3); plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig(f'{OUT}/G_all_years.png', dpi=110); plt.close("all")

fig, axs = plt.subplots(2, 5, figsize=(25, 9.5))
for ax, (_, row) in zip(axs.ravel(), index.iterrows()):
    b = B_und[row['baseline']]
    draw_graph(b, ax, f"{row['baseline']}  baseline for {row['date']}\n{b.number_of_nodes()} nodes, {b.number_of_edges()} links, 0 cycles", edge_color='#c55a11')
fig.legend(handles=legend, loc='lower center', ncol=3); plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig(f'{OUT}/B_all_years.png', dpi=110); plt.close("all")

# ## 4. Parameters over the years: G vs the baseline
# 
# For each year: the parameters of **G<sub>k</sub>** and of **1,000 random baseline trees** on that year's nodes (B<sub>k</sub> is one of them).
# The parameters are the same as in `the 2016 exploration notebooks in extras/` (definitions in `tree_nulls.py`), with one change.
# 
# **Flow direction:** RivGraph's direction heuristics leave directed loops in some years (see `direction loops` in the table above). Tidal channels flow both ways anyway.
# So for the time series every link is oriented **from the node farther from the sea to the node closer to it** (distance in links). Links joining two nodes at the same distance are left out of the direction-based parameters.
# This rule is the same for every year and for every baseline tree. It affects *alternative paths per inlet* and *outlets reachable per inlet*. In a tree it still gives exactly 1 of each.

N = 1000
rows, base_rows = [], []
for k, row in index.iterrows():
    g = row['graph']; G, D = G_und[g], G_dir[g]
    ins, outs = nets[g]['rn'].nodes['inlets'], nets[g]['rn'].nodes['outlets']
    rows.append({'graph': g, 'date': row['date'], **tn.compare_metrics(G, D, ins, outs, orientation='sea_distance')})
    rng = np.random.default_rng(1000 + k)
    for i in range(N):
        bd, bu = tn.random_parent_tree(G, ins, outs, rng)
        base_rows.append({'graph': g, 'date': row['date'], 'sample': i, **tn.compare_metrics(bu, bd, ins, outs, orientation='sea_distance')})
real = pd.DataFrame(rows).set_index('graph')
base = pd.DataFrame(base_rows)
real.to_csv(f'{OUT}/metrics_G.csv'); base.to_csv(f'{OUT}/metrics_baseline_samples.csv', index=False)
real.drop(columns='date').round(3).T

years = pd.to_datetime(real['date']).dt.year.values
params = ['independent cycles', 'meshedness', 'bridge share', 'alt. paths per inlet (log10)',
          'outlets reachable per inlet', 'mean inlet→sea path (links)', 'resistance inlet→sea (Ω)',
          'resistance / path length', 'global efficiency', 'algebraic connectivity λ2', 'edge betweenness Gini']
fig, axs = plt.subplots(3, 4, figsize=(20, 12))
for ax, p in zip(axs.ravel(), params):
    q = base.groupby('graph')[p].quantile([0.05, 0.5, 0.95]).unstack().loc[real.index]
    ax.fill_between(years, q[0.05], q[0.95], color='#e6550d', alpha=0.25, label='baseline trees 5–95%')
    ax.plot(years, q[0.5], color='#e6550d', lw=1.5, label='baseline median')
    ax.plot(years, real[p], 'k-o', lw=2, ms=5, label='G (real)')
    ax.set_title(p, fontsize=11); ax.set_xticks(years); ax.tick_params(axis='x', rotation=45)
axs.ravel()[-1].axis('off')
axs[0, 0].legend(fontsize=8)
plt.suptitle('Network parameters 2016–2026: real graphs G1–G10 vs random-tree baseline', y=1.0, fontsize=14)
plt.tight_layout(); plt.savefig(f'{OUT}/trends_G_vs_baseline.png', dpi=110); plt.close("all")

# ### Distance from the baseline over time (z-score)
# 
# z = (G − baseline mean) / baseline sd for the parameters that vary between random trees.
# This removes the effect of node count, which changes from year to year with tide and image quality. A change in z is a change in **structure** relative to a random tree with the same nodes.

zparams = ['mean inlet→sea path (links)', 'resistance inlet→sea (Ω)', 'global efficiency', 'algebraic connectivity λ2', 'edge betweenness Gini']
mu, sd = base.groupby('graph')[zparams].mean().loc[real.index], base.groupby('graph')[zparams].std().loc[real.index]
z = (real[zparams] - mu) / sd
z.to_csv(f'{OUT}/zscores_vs_baseline.csv')
fig, ax = plt.subplots(figsize=(11, 5))
for p, c in zip(zparams, plt.get_cmap('tab10').colors):
    ax.plot(years, z[p], '-o', color=c, label=p)
ax.axhspan(-2, 2, color='0.92', zorder=0); ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(years); ax.set_ylabel('z vs random-tree baseline'); ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
ax.set_title('How far each year’s network is from a random tree (grey = within ±2 sd)')
plt.tight_layout(); plt.savefig(f'{OUT}/zscores_vs_baseline.png', dpi=110); plt.close("all")
z.round(1)

# ### Simple trend test
# 
# Kendall's τ and Sen's slope for each parameter of G over the years. This is the Mann–Kendall test that is standard for short environmental series.
# With 10 points it only detects strong monotonic trends, so treat p < 0.05 as a lead to check, not a result.

from scipy.stats import kendalltau, theilslopes
trend = {}
for p in params:
    tau, pval = kendalltau(years, real[p])
    slope = theilslopes(real[p], years)[0]
    trend[p] = {'Kendall tau': tau, 'p-value': pval, 'Sen slope per year': slope,
                'G 2016': real[p].iloc[0], 'G 2026': real[p].iloc[-1]}
trend = pd.DataFrame(trend).T
trend.to_csv(f'{OUT}/trend_tests.csv')
trend.round(3)

# ## 5. Notes from this run
# 
# **Every year is far from the random-tree baseline, and the gap barely changes.**
# * 31–42 independent cycles every year (a tree has 0). Bridge share is 0.28–0.37 (a tree: 1). Resistance ÷ path length is 0.37–0.48 (a tree: 1).
# * Against 1,000 baseline trees per year:
#   * λ₂ (hard to cut in two) is **5–8 sd above** the baseline.
#   * Resistance to the sea is **3.5–5 sd below**.
#   * Global efficiency is **3–5 sd above**.
# * The real network has *longer* inlet → sea paths than the baseline (z = +2 to +4.6), because the baseline joins nodes regardless of distance. The edge-betweenness Gini is only modestly lower than in the trees (z = −0.8 to −2.6).
# 
# **No significant trend 2016–2026.**
# * Mann–Kendall p > 0.05 for every parameter.
# * The strongest is meshedness: τ = −0.47, p = 0.07, from 0.125 (2016) to 0.121 (2026). Most of that comes from a dip in 2023–2024.
# * The z-scores for λ₂ (8.0 → 5.6), efficiency (4.9 → 3.0) and Gini (−2.6 → −0.8) drift towards the baseline. That would mean slightly more tree-like over time, but 2016 is the most different year (fewest nodes), so this needs checking before reading it as a trend.
# 
# **Years to check by eye before interpreting:**
# * **2023 and 2024 (G7, G8)** drop to 31 cycles and ~170 links, and their bridge share rises to 0.36–0.37. 2024 also has only 14 inlets. Water fraction is normal, so check the masks for channels that are missing (narrow creeks broken at the image edge, haze) before calling this a real loss of loops.
# * **2016 (G1)** has the fewest nodes, which drives its high λ₂ and efficiency.
# * **2017** has haze or cloud patches on the western islands.
# 
# **Limits of this first pass:**
# * One scene per year, with tide stage unknown. Dates range from mid-October to mid-November.
# * ~30 m pixels and no 2018 image.
# * The alternative-paths and outlets-reachable numbers are small and noisy under the sea-distance orientation.
# * RivGraph's own flow directions contain loops in 2016, 2024 and 2025.
