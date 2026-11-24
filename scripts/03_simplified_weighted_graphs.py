"""Builds width/length-weighted, simplified versions of G1-G10 (common/sundarbans_simplified_network.py),
compares weighted and unweighted parameters with 10 random trees per year, and writes data_processed/timeseries_weighted/.
Converted from the notebook main_weighted.ipynb. Takes about 2 minutes.

Run from the top of this folder:  python scripts/03_simplified_weighted_graphs.py
"""
import sys
sys.path.insert(0, 'common')
import matplotlib
matplotlib.use('Agg')
# # Sundarbans channel networks 2016–2026: weighted, simplified G1…G10 and the baseline
# 
# This script redoes `scripts/01_build_graphs_and_metrics.py` on the **simplified, weighted** graphs. `scripts/01_build_graphs_and_metrics.py` and the `data_processed/timeseries/` folder are left as they were; everything here is saved in `data_processed/timeseries_weighted/`.
# 
# For each image in `data_raw/images/` (October–November, 2016–2026, no 2018):
# 
# 1. extract the cleaned RivGraph network, the same way as in `scripts/01_build_graphs_and_metrics.py`,
# 2. add weights: `conductance` = width / length, and `weight` = conductance ÷ the graph's largest conductance,
# 3. contract redundant links with the thresholds in `sundarbans_simplified_network.py`, which gives **G1 … G10**,
# 4. build the **baseline B1 … B10**: a random tree on the same year's nodes,
# 5. compare them over the years.
# 
# **Simplification settings (the file's defaults):**
# * Pass 1: contract the highest-weight links above median + 2 × 1.4826 × MAD, at most 7 per graph.
# * Pass 2: contract links shorter than 120 m.
# * Inlets and outlets are never contracted, real loops are never deleted, and **no node has a self-edge**.
# 
# **Which graph is used for which parameter:**
# * Parameters that **don't need weights** (cycles, bridges, paths in links, and so on) use the **unweighted** graphs.
# * Parameters that **do need weights** use **`conductance`**, not `weight`. `weight` is scaled by each graph's own maximum, so it can't be compared between years.
# 
# Kernel: *Python (rivgraph)*.

import glob, re, os, pickle, warnings
from datetime import datetime
import numpy as np, pandas as pd, networkx as nx, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sundarbans_rg as srg
import tree_nulls as tn
from sundarbans_simplified_network import (build_simplified_network, to_unweighted, load_graphs,
                                           assign_baseline_weights, count_cycles)
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})

OUT = 'data_processed/timeseries_weighted'
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

# ## 2. Build G1 … G10 and the baselines B1 … B10
# 
# The variables are dictionaries keyed by name, for example `G_und['G3']` or `B_und['B3']`:
# 
# | variable | contents |
# |---|---|
# | `G_und` | weighted, simplified G1–G10 (undirected) |
# | `G_unw` | the same graphs without weights |
# | `G_dir` | directed copies, same contractions |
# | `B_und` | baseline trees, with geometry |
# | `B_unw` | baseline trees without weights |
# 
# **Baseline geometry:** a random tree's links join arbitrary nodes, so they have no width or length of their own. Each baseline link is given the width and length of a different real link from that year, chosen at random. The baseline therefore has the same kind of channels as the real network, placed at random.

KIND_COL = {'inlet': '#2ca02c', 'outlet': '#ff7f0e', 'junction': '#444444'}
nets, G_und, G_unw, G_dir, B_und, B_unw, B_dir, inlets, outlets = {}, {}, {}, {}, {}, {}, {}, {}, {}
for k, row in index.iterrows():
    gname, bname = row['graph'], row['baseline']
    net = build_simplified_network(row['file'], f"{OUT}/{row['date']}", grid=grid, shoreline_path=f'{OUT}/shoreline_common.gpkg')
    nets[gname], G_und[gname], G_dir[gname] = net, net['G'], net['D']
    G_unw[gname] = to_unweighted(net['G'])
    inlets[gname], outlets[gname] = net['inlets'], net['outlets']
    rng = np.random.default_rng(k)                      # fixed seed -> the saved baseline is reproducible
    B_dir[bname], tree = tn.random_parent_tree(net['G'], net['inlets'], net['outlets'], rng)
    B_und[bname] = assign_baseline_weights(tree, net['G'], rng)
    B_unw[bname] = to_unweighted(tree)

assert all(nx.number_of_selfloops(g) == 0 for d in (G_und, G_unw, G_dir, B_und, B_unw) for g in d.values())
print('no graph has a self-edge')

counts = pd.DataFrame({g: {'date': index.set_index('graph').loc[g, 'date'],
                           'nodes before': nets[g]['G_raw'].number_of_nodes(), 'links before': nets[g]['G_raw'].number_of_edges(),
                           'nodes': G.number_of_nodes(), 'links': G.number_of_edges(),
                           'cycles before': count_cycles(nets[g]['G_raw']), 'cycles': count_cycles(G),
                           'contracted by weight': len(nets[g]['history_df']), 'contracted by length': len(nets[g]['spatial_history_df']),
                           'self-edges removed': nets[g]['self_edges_removed'],
                           'inlets': len(inlets[g]), 'outlets': len(outlets[g]),
                           'baseline links': B_und[g.replace('G', 'B')].number_of_edges()}
                       for g, G in G_und.items()}).T
counts.to_csv(f'{OUT}/graph_counts.csv')
counts

# ### Save the graphs
# 
# | file | contents |
# |---|---|
# | `graphs/G{k}_{date}_weighted.graphml` | simplified graph. Nodes: `kind`, `x`, `y` (UTM 45N, m), `merged`. Links: `weight`, `conductance`, `length_m`, `width_m` |
# | `graphs/G{k}_{date}_unweighted.graphml` | the same graph, links without attributes |
# | `graphs/B{k}_{date}_weighted.graphml`, `..._unweighted.graphml` | the baseline tree, both ways |
# | `graphs/all_graphs.pkl` | everything as networkx objects, weighted (`G_directed` and `B_directed` have no weights on B) |
# | `graphs/index.csv` | G/B number ↔ date ↔ image file |

def for_graphml(g, weighted=True):
    h = nx.MultiGraph()
    for n, d in g.nodes(data=True):
        h.add_node(n, kind=d['kind'], x=float(d['pos'][0]), y=float(d['pos'][1]), merged=bool(d.get('merged', False)))
    for u, v, k, d in g.edges(keys=True, data=True):
        h.add_edge(u, v, key=k, **({'weight': float(d['weight']), 'conductance': float(d['conductance']),
                                    'length_m': float(d['length']), 'width_m': float(d['width'])} if weighted else {}))
    return h

for _, row in index.iterrows():
    for name, d in ((row['graph'], G_und), (row['baseline'], B_und)):
        nx.write_graphml(for_graphml(d[name]), f"{OUT}/graphs/{name}_{row['date']}_weighted.graphml")
        nx.write_graphml(for_graphml(d[name], weighted=False), f"{OUT}/graphs/{name}_{row['date']}_unweighted.graphml")
with open(f'{OUT}/graphs/all_graphs.pkl', 'wb') as fh:
    pickle.dump({'index': index, 'G': G_und, 'G_directed': G_dir, 'B': B_und, 'B_directed': B_dir,
                 'inlets': inlets, 'outlets': outlets}, fh)
index.to_csv(f'{OUT}/graphs/index.csv', index=False)
len(os.listdir(f'{OUT}/graphs'))

# ### Getting weighted or unweighted graphs later
# 
# You don't need to re-run this notebook. Either load the saved graphs, or strip the weights from a graph you already have:

PKL = f'{OUT}/graphs/all_graphs.pkl'
weighted = load_graphs(PKL)                          # {'G1': graph, ...} with weights
unweighted = load_graphs(PKL, weighted=False)        # same graphs, no weights
baseline_unweighted = load_graphs(PKL, weighted=False, baseline=True)

g = weighted['G3']; u = unweighted['G3']
u_simple = to_unweighted(g, simple=True)             # plain nx.Graph: merges parallel links, so it hides their loops
print('weighted link:  ', {k: round(v, 3) for k, v in next(iter(g.edges(data=True)))[2].items() if k in ('weight', 'conductance', 'length', 'width')})
print('unweighted link:', next(iter(u.edges(data=True)))[2])
print(f'cycles: weighted {count_cycles(g)}, unweighted {count_cycles(u)}, simple {count_cycles(u_simple)}')

# ## 3. The graphs: G1 … G10 (link thickness shows weight rank)

def km_pos(g):
    return {n: ((d['pos'][0] - T.c) / 1e3, (d['pos'][1] - (T.f + H * T.e)) / 1e3) for n, d in g.nodes(data=True)}

def draw_graph(g, ax, title, edge_color='#333'):
    pos = km_pos(g)
    rank = pd.Series([d['weight'] for *_, d in g.edges(data=True)]).rank(pct=True).values   # raw weights are dominated by a few estuary links
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=edge_color, width=0.3 + 2.2 * rank ** 2)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=[14 if g.nodes[n]['kind'] != 'junction' else 4 for n in g],
                           node_color=[KIND_COL[g.nodes[n]['kind']] for n in g], linewidths=0)
    merged = [n for n in g if g.nodes[n].get('merged')]
    nx.draw_networkx_nodes(g, pos, nodelist=merged, ax=ax, node_size=14, node_color='red', linewidths=0)
    ax.set_xlim(EXT[:2]); ax.set_ylim(EXT[2:]); ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title, fontsize=10)

legend = [Line2D([], [], marker='o', ls='', color=c, label=k) for k, c in {**KIND_COL, 'merged junction': 'red'}.items()]
fig, axs = plt.subplots(2, 5, figsize=(25, 9.5))
for ax, (_, row) in zip(axs.ravel(), index.iterrows()):
    g = G_und[row['graph']]
    draw_graph(g, ax, f"{row['graph']}  ({row['date']})\n{g.number_of_nodes()} nodes, {g.number_of_edges()} links, {count_cycles(g)} cycles")
fig.legend(handles=legend, loc='lower center', ncol=4); plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig(f'{OUT}/G_all_years.png', dpi=110); plt.close("all")

fig, axs = plt.subplots(2, 5, figsize=(25, 9.5))
for ax, (_, row) in zip(axs.ravel(), index.iterrows()):
    b = B_und[row['baseline']]
    draw_graph(b, ax, f"{row['baseline']}  baseline for {row['date']}\n{b.number_of_nodes()} nodes, {b.number_of_edges()} links, 0 cycles", edge_color='#c55a11')
fig.legend(handles=legend[:3], loc='lower center', ncol=3); plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig(f'{OUT}/B_all_years.png', dpi=110); plt.close("all")

# ## 4. Parameters over the years: G vs the baseline
# 
# For each year: the parameters of **G<sub>k</sub>** and of **1,000 random baseline trees** on that year's nodes.
# 
# **Unweighted parameters** use the unweighted graphs: every link counts as 1. These are the same parameters as `scripts/01_build_graphs_and_metrics.py`, with direction from the node farther from the sea to the node closer to it.
# 
# **Weighted parameters** use `conductance` = width / length, so each link has resistance L/w. A short, wide channel conducts easily and a long, narrow one doesn't.
# 
# | weighted parameter | unweighted version | what the weights change |
# |---|---|---|
# | resistance inlet→sea (L/w) | resistance inlet→sea (Ω) | wide channels now carry more than narrow creeks |
# | global efficiency (L/w) | global efficiency | distance is measured in L/w, not in links |
# | algebraic connectivity λ₂ (w/L) | algebraic connectivity λ₂ | cutting a wide channel costs more than cutting a creek |
# | edge betweenness Gini (L/w) | edge betweenness Gini | shortest routes prefer wide channels |
# 
# In the weighted parameters the sea is one perfectly conducting node, with all outlets merged into it.

N = 10
rows_u, rows_w, base_u, base_w = [], [], [], []
for k, row in index.iterrows():
    g = row['graph']
    ins, outs = inlets[g], outlets[g]
    rows_u.append({'graph': g, 'date': row['date'], **tn.compare_metrics(G_unw[g], None, ins, outs, orientation='sea_distance')})
    rows_w.append({'graph': g, 'date': row['date'], **tn.weighted_metrics(G_und[g], ins, outs)})
    rng = np.random.default_rng(1000 + k)
    for i in range(N):
        bd, bu = tn.random_parent_tree(G_und[g], ins, outs, rng)
        base_u.append({'graph': g, 'sample': i, **tn.compare_metrics(to_unweighted(bu), None, ins, outs, orientation='sea_distance')})
        base_w.append({'graph': g, 'sample': i, **tn.weighted_metrics(assign_baseline_weights(bu, G_und[g], rng), ins, outs)})
real_u, real_w = pd.DataFrame(rows_u).set_index('graph'), pd.DataFrame(rows_w).set_index('graph')
base_u, base_w = pd.DataFrame(base_u), pd.DataFrame(base_w)
real_u.to_csv(f'{OUT}/metrics_G_unweighted.csv'); real_w.to_csv(f'{OUT}/metrics_G_weighted.csv')
base_u.to_csv(f'{OUT}/metrics_baseline_unweighted_samples.csv', index=False)
base_w.to_csv(f'{OUT}/metrics_baseline_weighted_samples.csv', index=False)
real_u.drop(columns='date').round(3).T

real_w.drop(columns='date').round(4).T

# ### Parameters that don't need weights (unweighted graphs)

years = pd.to_datetime(real_u['date']).dt.year.values

def trend_grid(real, base, params, shape, figsize, title, path):
    fig, axs = plt.subplots(*shape, figsize=figsize)
    for ax, p in zip(axs.ravel(), params):
        q = base.groupby('graph')[p].quantile([0.05, 0.5, 0.95]).unstack().loc[real.index]
        ax.fill_between(years, q[0.05], q[0.95], color='#e6550d', alpha=0.25, label='baseline trees 5–95%')
        ax.plot(years, q[0.5], color='#e6550d', lw=1.5, label='baseline median')
        ax.plot(years, real[p], 'k-o', lw=2, ms=5, label='G (real)')
        ax.set_title(p, fontsize=11); ax.set_xticks(years); ax.tick_params(axis='x', rotation=45)
    for ax in axs.ravel()[len(params):]:
        ax.axis('off')
    axs.ravel()[0].legend(fontsize=8)
    plt.suptitle(title, y=1.0, fontsize=14)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close("all")

params_u = ['independent cycles', 'meshedness', 'bridge share', 'alt. paths per inlet (log10)',
            'outlets reachable per inlet', 'mean inlet→sea path (links)', 'resistance inlet→sea (Ω)',
            'resistance / path length', 'global efficiency', 'algebraic connectivity λ2', 'edge betweenness Gini']
trend_grid(real_u, base_u, params_u, (3, 4), (20, 12), 'Unweighted parameters 2016–2026: G1–G10 vs random-tree baseline', f'{OUT}/trends_unweighted.png')

# ### Parameters that need weights (conductance = width / length)

params_w = ['resistance inlet→sea (L/w)', 'global efficiency (L/w)', 'algebraic connectivity λ2 (w/L)', 'edge betweenness Gini (L/w)']
trend_grid(real_w, base_w, params_w, (1, 4), (20, 4.6), 'Weighted parameters 2016–2026: G1–G10 vs random-tree baseline', f'{OUT}/trends_weighted.png')

# ### Distance from the baseline over time (z-score)
# 
# z = (G − baseline mean) / baseline sd, for parameters that vary between random trees.
# A change in z is a change in **structure** relative to a random tree on the same nodes. The weighted panel compares against trees with the same year's channel geometry.

def zscores(real, base, ps):
    gb = base.groupby('graph')[ps]
    return (real[ps] - gb.mean().loc[real.index]) / gb.std().loc[real.index]

z_u = zscores(real_u, base_u, ['mean inlet→sea path (links)', 'resistance inlet→sea (Ω)', 'global efficiency', 'algebraic connectivity λ2', 'edge betweenness Gini'])
z_w = zscores(real_w, base_w, params_w)
z_u.to_csv(f'{OUT}/zscores_unweighted.csv'); z_w.to_csv(f'{OUT}/zscores_weighted.csv')
fig, axs = plt.subplots(1, 2, figsize=(18, 5))
for ax, z, title in zip(axs, (z_u, z_w), ('unweighted parameters', 'weighted parameters')):
    for p, c in zip(z.columns, plt.get_cmap('tab10').colors):
        ax.plot(years, z[p], '-o', color=c, label=p)
    ax.axhspan(-2, 2, color='0.92', zorder=0); ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(years); ax.set_ylabel('z vs random-tree baseline'); ax.legend(fontsize=8); ax.set_title(title)
plt.tight_layout(); plt.savefig(f'{OUT}/zscores.png', dpi=110); plt.close("all")
pd.concat([z_u, z_w], axis=1).round(1).T

# ### Simple trend test
# 
# Kendall's τ and Sen's slope for each parameter of G over the years, as in `scripts/01_build_graphs_and_metrics.py`.
# With 10 points it only detects strong monotonic trends. There are 15 tests here, so about one could fall below p = 0.05 by chance: treat a low p as a lead to check, not a result.

from scipy.stats import kendalltau, theilslopes
trend = {}
for real, ps in ((real_u, params_u), (real_w, params_w)):
    for p in ps:
        tau, pval = kendalltau(years, real[p])
        trend[p] = {'Kendall tau': tau, 'p-value': pval, 'Sen slope per year': theilslopes(real[p], years)[0],
                    'G 2016': real[p].iloc[0], 'G 2026': real[p].iloc[-1]}
trend = pd.DataFrame(trend).T
trend.to_csv(f'{OUT}/trend_tests.csv')
trend.round(4)

# ## 5. Notes from this run
# 
# **Simplification.** Every year had exactly 7 contractions by weight and none by length. The 7 is the `max_contractions` limit, not the threshold. Each graph loses 7 nodes and 7 links and keeps all its cycles, inlets and outlets. No self-edges existed, so none were removed, and none exist now.
# 
# Compared with `scripts/01_build_graphs_and_metrics.py`, the mean inlet → sea path falls from 5.9 to 5.4 links, as expected with 7 links fewer. Mean cycles (38.2) and bridge share (0.31) stay the same, so the earlier conclusions don't depend on these contractions.
# 
# **Unweighted parameters: same picture as `scripts/01_build_graphs_and_metrics.py`.**
# * 31–42 independent cycles every year (a tree has 0), and bridge share 0.28–0.36 (a tree: 1).
# * λ₂ is 5.2–8.5 sd above the baseline, resistance to the sea 3.9–5.2 sd below it, and global efficiency 3.5–5.8 sd above it.
# * Edge betweenness Gini is only 1.0–2.5 sd below the baseline: counting every link equally, routes look a little more evenly spread than in a random tree.
# 
# **Weighted parameters: what the widths add.**
# * **Edge betweenness Gini reverses.** With widths included, the real Gini is 0.61–0.64 against about 0.45 for the baseline (6.3–8.4 sd *above*). Short, wide routes concentrate on a few trunk channels, so wide channels are connected to each other instead of being scattered through the network. This is the most stable weighted result: it stays within 0.61–0.64 in all ten years.
# * **Efficiency and λ₂ are also far above the baseline** (5.9–10.6 and 5.5–12.4 sd), which fits the same picture of a connected network of wide channels.
# * **Resistance to the sea is less unusual** once widths count: 1.6–2.6 sd below the baseline, against 3.9–5.2 sd unweighted. Part of the unweighted advantage comes from counting narrow side channels as equal to wide ones.
# * The weighted baseline uses the same year's channel widths and lengths at random positions, so these z-scores measure **how the channels are arranged**, not how wide they are.
# 
# **No significant trend 2016–2026.** All 15 parameters have p > 0.1. The strongest is meshedness: τ = −0.42, p = 0.11, from 0.132 to 0.126.
# 
# **Read these with care:**
# * **Weighted efficiency and λ₂ are noisy.** Both swing from year to year, and the baseline median swings with them (peaks in 2017 and 2024), which suggests a few very wide estuary links drive those years. Prefer the weighted Gini for trends.
# * **2023 and 2024 (G7, G8)** still show the dip to 31 cycles and bridge share 0.35–0.36. This has not been checked against the masks, so it could be missing channels, not real change.
# * `conductance` = width / length is a geometric stand-in. It ignores depth, tide and bed roughness, and links several km wide are really estuary.
# * One scene per year, tide stage unknown, ~30 m pixels, no 2018 image, and 2017 has haze on the western islands.
