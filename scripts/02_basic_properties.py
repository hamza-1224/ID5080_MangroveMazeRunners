"""Basic network properties of G1-G10 (report Section 5) and comparison with 100 Erdos-Renyi G(n, m) graphs per year.

Run from the top of this folder:  python scripts/02_basic_properties.py
Reads data_processed/timeseries/graphs/all_graphs.pkl; writes data_processed/basic_properties/. Takes about 2 minutes.
"""
import os
import pickle, sys
import numpy as np, pandas as pd, networkx as nx
from pyproj import Transformer

pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
d = pickle.load(open('data_processed/timeseries/graphs/all_graphs.pkl', 'rb'))
idx = d['index']
tf = Transformer.from_crs(32645, 4326, always_xy=True)
N_ER = 100

def props(S):
    n, m = S.number_of_nodes(), S.number_of_edges()
    comps = sorted((len(c) for c in nx.connected_components(S)), reverse=True)
    L = S.subgraph(max(nx.connected_components(S), key=len))
    deg = np.array([k for _, k in S.degree()])
    return dict(n=n, m=m, density=2 * m / (n * (n - 1)), mean_degree=deg.mean(), components=len(comps), largest=comps[0],
                mean_path=nx.average_shortest_path_length(L), diameter=nx.diameter(L), clustering=nx.average_clustering(S),
                transitivity=nx.transitivity(S), loops=m - n + len(comps), planar=nx.check_planarity(S)[0],
                deg1=np.mean(deg == 1), deg3=np.mean(deg == 3), deg_other=np.mean((deg != 1) & (deg != 3)))

rows, er_rows, top = [], [], []
for _, row in idx.iterrows():
    g = row['graph']; M = d['G'][g]
    S = nx.Graph(); S.add_nodes_from(M.nodes(data=True))
    for u, v, a in M.edges(data=True):
        if u != v and not S.has_edge(u, v):
            S.add_edge(u, v, **a)
    p = props(S); p['graph'] = g; p['year'] = str(row['date'])[:4]; p['parallel_links_merged'] = M.number_of_edges() - S.number_of_edges()
    rows.append(p)
    rng = np.random.default_rng(int(g[1:]))
    for i in range(N_ER):
        E = nx.gnm_random_graph(p['n'], p['m'], seed=int(rng.integers(1e9)))
        q = props(E); q['graph'] = g; er_rows.append(q)
    # edge betweenness on the real network, length-free (hops)
    eb = nx.edge_betweenness_centrality(S)
    (u, v), val = max(eb.items(), key=lambda kv: kv[1])
    a = S[u][v]; r, c = np.unravel_index(np.asarray(a['idx']), (1830, 2414))
    T = (30.215793497600036, 652532.54, -30.457631295584385, 2438313.46)
    x = T[1] + (np.median(c) + .5) * T[0]; y = T[3] + (np.median(r) + .5) * T[2]
    lon, lat = tf.transform(x, y)
    nb = nx.betweenness_centrality(S)
    nmax = max(nb, key=nb.get); lon2, lat2 = tf.transform(*S.nodes[nmax]['pos'])
    top.append(dict(graph=g, year=p['year'], edge_betw=round(val, 3), edge_lat=round(lat, 3), edge_lon=round(lon, 3), edge_width_m=round(a['width']),
                    edge_len_km=round(a['length'] / 1e3, 1), node_betw=round(nb[nmax], 3), node_lat=round(lat2, 3), node_lon=round(lon2, 3)))

real = pd.DataFrame(rows).set_index('graph'); er = pd.DataFrame(er_rows)
print(real.round(3).to_string())
print(er.groupby('graph')[['components', 'largest', 'mean_path', 'diameter', 'clustering', 'transitivity', 'loops', 'planar', 'deg1', 'deg3', 'deg_other']].mean().loc[real.index].round(3).to_string())
print(pd.DataFrame(top).to_string())
OUT = 'data_processed/basic_properties'; os.makedirs(OUT, exist_ok=True)
real.to_csv(OUT + '/basic_properties_G.csv'); er.to_csv(OUT + '/basic_properties_ER_samples.csv', index=False); pd.DataFrame(top).to_csv(OUT + '/top_betweenness.csv', index=False)
