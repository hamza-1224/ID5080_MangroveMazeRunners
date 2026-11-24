"""Deciding tests: image seam, gaps along lost channels, threshold / gap-closing sensitivity."""
import sys, pickle, os, numpy as np, pandas as pd, rasterio, networkx as nx
from scipy import ndimage as ndi
from skimage import morphology
sys.path.insert(0, 'common')
import sundarbans_rg as srg, tree_nulls as tn
from sundarbans_simplified_network import count_cycles
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)
OUT = sys.argv[1]
d = pickle.load(open('data_processed/timeseries_weighted/graphs/all_graphs.pkl', 'rb'))
idx = d['index']; files = list(idx['file']); names = list(idx['graph']); dates = idx['date'].astype(str).tolist()
grid = srg.common_grid(files); T = grid['transform']; H, W = grid['height'], grid['width']
Gs = d['G']
rgbs = {g: srg.read_rgb(f, grid)[0].astype(np.int16) for g, f in zip(names, files)}
masks = {g: rasterio.open(f"data_processed/timeseries_weighted/{r}/mask.tif").read(1).astype(bool) for g, r in zip(names, dates)}
consensus = sum(masks.values()) >= 5
land_core = ndi.binary_erosion(~consensus, iterations=5)

# a) step across the UTM northing 2,400,000 m tile boundary
seam_row = int(round((T.f - 2400000) / -T.e))
print('northing 2400000 is grid row', seam_row)
rows = []
for g, dt in zip(names, dates):
    rgb = rgbs[g]
    f = lambda a, b, ch: np.median(rgb[ch][a:b][land_core[a:b]])
    rows.append(dict(g=g, date=dt, n_red=f(seam_row - 200, seam_row - 5, 0), s_red=f(seam_row + 5, seam_row + 200, 0),
                     n_green=f(seam_row - 200, seam_row - 5, 1), s_green=f(seam_row + 5, seam_row + 200, 1),
                     north_all_red=f(0, seam_row - 5, 0), south_all_red=f(seam_row + 5, H, 0)))
seam = pd.DataFrame(rows); seam['step_red'] = seam.n_red - seam.s_red; seam['step_green'] = seam.n_green - seam.s_green
print(seam.round(1).to_string()); seam.to_csv(f'{OUT}/seam.csv', index=False)
# locate the largest step for each year from the row profile of land red
for g in ['G6', 'G7', 'G8', 'G9']:
    prof = np.array([np.median(rgbs[g][0][r][land_core[r]]) if land_core[r].sum() > 50 else np.nan for r in range(H)])
    prof = pd.Series(prof).interpolate().rolling(9, center=True).median()
    step = (prof.shift(-14) - prof.shift(14)).abs()
    print(g, 'largest red step at row', int(step.idxmax()), round(step.max(), 1))

# b) lost channels: gaps in the mask along their course, position vs seam
def link_mask(G):
    m = np.zeros((H, W), bool)
    for *_, a in G.edges(data=True): m.flat[a['idx']] = True
    return m
near = {g: ndi.binary_dilation(link_mask(Gs[g]), iterations=6) for g in names}
rows = []
for other in ['G7', 'G8']:
    for ref in ['G6']:
        for u, v, k, a in Gs[ref].edges(keys=True, data=True):
            px = np.asarray(a['idx'])
            if near[other].flat[px].mean() >= .5 or a['width'] > 700: continue     # 700 m+: estuary, not a creek
            wet = ndi.binary_dilation(masks[other], iterations=1).flat[px]
            lab, n = ndi.label(~wet); sizes = ndi.sum(~wet, lab, range(1, n + 1)) if n else np.array([0])
            r, c = np.unravel_index(px, (H, W))
            bg = (rgbs[other][2] - rgbs[other][1]).flat[px]
            gapvals = bg[~wet] if (~wet).any() else np.array([np.nan])
            rows.append(dict(year=other, u=u, v=v, km=a['length'] / 1e3, width_m=a['width'], gaps=n, longest_gap_px=sizes.max(), dry_frac=(~wet).mean(),
                             gap_BG_med=np.nanmedian(gapvals), north_of_seam=np.median(r) < seam_row, r=int(np.median(r)), c=int(np.median(c))))
lost = pd.DataFrame(rows); lost.to_csv(f'{OUT}/lost_creeks.csv', index=False)
print(lost.sort_values(['year', 'km'], ascending=[True, False]).round(2).to_string())
print(lost.groupby('year').agg(n=('km', 'size'), km=('km', 'sum'), med_gap=('longest_gap_px', 'median'), med_dry=('dry_frac', 'median'), north_km=('km', lambda s: s[lost.loc[s.index, 'north_of_seam']].sum())).round(2))
print('share of G6 creek km north of seam:', end=' ')
tot = n = 0
for *_, a in Gs['G6'].edges(data=True):
    if a['width'] > 700: continue
    tot += a['length']; n += a['length'] * (np.median(np.unravel_index(a['idx'], (H, W))[0]) < seam_row)
print(round(n / tot, 2))

# c) sensitivity: re-run every year under other mask rules
shore = 'data_processed/timeseries_weighted/shoreline_common.gpkg'
orig = srg.water_mask
def mask_rule(t, close):
    def f(rgb, min_px=30):
        R, G, B = rgb.astype(np.int16)
        w = (B - G) > t
        if close:
            w = morphology.binary_closing(w, morphology.disk(close))
        w = morphology.remove_small_objects(w, min_px)
        return morphology.remove_small_holes(w, min_px)
    return f
rows = []
for label, t, close in [('B-G>0', 0, 0), ('B-G>-15', -15, 0), ('B-G>-30', -30, 0), ('B-G>+15', 15, 0), ('close 1 px', 0, 1), ('close 2 px', 0, 2)]:
    srg.water_mask = mask_rule(t, close)
    for g, f, dt in zip(names, files, dates):
        try:
            net = srg.extract_network(f, f'{OUT}/s/{label.replace(" ", "_").replace(">", "")}_{g}', grid=grid, shoreline_path=shore)
            G = net['G']; G.remove_edges_from(list(nx.selfloop_edges(G, keys=True)))
            rows.append(dict(rule=label, g=g, date=dt, year=int(dt[:4]), nodes=G.number_of_nodes(), links=G.number_of_edges(), cycles=count_cycles(G),
                             bridge=tn._n_bridges(G) / G.number_of_edges(), inlets=len(net['rn'].nodes['inlets']), outlets=len(net['rn'].nodes['outlets']), comp=nx.number_connected_components(G)))
        except Exception as e:
            rows.append(dict(rule=label, g=g, date=dt, year=int(dt[:4]), error=repr(e)[:80]))
        pd.DataFrame(rows).to_csv(f'{OUT}/sensitivity.csv', index=False)
    print(label, 'done', flush=True)
srg.water_mask = orig
s = pd.DataFrame(rows)
print(s.pivot(index='rule', columns='year', values='cycles').to_string())
print(s.pivot(index='rule', columns='year', values='bridge').round(3).to_string())
print(s.pivot(index='rule', columns='year', values='inlets').to_string())
