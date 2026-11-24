"""Is the 2023-24 dip real? Compare 2022, 2023, 2024, 2025 networks and masks on the common grid."""
import sys, pickle, numpy as np, pandas as pd, rasterio, networkx as nx
from scipy import ndimage as ndi
sys.path.insert(0, 'common')
import sundarbans_rg as srg
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)
OUT = sys.argv[1]
d = pickle.load(open('data_processed/timeseries_weighted/graphs/all_graphs.pkl', 'rb'))
idx = d['index']; Gs = d['G']
files = list(idx['file']); grid = srg.common_grid(files)
H, W = grid['height'], grid['width']
rgbs, B_G = {}, {}
for g, f in zip(idx['graph'], files):
    rgb = srg.read_rgb(f, grid)[0].astype(np.int16); B_G[g] = rgb[2] - rgb[1]; rgbs[g] = rgb
masks = {g: rasterio.open(f"data_processed/timeseries_weighted/{r}/mask.tif").read(1).astype(bool) for g, r in zip(idx['graph'], idx['date'].astype(str))}

# 1. colour statistics per image: land and water level, gap between them, per horizontal band
rows = []
consensus = sum(masks.values()) >= 5
land_core = ndi.binary_erosion(~consensus, iterations=5); water_core = ndi.binary_erosion(consensus, iterations=5)
for g, dt in zip(idx['graph'], idx['date']):
    v = B_G[g]; rgb = rgbs[g]
    rows.append(dict(g=g, date=dt, land_BminusG=np.median(v[land_core]), land_p95=np.percentile(v[land_core], 95), water_BminusG=np.median(v[water_core]),
                     land_green=np.median(rgb[1][land_core]), land_red=np.median(rgb[0][land_core]),
                     frac_near0=np.mean(np.abs(v) < 15), water=masks[g].mean()))
print(pd.DataFrame(rows).round(3).to_string())
# 2. seam: row where mean land colour jumps
for g in idx['graph']:
    m = np.ma.array(rgbs[g][1], mask=~land_core).mean(axis=1).filled(np.nan)
    dif = np.abs(np.nan_to_num(m[12:] - m[:-12]))
    print(g, 'largest land-green jump at row', int(np.argmax(dif)), round(float(dif.max()), 1), 'north/south land green', round(float(np.nanmedian(m[:700])), 1), round(float(np.nanmedian(m[1100:])), 1))

# 3. channel pixel inventory by width class on land side
land = None
rows = []
for g in idx['graph']:
    w = masks[g]; dist = ndi.distance_transform_edt(w)
    from skimage.morphology import skeletonize
    sk = skeletonize(w); widths = 2 * dist[sk] * 30.3
    rows.append(dict(g=g, skel_km=sk.sum() * .0303, lt60=(widths < 60).sum() * .0303, w60_120=((widths >= 60) & (widths < 120)).sum() * .0303, w120_300=((widths >= 120) & (widths < 300)).sum() * .0303, gt300=(widths >= 300).sum() * .0303, n_comp=ndi.label(w, structure=np.ones((3, 3)))[1]))
print(pd.DataFrame(rows).round(0).to_string())

# 4. which links of a reference year are missing in another year
def link_mask(G):
    m = np.zeros((H, W), bool)
    for *_, a in G.edges(data=True):
        m.flat[a['idx']] = True
    return m
lm = {g: link_mask(Gs[g]) for g in Gs}
near = {g: ndi.binary_dilation(lm[g], iterations=6) for g in lm}   # 180 m tolerance
def missing(ref, other):
    out = []
    for u, v, k, a in Gs[ref].edges(keys=True, data=True):
        px = np.asarray(a['idx']); frac = near[other].flat[px].mean()
        if frac < 0.5:
            water = masks[other].flat[px].mean()
            # is the channel itself still water in `other`?  (dilate by 1 px for small shifts)
            wd = ndi.binary_dilation(masks[other], iterations=2).flat[px].mean()
            out.append(dict(ref=ref, other=other, u=u, v=v, len_km=a['length'] / 1e3, width_m=a['width'], covered=frac, water_exact=water, water_2px=wd,
                            r=int(np.median(np.unravel_index(px, (H, W))[0])), c=int(np.median(np.unravel_index(px, (H, W))[1]))))
    return pd.DataFrame(out)
res = []
for ref, other in [('G6', 'G7'), ('G6', 'G8'), ('G9', 'G7'), ('G9', 'G8'), ('G6', 'G9'), ('G9', 'G6'), ('G6', 'G5'), ('G10', 'G6')]:
    m = missing(ref, other); res.append(m)
    print(ref, '->', other, 'links missing', len(m), 'km', round(m.len_km.sum(), 1) if len(m) else 0, 'median width', round(m.width_m.median(), 0) if len(m) else 0, 'still water in mask (within 2px)', round(m.water_2px.mean(), 2) if len(m) else 0)
pd.concat(res).to_csv(f'{OUT}/missing_links.csv', index=False)
m = res[0]; print(m.sort_values('len_km', ascending=False).round(2).head(25).to_string())

# 5. threshold sensitivity: raw and cleaned cycles for B-G > t  and with 1 px closing
pickle.dump({'grid': grid}, open(f'{OUT}/grid.pkl', 'wb'))
