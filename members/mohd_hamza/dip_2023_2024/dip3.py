"""Which ten-year changes survive threshold changes and appear south of the tile seam (the control region)?"""
import sys, pickle, numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.stats import kendalltau, theilslopes
from skimage import morphology
from skimage.morphology import skeletonize
sys.path.insert(0, 'common')
import sundarbans_rg as srg
pd.set_option('display.width', 240); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
OUT = sys.argv[1]
d = pickle.load(open('data_processed/timeseries_weighted/graphs/all_graphs.pkl', 'rb'))
idx = d['index']; files = list(idx['file']); names = list(idx['graph']); years = idx['date'].astype(str).str[:4].astype(int).values
grid = srg.common_grid(files); T = grid['transform']; H, W = grid['height'], grid['width']
pix_km2 = T.a * -T.e / 1e6; seam = int(round((T.f - 2400000) / -T.e))
BG = [(lambda a: a[2] - a[1])(srg.read_rgb(f, grid)[0].astype(np.int16)) for f in files]

def mask(v, t):
    w = v > t
    w = morphology.remove_small_objects(w, 30)
    return morphology.remove_small_holes(w, 30)

base = [mask(v, 0) for v in BG]
consensus = sum(base) >= 5
land_shape = srg.closed_land(consensus, r=80)          # delta plain, sea excluded (same for every year)
sea = ~land_shape
zones = {'all': np.ones((H, W), bool), 'south of seam': np.arange(H)[:, None] > seam + 5, 'north of seam': np.arange(H)[:, None] < seam - 5}
zones = {k: np.broadcast_to(v, (H, W)) for k, v in zones.items()}

rows = []
for t in (0, -15, -30):
    for y, v in zip(years, BG):
        w = mask(v, t) if t else base[list(years).index(y)]
        dist = ndi.distance_transform_edt(w); sk = skeletonize(w)
        wid = 2 * dist * 30.3
        for zn, z in zones.items():
            zl = z & land_shape
            s = sk & zl
            rows.append(dict(t=t, zone=zn, year=y,
                             creek_lt120_km=(s & (wid < 120)).sum() * .0303, creek_120_300_km=(s & (wid >= 120) & (wid < 300)).sum() * .0303,
                             channel_gt300_km=(s & (wid >= 300)).sum() * .0303, inland_water_km2=(w & zl).sum() * pix_km2, land_km2=(~w & zl).sum() * pix_km2,
                             water_pieces=ndi.label(w & zl, structure=np.ones((3, 3)))[1]))
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/ten_year_measures.csv', index=False)
print(df[(df.t == 0) & (df.zone == 'all')].drop(columns=['t', 'zone']).round(0).to_string())
print(df[(df.t == 0) & (df.zone == 'south of seam')].drop(columns=['t', 'zone']).round(0).to_string())

trend = []
for (t, zn), g in df.groupby(['t', 'zone']):
    for c in ['creek_lt120_km', 'creek_120_300_km', 'channel_gt300_km', 'inland_water_km2', 'land_km2', 'water_pieces']:
        tau, p = kendalltau(g.year, g[c]); sl = theilslopes(g[c], g.year)[0]
        trend.append(dict(t=t, zone=zn, measure=c, tau=tau, p=p, slope=sl, pct_per_yr=100 * sl / g[c].median()))
tr = pd.DataFrame(trend); tr.to_csv(f'{OUT}/ten_year_trends.csv', index=False)
print(tr.pivot_table(index=['measure', 'zone'], columns='t', values='tau').round(2).to_string())
print(tr.pivot_table(index=['measure', 'zone'], columns='t', values='p').round(3).to_string())
print(tr[tr.t == 0].pivot_table(index='measure', columns='zone', values='pct_per_yr').round(2).to_string())

# network parameters: trend under other thresholds (reads whatever dip2 has finished)
try:
    s = pd.read_csv(f'{OUT}/sensitivity.csv')
    for rule, g in s.dropna(subset=['cycles']).groupby('rule'):
        print(rule, 'n', len(g), *[f"{c}: tau {kendalltau(g.year, g[c])[0]:+.2f} p {kendalltau(g.year, g[c])[1]:.2f}" for c in ('cycles', 'links', 'bridge', 'outlets')])
    print(s.pivot(index='rule', columns='year', values='cycles').to_string())
    print(s.pivot(index='rule', columns='year', values='bridge').round(3).to_string())
except Exception as e:
    print('no sensitivity yet', e)

# land change between two five-image composites, which damp single-scene tide effects
early, late = sum(base[:5]) >= 3, sum(base[5:]) >= 3        # water in most of 2016-21 / most of 2022-26
lost_land = morphology.remove_small_objects(late & ~early, 11)      # land -> water, >= 1 ha
gained = morphology.remove_small_objects(early & ~late, 11)         # water -> land
to_sea = ndi.distance_transform_edt(~sea) * .0303                   # km from open sea
for name, m in (('land lost', lost_land), ('land gained', gained)):
    print(name, round(m.sum() * pix_km2, 1), 'km2; on sea face (<2 km from sea)', round((m & (to_sea < 2)).sum() * pix_km2, 1),
          '; inland', round((m & (to_sea >= 2)).sum() * pix_km2, 1), '| south of seam', round(m[seam:].sum() * pix_km2, 1))
print('mid-period gap between composites: about 5.5 years')
lab, n = ndi.label(lost_land, structure=np.ones((3, 3)))
sizes = ndi.sum(lost_land, lab, range(1, n + 1)) * pix_km2
top = np.argsort(sizes)[::-1][:10]
com = ndi.center_of_mass(lost_land, lab, [i + 1 for i in top])
from pyproj import Transformer
tf = Transformer.from_crs(32645, 4326, always_xy=True)
for i, (r, c) in zip(top, com):
    lon, lat = tf.transform(T.c + c * T.a, T.f + r * T.e)
    print(f'lost patch {sizes[i]:.2f} km2 at {lat:.3f}N {lon:.3f}E  row {int(r)} col {int(c)} dist_sea_km {to_sea[int(r), int(c)]:.1f}')
np.savez_compressed(f'{OUT}/land_change.npz', early=early, late=late, lost=lost_land, gained=gained, sea=sea, consensus=consensus)
