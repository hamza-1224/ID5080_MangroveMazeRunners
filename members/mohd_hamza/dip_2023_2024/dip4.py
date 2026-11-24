"""Stress-test the two candidate long-term signals, and look at them in the images."""
import sys, pickle, numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.stats import kendalltau, theilslopes, spearmanr
from skimage import morphology
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'common')
import sundarbans_rg as srg
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)
OUT = sys.argv[1]
d = pickle.load(open('data_processed/timeseries_weighted/graphs/all_graphs.pkl', 'rb'))
idx = d['index']; files = list(idx['file']); dates = pd.to_datetime(idx['date']); years = dates.dt.year.values
grid = srg.common_grid(files); T = grid['transform']; H, W = grid['height'], grid['width']
pix = T.a * -T.e / 1e6; seam = int(round((T.f - 2400000) / -T.e))
rgb = [srg.read_rgb(f, grid)[0] for f in files]
BG = [r[2].astype(np.int16) - r[1].astype(np.int16) for r in rgb]
def mask(v, t=0):
    return morphology.remove_small_holes(morphology.remove_small_objects(v > t, 30), 30)
z = np.load(f'{OUT}/land_change.npz'); sea = z['sea']
to_sea = ndi.distance_transform_edt(~sea) * .0303

def change(ms, e_i, l_i, label):
    e = sum(ms[i] for i in e_i) >= (len(e_i) + 1) // 2 + (0 if len(e_i) % 2 else 1) * 0
    l = sum(ms[i] for i in l_i) >= (len(l_i) + 1) // 2
    lost = morphology.remove_small_objects(l & ~e, 11); gain = morphology.remove_small_objects(e & ~l, 11)
    f = lambda m, c: round((m & c).sum() * pix, 1)
    print(f'{label:46s} sea face lost {f(lost, to_sea < 2):5} gained {f(gain, to_sea < 2):5} | inland lost {f(lost, to_sea >= 2):5} gained {f(gain, to_sea >= 2):5}')
    return lost, gain
print('--- land change between composites (km2)')
for t in (0, -15, -30):
    ms = [mask(v, t) for v in BG]
    change(ms, range(5), range(5, 10), f'B-G>{t}: 2016-21 vs 2022-26')
    change(ms, range(5), [5, 8, 9], f'B-G>{t}: 2016-21 vs 2022,25,26 (no pale years)')
ms = [mask(v) for v in BG]
lost, gain = change(ms, [0, 1, 2], [7 - 2 + 3, 8, 9] if False else [6 + 2, 9, 5], 'B-G>0: 2016-19 vs 2022,25,26')
print('--- placebo: split with no time order, odd vs even images')
change(ms, [0, 2, 4, 6, 8], [1, 3, 5, 7, 9], 'odd vs even')
change(ms, [0, 3, 5, 8, 9], [1, 2, 4, 6, 7], 'shuffled split')

print('--- annual sea-face land (within 2 km of open sea), km2')
face = [((~m) & (to_sea < 2) & ~sea).sum() * pix for m in ms]
wet_face = pd.Series(face, index=years).round(1); print(wet_face.to_dict())
face_all = [((~m) & (to_sea < 2)).sum() * pix for m in ms]
print('tau', [round(x, 3) for x in kendalltau(years, face)], 'sen km2/yr', round(theilslopes(face, years)[0], 2))

m = pd.read_csv(f'{OUT}/ten_year_measures.csv')
print('--- leave-out checks: creek_120_300_km trend')
for t in (0, -15, -30):
    for zn in ('all', 'south of seam'):
        g = m[(m.t == t) & (m.zone == zn)].set_index('year')['creek_120_300_km']
        loo = [kendalltau(g.drop(y).index, g.drop(y).values)[1] for y in g.index]
        drop = g.drop([2023, 2024]); dropf = g.drop([2016, 2017])
        print(t, zn, 'all p', round(kendalltau(g.index, g.values)[1], 3), 'worst leave-one-out p', round(max(loo), 3), '| without 2023-24 tau', round(kendalltau(drop.index, drop.values)[0], 2), 'p', round(kendalltau(drop.index, drop.values)[1], 3), '| without 2016-17 tau', round(kendalltau(dropf.index, dropf.values)[0], 2), 'p', round(kendalltau(dropf.index, dropf.values)[1], 3))
print('--- do creek numbers depend on date in season or lunar phase?')
g = m[(m.t == 0) & (m.zone == 'all')].sort_values('year')
age = np.array([16.2, 27.0, 19.0, 29.4, 5.3, 25.3, 1.5, 12.7, 0.7, 9.5])
doy = dates.dt.dayofyear.values
semi = np.cos(2 * np.pi * age / 14.765)
for c in ('creek_120_300_km', 'creek_lt120_km', 'land_km2'):
    print(c, 'vs day of year rho', round(spearmanr(doy, g[c])[0], 2), 'p', round(spearmanr(doy, g[c])[1], 2), '| vs spring-neap cos rho', round(spearmanr(semi, g[c])[0], 2), 'p', round(spearmanr(semi, g[c])[1], 2))
print('year vs day of year rho', round(spearmanr(years, doy)[0], 2))

print('--- network parameters without the pale years')
u = pd.read_csv('data_processed/timeseries_weighted/metrics_G_unweighted.csv', index_col=0)
u['year'] = pd.to_datetime(u['date']).dt.year
w = pd.read_csv('data_processed/timeseries_weighted/metrics_G_weighted.csv', index_col=0); u = u.join(w.drop(columns='date'))
for c in ['independent cycles', 'meshedness', 'bridge share', 'mean inlet→sea path (links)', 'edge betweenness Gini (L/w)', 'global efficiency']:
    a = kendalltau(u.year, u[c]); s = u[~u.year.isin([2023, 2024])]; b = kendalltau(s.year, s[c])
    print(f'{c:32s} all tau {a[0]:+.2f} p {a[1]:.2f} | no 2023-24 tau {b[0]:+.2f} p {b[1]:.2f}')

print('--- around the cyclones: Amphan May 2020, Yaas May 2021, Remal May 2024')
print(u.set_index('year')[['independent cycles', 'bridge share', 'edge betweenness Gini (L/w)']].round(3).T.to_string())

# maps
e5 = sum(ms[:5]) >= 3; l3 = sum(ms[i] for i in (5, 8, 9)) >= 2
los = morphology.remove_small_objects(l3 & ~e5, 11); gai = morphology.remove_small_objects(e5 & ~l3, 11)
base = np.ones((H, W, 3)) * .95; base[e5 & l3] = (.72, .82, .92)
base[ndi.binary_dilation(los, iterations=3)] = (.80, .15, .15); base[ndi.binary_dilation(gai, iterations=3)] = (.10, .55, .25)
fig, ax = plt.subplots(figsize=(11, 8.6)); ax.imshow(base, extent=[0, W * T.a / 1e3, 0, H * -T.e / 1e3]); ax.set_xlabel('km'); ax.set_ylabel('km')
ax.axhline((H - seam) * -T.e / 1e3, color='.4', ls='--', lw=.8); ax.text(1, (H - seam) * -T.e / 1e3 + .6, 'tile line (2,400,000 m N)', fontsize=8, color='.3')
ax.set_title('Land change, 2016–21 vs 2022, 2025, 2026 (red = land lost, green = land gained, patches thickened to be visible)')
fig.tight_layout(); fig.savefig(f'{OUT}/land_change_map.png', dpi=130); plt.close(fig)

def crops(mm, n, name):
    lab, k = ndi.label(mm, structure=np.ones((3, 3))); sz = ndi.sum(mm, lab, range(1, k + 1))
    cen = ndi.center_of_mass(mm, lab, [int(i) + 1 for i in np.argsort(sz)[::-1][:n]])
    fig, axs = plt.subplots(n, 4, figsize=(13, 3.2 * n))
    for row, (r, c) in zip(np.atleast_2d(axs), cen):
        r, c = int(r), int(c); a, b = max(r - 70, 0), max(c - 90, 0)
        for ax, i in zip(row, (0, 2, 5, 9)):
            ax.imshow(np.moveaxis(rgb[i][:, a:a + 140, b:b + 180], 0, -1)); ax.set_title(f'{years[i]} (row {r}, col {c})', fontsize=9); ax.axis('off')
    fig.tight_layout(); fig.savefig(f'{OUT}/{name}.png', dpi=75); plt.close(fig)
crops(los & (to_sea < 2), 3, 'crops_seaface_loss'); crops(gai & (to_sea >= 2), 3, 'crops_inland_gain')
