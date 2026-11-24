"""Temporal analysis figures (Mohd Hamza): real graphs G1-G10 vs 10 random trees per year.

Run from the project folder:  python members/mohd_hamza/temporal_analysis/make_temporal_figures.py members/mohd_hamza/temporal_analysis/figures
Reads data_processed/timeseries/metrics_G.csv and metrics_baseline_samples.csv (made by scripts/01_build_graphs_and_metrics.py).
Writes comparison_5.pdf, zscores.pdf and trend_tests_5params.csv.
"""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import kendalltau
import os
OUT = sys.argv[1] if len(sys.argv) > 1 else 'members/mohd_hamza/temporal_analysis/figures'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'legend.frameon': False, 'font.family': 'DejaVu Sans'})
r = pd.read_csv('data_processed/timeseries/metrics_G.csv', index_col=0); b = pd.read_csv('data_processed/timeseries/metrics_baseline_samples.csv')
b = b[b['sample'] < 10]          # the first 10 trees per year = a 10-tree run with the same seeds
assert b.groupby('graph').size().eq(10).all()
years = pd.to_datetime(r['date']).dt.year.values
REAL, BASE = '#1a1a1a', '#e6550d'
def axes_years(ax):
    ax.set_xticks(years); ax.set_xticklabels([f"'{str(y)[2:]}" for y in years], fontsize=9.5); ax.grid(axis='y', color='0.9', lw=.8)
    ax.axvspan(2022.6, 2024.4, color='0.93', zorder=0)

panels = [('independent cycles', 'Independent cycles (loops)', 0, (-3, 48)), ('bridge share', 'Bridge share (cut-channels)', 1, (0, 1.08)),
          ('global efficiency', 'Global efficiency (ease of reach)', None, None), ('algebraic connectivity λ2', 'λ₂ (resistance to splitting)', None, None),
          ('mean inlet→sea path (links)', 'Inlet→sea path (links)', None, None)]
fig, axs = plt.subplots(2, 3, figsize=(12, 6.2)); axs = axs.ravel()
for ax, (p, title, tree, ylim) in zip(axs, panels):
    if tree is not None:
        ax.axhline(tree, color=BASE, lw=2.2)
        ax.text(years[0], tree + (2 if tree == 0 else -.07), f'every tree = {tree}', color=BASE, fontsize=9.5, va='bottom' if tree == 0 else 'top')
        ax.set_ylim(*ylim)
    else:
        q = b.groupby('graph')[p].quantile([0, .5, 1]).unstack().loc[r.index]
        ax.fill_between(years, q[0], q[1], color=BASE, alpha=.22, lw=0); ax.plot(years, q[.5], color=BASE, lw=1.8)
    ax.plot(years, r[p], '-o', color=REAL, lw=2, ms=4.5)
    ax.set_title(title); axes_years(ax)
axs[5].axis('off')
axs[5].legend(handles=[Line2D([], [], color=REAL, marker='o', lw=2, label='real network'), Line2D([], [], color=BASE, lw=1.8, label='10 random trees: median'),
                       Patch(color=BASE, alpha=.22, label='10 random trees: range'), Patch(color='0.93', label='2023–24: threshold-sensitive')],
              loc='center', fontsize=11)
fig.tight_layout(h_pad=1.6); fig.savefig(f'{OUT}/comparison_5.pdf'); plt.close(fig)

# distance from the random trees, in standard deviations
ps = [('algebraic connectivity λ2', 'Algebraic connectivity λ₂', '#3182bd'), ('global efficiency', 'Global efficiency', '#31a354'),
      ('mean inlet→sea path (links)', 'Inlet→sea path', '#756bb1')]
cols = [p for p, *_ in ps]
mu = b.groupby('graph')[cols].mean().loc[r.index]; sd = b.groupby('graph')[cols].std().loc[r.index]
z = (r[cols] - mu) / sd
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.axhspan(-2, 2, color='0.9', zorder=0); ax.axhline(0, color='0.4', lw=.8)
ax.text(years[0] - .2, 1.5, 'range typical of random trees (±2 sd)', va='center', fontsize=9.5, color='0.35')
for p, lab, c in ps:
    ax.plot(years, z[p], '-o', color=c, lw=2, ms=5); ax.text(years[-1] + .25, z[p].iloc[-1], lab, color=c, va='center', fontsize=10)
ax.set_xticks(years); ax.set_xticklabels([f"'{str(y)[2:]}" for y in years]); ax.set_xlim(years[0] - .4, years[-1] + 4.6)
ax.set_ylabel('distance from random trees (sd)'); ax.grid(axis='y', color='0.92'); ax.set_ylim(-3, 12)
ax.spines['bottom'].set_bounds(years[0], years[-1])
fig.tight_layout(); fig.savefig(f'{OUT}/zscores.pdf'); plt.close(fig)

rows = [dict(parameter=p, kendall_tau=kendalltau(years, r[p])[0], p_value=kendalltau(years, r[p])[1])
        for p in ['independent cycles', 'bridge share', 'global efficiency', 'algebraic connectivity λ2', 'mean inlet→sea path (links)']]
pd.DataFrame(rows).to_csv(f'{OUT}/trend_tests_5params.csv', index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(z.round(1).agg(['min', 'max']).T)
