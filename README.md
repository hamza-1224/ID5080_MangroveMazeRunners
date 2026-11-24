# Mangrove Maze: loops in the Indian Sundarbans tidal channel network, 2016–2026

ID5080 Complex Networks, project P0X, team Mangrove Maze Runners.

We turn ten yearly Sentinel-2 images of the Indian Sundarbans into channel graphs (junctions = nodes, channels = edges), compare them with random trees and Erdős–Rényi graphs, and follow their structure over 2016–2026. The report is in `report/`, the final figure in `figure/`, and every file is listed in `MANIFEST.md`.

## 1. Install

Needs conda (Miniconda or Anaconda). From the top of this folder:

```
conda env create -f environment.yml
conda activate rivgraph
```

This installs Python 3.12, the geospatial stack and RivGraph (pinned to commit f8745a6 from GitHub). Takes about 5–10 minutes.

## 2. Run

Run every command from the top of this folder. All paths are relative, and all random draws use fixed seeds, so the numbers come out the same each time.

| Step | Command | Makes | Time |
|---|---|---|---|
| 1 | `python scripts/01_build_graphs_and_metrics.py` | graphs G1–G10, baselines B1–B10, parameters, trend tests (`data_processed/timeseries/`) | ~3 min |
| 2 | `python scripts/02_basic_properties.py` | basic properties and Erdős–Rényi comparison (`data_processed/basic_properties/`) | ~2 min |
| 3 | `python scripts/03_simplified_weighted_graphs.py` | weighted, simplified graphs (`data_processed/timeseries_weighted/`) | ~2 min |
| 4 | `python scripts/04_final_figure.py` | `figure/ID5080_MangroveMazeRunners_fig.pdf` | <1 min |
| 5 | `python members/mohd_hamza/temporal_analysis/make_temporal_figures.py` | report Figure 2 and trend tests | <1 min |

Step 1 must run first; steps 2–5 use its output. The outputs of every step are already included in `data_processed/` and `figure/`, so any step can also be run on its own. The 2023–24 checks in `members/mohd_hamza/dip_2023_2024/` need steps 1 and 3; see the README there (one of them takes about 15 minutes).

## 3. Load the graphs in your own code

```python
import sys, pickle, networkx as nx
sys.path.insert(0, 'common')
d = pickle.load(open('data_processed/timeseries/graphs/all_graphs.pkl', 'rb'))
G6 = d['G']['G6']                     # 2022 graph: nx.MultiGraph
inlets, outlets = d['inlets']['G6'], d['outlets']['G6']
G1 = nx.read_graphml('data_processed/timeseries/graphs/G1_2016-10-17.graphml')   # same graphs as GraphML
```

Graph numbers and dates: `data_processed/timeseries/graphs/index.csv`. Nodes carry `kind` (inlet, outlet, junction) and `pos` (UTM zone 45N, metres); edges carry `length` and `width` in metres.

## 4. Folder layout

| Folder | Contents |
|---|---|
| `report/` | The report PDF and its LaTeX source |
| `figure/` | The final network figure (vector PDF) |
| `common/` | Shared pipeline: image → graph (`sundarbans_rg.py`), random trees and parameters (`tree_nulls.py`), weights and simplification (`sundarbans_simplified_network.py`) |
| `scripts/` | The four scripts above |
| `data_raw/` | The Copernicus images exactly as downloaded, never edited |
| `data_processed/` | Everything the scripts produce |
| `members/` | One folder per team member |
| `extras/` | Exploration notebooks, slides, teammates' figures, the reference paper |
