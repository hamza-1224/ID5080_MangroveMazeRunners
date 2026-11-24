# Mohd Hamza (PH22B016)

What I did: downloaded the 10 Sentinel-2 images, ran the team's masking and graph pipeline to build G1–G10, did the temporal analysis against the random-tree baseline, investigated the 2023–24 dip, and wrote the report. Details: `Individual_Contribution_Report.pdf`.

All commands are run from the project folder, with the `rivgraph` conda environment.

## temporal_analysis/

| File | What it is |
|---|---|
| `make_temporal_figures.py` | Makes the two temporal figures and the trend tests. Reads `timeseries/metrics_G.csv` and `timeseries/metrics_baseline_samples.csv` (from `main.ipynb`) and uses the first 10 random trees per year. |
| `figures/comparison_5.pdf` | Five parameters, 2016–2026, real network vs 10 random trees (report Fig. 2, slide "Real network vs random trees"). |
| `figures/zscores.pdf` | Distance of the real network from the random trees, in standard deviations (slide "Trends over ten years"). |
| `figures/trend_tests_5params.csv` | Kendall's tau and p-value for each parameter. |
| `temporal_slides.tex`, `.pdf` | The two Beamer slides. |

Run: `python members/mohd_hamza/temporal_analysis/make_temporal_figures.py members/mohd_hamza/temporal_analysis/figures` (a few seconds).

## dip_2023_2024/

Checks on why loops fall to 31 in 2023–24. Each script takes an output folder as its argument and reads the images and `timeseries_weighted/` outputs.

| File | What it is |
|---|---|
| `dip1.py` | Image colour statistics, channel width inventory, and which links are missing in 2023–24. |
| `dip2.py` | Colour step at the tile line, gaps along the lost creeks, and the rerun of all years under other water thresholds (slow: about 15 min). Writes `seam.csv`, `lost_creeks.csv`, `sensitivity.csv`. |
| `dip3.py` | Ten-year water and creek measures and their trends. Writes `ten_year_measures.csv`, `ten_year_trends.csv`. |
| `dip4.py` | Land change between multi-year composites, and checks without the 2023–24 images. Writes `land_change_map.png`. |
| `sensitivity.csv` | Loops, bridge share, inlets and outlets per year under each water rule (report Sec. 6 table). |
