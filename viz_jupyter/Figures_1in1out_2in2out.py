"""
Figures_1in1out_2in2out.py
Compare 1in1out and 2in2out emulator outputs.
All figures are saved as PDF vector graphics (dpi=200).
"""

import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='OM4 resolution analysis: movies and static plots')
parser.add_argument('--freq', choices=['yearly', 'monthly','all'], default='yearly',
                    help='Movie frequency: yearly or monthly')
parser.add_argument('--var', choices=['temperature', 'ohc'], default='temperature',
                    help='What to generate: temperature movies, ohc movies')
args = parser.parse_args()

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures

import xarray as xr
import torch
import xrft
import numpy as np
import datetime
import cftime
import matplotlib.pyplot as plt
import sys
import re
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import fsspec
import xesmf as xe
from scipy.stats import pearsonr
import os
from functions import *
from functions import _display_name, reorder_legend_paired
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

# ============================================================
# Helper: save current figure as PDF
# ============================================================
def save_current_fig(name, output_dir, dpi=200):
    """Save the current matplotlib figure as a PDF vector graphic."""
    path = os.path.join(output_dir, f"{name}.pdf")
    plt.savefig(path, format='pdf', dpi=dpi, bbox_inches='tight')
    plt.close('all')
    print(f"  -> Saved: {path}")


# ============================================================
# Configuration
# ============================================================
# Shared constants (DEPTH_SLICES, DEPTH_LAYERS, BASINS_TO_PLOT,
# VAR_CONFIG, RHO, CP, SCALE_FACTOR, OCEAN_FIELD_NAME_PREFIXES)
# are imported from functions.py via `from functions import *`.

TIME_INDICES = [0, 300, 580]

# ============================================================
# 1. Output Directory (Configurable Name)
# ============================================================
OUTPUT_DIR = "Figures_1in1out_2in2out"
PDF_DIR = os.path.join(OUTPUT_DIR, "pdf_figures")

my_key = "XL8EJRUDQ7QPDJSUUC8Q"
my_secret = "2HaAeDBeiyGBxeeS60t6iOrrwgW6iebpiqoYkAHM"
fs = fsspec.filesystem("s3", key=my_key, secret=my_secret, client_kwargs={"endpoint_url": "https://nyu1.osn.mghpcc.org"})
storage_options = {"key": my_key, "secret": my_secret, "client_kwargs": {"endpoint_url": "https://nyu1.osn.mghpcc.org"}}

# 2. Emulator Configuration
EMULATOR_CONFIGS = [
    (
        '/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/checkpoints/2026-06-26-samudra_om4_v2_hist0_steps8_1deg_mem467_eval_ema/predictions.zarr',
        "1in1out",
        "tab:gray"
    ),
    (
        '/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/checkpoints/ablation_studies/2026-01-30-samudra_om4_onedeg_20_cap_eval/predictions.zarr',
        "2in2out",
        "tab:orange"
    )
]

TRUTH_CONFIGS = [
    ("/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/onedeg/OM4.zarr", "OM4 (1$^\\circ$)"),
    ("/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/onedeg/OM4.zarr", "OM4 (1$^\\circ$)")
]

LEVS_PATH = '/pscratch/sd/a/asubel/Data/Chapter_2/CM4_CMIP_New.zarr/'
OCEAN_SAMPLE_PATH = '/pscratch/sd/a/asubel/full-model/scratch/ocean_sample.nc'
STATIC_GRID_PATH = 's3://m2lines-pubs/FOMO/raw/ocean_static_no_mask_table.zarr'

print(f"Output Directory: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

# Create subdirectories for organized figure storage
SPECTRUM_DIR = os.path.join(PDF_DIR, "spectrum")
TIMESERIES_DIR = os.path.join(PDF_DIR, "timeseries")
VARIANCE_DIR = os.path.join(PDF_DIR, "variance")
ZONAL_BIAS_DIR = os.path.join(PDF_DIR, "zonal_bias")
SPATIAL_BIAS_DIR = os.path.join(PDF_DIR, "spatial_bias")
ZONAL_MEAN_DIR = os.path.join(PDF_DIR, "zonal_mean")
SPATIAL_DISTRIBUTION_DIR = os.path.join(PDF_DIR, "spatial_distribution")

for d in [SPECTRUM_DIR, TIMESERIES_DIR, VARIANCE_DIR, ZONAL_BIAS_DIR,
          SPATIAL_BIAS_DIR, ZONAL_MEAN_DIR, SPATIAL_DISTRIBUTION_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Load Data
# ============================================================
print("Loading Truth Data...")
data_truth_ocean_list = []
for truth_path, truth_label in TRUTH_CONFIGS:
    print(f" - Loading {truth_label} from {truth_path}...")
    if "s3" in truth_path:
        data_truth = xr.open_zarr(truth_path, storage_options=storage_options)
    else:
        data_truth = xr.open_zarr(truth_path)
    data_truth_ocean_list.append(data_truth)

print("Loading Emulators...")
emulators_raw = []
emulator_labels = []
emulator_colors = []

for path, label, color in EMULATOR_CONFIGS:
    try:
        print(f" - Loading {label} from {path}...")
        ds = xr.open_zarr(path).rename({'lat':'y','lon':'x'})
        emulators_raw.append(ds)
        emulator_labels.append(label)
        emulator_colors.append(color)
    except Exception as e:
        print(f"   Error loading {label}: {e}")

if not emulators_raw:
    print("No emulators loaded. Exiting.")
    sys.exit(1)

# Sync each truth dataset to the matching emulator rollout. The two rollouts
# start one output step apart, so using the first emulator for all truths leaves
# the second truth with an extra timestamp.
for i in range(len(data_truth_ocean_list)):
    time_slice = slice(emulators_raw[i].time.values[0], emulators_raw[i].time.values[-1])
    data_truth_ocean_list[i] = data_truth_ocean_list[i].sel(time=time_slice)

# Load Aux Data
print("Loading Auxiliary Data...")
# levs = xr.open_zarr(LEVS_PATH)['lev'].values
# ocean_test = xr.open_dataset(OCEAN_SAMPLE_PATH)
# depth_levels = concat_levels(ocean_test, 'idepth', 'levels')
# dz_from_data = (depth_levels[1:] - depth_levels[:-1]).data

levs = np.array([
    2.5, 10, 22.5, 40, 65, 105, 165, 250, 375,
    550, 775, 1050, 1400, 1850, 2400, 3100, 4000, 5000, 6000,
])
depth_levels = np.array([
    0, 5, 15, 30, 50, 80, 130, 200, 300, 450, 650,
    900, 1200, 1600, 2100, 2700, 3500, 4500, 5500, 6750,
], dtype=float)
dz_from_data = (depth_levels[1:] - depth_levels[:-1])

# Load Grid & Generate Masks
shape_processed = (emulators_raw[0].y.size, emulators_raw[0].x.size)
print(f"Loading target grid for shape {shape_processed}...")
target_grid_path = f"s3://m2lines-pubs/FOMO/raw/grids/gaussian_grid_{shape_processed[0]}_by_{shape_processed[1]}.zarr"
target_grid = xr.open_zarr(target_grid_path, storage_options={"anon": True, "client_kwargs": {"endpoint_url": "https://nyu1.osn.mghpcc.org"}})
target_grid = target_grid.rename({
    "grid_x": "x_b", "grid_y": "y_b", "grid_xt": "x", "grid_yt": "y",
    "grid_lon": "lon_b", "grid_lat": "lat_b", "grid_lont": "lon", "grid_latt": "lat",
}).compute()

print("Regridding basin mask...")
ds_static = xr.open_zarr(STATIC_GRID_PATH, storage_options={"anon": True, "client_kwargs": {"endpoint_url": "https://nyu1.osn.mghpcc.org"}})
ds_source = ds_static[['basin', 'geolat', 'geolon']].rename({'geolat': 'lat', 'geolon': 'lon'})
ds_target = target_grid[['lat', 'lon']]
regridder = xe.Regridder(ds_source, ds_target, method='nearest_s2d', periodic=True, reuse_weights=False)
basin_regridded = regridder(ds_source['basin'])

# Grid Metrics
R = 6.371e6
dx_deg = target_grid['lon_b'].diff('x_b')[:-1].values
dy_deg = target_grid['lat_b'].diff('y_b')[:,:-1].values
dx_cal = np.deg2rad(dx_deg)*np.cos(np.deg2rad(target_grid['lat'].values))*R
dy_cal = np.deg2rad(dy_deg)*R

grid_metrics = {
    'dx': (['y','x'], dx_cal),
    'dy': (['y','x'], dy_cal),
    'areacello': (['y','x'], xe.util.cell_area(target_grid, R).compute().data),
    'basin': (['y','x'], basin_regridded.compute().data)
}

# Process Data
print("Processing Datasets...")
processed_emulators = []
processed_truths = []

for idx, emu_ds in enumerate(emulators_raw):
    shape = (emu_ds.y.size, emu_ds.x.size)
    print(f"Processing resolution {shape}...")

    target_grid_path = f"s3://m2lines-pubs/FOMO/raw/grids/gaussian_grid_{shape[0]}_by_{shape[1]}.zarr"
    target_grid_res = xr.open_zarr(target_grid_path, storage_options={"anon": True, "client_kwargs": {"endpoint_url": "https://nyu1.osn.mghpcc.org"}})
    target_grid_res = target_grid_res.rename({
        "grid_x": "x_b", "grid_y": "y_b", "grid_xt": "x", "grid_yt": "y",
        "grid_lon": "lon_b", "grid_lat": "lat_b", "grid_lont": "lon", "grid_latt": "lat",
    }).compute()

    ds_target_res = target_grid_res[['lat', 'lon']]
    regridder_res = xe.Regridder(ds_source, ds_target_res, method='nearest_s2d', periodic=True, reuse_weights=False)
    basin_regridded_res = regridder_res(ds_source['basin'])

    dx_deg_res = target_grid_res['lon_b'].diff('x_b')[:-1].values
    dy_deg_res = target_grid_res['lat_b'].diff('y_b')[:,:-1].values
    dx_cal_res = np.deg2rad(dx_deg_res)*np.cos(np.deg2rad(target_grid_res['lat'].values))*R
    dy_cal_res = np.deg2rad(dy_deg_res)*R

    grid_metrics_res = {
        'dx': (['y','x'], dx_cal_res),
        'dy': (['y','x'], dy_cal_res),
        'areacello': (['y','x'], xe.util.cell_area(target_grid_res, R).compute().data),
        'basin': (['y','x'], basin_regridded_res.compute().data)
    }

    processed_emulators.append(process_dataset(emu_ds, levs, dz_from_data, grid_metrics_res))
    processed_truths.append(process_dataset(data_truth_ocean_list[idx], levs, dz_from_data, grid_metrics_res))

# Construct Emulator Dictionary
emulators_to_analyze = {}
for i, ds in enumerate(processed_emulators):
    emulators_to_analyze[emulator_labels[i]] = {
        'ds': ds,
        'color': emulator_colors[i],
        'truth': processed_truths[i],
        'truth_label': TRUTH_CONFIGS[i][1]
    }

# Basin masks, zonal mean, ENSO, and time series
from functions import extract_basin_masks
masks_list = []
for label, emu_data in emulators_to_analyze.items():
    ds_truth = emu_data['truth']
    masks = extract_basin_masks(ds_truth)
    masks_list.append(masks)


# ============================================================
# Zonal Mean Distribution (per basin)
# ============================================================
from functions import plot_zonal_mean_comparison_together
plot_zonal_mean_comparison_together(
    emulators_to_analyze,
    variable_name='thetao',
    units='$^\\circ$C',
    masks_list=masks_list,
    colorbar_fix=True,
    deseason_metrics=True
)
save_current_fig("zonal_mean_thetao", ZONAL_MEAN_DIR)

# ============================================================
# Global Zonal Mean Distribution (all ocean, no basin split)
# ============================================================
print("\n" + "="*60)
print("Global Zonal Mean Distribution (All Ocean)")
print("="*60)

from functions import plot_zonal_mean_global_comparison_together
plot_zonal_mean_global_comparison_together(
    emulators_to_analyze,
    variable_name='thetao',
    units='$^\\circ$C',
    colorbar_fix=True,
    deseason_metrics=True
)
save_current_fig("zonal_mean_global_thetao", ZONAL_MEAN_DIR)

# ============================================================
# ENSO Evaluation
# ============================================================
print("\n" + "="*60)
print("ENSO Evaluation (Niño 3.4)")
print("="*60)

emulators_to_analyze_enso = emulators_to_analyze

from functions import plot_nino34_comparison_together, compute_nino34_metrics

ENSO_DIR = os.path.join(PDF_DIR, "enso")
os.makedirs(ENSO_DIR, exist_ok=True)

# Niño 3.4 Index time series
plot_nino34_comparison_together(emulators_to_analyze_enso)
save_current_fig("nino34_index", ENSO_DIR)

# Niño 3.4 numerical metrics
nino_metrics = compute_nino34_metrics(emulators_to_analyze_enso)
print("\nNiño 3.4 Metrics:")
with pd.option_context('display.float_format', '{:.4f}'.format):
    print(nino_metrics.to_string(index=False))
csv_path = os.path.join(ENSO_DIR, "nino34_metrics.csv")
nino_metrics.to_csv(csv_path, index=False)
print(f"  -> Saved: {csv_path}")

# ============================================================
# Time Series
# ============================================================
print("\n" + "="*60)
print("Time Series Plots")
print("="*60)

# Time Series (Thetao)
plot_mean_timeseries_comparison_together(
    emulators_to_analyze,
    variable_name='thetao',
    var_units='$^\\circ$C',
    var_label='Temperature',
    output_dir=OUTPUT_DIR,
    overlay=True
)
save_current_fig("timeseries_thetao_only", TIMESERIES_DIR)

# Time Series (Salinity)
plot_mean_timeseries_comparison_together(
    emulators_to_analyze,
    variable_name='so',
    var_units='psu',
    var_label='Salinity',
    output_dir=OUTPUT_DIR,
    overlay=True
)
save_current_fig("timeseries_so_only", TIMESERIES_DIR)

# Detrended Time Series (Thetao)
plot_mean_timeseries_detrended_comparison_together(
    emulators_to_analyze,
    variable_name='thetao',
    var_units='$^\\circ$C',
    var_label='Temperature',
    output_dir=OUTPUT_DIR,
    overlay=True
)
save_current_fig("timeseries_thetao_detrended", TIMESERIES_DIR)

# Detrended Time Series (Salinity)
plot_mean_timeseries_detrended_comparison_together(
    emulators_to_analyze,
    variable_name='so',
    var_units='psu',
    var_label='Salinity',
    output_dir=OUTPUT_DIR,
    overlay=True
)
save_current_fig("timeseries_so_detrended", TIMESERIES_DIR)

# ============================================================
# Temporal Variance
# ============================================================
print("\n" + "="*60)
print("Temporal Variance")
print("="*60)

plot_variance_comparison_together(
    emulators_to_analyze,
    variable_name='thetao',
    units='$^\\circ$C',
    output_dir=OUTPUT_DIR,
    colorbar_fix=True
)
save_current_fig("variance_thetao", VARIANCE_DIR)

plot_variance_comparison_together(
    emulators_to_analyze,
    variable_name='so',
    units='psu',
    output_dir=OUTPUT_DIR,
    colorbar_fix=True
)
save_current_fig("variance_so", VARIANCE_DIR)

plot_ke_variance_comparison_together(
    emulators_to_analyze,
    var_u='uo',
    var_v='vo',
    output_dir=OUTPUT_DIR,
    colorbar_fix=True,
    n_levels=12
)
save_current_fig("variance_ke", VARIANCE_DIR)

# ============================================================
# Temporal Variance Numerical Metrics
# ============================================================
print("\n" + "="*60)
print("Temporal Variance Numerical Metrics")
print("="*60)

from functions import compute_temporal_variance_metrics, compute_ke_temporal_variance_metrics, format_variance_metrics_table

for var_name in ['thetao', 'so']:
    print(f"\n--- {var_name} ---")
    raw = compute_temporal_variance_metrics(
        emulators_dict=emulators_to_analyze,
        variable_name=var_name,
        spectrum_regions=SPECTRUM_REGIONS,
    )
    table = format_variance_metrics_table(raw)
    with pd.option_context('display.float_format', '{:.4e}'.format,
                           'display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', 300):
        print(table.to_string())
    csv_path = os.path.join(PDF_DIR, f"temporal_variance_metrics_{var_name}.csv")
    table.to_csv(csv_path)
    print(f"  -> Saved: {csv_path}")

print("\n--- KE ---")
ke_raw = compute_ke_temporal_variance_metrics(
    emulators_dict=emulators_to_analyze,
    spectrum_regions=SPECTRUM_REGIONS,
)
ke_table = format_variance_metrics_table(ke_raw)
with pd.option_context('display.float_format', '{:.4e}'.format,
                       'display.max_rows', None,
                       'display.max_columns', None,
                       'display.width', 300):
    print(ke_table.to_string())
csv_path = os.path.join(PDF_DIR, "temporal_variance_metrics_ke.csv")
ke_table.to_csv(csv_path)
print(f"  -> Saved: {csv_path}")

# ============================================================
# Zonal Mean Bias
# ============================================================
print("\n" + "="*60)
print("Zonal Mean Bias")
print("="*60)

plot_ohc_zonal_bias_ocean_comparison_together(
    emulators_to_analyze,
    masks_list=masks_list,
    colorbar_fix=True
)
save_current_fig("ohc_zonal_bias_ocean", ZONAL_BIAS_DIR)

plot_zonal_bias_ocean_comparison_together(
    emulators_to_analyze,
    'thetao',
    '$^\\circ$C',
    masks_list,
    OUTPUT_DIR,
    colorbar_fix=True
)
save_current_fig("zonal_bias_ocean_thetao", ZONAL_BIAS_DIR)

plot_snapshot_zonal_bias_ocean_comparison_together(
    emulators_dict=emulators_to_analyze,
    variable_name='thetao',
    units='°C',
    masks_list=masks_list,
    TIME_INDICES=TIME_INDICES,
    colorbar_fix=True
)
save_current_fig("snapshot_zonal_bias_ocean_thetao", ZONAL_BIAS_DIR)

plot_snapshot_ohc_zonal_bias_ocean_comparison_together(
    emulators_dict=emulators_to_analyze,
    masks_list=masks_list,
    TIME_INDICES=TIME_INDICES,
    colorbar_fix=True
)
save_current_fig("snapshot_ohc_zonal_bias_ocean", ZONAL_BIAS_DIR)

# ============================================================
# Global Spatial Bias Maps
# ============================================================
print("\n" + "="*60)
print("Global Spatial Bias Maps")
print("="*60)

plot_ohc_bias_comparison_together(emulators_to_analyze, colorbar_fix=True)
save_current_fig("ohc_bias", SPATIAL_BIAS_DIR)

plot_temperature_bias_comparison_together(emulators_to_analyze, colorbar_fix=True)
save_current_fig("temperature_bias", SPATIAL_BIAS_DIR)

plot_ohc_snapshot_bias_comparison_together(emulators_to_analyze, TIME_INDICES, colorbar_fix=True)
save_current_fig("ohc_snapshot_bias", SPATIAL_BIAS_DIR)

plot_temperature_snapshot_bias_comparison_together(emulators_to_analyze, TIME_INDICES, colorbar_fix=True)
save_current_fig("temperature_snapshot_bias", SPATIAL_BIAS_DIR)

from functions import plot_ke_snapshot_bias_comparison_together
plot_ke_snapshot_bias_comparison_together(emulators_to_analyze, TIME_INDICES, DEPTH_LAYERS)
save_current_fig("ke_snapshot_bias", SPATIAL_BIAS_DIR)


# ============================================================
# Global Spatial Distribution Maps (non-deseason, metrics on deseasoned data)
# ============================================================
print("\n" + "="*60)
print("Global Spatial Distribution Maps")
print("="*60)

plot_ohc_mean_comparison_together(
    emulators_to_analyze,
    colorbar_fix=True
)
save_current_fig("ohc_mean", SPATIAL_DISTRIBUTION_DIR)

plot_temperature_mean_comparison_together(
    emulators_to_analyze,
    colorbar_fix=True
)
save_current_fig("temperature_mean", SPATIAL_DISTRIBUTION_DIR)

VARIABLE_TO_PLOT = 'thetao'

plot_snapshot_comparison_multidepth_together(
    emulators_to_analyze,
    var_name=VARIABLE_TO_PLOT,
    time_indices=TIME_INDICES,
    depth_configs=DEPTH_LAYERS,
    fix_colorbar=False,
    remove_seasonal=False,
    deseason_metrics=True,
    n_levels=12
)
save_current_fig("snapshot_multidepth_thetao", SPATIAL_DISTRIBUTION_DIR)

# Velocity snapshots
for vel_var in ['uo', 'vo']:
    plot_snapshot_comparison_multidepth_together(
        emulators_to_analyze,
        var_name=vel_var,
        time_indices=TIME_INDICES,
        depth_configs=DEPTH_LAYERS,
        fix_colorbar=False,
        remove_seasonal=False,
        deseason_metrics=True,
        n_levels=12
    )
    save_current_fig(f"snapshot_multidepth_{vel_var}", SPATIAL_DISTRIBUTION_DIR)

plot_ohc_snapshot_comparison_multidepth_together(
    emulators_to_analyze,
    time_indices=TIME_INDICES,
    depth_configs=DEPTH_SLICES,
    colorbar_fix=True
)
save_current_fig("ohc_snapshot_multidepth", SPATIAL_DISTRIBUTION_DIR)

from functions import plot_ke_snapshot_comparison_multidepth_together
plot_ke_snapshot_comparison_multidepth_together(
    emulators_to_analyze,
    time_indices=TIME_INDICES,
    depth_configs=DEPTH_LAYERS,
)
save_current_fig("ke_snapshot_multidepth", SPATIAL_DISTRIBUTION_DIR)

# ============================================================
# Global Spatial Distribution Maps (deseasoned anomalies)
# ============================================================
print("\n" + "="*60)
print("Global Spatial Distribution Maps (Deseasoned)")
print("="*60)

plot_snapshot_comparison_multidepth_together(
    emulators_to_analyze,
    var_name=VARIABLE_TO_PLOT,
    time_indices=TIME_INDICES,
    depth_configs=DEPTH_LAYERS,
    fix_colorbar=False,
    remove_seasonal=True,
    n_levels=12
)
save_current_fig("snapshot_multidepth_thetao_deseasoned", SPATIAL_DISTRIBUTION_DIR)

for vel_var in ['uo', 'vo']:
    plot_snapshot_comparison_multidepth_together(
        emulators_to_analyze,
        var_name=vel_var,
        time_indices=TIME_INDICES,
        depth_configs=DEPTH_LAYERS,
        fix_colorbar=False,
        remove_seasonal=True,
        n_levels=12
    )
    save_current_fig(f"snapshot_multidepth_{vel_var}_deseasoned", SPATIAL_DISTRIBUTION_DIR)

plot_ke_snapshot_comparison_multidepth_together(
    emulators_to_analyze,
    time_indices=TIME_INDICES,
    depth_configs=DEPTH_LAYERS,
    remove_seasonal=True,
    n_levels=12
)
save_current_fig("ke_snapshot_multidepth_deseasoned", SPATIAL_DISTRIBUTION_DIR)


# ============================================================
# T vs U Anomaly Snapshot: 2 rows (T, uo) x 3 cols (OM4, 1in1out, 2in2out)
# ============================================================
print("\n" + "="*60)
print("T vs U Anomaly Snapshot")
print("="*60)

from matplotlib.gridspec import GridSpec

# Remove seasonal cycle
def _deseason(da):
    clim = da.groupby('time.dayofyear').mean('time')
    return (da.groupby('time.dayofyear') - clim).drop_vars('dayofyear')

# Get datasets
emu_labels_list = list(emulators_to_analyze.keys())
ds_truth = emulators_to_analyze[emu_labels_list[0]]['truth']
ds_one_in_one_out = emulators_to_analyze[emu_labels_list[0]]['ds']
ds_two_in_two_out = emulators_to_analyze[emu_labels_list[1]]['ds']

# Deseason all fields
print("Removing seasonal cycle...")
truth_thetao_anom = _deseason(ds_truth['thetao'])
truth_uo_anom = _deseason(ds_truth['uo'])
one_in_one_thetao_anom = _deseason(ds_one_in_one_out['thetao'])
one_in_one_uo_anom = _deseason(ds_one_in_one_out['uo'])
two_in_two_thetao_anom = _deseason(ds_two_in_two_out['thetao'])
two_in_two_uo_anom = _deseason(ds_two_in_two_out['uo'])

# Depth: deep ocean (3100m, index 15) where imprinting is most visible
DEPTH_IDX = 15
DEPTH_LABEL = '3100 m'

for time_idx in TIME_INDICES:
    time_val = ds_truth.time.isel(time=time_idx).values
    time_str = str(time_val)[:10]
    print(f"Plotting T vs U snapshot at t={time_str}, depth={DEPTH_LABEL}...")

    # Transposed layout: rows=variables, cols=datasets
    # panels[row, col] where row=variable, col=dataset
    panels = {
        (0, 0): truth_thetao_anom.isel(time=time_idx, lev=DEPTH_IDX).compute(),
        (0, 1): one_in_one_thetao_anom.isel(time=time_idx, lev=DEPTH_IDX).compute(),
        (0, 2): two_in_two_thetao_anom.isel(time=time_idx, lev=DEPTH_IDX).compute(),
        (1, 0): truth_uo_anom.isel(time=time_idx, lev=DEPTH_IDX).compute(),
        (1, 1): one_in_one_uo_anom.isel(time=time_idx, lev=DEPTH_IDX).compute(),
        (1, 2): two_in_two_uo_anom.isel(time=time_idx, lev=DEPTH_IDX).compute(),
    }

    col_labels = ['OM4', '1in1out', '2in2out']
    row_labels = ['Temperature\nAnomaly', 'U-Velocity\nAnomaly']
    row_units = ['°C', 'm/s']
    nrows_fig, ncols_fig = 2, 3

    # Layout: 2 map rows + colorbar column on the right
    fig = plt.figure(figsize=(5 * ncols_fig + 0.8, 2.5 * nrows_fig))
    gs = GridSpec(nrows_fig, ncols_fig + 1, figure=fig,
                  height_ratios=[1, 1],
                  width_ratios=[1, 1, 1, 0.03],
                  hspace=0.08, wspace=0.08,
                  top=0.90, bottom=0.05, left=0.08, right=0.97)

    proj = ccrs.Robinson(central_longitude=210)
    axes = np.empty((nrows_fig, ncols_fig), dtype=object)
    for i in range(nrows_fig):
        for j in range(ncols_fig):
            axes[i, j] = fig.add_subplot(gs[i, j], projection=proj)

    # Place colorbar axes on the right side of each row (half height, centered)
    fig.canvas.draw()
    cbar_axes_row = []
    for i in range(nrows_fig):
        pos = axes[i, -1].get_position()
        cb_height = pos.height * 2 / 3
        cb_y0 = pos.y0 + (pos.height - cb_height) / 2
        cb_x0 = pos.x0 + pos.width + 0.03
        cax = fig.add_axes([cb_x0, cb_y0, 0.008, cb_height])
        cbar_axes_row.append(cax)

    # Compute symmetric color ranges per row (from truth, col=0)
    norms = []
    for i in range(nrows_fig):
        valid = panels[(i, 0)].values.flatten()
        valid = valid[~np.isnan(valid)]
        abs_max = np.percentile(np.abs(valid), 95)
        norms.append(colors.Normalize(vmin=-abs_max, vmax=abs_max))

    for i in range(nrows_fig):
        for j in range(ncols_fig):
            ax = axes[i, j]
            panels[(i, j)].plot.pcolormesh(
                ax=ax, transform=ccrs.PlateCarree(), x='x', y='y',
                cmap=plt.cm.RdBu_r, norm=norms[i], add_colorbar=False, rasterized=True)
            ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
            ax.coastlines(zorder=11)
            if i == 0:
                ax.set_title(col_labels[j], fontweight='bold', fontsize=16)
            else:
                ax.set_title("")
            if j == 0:
                ax.text(-0.07, 0.55, row_labels[i], va='center', ha='right', rotation=90,
                        transform=ax.transAxes, fontweight='bold', fontsize=16,
                        multialignment='center')
            # RMSE & Corr for emulator cols (vs OM4 truth, col=0)
            if j > 0:
                emu_vals = panels[(i, j)].values.flatten()
                tru_vals = panels[(i, 0)].values.flatten()
                valid = ~(np.isnan(emu_vals) | np.isnan(tru_vals))
                rmse = np.sqrt(np.mean((emu_vals[valid] - tru_vals[valid])**2))
                corr = np.corrcoef(emu_vals[valid], tru_vals[valid])[0, 1]
                ax.text(0.02, 0.95, f'RMSE: {rmse:.4f}\nCorr: {corr:.4f}',
                        transform=ax.transAxes, fontsize=13, fontweight='bold', color='black',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, zorder=20),
                        verticalalignment='top', linespacing=1.3, zorder=20)

    # Vertical colorbar on the right side of each row
    for i in range(nrows_fig):
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norms[i], cmap=plt.cm.RdBu_r),
                          cax=cbar_axes_row[i], orientation='vertical',
                          label=f'{row_units[i]}')
        cb.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))

    fig.suptitle(f"Anomaly Snapshot at {DEPTH_LABEL}, t={time_str}", y=1.02, fontsize=16)
    plt.show()
    save_current_fig(f"snapshot_T_vs_U_{DEPTH_LABEL.replace(' ','')}_{time_str}", SPATIAL_DISTRIBUTION_DIR)

# ============================================================
# Spectrum Evaluation
# ============================================================
print("\n" + "="*60)
print("Spectrum Evaluation")
print("="*60)

var_to_eval = 'thetao'
time_window = None
n_regions = len(SPECTRUM_REGIONS)
nrows = 2
ncols = (n_regions + 1) // 2

print("\n=== Evaluating isotropic spectrum at surface (all regions) ===")
fig_iso, axes_iso = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
axes_iso_flat = axes_iso.flatten()
for idx, (region_name, lon_slice, lat_slice) in enumerate(SPECTRUM_REGIONS):
    print(f"  -> {region_name}")
    plot_isotropic_spectrum_comparison_together(
        emulators_dict=emulators_to_analyze,
        show_ratios=False,
        var_to_eval=var_to_eval,
        lev_idx=0,
        time_window=time_window,
        lon_slice=lon_slice,
        lat_slice=lat_slice,
        target_wavenumbers=[0.01, 0.02],
        region_name=region_name,
        ax=axes_iso_flat[idx],
    )
oh, ol, nc = reorder_legend_paired(*axes_iso_flat[0].get_legend_handles_labels())
fig_iso.legend(oh, ol, loc='lower center', ncol=nc, fontsize=16,
               bbox_to_anchor=(0.5, -0.08), frameon=False)
fig_iso.suptitle(f'Isotropic Power Spectrum of {_display_name(var_to_eval)} (Surface)', fontsize=20, y=1.01)
fig_iso.tight_layout()
save_current_fig("isotropic_spectrum_all_regions", SPECTRUM_DIR)

print("\n=== Evaluating EKE spectrum at surface (all regions) ===")
fig_ke, axes_ke = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
axes_ke_flat = axes_ke.flatten()
for idx, (region_name, lon_slice, lat_slice) in enumerate(SPECTRUM_REGIONS):
    print(f"  -> {region_name}")
    plot_ke_spectrum_comparison_together(
        emulators_dict=emulators_to_analyze,
        show_ratios=False,
        var_u='uo',
        var_v='vo',
        lev_idx=0,
        time_window=time_window,
        lon_slice=lon_slice,
        lat_slice=lat_slice,
        target_wavenumbers=[0.01, 0.02],
        region_name=region_name,
        ax=axes_ke_flat[idx],
    )
oh, ol, nc = reorder_legend_paired(*axes_ke_flat[0].get_legend_handles_labels())
fig_ke.legend(oh, ol, loc='lower center', ncol=nc, fontsize=16,
              bbox_to_anchor=(0.5, -0.10), frameon=False)
fig_ke.suptitle('EKE Spectrum (Surface)', fontsize=20, y=1.01)
fig_ke.tight_layout()
save_current_fig("ke_spectrum_all_regions", SPECTRUM_DIR)

print("\n=== Evaluating temporal spectrum at surface (all regions) ===")
fig_ts, axes_ts = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
axes_ts_flat = axes_ts.flatten()
for idx, (region_name, lon_slice, lat_slice) in enumerate(SPECTRUM_REGIONS):
    print(f"  -> {region_name}")
    plot_temporal_spectrum_comparison_together(
        emulators_dict=emulators_to_analyze,
        var_to_eval='thetao',
        show_ratios=False,
        lev_idx=0,
        lon_slice=lon_slice,
        lat_slice=lat_slice,
        target_frequencies=[1.0, 4.0],
        region_name=region_name,
        ax=axes_ts_flat[idx],
    )
oh, ol, nc = reorder_legend_paired(*axes_ts_flat[0].get_legend_handles_labels())
fig_ts.legend(oh, ol, loc='lower center', ncol=nc, fontsize=16,
              bbox_to_anchor=(0.5, -0.10), frameon=False)
fig_ts.suptitle(f'Temporal Power Spectrum of {_display_name(var_to_eval)} (Surface)', fontsize=20, y=1.01)
fig_ts.tight_layout()
save_current_fig("temporal_spectrum_all_regions", SPECTRUM_DIR)

print("\n=== Evaluating EKE temporal spectrum at surface (all regions) ===")
fig_kets, axes_kets = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
axes_kets_flat = axes_kets.flatten()
for idx, (region_name, lon_slice, lat_slice) in enumerate(SPECTRUM_REGIONS):
    print(f"  -> {region_name}")
    plot_ke_temporal_spectrum_comparison_together(
        emulators_dict=emulators_to_analyze,
        show_ratios=False,
        var_u='uo',
        var_v='vo',
        lev_idx=0,
        lon_slice=lon_slice,
        lat_slice=lat_slice,
        target_frequencies=[1.0, 4.0],
        region_name=region_name,
        ax=axes_kets_flat[idx],
    )
oh, ol, nc = reorder_legend_paired(*axes_kets_flat[0].get_legend_handles_labels())
fig_kets.legend(oh, ol, loc='lower center', ncol=nc, fontsize=16,
                bbox_to_anchor=(0.5, -0.10), frameon=False)
fig_kets.suptitle('Temporal EKE Power Spectrum (Surface)', fontsize=20, y=1.01)
fig_kets.tight_layout()
save_current_fig("ke_temporal_spectrum_all_regions", SPECTRUM_DIR)

print("\n=== Isotropic spectrum + ratio (all regions) ===")
plot_isotropic_spectrum_with_ratio(
    emulators_dict=emulators_to_analyze,
    spectrum_regions=SPECTRUM_REGIONS,
    var_to_eval=var_to_eval,
    lev_idx=0,
    time_window=time_window,
)
save_current_fig("isotropic_spectrum_with_ratio", SPECTRUM_DIR)

print("\n=== EKE spectrum + ratio (all regions) ===")
plot_ke_spectrum_with_ratio(
    emulators_dict=emulators_to_analyze,
    spectrum_regions=SPECTRUM_REGIONS,
    lev_idx=0,
    time_window=time_window,
)
save_current_fig("ke_spectrum_with_ratio", SPECTRUM_DIR)

print("\n=== Temporal spectrum + ratio (all regions) ===")
plot_temporal_spectrum_with_ratio(
    emulators_dict=emulators_to_analyze,
    spectrum_regions=SPECTRUM_REGIONS,
    var_to_eval='thetao',
    lev_idx=0,
)
save_current_fig("temporal_spectrum_with_ratio", SPECTRUM_DIR)

print("\n=== EKE temporal spectrum + ratio (all regions) ===")
plot_ke_temporal_spectrum_with_ratio(
    emulators_dict=emulators_to_analyze,
    spectrum_regions=SPECTRUM_REGIONS,
    lev_idx=0,
)
save_current_fig("ke_temporal_spectrum_with_ratio", SPECTRUM_DIR)

print("\n" + "="*60)
print(f"All PDF figures saved to: {PDF_DIR}")
print("="*60)
