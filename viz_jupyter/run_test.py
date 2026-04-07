"""
OM4_reso_all.py
Auto-generated from OM4_reso_all.ipynb.
All _together figures are saved as PDF vector graphics (dpi=200).
"""

import argparse

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
import pandas as pd
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
OUTPUT_DIR = "Figures_multiscale"
PDF_DIR = os.path.join(OUTPUT_DIR, "pdf_figures")

my_key = "XL8EJRUDQ7QPDJSUUC8Q"
my_secret = "2HaAeDBeiyGBxeeS60t6iOrrwgW6iebpiqoYkAHM"
fs = fsspec.filesystem("s3", key=my_key, secret=my_secret, client_kwargs={"endpoint_url": "https://nyu1.osn.mghpcc.org"})
storage_options = {"key": my_key, "secret": my_secret, "client_kwargs": {"endpoint_url": "https://nyu1.osn.mghpcc.org"}}

# 2. Emulator Configuration (only 1° for imprinting analysis)
EMULATOR_CONFIGS = [
    (
        '/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/Ocean_Emulator/.LOCAL/2025-12-02-samudra_om4_v1_onedeg_eval/predictions.zarr',
        "Samudra (1$^\\circ$)",
        "tab:grey"
    ),
    (
        '/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/checkpoints/2026-01-30-samudra_om4_onedeg_20_cap_eval/predictions.zarr',
        "Samudra 2 (1$^\\circ$)",
        "tab:orange"
    ),
]

# 3. Truth & Aux Paths
TRUTH_CONFIGS = [
    ("/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/onedeg/OM4.zarr", "OM4 (1$^\\circ$)"),
    ("/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/onedeg/OM4.zarr", "OM4 (1$^\\circ$)"),
]

LEVS_PATH = '/pscratch/sd/a/asubel/Data/Chapter_2/CM4_CMIP_New.zarr/'
OCEAN_SAMPLE_PATH = '/pscratch/sd/a/asubel/full-model/scratch/ocean_sample.nc'
STATIC_GRID_PATH = 's3://emulators/am16581/ocean_static_no_mask_table.zarr'

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

# Sync Time
time_slice = slice(emulators_raw[0].time.values[0], emulators_raw[0].time.values[-1])
for i in range(len(data_truth_ocean_list)):
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
target_grid_path = f"s3://emulators/am16581/grids/gaussian_grid_{shape_processed[0]}_by_{shape_processed[1]}.zarr"
target_grid = xr.open_zarr(target_grid_path, storage_options=storage_options)
target_grid = target_grid.rename({
    "grid_x": "x_b", "grid_y": "y_b", "grid_xt": "x", "grid_yt": "y",
    "grid_lon": "lon_b", "grid_lat": "lat_b", "grid_lont": "lon", "grid_latt": "lat",
}).compute()

print("Regridding basin mask...")
ds_static = xr.open_zarr(STATIC_GRID_PATH, storage_options=storage_options)
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
    'areacello': (['y','x'], xe.util.cell_area(target_grid, R).values),
    'basin': (['y','x'], basin_regridded.values)
}

# Process Data
print("Processing Datasets...")
processed_emulators = []
processed_truths = []

for idx, emu_ds in enumerate(emulators_raw):
    shape = (emu_ds.y.size, emu_ds.x.size)
    print(f"Processing resolution {shape}...")

    target_grid_path = f"s3://emulators/am16581/grids/gaussian_grid_{shape[0]}_by_{shape[1]}.zarr"
    target_grid_res = xr.open_zarr(target_grid_path, storage_options=storage_options)
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
        'areacello': (['y','x'], xe.util.cell_area(target_grid_res, R).values),
        'basin': (['y','x'], basin_regridded_res.values)
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


# ============================================================
# Imprinting Snapshot Analysis
# ============================================================
print("\n" + "="*60)
print("Imprinting Snapshot Analysis")
print("="*60)

IMPRINTING_DIR = "Figures_imprinting"
os.makedirs(IMPRINTING_DIR, exist_ok=True)

# --- Configuration ---
# Depth indices into levs array: 2.5m→0, 1050m→11, 3100m→15
IMPRINT_DEPTHS = [
    {'title': '2.5 m (Surface)', 'index': 0},
    {'title': '1050 m (Intermediate)', 'index': 11},
    {'title': '3100 m (Deep)', 'index': 15},
]

# Use Samudra v1 (1°) and Samudra 2 (1°), both share the same 1° OM4 truth
samudra_v1_label = emulator_labels[0]
samudra_v2_label = emulator_labels[1]

ds_samudra_v1 = emulators_to_analyze[samudra_v1_label]['ds']
ds_samudra_v2 = emulators_to_analyze[samudra_v2_label]['ds']
ds_truth_1deg = emulators_to_analyze[samudra_v1_label]['truth']

# Time steps: late rollout, artifacts well accumulated
TIME_INDICES_IMPRINT = [500, 535, 570]

# Velocity variable
VEL_VAR = 'vo'

# --- Remove seasonal cycle (dayofyear climatology) ---
print("Removing seasonal cycle from all fields...")

def remove_seasonal_cycle(da):
    """Remove seasonal cycle using dayofyear climatology."""
    climatology = da.groupby('time.dayofyear').mean('time')
    return (da.groupby('time.dayofyear') - climatology).drop_vars('dayofyear')

ds_truth_thetao_anom = remove_seasonal_cycle(ds_truth_1deg['thetao'])
ds_v1_thetao_anom = remove_seasonal_cycle(ds_samudra_v1['thetao'])
ds_v2_thetao_anom = remove_seasonal_cycle(ds_samudra_v2['thetao'])
ds_truth_vo_anom = remove_seasonal_cycle(ds_truth_1deg[VEL_VAR])
ds_v1_vo_anom = remove_seasonal_cycle(ds_samudra_v1[VEL_VAR])
ds_v1_uo_anom = remove_seasonal_cycle(ds_samudra_v1['uo'])
print("Seasonal cycle removed.")

# --- Figure setup ---
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr as _pearsonr

# Layout: 5 rows × 3 cols
# Row 0: OM4 truth T anomaly
# Row 1: Samudra T anomaly
# Row 2: Samudra vo anomaly
# Row 3: OM4 truth vo anomaly
# Row 4: Samudra 2 T anomaly
row_labels_main = [
    'OM4\nT anom',
    'Samudra\nT anom',
    f'Samudra\n{VEL_VAR} anom',
    f'OM4\n{VEL_VAR} anom',
    'Samudra 2\nT anom',
]
nrows_main = 5
ncols = len(IMPRINT_DEPTHS)

def get_anom_snapshot(da_anom, time_idx, depth_idx):
    """Extract a 2D deseasoned snapshot at specific time and depth."""
    return da_anom.isel(time=time_idx, lev=depth_idx).compute()

def make_discrete_norm(vmin, vmax, n_levels=12):
    """Create a BoundaryNorm for discrete colorbar."""
    bounds = np.linspace(vmin, vmax, n_levels + 1)
    return colors.BoundaryNorm(bounds, plt.cm.RdBu_r.N)

for TIME_IDX in TIME_INDICES_IMPRINT:
    time_val = ds_truth_1deg.time.isel(time=TIME_IDX).values
    time_str = str(time_val)[:10]
    print(f"\n{'='*40}")
    print(f"Processing t={TIME_IDX} ({time_str})")
    print(f"{'='*40}")

    # Pre-compute all panels (deseasoned snapshots)
    print("Computing deseasoned snapshots...")
    truth_t_map = {}     # OM4 truth T anomaly
    sam_t_map = {}       # Samudra T anomaly
    sam_vel_map = {}     # Samudra vo anomaly
    truth_vel_map = {}   # OM4 truth vo anomaly
    sam2_t_map = {}      # Samudra 2 T anomaly
    for j, d_conf in enumerate(IMPRINT_DEPTHS):
        d_idx = d_conf['index']
        truth_t_map[j] = get_anom_snapshot(ds_truth_thetao_anom, TIME_IDX, d_idx)
        sam_t_map[j]   = get_anom_snapshot(ds_v1_thetao_anom, TIME_IDX, d_idx)
        sam2_t_map[j]  = get_anom_snapshot(ds_v2_thetao_anom, TIME_IDX, d_idx)
        sam_vel_map[j] = get_anom_snapshot(ds_v1_vo_anom, TIME_IDX, d_idx)
        truth_vel_map[j] = get_anom_snapshot(ds_truth_vo_anom, TIME_IDX, d_idx)

    # ================================================================
    # Main figure: 5 rows × 3 cols global snapshot
    # ================================================================
    print("Plotting main figure...")

    fig = plt.figure(figsize=(5 * ncols, 2.8 * nrows_main + 2.5))
    gs = GridSpec(nrows_main + 2, ncols, figure=fig,
                  height_ratios=[1] * nrows_main + [0.04, 0.04],
                  hspace=0.10, wspace=0.05,
                  top=0.93, bottom=0.05, left=0.10, right=0.97)

    proj = ccrs.Robinson(central_longitude=210)
    axes = np.empty((nrows_main, ncols), dtype=object)
    for i in range(nrows_main):
        for j in range(ncols):
            axes[i, j] = fig.add_subplot(gs[i, j], projection=proj)

    cbar_t_axes = [fig.add_subplot(gs[nrows_main, j]) for j in range(ncols)]
    cbar_v_axes = [fig.add_subplot(gs[nrows_main + 1, j]) for j in range(ncols)]
    for cax_list in [cbar_t_axes, cbar_v_axes]:
        for cax in cax_list:
            pos = cax.get_position()
            new_width = pos.width * 0.7
            offset = (pos.width - new_width) / 2
            cax.set_position([pos.x0 + offset, pos.y0, new_width, pos.height])

    for j, d_conf in enumerate(IMPRINT_DEPTHS):
        # Compute T colorbar range (shared across rows 0,1,4 for same depth)
        all_t = np.concatenate([truth_t_map[j].values.flatten(),
                                sam_t_map[j].values.flatten(),
                                sam2_t_map[j].values.flatten()])
        all_t = all_t[~np.isnan(all_t)]
        t_abs_max = np.percentile(np.abs(all_t), 98)

        # Compute velocity colorbar range (shared across rows 2,3)
        all_v = np.concatenate([sam_vel_map[j].values.flatten(),
                                truth_vel_map[j].values.flatten()])
        all_v = all_v[~np.isnan(all_v)]
        v_abs_max = np.percentile(np.abs(all_v), 98)

        # Use discrete colorbar for intermediate/deep T, continuous for surface
        if j == 0:
            t_norm = colors.Normalize(vmin=-t_abs_max, vmax=t_abs_max)
        else:
            t_norm = make_discrete_norm(-t_abs_max, t_abs_max, n_levels=12)
        v_norm = colors.Normalize(vmin=-v_abs_max, vmax=v_abs_max)

        # Data for each row
        row_data = [truth_t_map[j], sam_t_map[j], sam_vel_map[j],
                    truth_vel_map[j], sam2_t_map[j]]
        # Which norm to use per row: T rows (0,1,4) vs velocity rows (2,3)
        row_norms = [t_norm, t_norm, v_norm, v_norm, t_norm]

        for i in range(nrows_main):
            ax = axes[i, j]
            row_data[i].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='x', y='y',
                cmap=plt.cm.RdBu_r, norm=row_norms[i], add_colorbar=False, rasterized=True)
            ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
            ax.coastlines(zorder=11)
            ax.set_title(d_conf['title'] if i == 0 else "", fontweight='bold')
            if j == 0:
                ax.text(-0.07, 0.55, row_labels_main[i], va='center', ha='right',
                        rotation=90, transform=ax.transAxes, fontweight='bold',
                        fontsize=9, multialignment='center')

        cb_t = fig.colorbar(plt.cm.ScalarMappable(norm=t_norm, cmap=plt.cm.RdBu_r),
            cax=cbar_t_axes[j], orientation='horizontal', label='T anomaly (°C)')
        cb_t.ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        cb_v = fig.colorbar(plt.cm.ScalarMappable(norm=v_norm, cmap=plt.cm.RdBu_r),
            cax=cbar_v_axes[j], orientation='horizontal', label=f'{VEL_VAR} anomaly (m/s)')
        cb_v.ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))

    fig.suptitle(f"Imprinting: T vs Velocity Anomaly Snapshots at t={time_str}", y=0.97, fontsize=14)
    save_current_fig(f"imprinting_snapshot_t{TIME_IDX}", IMPRINTING_DIR)

    # ================================================================
    # Regional zoom: South Pacific only (most convincing for imprinting)
    # ================================================================
    ZOOM_REGIONS = [
        {'name': 'South Pacific', 'lon_min': -160, 'lon_max': -120,
         'lat_min': -40, 'lat_max': -20, 'tag': 'south_pacific'},
    ]

    for region in ZOOM_REGIONS:
        print(f"\nPlotting regional zoom: {region['name']}...")

        sel_lon_min = region['lon_min'] % 360
        sel_lon_max = region['lon_max'] % 360

        fig_r = plt.figure(figsize=(5 * ncols, 2.8 * nrows_main + 3.0))
        gs_r = GridSpec(nrows_main + 2, ncols, figure=fig_r,
                        height_ratios=[1] * nrows_main + [0.04, 0.04],
                        hspace=0.20, wspace=0.15,
                        top=0.92, bottom=0.06, left=0.12, right=0.97)

        proj_r = ccrs.PlateCarree()
        axes_r = np.empty((nrows_main, ncols), dtype=object)
        for i in range(nrows_main):
            for j in range(ncols):
                axes_r[i, j] = fig_r.add_subplot(gs_r[i, j], projection=proj_r)

        cbar_t_r = [fig_r.add_subplot(gs_r[nrows_main, j]) for j in range(ncols)]
        cbar_v_r = [fig_r.add_subplot(gs_r[nrows_main + 1, j]) for j in range(ncols)]
        for cax_list in [cbar_t_r, cbar_v_r]:
            for cax in cax_list:
                pos = cax.get_position()
                new_width = pos.width * 0.7
                offset = (pos.width - new_width) / 2
                cax.set_position([pos.x0 + offset, pos.y0, new_width, pos.height])

        extent = [region['lon_min'], region['lon_max'],
                  region['lat_min'], region['lat_max']]

        for j, d_conf in enumerate(IMPRINT_DEPTHS):
            # Slice regional data
            reg_truth_t = truth_t_map[j].sel(x=slice(sel_lon_min, sel_lon_max),
                                              y=slice(region['lat_min'], region['lat_max']))
            reg_sam_t   = sam_t_map[j].sel(x=slice(sel_lon_min, sel_lon_max),
                                            y=slice(region['lat_min'], region['lat_max']))
            reg_sam_vel = sam_vel_map[j].sel(x=slice(sel_lon_min, sel_lon_max),
                                              y=slice(region['lat_min'], region['lat_max']))
            reg_truth_vel = truth_vel_map[j].sel(x=slice(sel_lon_min, sel_lon_max),
                                                   y=slice(region['lat_min'], region['lat_max']))
            reg_sam2_t  = sam2_t_map[j].sel(x=slice(sel_lon_min, sel_lon_max),
                                             y=slice(region['lat_min'], region['lat_max']))

            reg_row_data = [reg_truth_t, reg_sam_t, reg_sam_vel,
                            reg_truth_vel, reg_sam2_t]

            # Shared T norm (discrete for intermediate/deep)
            all_t_r = np.concatenate([d.values.flatten() for d in
                                      [reg_truth_t, reg_sam_t, reg_sam2_t]])
            all_t_r = all_t_r[~np.isnan(all_t_r)]
            t_abs_r = np.percentile(np.abs(all_t_r), 98) if len(all_t_r) > 0 else 1.0
            if j == 0:
                t_norm_r = colors.Normalize(vmin=-t_abs_r, vmax=t_abs_r)
            else:
                t_norm_r = make_discrete_norm(-t_abs_r, t_abs_r, n_levels=12)

            all_v_r = np.concatenate([d.values.flatten() for d in
                                      [reg_sam_vel, reg_truth_vel]])
            all_v_r = all_v_r[~np.isnan(all_v_r)]
            v_abs_r = np.percentile(np.abs(all_v_r), 98) if len(all_v_r) > 0 else 1.0
            v_norm_r = colors.Normalize(vmin=-v_abs_r, vmax=v_abs_r)

            row_norms_r = [t_norm_r, t_norm_r, v_norm_r, v_norm_r, t_norm_r]

            for i in range(nrows_main):
                ax = axes_r[i, j]
                reg_row_data[i].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(),
                    x='x', y='y', cmap=plt.cm.RdBu_r, norm=row_norms_r[i],
                    add_colorbar=False, rasterized=True)
                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black',
                               facecolor='lightgray')
                ax.coastlines(zorder=11)
                gl = ax.gridlines(draw_labels=True, alpha=0.3, linewidth=0.5,
                    xlocs=np.arange(-180, 181, 10), ylocs=np.arange(-90, 91, 10))
                gl.top_labels = False
                gl.right_labels = False
                gl.left_labels = (j == 0)
                gl.bottom_labels = (i == nrows_main - 1)
                ax.set_title(d_conf['title'] if i == 0 else "", fontweight='bold')
                if j == 0:
                    ax.text(-0.22, 0.55, row_labels_main[i], va='center', ha='right',
                            rotation=90, transform=ax.transAxes, fontweight='bold',
                            fontsize=9, multialignment='center')

            cb_t = fig_r.colorbar(plt.cm.ScalarMappable(norm=t_norm_r, cmap=plt.cm.RdBu_r),
                cax=cbar_t_r[j], orientation='horizontal', label='T anomaly (°C)')
            cb_t.ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
            cb_v = fig_r.colorbar(plt.cm.ScalarMappable(norm=v_norm_r, cmap=plt.cm.RdBu_r),
                cax=cbar_v_r[j], orientation='horizontal', label=f'{VEL_VAR} anomaly (m/s)')
            cb_v.ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))

        fig_r.suptitle(f"Imprinting Zoom: {region['name']} at t={time_str}",
                       y=0.96, fontsize=14)
        save_current_fig(f"imprinting_{region['tag']}_t{TIME_IDX}", IMPRINTING_DIR)

    # ================================================================
    # Spatial correlation: T_error vs velocity by depth
    # Compute separately for global and South Pacific quiet region
    # ================================================================
    print("\nComputing spatial correlation: T_error vs velocity by depth...")

    n_levs = len(levs)

    # South Pacific quiet region for focused correlation
    sp_lon_min, sp_lon_max = 200, 240  # 0-360 convention
    sp_lat_min, sp_lat_max = -40, -20

    regions_corr = [
        ('Global', None),
        ('South Pacific', {'x': slice(sp_lon_min, sp_lon_max),
                           'y': slice(sp_lat_min, sp_lat_max)}),
    ]

    fig_corr, axes_corr = plt.subplots(1, len(regions_corr), figsize=(5 * len(regions_corr), 7),
                                        sharey=True)
    if len(regions_corr) == 1:
        axes_corr = [axes_corr]

    for r_idx, (reg_name, reg_sel) in enumerate(regions_corr):
        corr_sam = np.full(n_levs, np.nan)
        corr_sam2 = np.full(n_levs, np.nan)

        for k in range(n_levs):
            t_tr = ds_truth_thetao_anom.isel(time=TIME_IDX, lev=k)
            t_s1 = ds_v1_thetao_anom.isel(time=TIME_IDX, lev=k)
            t_s2 = ds_v2_thetao_anom.isel(time=TIME_IDX, lev=k)
            vel = ds_v1_vo_anom.isel(time=TIME_IDX, lev=k)

            if reg_sel is not None:
                t_tr = t_tr.sel(**reg_sel)
                t_s1 = t_s1.sel(**reg_sel)
                t_s2 = t_s2.sel(**reg_sel)
                vel = vel.sel(**reg_sel)

            t_tr_f = t_tr.values.flatten()
            t_s1_f = t_s1.values.flatten()
            t_s2_f = t_s2.values.flatten()
            vel_f = vel.values.flatten()
            e1, e2 = t_s1_f - t_tr_f, t_s2_f - t_tr_f
            valid1 = ~(np.isnan(e1) | np.isnan(vel_f))
            valid2 = ~(np.isnan(e2) | np.isnan(vel_f))
            if valid1.sum() > 30:
                corr_sam[k], _ = _pearsonr(e1[valid1], vel_f[valid1])
            if valid2.sum() > 30:
                corr_sam2[k], _ = _pearsonr(e2[valid2], vel_f[valid2])

        ax_c = axes_corr[r_idx]
        ax_c.plot(corr_sam, levs, 'o-', color='tab:grey', label='Samudra', lw=2, markersize=5)
        ax_c.plot(corr_sam2, levs, 's-', color='tab:orange', label='Samudra 2', lw=2, markersize=5)
        ax_c.axvline(0, color='black', ls=':', alpha=0.5)
        ax_c.invert_yaxis()
        ax_c.set_yscale('symlog', linthresh=100)
        ax_c.set_xlabel(f'Spatial Corr: T_error vs {VEL_VAR}', fontsize=12)
        if r_idx == 0:
            ax_c.set_ylabel('Depth (m)', fontsize=12)
        ax_c.set_title(reg_name, fontsize=13)
        ax_c.legend(fontsize=11)
        ax_c.grid(True, alpha=0.3)
        ax_c.set_yticks(levs)
        ax_c.set_yticklabels([f'{d:.0f}' for d in levs])

        print(f"  {reg_name} done.")

    fig_corr.suptitle(f'Imprinting: T Error–Velocity Spatial Correlation\nt={time_str}',
                      fontsize=14, y=1.02)
    plt.tight_layout()
    save_current_fig(f"imprinting_corr_profile_t{TIME_IDX}", IMPRINTING_DIR)

print("\n" + "="*60)
print(f"Imprinting analysis complete. Figures saved to: {os.path.abspath(IMPRINTING_DIR)}")
print("="*60)
