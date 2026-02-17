"""
OM4_reso_all.py
Auto-generated from OM4_reso_all.ipynb.
All _together figures are saved as PDF vector graphics (dpi=200).
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures

import xarray as xr
import torch
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

# 5. Variable Mappings
OCEAN_FIELD_NAME_PREFIXES = {
    'thetao': ['thetao_'], 'so': ['so_'], 'uo': ['uo_'], 'vo': ['vo_'],
    'zos': ['zos'], 'sea_surface_temperature': ['sst'], 'lev': ['idepth_']
}

# 6. Plotting Constants
BASINS_TO_PLOT = ['Atlantic', 'Pacific', 'Indian', 'Southern']

DEPTH_SLICES = [
    {'min': 0, 'max': 700, 'title': 'Surface (0-700m)'},
    {'min': 700, 'max': 2000, 'title': 'Intermediate (700-2000m)'},
    {'min': 2000, 'max': 7000, 'title': 'Deep (2000-7000m)'}
]
VAR_CONFIG = {
    'thetao': {'cmap': cmocean.cm.thermal, 'units': '$^\\circ$C', 'label': 'Temperature', 'centered': False},
    'so': {'cmap': cmocean.cm.haline, 'units': 'psu', 'label': 'Salinity', 'centered': False},
    'ohc': {'cmap': cmocean.cm.thermal, 'units': 'J', 'label': 'Total Heat Content', 'centered': False},
    'zos':    {'cmap': cmocean.cm.balance, 'units': 'm',         'label': 'SSH',         'centered': True},
    'uo':     {'cmap': cmocean.cm.balance, 'units': 'm/s',       'label': 'U-Velocity',  'centered': True},
    'vo':     {'cmap': cmocean.cm.balance, 'units': 'm/s',       'label': 'V-Velocity',  'centered': True},
}

# Constants
RHO = 1035          # Reference density (kg/m^3)
CP = 3850           # Specific heat capacity (J/kg/K)
SCALE_FACTOR = 1e21 # Scale to ZJ (10^21 J)

DEPTH_LAYERS = [
    {'title':'2.5m','index':0},
    {'title':'700m','index':10},
    {'title':'2000m','index':14},
]

# Configuration
BASINS_TO_PLOT = ['Atlantic', 'Pacific', 'Indian', 'Southern']

TIME_INDICES = [0, 300, 580]

# ============================================================
# 1. Output Directory (Configurable Name)
# ============================================================
OUTPUT_DIR = "Emulator_Evaluation_Figures"
PDF_DIR = os.path.join(OUTPUT_DIR, "pdf_figures")

my_key = "XL8EJRUDQ7QPDJSUUC8Q"
my_secret = "2HaAeDBeiyGBxeeS60t6iOrrwgW6iebpiqoYkAHM"
fs = fsspec.filesystem("s3", key=my_key, secret=my_secret, client_kwargs={"endpoint_url": "https://nyu1.osn.mghpcc.org"})
storage_options = {"key": my_key, "secret": my_secret, "client_kwargs": {"endpoint_url": "https://nyu1.osn.mghpcc.org"}}

# 2. Emulator Configuration
EMULATOR_CONFIGS = [
    (
        '/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/Ocean_Emulator/.LOCAL/2025-12-02-samudra_om4_v1_onedeg_eval/predictions.zarr',
        "Samudra",
        "tab:gray"
    ),
    (
        '/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/checkpoints/2026-01-30-samudra_om4_onedeg_20_cap_eval/predictions.zarr',
        "WDMSE-1deg",
        "tab:orange"
    ),
    (
        '/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/checkpoints/2026-02-05-samudra_om4_halfdeg_20_cap_fix_eval/predictions.zarr',
        "WDMSE-1/2deg",
        "tab:green"
    ),
    (
        '/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/checkpoints/2025-02-08-samudra_om4_quarterdeg_cap_20_eval_fix/predictions.zarr',
        "WDMSE-1/4deg",
        "tab:blue"
    )
]

# 3. Truth & Aux Paths
TRUTH_CONFIGS = [
    ("/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/onedeg/OM4.zarr", "OM4-1deg"),
    ("/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/onedeg/OM4.zarr", "OM4-1deg"),
    ("/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/halfdeg/OM4.zarr", "OM4-1/2deg"),
    ("/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/quarterdeg/OM4.zarr", "OM4-1/4deg")
]

LEVS_PATH = '/pscratch/sd/a/asubel/Data/Chapter_2/CM4_CMIP_New.zarr/'
OCEAN_SAMPLE_PATH = '/pscratch/sd/a/asubel/full-model/scratch/ocean_sample.nc'
STATIC_GRID_PATH = 's3://emulators/am16581/ocean_static_no_mask_table.zarr'

print(f"Output Directory: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

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
levs = xr.open_zarr(LEVS_PATH)['lev'].values
ocean_test = xr.open_dataset(OCEAN_SAMPLE_PATH)
depth_levels = concat_levels(ocean_test, 'idepth', 'levels')
dz_from_data = (depth_levels[1:] - depth_levels[:-1]).data

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

from functions import extract_basin_masks
masks_list = []
for label, emu_data in emulators_to_analyze.items():
    ds_truth = emu_data['truth']
    masks = extract_basin_masks(ds_truth)
    masks_list.append(masks)


# ============================================================
# Spectral Analysis
# ============================================================
print("\n" + "="*60)
print("Spectral Analysis")
print("="*60)

var_to_eval = 'thetao'
time_window = 100

for lev_idx in [0, 9]:
    for lon_slice, lat_slice in [(slice(180, 243), slice(-40, 35)), (slice(120, 280), slice(-15, 15))]:
        region_tag = f"lev{lev_idx}_lon{lon_slice.start}-{lon_slice.stop}_lat{lat_slice.start}-{lat_slice.stop}"

        print(f"\n=== Evaluating isotropic spectrum at level index {lev_idx} for longitude {lon_slice} and latitude {lat_slice} ===")
        plot_isotropic_spectrum_comparison_together(
            emulators_dict=emulators_to_analyze,
            show_ratios=False,
            var_to_eval=var_to_eval,
            lev_idx=lev_idx,
            time_window=time_window,
            lon_slice=lon_slice,
            lat_slice=lat_slice,
            target_wavenumbers=[0.01, 0.02],
        )
        save_current_fig(f"isotropic_spectrum_{region_tag}", PDF_DIR)

        print(f"\n=== Evaluating kinetic energy spectrum at level index {lev_idx} for longitude {lon_slice} and latitude {lat_slice} ===")
        plot_ke_spectrum_comparison_together(
            emulators_dict=emulators_to_analyze,
            show_ratios=False,
            var_u='uo',
            var_v='vo',
            lev_idx=lev_idx,
            time_window=time_window,
            lon_slice=lon_slice,
            lat_slice=lat_slice,
            target_wavenumbers=[0.01, 0.02],
        )
        save_current_fig(f"ke_spectrum_{region_tag}", PDF_DIR)

        print(f"\n=== Evaluating temporal spectrum at level index {lev_idx} for longitude {lon_slice} and latitude {lat_slice} ===")
        plot_temporal_spectrum_comparison_together(
            emulators_dict=emulators_to_analyze,
            var_to_eval='thetao',
            show_ratios=False,
            lev_idx=lev_idx,
            lon_slice=lon_slice,
            lat_slice=lat_slice,
            target_frequencies=[1.0, 4.0],
        )
        save_current_fig(f"temporal_spectrum_{region_tag}", PDF_DIR)

        print(f"\n=== Evaluating kinetic energy temporal spectrum at level index {lev_idx} for longitude {lon_slice} and latitude {lat_slice} ===")
        plot_ke_temporal_spectrum_comparison_together(
            emulators_dict=emulators_to_analyze,
            show_ratios=False,
            var_u='uo',
            var_v='vo',
            lev_idx=lev_idx,
            lon_slice=lon_slice,
            lat_slice=lat_slice,
            target_frequencies=[1.0, 4.0]
        )
        save_current_fig(f"ke_temporal_spectrum_{region_tag}", PDF_DIR)

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
    output_dir=OUTPUT_DIR
)
save_current_fig("timeseries_thetao", PDF_DIR)

# Time Series (Salinity)
plot_mean_timeseries_comparison_together(
    emulators_to_analyze,
    variable_name='so',
    var_units='psu',
    var_label='Salinity',
    output_dir=OUTPUT_DIR
)
save_current_fig("timeseries_so", PDF_DIR)


# ============================================================
# Temporal Variance Maps
# ============================================================
print("\n" + "="*60)
print("Temporal Variance Maps")
print("="*60)

plot_variance_comparison_together(
    emulators_to_analyze,
    variable_name='thetao',
    units='$^\\circ$C',
    output_dir=OUTPUT_DIR,
    colorbar_fix=True
)
save_current_fig("variance_thetao", PDF_DIR)

plot_variance_comparison_together(
    emulators_to_analyze,
    variable_name='so',
    units='psu',
    output_dir=OUTPUT_DIR,
    colorbar_fix=True
)
save_current_fig("variance_so", PDF_DIR)


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
save_current_fig("ohc_zonal_bias_ocean", PDF_DIR)

plot_zonal_bias_ocean_comparison_together(
    emulators_to_analyze,
    'thetao',
    '$^\\circ$C',
    masks_list,
    OUTPUT_DIR,
    colorbar_fix=True
)
save_current_fig("zonal_bias_ocean_thetao", PDF_DIR)

plot_snapshot_zonal_bias_ocean_comparison_together(
    emulators_dict=emulators_to_analyze,
    variable_name='thetao',
    units='°C',
    masks_list=masks_list,
    TIME_INDICES=TIME_INDICES,
    colorbar_fix=True
)
save_current_fig("snapshot_zonal_bias_ocean_thetao", PDF_DIR)

plot_snapshot_ohc_zonal_bias_ocean_comparison_together(
    emulators_dict=emulators_to_analyze,
    masks_list=masks_list,
    TIME_INDICES=TIME_INDICES,
    colorbar_fix=True
)
save_current_fig("snapshot_ohc_zonal_bias_ocean", PDF_DIR)


# ============================================================
# Global Spatial Bias Maps
# ============================================================
print("\n" + "="*60)
print("Global Spatial Bias Maps")
print("="*60)

plot_ohc_bias_comparison_together(emulators_to_analyze, colorbar_fix=True)
save_current_fig("ohc_bias", PDF_DIR)

plot_temperature_bias_comparison_together(emulators_to_analyze, colorbar_fix=True)
save_current_fig("temperature_bias", PDF_DIR)

plot_ohc_snapshot_bias_comparison_together(emulators_to_analyze, TIME_INDICES, colorbar_fix=True)
save_current_fig("ohc_snapshot_bias", PDF_DIR)

plot_temperature_snapshot_bias_comparison_together(emulators_to_analyze, TIME_INDICES, colorbar_fix=True)
save_current_fig("temperature_snapshot_bias", PDF_DIR)


# ============================================================
# Zonal Mean Distribution
# ============================================================
print("\n" + "="*60)
print("Zonal Mean Distribution")
print("="*60)

plot_zonal_mean_comparison_together(
    emulators_to_analyze,
    variable_name='thetao',
    units='$^\circ$C',
    masks_list=masks_list,
    colorbar_fix=True
)
save_current_fig("zonal_mean_thetao", PDF_DIR)

plot_ohc_zonal_mean_comparison_together(
    emulators_to_analyze,
    variable_name='thetao',
    units='$^\circ$C',
    masks_list=masks_list,
    colorbar_fix=True
)
save_current_fig("ohc_zonal_mean", PDF_DIR)

plot_zonal_snapshot_comparison_together(
    emulators_to_analyze,
    variable_name='thetao',
    units='$^\circ$C',
    masks_list=masks_list,
    time_indices=TIME_INDICES,
    colorbar_fix=True
)
save_current_fig("zonal_snapshot_thetao", PDF_DIR)

plot_ohc_zonal_snapshot_comparison_together(
    emulators_to_analyze,
    variable_name='thetao',
    units='$^\circ$C',
    masks_list=masks_list,
    TIME_INDICES=TIME_INDICES,
    colorbar_fix=True
)
save_current_fig("ohc_zonal_snapshot", PDF_DIR)


# ============================================================
# Global Spatial Distribution Maps
# ============================================================
print("\n" + "="*60)
print("Global Spatial Distribution Maps")
print("="*60)

plot_ohc_mean_comparison_together(
    emulators_to_analyze,
    colorbar_fix=True
)
save_current_fig("ohc_mean", PDF_DIR)

plot_temperature_mean_comparison_together(
    emulators_to_analyze,
    colorbar_fix=True
)
save_current_fig("temperature_mean", PDF_DIR)

VARIABLE_TO_PLOT = 'thetao'

plot_snapshot_comparison_multidepth_together(
    emulators_to_analyze,
    var_name=VARIABLE_TO_PLOT,
    time_indices=TIME_INDICES,
    depth_configs=DEPTH_LAYERS,
    fix_colorbar=True
)
save_current_fig("snapshot_multidepth_thetao", PDF_DIR)

plot_ohc_snapshot_comparison_multidepth_together(
    emulators_to_analyze,
    time_indices=TIME_INDICES,
    depth_configs=DEPTH_SLICES,
    colorbar_fix=True
)
save_current_fig("ohc_snapshot_multidepth", PDF_DIR)


print("\n" + "="*60)
print(f"All PDF figures saved to: {PDF_DIR}")
print("="*60)
