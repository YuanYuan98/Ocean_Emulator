import xarray as xr
import torch
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import fsspec
import xesmf as xe
import os
from functions import *

# Constants
OUTPUT_DIR = "Emulator_Evaluation_Figures_MultiRes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# S3 Config
my_key = "XL8EJRUDQ7QPDJSUUC8Q"
my_secret = "2HaAeDBeiyGBxeeS60t6iOrrwgW6iebpiqoYkAHM"
storage_options = {"key": my_key, "secret": my_secret, "client_kwargs": {"endpoint_url": "https://nyu1.osn.mghpcc.org"}}

# Aux Paths
LEVS_PATH = '/pscratch/sd/a/asubel/Data/Chapter_2/CM4_CMIP_New.zarr/'
OCEAN_SAMPLE_PATH = '/pscratch/sd/a/asubel/full-model/scratch/ocean_sample.nc'
STATIC_GRID_PATH = 's3://emulators/am16581/ocean_static_no_mask_table.zarr'

# Resolution Configurations
CONFIGS = [
    {
        'name': '1-deg',
        'truth_path': "/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/onedeg/OM4.zarr",
        'emu_path': '/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/Ocean_Emulator/.LOCAL/2025-12-03-samudra_om4_onedeg/predictions.zarr',
        'label': 'WDMSE-1',
        'truth_label': 'OM4-1',
        'color': 'tab:blue'
    },
    {
        'name': '1/2-deg',
        'truth_path': "/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/halfdeg/OM4.zarr",
        'emu_path': '/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/Ocean_Emulator/.LOCAL/2025-12-02-samudra_om4_halfdeg/predictions.zarr',
        'label': 'WDMSE-1/2',
        'truth_label': 'OM4-1/2',
        'color': 'tab:orange'
    },
    {
        'name': '1/4-deg',
        'truth_path': "/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/quarterdeg/OM4.zarr",
        'emu_path': '/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/checkpoints/2025-12-19-samudra_om4_quarterdeg_eval5/predictions.zarr',
        'label': 'WDMSE-1/4',
        'truth_label': 'OM4-1/4',
        'color': 'tab:green'
    }
]

# Analysis settings
var_to_eval = 'thetao'
lev_idx = 0
time_window = 100
var_u = 'uo'
var_v = 'vo'

lon_slice = slice(180, 243)
lat_slice = slice(-40, 35)

# Initialize storage for results
results = {}

# Aux data (loaded once or reused where appropriate, but need grid for each)
levs = xr.open_zarr(LEVS_PATH)['lev'].values
ocean_test = xr.open_dataset(OCEAN_SAMPLE_PATH)
depth_levels = concat_levels(ocean_test, 'idepth', 'levels')
dz_from_data = (depth_levels[1:] - depth_levels[:-1]).data
R = 6.371e6

print(f"Output Directory: {OUTPUT_DIR}")

# --- Helper Functions ---

def get_spectrum(ds_target, var_name, lev_i, t_window, x_sl, y_sl, dx_mean, dy_mean):
    data_slice_raw = ds_target[var_name].isel(lev=lev_i, time=slice(None, t_window))
    data_slice_raw = data_slice_raw.transpose('time', ...).sel({'x': x_sl, 'y': y_sl})
    
    time_mean = data_slice_raw.mean(dim='time')
    data_slice_anom = data_slice_raw - time_mean
    data_prepared = data_slice_anom.fillna(0).values
    
    k_cent, spec = compute_isotropic_spectrum_torch(
        torch.as_tensor(data_prepared),
        dx=dx_mean, dy=dy_mean, n_factor=2,
        detrend='linear', window='hann', cutoff_before_bins=False
    )
    return k_cent.cpu().numpy() * 1000 * 2 * np.pi, spec.cpu().numpy()

def get_ke_spectrum(ds_target, lev_i, t_window, x_sl, y_sl, dx_mean, dy_mean):
    def process_var(var_name):
        data = ds_target[var_name].isel(lev=lev_i, time=slice(None, t_window))
        data = data.transpose('time', ...).sel({'x': x_sl, 'y': y_sl})
        data_anom = data - data.mean(dim='time')
        return torch.as_tensor(data_anom.fillna(0).values)

    u_tensor = process_var(var_u)
    v_tensor = process_var(var_v)
    
    k_cent, spec_u = compute_isotropic_spectrum_torch(u_tensor, dx=dx_mean, dy=dy_mean, n_factor=2, detrend='linear', window='hann')
    _, spec_v = compute_isotropic_spectrum_torch(v_tensor, dx=dx_mean, dy=dy_mean, n_factor=2, detrend='linear', window='hann')
    
    spec_ke = 0.5 * (spec_u + spec_v)
    
    k_km = k_cent.cpu().numpy() * 1000 * 2 * np.pi
    return k_km, spec_ke.cpu().numpy()

# --- Main Loop ---

for config in CONFIGS:
    print(f"Processing {config['name']}...")
    
    # 1. Load Truth
    print(f"  Loading Truth: {config['truth_path']}")
    if "s3" in config['truth_path']:
        ds_truth = xr.open_zarr(config['truth_path'], storage_options=storage_options)
    else:
        ds_truth = xr.open_zarr(config['truth_path'])
        
    # 2. Load Emulator
    print(f"  Loading Emulator: {config['emu_path']}")
    ds_emu = xr.open_zarr(config['emu_path']).rename({'lat':'y','lon':'x'})
    
    # Sync Time
    try:
        # Sometimes time indices might not exactly match or need interception
        time_slice = slice(ds_emu.time.values[0], ds_emu.time.values[-1])
        ds_truth = ds_truth.sel(time=time_slice)
    except Exception as e:
        print(f"  Warning: time sync issue: {e}")
    
    # 3. Grid & Metrics
    shape = (ds_emu.y.size, ds_emu.x.size)
    print(f"  Loading grid for shape {shape}...")
    target_grid_path = f"s3://emulators/am16581/grids/gaussian_grid_{shape[0]}_by_{shape[1]}.zarr"
    target_grid = xr.open_zarr(target_grid_path, storage_options=storage_options).rename({
        "grid_x": "x_b", "grid_y": "y_b", "grid_xt": "x", "grid_yt": "y",
        "grid_lon": "lon_b", "grid_lat": "lat_b", "grid_lont": "lon", "grid_latt": "lat",
    }).compute()
    
    # Regrid Basin
    print("  Regridding basin...")
    ds_static = xr.open_zarr(STATIC_GRID_PATH, storage_options=storage_options)
    ds_source = ds_static[['basin', 'geolat', 'geolon']].rename({'geolat': 'lat', 'geolon': 'lon'})
    ds_target_grid = target_grid[['lat', 'lon']]
    regridder = xe.Regridder(ds_source, ds_target_grid, method='nearest_s2d', periodic=True, reuse_weights=False)
    basin_regridded = regridder(ds_source['basin'])
    
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
    
    # Process Datasets
    print("  Processing datasets...")
    ds_truth = process_dataset(ds_truth, levs, dz_from_data, grid_metrics)
    ds_emu = process_dataset(ds_emu, levs, dz_from_data, grid_metrics)
    
    # Calculate Mean DX/DY for Spectrum
    dx_mean = float(ds_truth.dx.sel({'x': lon_slice, 'y': lat_slice}).mean().values)
    dy_mean = float(ds_truth.dy.sel({'x': lon_slice, 'y': lat_slice}).mean().values)
    
    print(f"  Computing Spectra (dx_mean={dx_mean:.1f}, dy_mean={dy_mean:.1f})...")
    
    # Compute Spectra
    k_truth, spec_truth = get_spectrum(ds_truth, var_to_eval, lev_idx, time_window, lon_slice, lat_slice, dx_mean, dy_mean)
    k_emu, spec_emu = get_spectrum(ds_emu, var_to_eval, lev_idx, time_window, lon_slice, lat_slice, dx_mean, dy_mean)
    
    k_ke_truth, spec_ke_truth = get_ke_spectrum(ds_truth, lev_idx, time_window, lon_slice, lat_slice, dx_mean, dy_mean)
    k_ke_emu, spec_ke_emu = get_ke_spectrum(ds_emu, lev_idx, time_window, lon_slice, lat_slice, dx_mean, dy_mean)
    
    # Store Results
    results[config['name']] = {
        'config': config,
        'temp': {
            'truth': {'k': k_truth, 'spec': spec_truth},
            'emu': {'k': k_emu, 'spec': spec_emu}
        },
        'ke': {
            'truth': {'k': k_ke_truth, 'spec': spec_ke_truth},
            'emu': {'k': k_ke_emu, 'spec': spec_ke_emu}
        }
    }

# --- Plotting 1: Temperature ---
print("Plotting Temperature Spectra...")
plt.figure(figsize=(6, 4),dpi=200)

for name, res in results.items():
    c = res['config']['color']
    # Plot Truth
    plt.loglog(res['temp']['truth']['k'], res['temp']['truth']['spec'].mean(0), '--', label=f"{res['config']['truth_label']}", color=c, alpha=0.7)
    # Plot Emulator
    plt.loglog(res['temp']['emu']['k'], res['temp']['emu']['spec'].mean(0), '-', label=f"{res['config']['label']}", color=c, linewidth=2)

plt.xlabel(r'Wavenumber $\kappa$ (km$^{-1}$)')
plt.ylabel('Power Spectral Density')
plt.title(f'Isotropic Power Spectrum of {var_to_eval} ({levs[lev_idx]}m)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add ticks for wavelengths
target_ks = [0.01, 0.02, 0.05]
labels = ['$10^{-2}$', '$2\\times10^{-2}$', '$5\\times10^{-2}$']
plt.xticks(target_ks, labels)

plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/temp_spectrum_comparison.png")
plt.show()

# --- Plotting 2: Kinetic Energy ---
print("Plotting Kinetic Energy Spectra...")
plt.figure(figsize=(6, 4),dpi=200)

for name, res in results.items():
    c = res['config']['color']
    # Plot Truth
    plt.loglog(res['ke']['truth']['k'], res['ke']['truth']['spec'].mean(0), '--', label=f"{res['config']['truth_label']}", color=c, alpha=0.7)
    # Plot Emulator
    plt.loglog(res['ke']['emu']['k'], res['ke']['emu']['spec'].mean(0), '-', label=f"{res['config']['label']}", color=c, linewidth=2)

plt.xlabel(r'Wavenumber $\kappa$ (km$^{-1}$)')
plt.ylabel('Kinetic Energy Density')
plt.title(f'Kinetic Energy Spectrum (Anomalies) at {levs[lev_idx]}m')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.xticks(target_ks, labels)

plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/ke_spectrum_comparison.png")
plt.show()

print("Script completed.")