import os
import re
import sys
import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import fsspec
import xesmf as xe
from itertools import cycle
from scipy.stats import pearsonr
# ==========================================
# --- USER CONFIGURATION ---
# ==========================================

# 1. Output Directory (Configurable Name)
OUTPUT_DIR = "Emulator_Evaluation_Figures_1deg_All_New"

# 2. Emulator Configuration 
# Format: (Path, Label, Color)
# You can add or remove entries here to compare any number of emulators.
EMULATOR_CONFIGS = [
    (
        '/pscratch/sd/a/asubel/Ocean_Emulator/notebooks/temp/Sample_Rollouts/onedeg_eval_preds.zarr', 
        "Emulator WDMSE", 
        "tab:orange"
    ),
    (
        '/pscratch/sd/a/asubel/Ocean_Emulator/notebooks/temp/Sample_Rollouts/onedeg_eval_preds_100_limit.zarr', 
        "Emulator WDSME Cap", 
        "tab:green"
    ),
    # Add a third emulator here if needed:
    (
'/pscratch/sd/a/asubel/Ocean_Emulator/notebooks/temp/Sample_Rollouts/onedeg_eval_preds_v1.zarr',
 "Emulator v1",
 "tab:blue"
    ),
]

# 3. Truth & Aux Paths
TRUTH_PATH = "/pscratch/sd/a/asubel/Ocean_Emulator/notebooks/temp/Sample_Rollouts/One_Degree_Subset.zarr"
TRUTH_LABEL = "(OM4)"

LEVS_PATH = '/pscratch/sd/a/asubel/Data/Chapter_2/CM4_CMIP_New.zarr/'
OCEAN_SAMPLE_PATH = '/pscratch/sd/a/asubel/full-model/scratch/ocean_sample.nc'
STATIC_GRID_PATH = 's3://emulators/am16581/ocean_static_no_mask_table.zarr'

# 4. Storage Options
fs = fsspec.filesystem("s3", profile="osn", client_kwargs={"endpoint_url": "https://nyu1.osn.mghpcc.org"})
storage_options = {"profile": "osn", "client_kwargs": {"endpoint_url": "https://nyu1.osn.mghpcc.org"}}

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
}

# ==========================================
# --- HELPER FUNCTIONS ---
# ==========================================

def concat_levels(ds: xr.Dataset, prefix: str, xg_name: str | None = None, new_coord: None = None) -> xr.DataArray:
    """Concatenates the levels of a 3D variable into a single DataArray."""
    levels = []
    names = []
    level_pattern = re.compile(rf"{prefix}_(\d+)$")
    for name in ds.data_vars:
        match = level_pattern.search(name)
        if match:
            levels.append(int(match.group(1)))
            names.append(name)
    names = sorted(names, key=lambda name: levels[names.index(name)])
    da = xr.concat([ds[name] for name in names], dim="lev")
    da.name = prefix
    if xg_name is not None:
        da.attrs["xg_name"] = xg_name
    if new_coord is not None:
        new_coord_name, new_coord = new_coord
        da[new_coord_name] = new_coord
        da = da.swap_dims({"lev": new_coord_name})
    return da

def extract_basin_masks(ds_with_basin_var):
    """Extracts boolean masks from the integer 'basin' variable."""
    basin_code = ds_with_basin_var['basin']
    return {
        'Atlantic': xr.where(basin_code == 2, 1.0, np.nan),
        'Pacific': xr.where(basin_code == 3, 1.0, np.nan),
        'Indian': xr.where(basin_code == 5, 1.0, np.nan),
        'Southern': xr.where(basin_code == 1, 1.0, np.nan),
    }

def process_dataset(ds_input, truth_levs, dz_arr, grid_metrics_dict):
    """Encapsulates the logic to turn raw rollout/truth into a stacked dataset with grid metrics."""
    # Initialize basic coordinate structure
    ds_stacked = xr.Dataset(coords={
        'time': (['time'], ds_input.time.values),
        'y': (['y'], ds_input.y.values),
        'x': (['x'], ds_input.x.values),
        'lev': (['lev'], truth_levs),
        'dz': (['lev'], dz_arr),
    })

    # Process 3D vars
    for var in ['uo', 'vo', 'thetao', 'so']:
        if var in OCEAN_FIELD_NAME_PREFIXES:
            prefix = OCEAN_FIELD_NAME_PREFIXES[var][0][:-1]
            try:
                data_processed = concat_levels(ds_input, prefix, var).squeeze().transpose('time', 'y', 'x', 'lev').data
                ds_stacked[var] = (['time', 'y', 'x', 'lev'], data_processed)
            except Exception as e:
                pass # Variable might not exist

    # Process 2D vars
    if 'zos' in ds_input:
        ds_stacked['zos'] = (['time', 'y', 'x'], ds_input['zos'].squeeze().data)

    # Apply Dummy Mask to dz if velocity exists to ensure consistent masking
    if 'vo' in ds_stacked:
        ds_stacked['dz'] = (ds_stacked['dz'] * (ds_stacked['vo'].isel(time=0) * 0 + 1)).compute()

    # Add Grid Metrics
    for metric, data in grid_metrics_dict.items():
        ds_stacked[metric] = data
        ds_stacked = ds_stacked.set_coords(metric)

    return ds_stacked

def calculate_zonal_mean_simple(field_mean, mask, dx, wet):
    """Simple Zonal Mean: Sum(Field * Mask * dx) / Sum(Mask * Wet_Truth * dx)"""
    numerator = (field_mean * mask * dx).sum('x')
    denominator = (mask * wet * dx).sum('x')
    return numerator / denominator

def calculate_vertical_mean_simple(field_mean, dz, wet, d_min, d_max):
    """Computes depth-weighted mean for a 2D map over a specific depth range."""
    f_slice = field_mean.sel(lev=slice(d_min, d_max))
    dz_slice = dz.sel(lev=slice(d_min, d_max))
    wet_slice = wet.sel(lev=slice(d_min, d_max))
    
    num = (f_slice * dz_slice * wet_slice).sum('lev')
    den = (dz_slice * wet_slice).sum('lev')
    return num / den

def rmse_spatial(truth, pred, area_weights):
    """Calculates area-weighted RMSE between two 2D maps."""
    diff_sq = (truth - pred)**2
    valid_mask = (~np.isnan(diff_sq))
    w_sum = area_weights.where(valid_mask).sum()
    rmse = np.sqrt((diff_sq * area_weights).sum() / w_sum)
    return rmse.item()

def calculate_snapshot_map(dataset, var_name, dz, wet_mask, d_min, d_max, time_idx):
    """Computes a 2D map for a specific variable at a specific time index."""
    ds_t = dataset.isel(time=time_idx)
    
    if var_name == 'ohc':
        rho = 1026.0
        cp = 3990.0
        field = ds_t['thetao']
        area = dataset['areacello']
        f_slice = field.sel(lev=slice(d_min, d_max))
        dz_slice = dz.sel(lev=slice(d_min, d_max))
        wet_slice = wet_mask.sel(lev=slice(d_min, d_max))
        heat_content_cell = rho * cp * f_slice * area * dz_slice * wet_slice
        return heat_content_cell.sum(dim='lev').compute()

    field = ds_t[var_name]
    if 'lev' in field.dims:
        f_slice = field.sel(lev=slice(d_min, d_max))
        dz_slice = dz.sel(lev=slice(d_min, d_max))
        wet_slice = wet_mask.sel(lev=slice(d_min, d_max))
        numerator = (f_slice * dz_slice * wet_slice).sum('lev')
        denominator = (dz_slice * wet_slice).sum('lev')
        result = numerator / denominator
    else:
        mask_2d = wet_mask.isel(lev=0) if 'lev' in wet_mask.dims else wet_mask
        result = field * mask_2d
    return result.compute()

# ==========================================
# --- PLOTTING FUNCTIONS ---
# ==========================================

def plot_zonal_bias_comparison(ds_truth, emulators_dict, variable_name, units, masks, output_dir):
    """Generates Zonal Mean Bias plots."""
    truth_field_mean = ds_truth[variable_name].mean('time')
    dx_truth = ds_truth['dx']
    wet_mask = (ds_truth[variable_name].isel(time=0) * 0 + 1).compute()
    
    print(f"Calculating Truth Zonal Profiles for {variable_name}...")
    truth_profiles = {}
    for basin in BASINS_TO_PLOT:
        truth_profiles[basin] = calculate_zonal_mean_simple(
            truth_field_mean, masks[basin], dx_truth, wet_mask
        ).compute()

    for label, emu_data in emulators_dict.items():
        ds_emu = emu_data['ds']
        emu_field_mean = ds_emu[variable_name].mean('time')
        
        print(f"Generating Zonal Bias Plot for {label}...")
        bias_profiles = {}
        rmse_vals = {}
        weight_2d = (ds_truth['dy'] * ds_truth['dz']).mean('x').compute()
        all_biases = []

        for basin in BASINS_TO_PLOT:
            prof_emu = calculate_zonal_mean_simple(
                emu_field_mean, masks[basin], dx_truth, wet_mask
            ).compute()
            prof_truth = truth_profiles[basin]
            
            bias = prof_emu - prof_truth
            bias_profiles[basin] = bias
            all_biases.append(bias.values.flatten())
            
            sq_diff = bias**2
            rmse = np.sqrt((sq_diff * weight_2d).sum(['y','lev']) / weight_2d.sum(['y','lev']))
            rmse_vals[basin] = rmse.item()

        flat_biases = np.concatenate(all_biases)
        flat_biases = flat_biases[~np.isnan(flat_biases)]
        max_abs_bias = np.percentile(np.abs(flat_biases), 98) if len(flat_biases) > 0 else 1.0
        
        norm = colors.Normalize(vmin=-max_abs_bias, vmax=max_abs_bias)
        cmap = cmocean.cm.balance

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True, constrained_layout=True)
        axes = axes.flatten()

        for i, basin in enumerate(BASINS_TO_PLOT):
            ax = axes[i]
            p_mesh = bias_profiles[basin].plot.contourf(
                ax=ax, cmap=cmap, norm=norm, add_colorbar=False, 
                levels=np.linspace(-max_abs_bias, max_abs_bias, 25), yincrease=False,
                x='y', y='lev' 
            )
            ax.set_title(f"{basin} (RMSE: {rmse_vals[basin]:.3f} {units})", fontsize=16, loc='left')
            ax.set_facecolor('grey')
            ax.set_xlabel(''); ax.set_ylabel('')
            axes[0].set_ylabel('Depth (m)'); axes[2].set_ylabel('Depth (m)')
            axes[2].set_xlabel('Latitude index'); axes[3].set_xlabel('Latitude index')
        
        cbar = fig.colorbar(p_mesh, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label(f"Bias ({units})")
        fig.suptitle(f"Zonal Mean Bias: {label} vs Truth ({variable_name})", fontsize=20, y=1.03)
        
        save_path = os.path.join(output_dir, f"Zonal_Bias_{label.replace(' ', '_')}_{variable_name}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")

def plot_spatial_bias_comparison(ds_truth, emulators_dict, variable_name, units, output_dir):
    """Generates Spatial Bias Map plots."""
    if variable_name == 'ohc': return # OHC handled in snapshots typically for this notebook flow

    truth_field_mean = ds_truth[variable_name].mean('time')
    dz_truth = ds_truth['dz']
    wet_mask = (ds_truth[variable_name].isel(time=0) * 0 + 1).compute()
    area = ds_truth['areacello']

    truth_maps = {}
    print(f"Calculating Truth Spatial Maps for {variable_name}...")
    for Slices in DEPTH_SLICES:
        key = Slices['title']
        truth_maps[key] = calculate_vertical_mean_simple(
            truth_field_mean, dz_truth, wet_mask, Slices['min'], Slices['max']
        ).compute()

    for label, emu_data in emulators_dict.items():
        ds_emu = emu_data['ds']
        emu_field_mean = ds_emu[variable_name].mean('time')
        
        print(f"Generating Spatial Bias Maps for {label}...")
        bias_maps = {}
        rmse_vals = {}
        all_biases_flat = []
        
        for Slices in DEPTH_SLICES:
            key = Slices['title']
            map_emu = calculate_vertical_mean_simple(
                emu_field_mean, dz_truth, wet_mask, Slices['min'], Slices['max']
            ).compute()
            map_truth = truth_maps[key]
            
            bias = map_emu - map_truth
            bias_maps[key] = bias
            all_biases_flat.append(bias.values.flatten())
            rmse_vals[key] = rmse_spatial(map_truth, map_emu, area)

        flat_data = np.concatenate(all_biases_flat)
        flat_data = flat_data[~np.isnan(flat_data)]
        max_abs_bias = np.percentile(np.abs(flat_data), 98) if len(flat_data) > 0 else 1.0
        
        norm = colors.Normalize(vmin=-max_abs_bias, vmax=max_abs_bias)
        cmap = cmocean.cm.balance

        fig, axes = plt.subplots(1, len(DEPTH_SLICES), figsize=(6 * len(DEPTH_SLICES), 5),
                                 subplot_kw={'projection': ccrs.Robinson(central_longitude=210)},
                                 constrained_layout=True)
        if len(DEPTH_SLICES) == 1: axes = [axes]
        
        for i, Slices in enumerate(DEPTH_SLICES):
            key = Slices['title']
            ax = axes[i]
            bias = bias_maps[key]
            p = bias.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='x', y='y',
                                     cmap=cmap, norm=norm, add_colorbar=False, rasterized=True)
            ax.set_title(f"{key}\nRMSE: {rmse_vals[key]:.3f} {units}", fontsize=14)
            ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
            ax.coastlines(zorder=11)

        cbar = fig.colorbar(p, ax=axes, orientation='horizontal', shrink=0.4, pad=0.05, aspect=30)
        cbar.set_label(f"Bias ({units})", fontsize=12)
        fig.suptitle(f"Time-Mean Bias: {label} vs Truth ({variable_name})", fontsize=18, y=1.1)
        
        save_path = os.path.join(output_dir, f"Spatial_Bias_{label.replace(' ', '_')}_{variable_name}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")

def plot_snapshot_comparison(ds_truth, emulators_dict, var_name, time_indices, depth_config, output_dir):
    """Generates Snapshot comparison maps."""
    config = VAR_CONFIG.get(var_name, {'cmap': 'viridis', 'units': '', 'label': var_name, 'centered': False})
    units = config['units']
    
    print(f"Preparing Snapshot Maps for {config['label']} ({depth_config['title']})...")
    dz_truth = ds_truth['dz']
    
    ref_var = 'thetao' if var_name == 'ohc' else var_name
    wet_mask = (ds_truth[ref_var].isel(time=0) * 0 + 1).compute()

    maps_data = {}
    datasets = ["Truth"] + list(emulators_dict.keys())
    
    # Truth Maps
    maps_data["Truth"] = {}
    for t in time_indices:
        maps_data["Truth"][t] = calculate_snapshot_map(
            ds_truth, var_name, dz_truth, wet_mask, 
            depth_config['min'], depth_config['max'], t
        )
    
    # Emulator Maps
    for label, emu_data in emulators_dict.items():
        ds_emu = emu_data['ds']
        maps_data[label] = {}
        for t in time_indices:
            maps_data[label][t] = calculate_snapshot_map(
                ds_emu, var_name, dz_truth, wet_mask, 
                depth_config['min'], depth_config['max'], t
            )

    nrows = len(datasets)
    ncols = len(time_indices)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             subplot_kw={'projection': ccrs.Robinson(central_longitude=210)},
                             constrained_layout=True)
    
    if nrows == 1: axes = np.expand_dims(axes, 0)
    if ncols == 1: axes = np.expand_dims(axes, 1)

    cmap = config['cmap']
    
    for j, t_idx in enumerate(time_indices):
        # Determine Color Scaling based on Truth
        truth_slice = maps_data["Truth"][t_idx].values
        valid_data = truth_slice[~np.isnan(truth_slice)]
        
        if len(valid_data) > 0:
            if config['centered']:
                abs_max = np.percentile(np.abs(valid_data), 99.75)
                vmin, vmax = -abs_max, abs_max
            else:
                vmin = np.percentile(valid_data, .25)
                vmax = np.percentile(valid_data, 98)
        else:
            vmin, vmax = 0, 1
        
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        try:
            time_val = ds_truth.time[t_idx].values
            time_str = str(time_val)[:10]
        except:
            time_str = f"Index {t_idx}"

        for i, ds_name in enumerate(datasets):
            ax = axes[i, j]
            data_to_plot = maps_data[ds_name][t_idx]
            
            p = data_to_plot.plot.pcolormesh(
                ax=ax, transform=ccrs.PlateCarree(), x='x', y='y',
                cmap=cmap, norm=norm, add_colorbar=False, rasterized=True
            )
            ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
            ax.coastlines(zorder=11)
            
            if i == 0: ax.set_title(f"Time: {time_str}", fontsize=16, fontweight='bold')
            else: ax.set_title("")
            
            if j == 0:
                ax.text(-0.07, 0.55, ds_name, va='center', ha='right',
                        rotation='vertical', transform=ax.transAxes, fontsize=14, fontweight='bold')

        cbar = fig.colorbar(p, ax=axes[:, j], orientation='horizontal', shrink=0.6, pad=0.03, aspect=25)
        cbar.set_label(f"{config['label']} ({units})", fontsize=12)

    main_title = f"{config['label']} Snapshots ({depth_config['title']})"
    fig.suptitle(main_title, fontsize=20, y=1.05)
    
    save_path = os.path.join(output_dir, f"Snapshots_{var_name}_{depth_config['min']}_{depth_config['max']}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def get_trend(da):
    """
    Calculates a linear trend for a DataArray with a time dimension.
    Returns:
        slope_per_decade (float): The slope scaled to units/decade.
        trend_line (xr.DataArray): The trend line with the same shape/coords as input.
    """
    # 1. Convert time to fractional years for linear fitting
    times = da.time.values
    # Handle cftime or numpy datetime objects
    if hasattr(times[0], 'year'): # cftime
        years = np.array([t.year + (t.month-1)/12.0 for t in times])
    elif np.issubdtype(times.dtype, np.datetime64): # numpy datetime64
        # approximate conversion
        years = pd.to_datetime(times).year + (pd.to_datetime(times).month-1)/12.0
    else:
        # Fallback: assume monthly steps if no time info
        years = np.arange(len(times)) / 12.0

    y = da.values
    
    # 2. Fit linear trend (ignoring NaNs)
    valid_idx = np.isfinite(y)
    if not np.any(valid_idx):
        return np.nan, da * np.nan
        
    slope, intercept = np.polyfit(years[valid_idx], y[valid_idx], 1)
    
    # 3. Construct trend line
    trend_vals = slope * years + intercept
    trend_line = xr.DataArray(trend_vals, coords=da.coords, dims=da.dims)
    
    return slope * 10, trend_line # * 10 for per-decade scaling

def plot_mean_timeseries_comparison(ds_truth, emulators_dict, variable_name, var_units, var_label, output_dir):
    """
    Generates a multi-panel plot comparing emulator and ground truth for the 
    area/volume-weighted mean of a specified variable.
    """
    print(f"Generating Time Series for {var_label}...")
    num_plots = len(DEPTH_SLICES)
    area = ds_truth['areacello']
    
    # Create a wet mask based on Truth (NaN check on first time step)
    # 1.0 where water exists, 0.0 where land/NaN
    wet_mask = xr.where(np.isnan(ds_truth[variable_name].isel(time=0)), 0.0, 1.0)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(
        num_plots, 1, 
        figsize=(10, 3.5 * num_plots), 
        sharex=True,
        constrained_layout=True
    )
    if num_plots == 1: axes = [axes]
    
    for i, Slices in enumerate(DEPTH_SLICES):
        ax = axes[i]
        min_d, max_d = Slices['min'], Slices['max']
        depth_slice = slice(min_d, max_d)
        
        # --- Process Truth ---
        field_truth = ds_truth[variable_name].sel(lev=depth_slice)
        dz_truth = ds_truth['dz'].sel(lev=depth_slice)
        vol_weight_truth = dz_truth * area
        
        # 1. Calculate Numerator (Sum of Field * Volume) -> Reduces 4D to 1D
        numerator_truth = (field_truth * vol_weight_truth).sum(['x', 'y', 'lev']).compute()
        
        # 2. Calculate Denominator (Total Valid Volume)
        # Using wet_mask to ensure we sum the same volume domain
        volume_denominator_truth = (vol_weight_truth * wet_mask.sel(lev=depth_slice)).sum(['x', 'y', 'lev']).compute()
        
        # 3. Compute Mean Time Series
        ts_truth = numerator_truth / volume_denominator_truth
        
        trend_val_truth, trend_line_truth = get_trend(ts_truth)
        
        # Plot Truth
        # Use simple plotting if xarray plot fails with specific time formats
        try:
            ts_truth.plot(ax=ax, label=TRUTH_LABEL, color='black', lw=2.5, zorder=10)
        except:
            ax.plot(ts_truth.time, ts_truth, label=TRUTH_LABEL, color='black', lw=2.5, zorder=10)

        ax.plot(trend_line_truth.time, trend_line_truth, 
                label=f'Truth Trend ({trend_val_truth:+.2e} {var_units}/dec)', 
                color='gray', ls='--', lw=2, zorder=10)
        
        # --- Process Emulators ---
        stats_text = []
        
        for label, emu_data in emulators_dict.items():
            ds_emu = emu_data['ds']
            color = emu_data['color']
            
            field_emu = ds_emu[variable_name].sel(lev=depth_slice)
            dz_emu = ds_emu['dz'].sel(lev=depth_slice)
            vol_weight_emu = dz_emu * area
            
            # 1. Calculate Numerator
            numerator_emu = (field_emu * vol_weight_emu).sum(['x', 'y', 'lev']).compute()
            
            # 2. Calculate Denominator 
            # (Using emulator dz but truth mask to compare consistent domains)
            volume_denominator_emu = (vol_weight_emu * wet_mask.sel(lev=depth_slice)).sum(['x', 'y', 'lev']).compute()
            
            # 3. Compute Mean
            ts_emu = numerator_emu / volume_denominator_emu
            
            trend_val_emu, trend_line_emu = get_trend(ts_emu)
            
            # Metrics
            min_len = min(len(ts_truth), len(ts_emu))
            r_val, _ = pearsonr(ts_truth.values[:min_len], ts_emu.values[:min_len])
            r2_val = r_val**2
            stats_text.append(f"{label} R²={r2_val:.2f}")
            
            # Plot Emulator
            try:
                ts_emu.plot(ax=ax, label=label, color=color, lw=2)
            except:
                ax.plot(ts_emu.time, ts_emu, label=label, color=color, lw=2)

            ax.plot(trend_line_emu.time, trend_line_emu, 
                    label=f'{label} Trend ({trend_val_emu:+.2e})', 
                    color=color, ls=':', lw=1.5)
            
        # --- Formatting ---
        stats_title = " | ".join(stats_text)
        ax.set_title(f"{Slices['title']}\n{stats_title}", fontsize=12, loc='left')
        ax.set_ylabel(f"Global Mean {var_label} [{var_units}]", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_xlabel('')

    axes[-1].set_xlabel("Year", fontsize=12)
    fig.suptitle(f"Global Mean {var_label} Time Series by Depth", fontsize=14, y=1.02)
    
    # Save Figure
    save_path = os.path.join(output_dir, f"TimeSeries_Comparison_{var_label}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_variance_comparison(ds_truth, emulators_dict, variable_name, units, output_dir=None):
    """
    Generates a grid of Time-Variance maps.
    Rows: Truth, then Emulators.
    Cols: Depth Slices.
    Colorbar: Shared per COLUMN (Depth Slice).
    """
    print(f"\nGenerating Variance Comparison for {variable_name} (Per-Column Colorbars)...")
    
    # 1. Setup Metrics
    dz_truth = ds_truth['dz']
    wet_mask = (ds_truth[variable_name].isel(time=0) * 0 + 1).compute()
    
    # 2. Calculate Truth Variance (3D)
    print("Calculating Truth Variance...")
    truth_var_3d = ds_truth[variable_name].var('time') 
    
    # 3. Pre-compute Maps
    plot_data = {}
    plot_data["Truth"] = {}
    
    # Process Truth Slices
    for Slices in DEPTH_SLICES:
        key = Slices['title']
        plot_data["Truth"][key] = calculate_vertical_mean_simple(
            truth_var_3d, dz_truth, wet_mask, Slices['min'], Slices['max']
        ).compute()

    # Process Emulators
    for label, emu_data in emulators_dict.items():
        print(f"Calculating Variance for {label}...")
        ds_emu = emu_data['ds']
        emu_var_3d = ds_emu[variable_name].var('time')
        
        plot_data[label] = {}
        for Slices in DEPTH_SLICES:
            key = Slices['title']
            plot_data[label][key] = calculate_vertical_mean_simple(
                emu_var_3d, dz_truth, wet_mask, Slices['min'], Slices['max']
            ).compute()

    # 4. Plotting Setup
    datasets = ["Truth"] + list(emulators_dict.keys())
    nrows = len(datasets)
    ncols = len(DEPTH_SLICES)
    
    cmap = cmocean.cm.thermal 

    fig, axes = plt.subplots(
        nrows, ncols, 
        figsize=(5 * ncols, 4 * nrows), # Increased height slightly for individual cbars
        subplot_kw={'projection': ccrs.Robinson(central_longitude=210)},
        constrained_layout=True
    )
    
    if nrows == 1: axes = np.expand_dims(axes, 0)
    if ncols == 1: axes = np.expand_dims(axes, 1)

    # 5. Iterate by COLUMN to determine column-specific color scaling
    for j, Slices in enumerate(DEPTH_SLICES):
        key = Slices['title']
        
        # Gather all data for this column to find the 99th percentile
        col_data_flat = []
        for ds_name in datasets:
            col_data_flat.append(plot_data[ds_name][key].values.flatten())
            
        flat_data = np.concatenate(col_data_flat)
        flat_data = flat_data[~np.isnan(flat_data)]
        
        if len(flat_data) > 0:
            vmax = np.percentile(flat_data, 99)
        else:
            vmax = 1.0
        vmin = 0
        
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        # Plot every Row in this Column
        p_mesh = None # Placeholder for the mappable
        for i, ds_name in enumerate(datasets):
            ax = axes[i, j]
            data = plot_data[ds_name][key]
            
            p_mesh = data.plot.pcolormesh(
                ax=ax, transform=ccrs.PlateCarree(),
                x='x', y='y',
                cmap=cmap, norm=norm,
                add_colorbar=False, rasterized=True
            )
            
            ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
            ax.coastlines(zorder=11)
            
            # Row Labels
            if j == 0:
                ax.text(-0.07, 0.55, ds_name, va='center', ha='right',
                        rotation='vertical', transform=ax.transAxes, 
                        fontsize=14, fontweight='bold')
            
            # Column Titles
            if i == 0:
                ax.set_title(f"{key}", fontsize=14, fontweight='bold')
            else:
                ax.set_title("")

        # Add Colorbar for this COLUMN (Placed below the column)
        if p_mesh:
            cbar = fig.colorbar(p_mesh, ax=axes[:, j], orientation='horizontal', shrink=0.7, pad=0.03, aspect=20)
            cbar.set_label(f"Variance ({units}$^2$)\nMax: {vmax:.2e}", fontsize=10)

    fig.suptitle(f"Temporal Variance Comparison: {variable_name}", fontsize=18, y=1.05)
    
    if output_dir:
        import os
        save_path = os.path.join(output_dir, f"Variance_Comparison_{variable_name}.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()

# ==========================================
# --- MAIN EXECUTION ---
# ==========================================

def main():
    print(f"Output Directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Data
    print("Loading Truth Data...")
    if "s3" in TRUTH_PATH:
        data_truth_ocean = xr.open_zarr(TRUTH_PATH, storage_options=storage_options)
    else:
        data_truth_ocean = xr.open_zarr(TRUTH_PATH)

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
        return

    # Sync Time (Assume all emulators align with first emulator, truth aligns with them)
    time_slice = slice(emulators_raw[0].time.values[0], emulators_raw[0].time.values[-1])
    data_truth_ocean = data_truth_ocean.sel(time=time_slice)
    
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
    dataset_truth = data_truth_ocean
    for metric, data in grid_metrics.items():
        dataset_truth[metric] = data
        dataset_truth = dataset_truth.set_coords(metric)

    processed_emulators = []
    for emu_ds in emulators_raw:
        processed_emulators.append(process_dataset(emu_ds, levs, dz_from_data, grid_metrics))

    # Construct Emulator Dictionary
    emulators_to_analyze = {}
    for i, ds in enumerate(processed_emulators):
        emulators_to_analyze[emulator_labels[i]] = {'ds': ds, 'color': emulator_colors[i]}

    # Basin Masks
    basin_masks = extract_basin_masks(dataset_truth)

    # --- Run Plots ---
    print("\nStarting Plotting Sequence...")
    
    # 1. Zonal Bias (Thetao)
    plot_zonal_bias_comparison(dataset_truth, emulators_to_analyze, 'thetao', '$^\\circ$C', basin_masks, OUTPUT_DIR)
    
    # 2. Spatial Bias (Thetao)
    plot_spatial_bias_comparison(dataset_truth, emulators_to_analyze, 'thetao', '$^\\circ$C', OUTPUT_DIR)
    
    # 3. Spatial Bias (Salinity)
    plot_spatial_bias_comparison(dataset_truth, emulators_to_analyze, 'so', 'psu', OUTPUT_DIR)

    # 4. Snapshots (Thetao, Surface)
    time_indices = [-100, -50, -1] # As per notebook
    plot_snapshot_comparison(dataset_truth, emulators_to_analyze, 'thetao', time_indices, DEPTH_SLICES[0], OUTPUT_DIR)
    
    # 5. Snapshots (OHC, Upper Ocean)
    plot_snapshot_comparison(dataset_truth, emulators_to_analyze, 'ohc', time_indices, {'min': 0, 'max': 700, 'title': 'Upper Ocean (0-700m)'}, OUTPUT_DIR)

# 6. Time Series (Thetao)
    plot_mean_timeseries_comparison(
        dataset_truth, 
        emulators_to_analyze, 
        variable_name='thetao', 
        var_units='$^\\circ$C', 
        var_label='Temperature',
        output_dir=OUTPUT_DIR
    )

    # 7. Time Series (Salinity)
    plot_mean_timeseries_comparison(
        dataset_truth, 
        emulators_to_analyze, 
        variable_name='so', 
        var_units='psu', 
        var_label='Salinity',
        output_dir=OUTPUT_DIR
    )

    plot_variance_comparison(
        dataset_truth, 
        emulators_to_analyze, 
        variable_name='thetao', 
        units='$^\\circ$C',
        output_dir=OUTPUT_DIR 
    )
    
    plot_variance_comparison(
        dataset_truth, 
        emulators_to_analyze, 
        variable_name='so', 
        units='psu',
        output_dir=OUTPUT_DIR 
    )    
    print(f"\nAll plots generated successfully in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()