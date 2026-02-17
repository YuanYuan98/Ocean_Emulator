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
    #{'title':'3000m','index':15}
]

# # Visualization Settings
# VAR_CONFIG = {
#     'thetao': {'cmap': cmocean.cm.thermal, 'units': '$^\circ$C', 'label': 'Temperature', 'centered': False},
#     'so':     {'cmap': cmocean.cm.haline,  'units': 'psu',       'label': 'Salinity',    'centered': False},
#     'zos':    {'cmap': cmocean.cm.balance, 'units': 'm',         'label': 'SSH',         'centered': True},
#     'uo':     {'cmap': cmocean.cm.balance, 'units': 'm/s',       'label': 'U-Velocity',  'centered': True},
#     'vo':     {'cmap': cmocean.cm.balance, 'units': 'm/s',       'label': 'V-Velocity',  'centered': True},
# }

# Configuration
BASINS_TO_PLOT = ['Atlantic', 'Pacific', 'Indian', 'Southern']

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

def _detrend_linear_torch(data):
    """
    Removes a linear plane of best fit from 4D (B, C, H, W) data.
    """
    B, C, H, W = data.shape
    device = data.device
    dtype = data.dtype
    
    y_coords = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    x_coords = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    A = torch.stack([X.flatten(), Y.flatten(), torch.ones_like(X).flatten()], dim=1)
    
    B_prime = B * C
    data_flat = data.reshape(B_prime, H * W)
    

    coeffs, _, _, _ = torch.linalg.lstsq(A, data_flat.T)

    plane = (A @ coeffs.permute(1, 0).unsqueeze(-1)).reshape(B_prime, H, W)
    
    detrended_data = data.reshape(B_prime, H, W) - plane
    
    return detrended_data.reshape(B, C, H, W)


def compute_isotropic_spectrum_torch(data, dx=1.0, dy=1.0, num_bins=None, n_factor = 4,
                                     remove_mean=True, detrend=None, window='Hann',
                                     truncate=True, cutoff_before_bins: bool = True):
    """
    Computes the isotropic 1D power spectrum from 2D (H,W), 3D (B,H,W),
    or 4D (B,C,H,W) data. Matches `xrft.isotropic_power_spectrum(scaling="density")`.

    The output spectrum is computed for each batch and channel element.

    Parameters:
    ----------
    data : torch.Tensor
        Input data tensor. Can be 2D, 3D, or 4D.
    dx : float, optional
        Grid spacing in the x-dimension.
    dy : float, optional
        Grid spacing in the y-dimension.
    num_bins : int, optional
        Number of bins. If None, defaults to min(H, W) // n_factor.
    n_factor : int, optional
        Factor to determine number of bins.
    remove_mean : bool, optional
        If True, removes spatial mean. Overridden by `detrend`.
    detrend : str, optional
        'linear' or 'constant'.
    window : str, optional
        'hann' or 'Hann'.
    truncate : bool, optional
        If True, truncates spectrum at the smallest Nyquist frequency.
    cutoff_before_bins: bool, optional
        If True, truncates the spectrum after already computing the bin locations. Matches xrft
    Returns:
    -------
    k_bins_centers : torch.Tensor
        1D tensor of bin center wavenumbers. Shape: (num_bins,)
    iso_spectrum : torch.Tensor
        1D tensor of the (k * P(k)) spectrum.
        Shape: (B, C, num_bins), (B, num_bins), or (num_bins,)
    """
    
    device = data.device
    dtype = data.dtype
    orig_dim = data.dim()
    
    if orig_dim == 2:
        data = data.reshape(1, 1, *data.shape)  
    elif orig_dim == 3:
        data = data.unsqueeze(1)                
    elif orig_dim != 4:
        raise ValueError("Input data must be 2D, 3D, or 4D (B, C, H, W)")
        
    B, C, H, W = data.shape
    B_prime = B * C 
    Lx = W * dx
    Ly = H * dy
    
    if num_bins is None:
        num_bins = min(H, W) // n_factor 

    if detrend == 'linear':
        data = _detrend_linear_torch(data)
    elif detrend == 'constant' or remove_mean:
        data = data - torch.mean(data, dim=(-2, -1), keepdim=True)

    if window and window.lower() == 'hann':
        win_y = torch.hann_window(H, device=device, dtype=dtype).unsqueeze(1)
        win_x = torch.hann_window(W, device=device, dtype=dtype).unsqueeze(0)
        win_2d = (win_y * win_x).reshape(1, 1, H, W)
        
        window_correction = torch.mean(win_2d**2).item()
        data = data * win_2d
    else:
        window_correction = 1.0

    fft_2d = torch.fft.rfft2(data, norm='forward')
    
    power_2d = torch.abs(fft_2d)**2
    power_2d = power_2d / window_correction
    
    psd_2d = power_2d * (Lx * Ly)
    
    k_x = torch.fft.rfftfreq(W, d=dx, device=device, dtype=dtype)
    k_y = torch.fft.fftfreq(H, d=dy, device=device, dtype=dtype)
    
    k_x_nyq = 1.0 / (2.0 * dx)
    k_y_nyq = 1.0 / (2.0 * dy)

    k_Y, k_X = torch.meshgrid(k_y, k_x, indexing='ij')
    k_mag = torch.sqrt(k_X**2 + k_Y**2)
    
    k_max_domain = k_mag.max()
    
    if truncate and cutoff_before_bins:
        k_max_cutoff = min(k_x_nyq, k_y_nyq)
        k_max = min(k_max_domain, k_max_cutoff)
    else:
        k_max = k_max_domain
        
    k_bins = torch.linspace(0, k_max, num_bins + 1, device=device, dtype=dtype)
    if truncate and not cutoff_before_bins:
        k_max_cutoff = min(k_x_nyq, k_y_nyq) 
        k_max = min(k_max_domain, k_max_cutoff)        
        k_bins = k_bins[k_bins<k_max_cutoff]
        num_bins = k_bins.numel()-1
    k_bins_centers = (k_bins[:-1] + k_bins[1:]) / 2

    k_mag_flat = k_mag.flatten()
    bin_edges = k_bins[1:-1] 
    
    bin_indices = torch.bucketize(k_mag_flat, bin_edges, right=True)
    
    N_flat = k_mag_flat.shape[0]
    psd_flat_batched = psd_2d.reshape(B_prime, N_flat)

    bin_indices_batched = bin_indices.expand(B_prime, -1)
    binned_psd_sum = torch.zeros(B_prime, num_bins, device=device, dtype=dtype)
    binned_psd_sum.scatter_add_(dim=1, index=bin_indices_batched, src=psd_flat_batched)
    binned_counts = torch.bincount(
        bin_indices, 
        minlength=num_bins
    )
    
    binned_counts_safe = binned_counts.float()
    binned_counts_safe[binned_counts_safe == 0] = torch.nan

    iso_psd_binned = binned_psd_sum / binned_counts_safe.unsqueeze(0)
    iso_spectrum = iso_psd_binned * k_bins_centers.unsqueeze(0)
    iso_spectrum = iso_spectrum.reshape(B, C, num_bins)
    iso_spectrum[..., 0] = torch.nan
    
    if orig_dim == 2:
        iso_spectrum = iso_spectrum.squeeze(0).squeeze(0) 
    elif orig_dim == 3:
        iso_spectrum = iso_spectrum.squeeze(1)            
    return k_bins_centers, iso_spectrum





def rmse_spatial(truth_map, pred_map, area):
    """Area-weighted RMSE for 2D spatial maps."""
    sq_diff = (truth_map - pred_map)**2
    
    # Mask weights where data is missing
    weights = area.where(sq_diff.notnull())
    
    numerator = (sq_diff * weights).sum(['y', 'x'])
    denominator = weights.sum(['y', 'x'])
    
    return np.sqrt(numerator / denominator).values.item()

def calculate_ohc_2d(ds):
    """
    Calculates Ocean Heat Content (J/m^2) by integrating thetao over depth.
    Formula: Integral(rho * cp * thetao * dz)
    """
    # Ensure dz is present and correct shape
    # Sum over 'lev' dimension
    ohc = (ds['thetao'] * RHO * CP * ds['dz'])
    ohc.attrs['units'] = 'J/m^2'
    return ohc

def calculate_zonal_mean_2d(field_2d, mask, dx, wet_2d):
    """
    Zonal Mean for a 2D field (e.g. OHC).
    """
    numerator = (field_2d * mask * dx).sum('x')
    denominator = (mask * wet_2d * dx).sum('x')
    return numerator / denominator

def plot_ohc_zonal_bias_comparison(ds_truth, emulators_dict, masks,colorbar_fix = False):
    """Generates a 2x2 plot of Zonal Mean OHC Bias (Line Plots) for EACH emulator."""
    
    variable_name = 'OHC'
    units = 'J/m^2'
    
    # 0. Setup Truth Metrics
    # Calculate OHC Mean (Time averaged)
    truth_field_2d = calculate_ohc_2d(ds_truth).mean('time').compute()
    dx_truth = ds_truth['dx']
    
    # Define 2D "Wet" mask from Truth (surface)
    wet_mask_2d = (ds_truth['thetao'].isel(time=0, lev=0) * 0 + 1).compute()
    
    # 1. Calculate Truth Zonal Profiles (1D: y)
    truth_profiles = {}
    print("Calculating Truth OHC Zonal Profiles...")
    for basin in BASINS_TO_PLOT:
        truth_profiles[basin] = calculate_zonal_mean_2d(
            truth_field_2d, masks[basin], dx_truth, wet_mask_2d
        ).compute()

    # 2. Iterate over Emulators
    for iindex, (label, emu_data) in enumerate(emulators_dict.items()):
        ds_emu = emu_data['ds']
        # Calculate Emulator OHC Mean
        emu_field_2d = calculate_ohc_2d(ds_emu).mean('time').compute()
        
        print(f"\nGenerating OHC Zonal Bias Plot for {label}...")
        
        bias_profiles = {}

        rmse_vals = {}
        
        # Weights for RMSE (Mean across X)
        weight_2d = (ds_truth['dy'] * ds_truth['dz']).mean('x').compute()

        all_biases = []

        for basin in BASINS_TO_PLOT:
            # Calculate Emulator Profile using TRUTH metrics
            prof_emu = calculate_zonal_mean_2d(
                emu_field_2d, masks[basin], dx_truth, wet_mask_2d
            ).compute()
            
            prof_truth = truth_profiles[basin]
            
            # Simple Subtraction
            bias = prof_emu - prof_truth
            bias_profiles[basin] = bias.drop('time')
            all_biases.append(bias.values.flatten())
            
            # RMSE Calculation
            sq_diff = bias**2
            rmse = np.sqrt((sq_diff * weight_2d).sum(['y','lev']) / weight_2d.sum(['y','lev']))
            rmse_vals[basin] = rmse.item()

        # Dynamic Color Scaling
        if iindex==0:
            flat_biases = np.concatenate(all_biases)
            flat_biases = flat_biases[~np.isnan(flat_biases)]
            max_abs_bias = np.percentile(np.abs(flat_biases), 98) if len(flat_biases) > 0 else 1.0

        if colorbar_fix:
            max_abs_bias = 2.5*10**8
        
        norm = colors.Normalize(vmin=-max_abs_bias, vmax=max_abs_bias)
        cmap = cmocean.cm.balance

        # --- Plotting ---
        plt.style.use('seaborn-v0_8-talk')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True, constrained_layout=True)
        axes = axes.flatten()

        for i, basin in enumerate(BASINS_TO_PLOT):
            ax = axes[i]
            print(f"Shape: {bias_profiles[basin].shape}")
            print(f"Dims: {bias_profiles[basin].dims}")
            
            p_mesh = bias_profiles[basin].plot.contourf(
                ax=ax, cmap=cmap, norm=norm, add_colorbar=False, 
                levels=np.linspace(-max_abs_bias, max_abs_bias, 25), yincrease=False,
                x='y', y='lev' 
            )
            ax.set_title(f"{basin} (RMSE: {rmse_vals[basin]:.3e} {units})", fontsize=16, loc='left')
            ax.set_facecolor('grey')
            ax.set_xlabel(''); ax.set_ylabel('')

        axes[0].set_ylabel('Depth (m)'); axes[2].set_ylabel('Depth (m)')
        axes[2].set_xlabel('Latitude index'); axes[3].set_xlabel('Latitude index')
        cbar = fig.colorbar(p_mesh, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label(f"Bias ({units})")
        fig.suptitle(f"Zonal Mean Bias of OHC: {label} vs Truth", fontsize=20, y=1.03)
        plt.show()


def calculate_vertical_simple(field, dz, wet_mask, depth_index):
    """
    Calculates the vertical simple map using Truth's geometry.
    Formula: Sum(Field * dz * wet) / Sum(dz * wet) over depth slice.
    """
    # Select depth slices
    f_slice = field.isel(lev=depth_index)
    dz_slice = dz.isel(lev=depth_index)
    wet_slice = wet_mask.isel(lev=depth_index)
    
    numerator = (f_slice * dz_slice * wet_slice)
    denominator = (dz_slice * wet_slice)
    
    return numerator / denominator

def calculate_vertical_mean_simple(field, dz, wet_mask, min_depth, max_depth):
    """
    Calculates the vertical mean map using Truth's geometry.
    Formula: Sum(Field * dz * wet) / Sum(dz * wet) over depth slice.
    """
    # Select depth slices
    f_slice = field.sel(lev=slice(min_depth, max_depth))
    dz_slice = dz.sel(lev=slice(min_depth, max_depth))
    wet_slice = wet_mask.sel(lev=slice(min_depth, max_depth))
    
    numerator = (f_slice * dz_slice * wet_slice).sum('lev')
    denominator = (dz_slice * wet_slice).sum('lev')
    
    return numerator / denominator.where(denominator != 0)


def calculate_ohc_map(dataset, dz, wet_mask, min_depth, max_depth):
    """
    Calculates the Time-Mean OHC map (Vertically Integrated).
    Formula: Sum(rho * cp * thetao * dz) / Scale
    Units: GJ/m^2
    """
    # Select slices
    thetao_slice = dataset['thetao'].sel(lev=slice(min_depth, max_depth))
    dz_slice = dz.sel(lev=slice(min_depth, max_depth))
    wet_slice = wet_mask.sel(lev=slice(min_depth, max_depth))
    
    # 1. Integrate Vertically (Sum over lev)
    # Result is (time, y, x)
    #ohc_integrated = (RHO * CP * thetao_slice * dz_slice *dataset['areacello']* wet_slice).sum('lev')

    ohc_integrated = (RHO * CP * thetao_slice * dz_slice * wet_slice).sum('lev')

    
    # 2. Time Average
    ohc_mean = ohc_integrated.mean('time')
    
    # 3. Scale
    return ohc_mean / SCALE_FACTOR

def plot_ohc_mean_comparison(ds_truth, emulators_dict,TRUTH_LABEL='Truth',colorbar_fix=False):
    """
    Generates a grid of OHC maps:
    Rows: Datasets (Truth, Emulator 1, Emulator 2...)
    Cols: Depth Slices (Upper, Mid, Deep)
    """
    
    # Setup Metrics
    print("Preparing Truth Metrics...")
    dz_truth = ds_truth['dz']
    # Wet mask from Truth (1 for Ocean, NaN for Land)
    wet_mask = (ds_truth['thetao'].isel(time=0) * 0 + 1).compute()
    
    # 1. Compute Truth Maps (Row 1)
    truth_maps = {}
    print("Calculating Truth OHC Maps...")
    for Slices in DEPTH_SLICES:
        key = Slices['title']
        truth_maps[key] = calculate_ohc_map(
            ds_truth, dz_truth, wet_mask, Slices['min'], Slices['max']
        ).compute()
        
    # 2. Compute Emulator Maps
    emu_maps = {}
    for label, emu_data in emulators_dict.items():
        print(f"Calculating OHC Maps for {label}...")
        ds_emu = emu_data['ds']
        emu_maps[label] = {}
        for Slices in DEPTH_SLICES:
            key = Slices['title']
            # Note: We use Truth's dz and wet_mask to ensure integration domain consistency
            emu_maps[label][key] = calculate_ohc_map(
                ds_emu, dz_truth, wet_mask, Slices['min'], Slices['max']
            ).compute()

    # --- Plotting Setup ---
    datasets = [TRUTH_LABEL] + list(emulators_dict.keys())
    nrows = len(datasets)
    ncols = len(DEPTH_SLICES)
    
    fig, axes = plt.subplots(
        nrows, ncols, 
        figsize=(4.5 * ncols, 3 * nrows),
        subplot_kw={'projection': ccrs.Robinson(central_longitude=210)},
        constrained_layout=True
    )
    
    # Ensure axes is 2D array even if 1 row
    if nrows == 1: axes = np.expand_dims(axes, 0)
    if ncols == 1: axes = np.expand_dims(axes, 1)

    cmap = cmocean.cm.thermal
    
    # 3. Iterate Column-by-Column (Depth Slice) to lock colorbars
    for j, Slices in enumerate(DEPTH_SLICES):
        key = Slices['title']
        
        # Determine Color Scale based on TRUTH (or global max)
        data_truth = truth_maps[key].values
        valid_data = data_truth[~np.isnan(data_truth)]
        
        vmin = np.percentile(valid_data, .5)
        vmax = np.percentile(valid_data, 99.5)
        
        if colorbar_fix:
            if key == 'Surface (0-700m)':
                vmin, vmax = 0, 5*10**(-11)
            elif key == 'Intermediate (700-2000m)':
                vmin, vmax = 0, 5*10**(-11)
            elif key == 'Deep (2000-7000m)':
                vmin, vmax = 0, 4*10**(-11)
        
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        # Iterate Row-by-Row (Datasets)
        for i, ds_name in enumerate(datasets):
            ax = axes[i, j]
            
            # Select Data
            if ds_name == TRUTH_LABEL:
                data_to_plot = truth_maps[key]
                row_label = "Ground Truth"
            else:
                # Map back from display label to dictionary key if needed, 
                # but here ds_name is the label
                data_to_plot = emu_maps[ds_name][key]
                row_label = ds_name
            
            # Plot
            p = data_to_plot.plot.pcolormesh(
                ax=ax, transform=ccrs.PlateCarree(),
                x='x', y='y',
                cmap=cmap, norm=norm,
                add_colorbar=False,
                rasterized=True
            )
            
            # Styling
            ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
            ax.coastlines(zorder=11)
            
            # Titles/Labels
            if i == 0:
                ax.set_title(key, fontsize=16, fontweight='bold')
            else:
                ax.set_title("")
                
            if j == 0:
                # Add Row Label on the left
                ax.text(-0.07, 0.55, row_label, va='center', ha='right',
                        rotation='vertical', transform=ax.transAxes, 
                        fontsize=14, fontweight='bold')

        # Add ONE Colorbar per Column (at the bottom of the column)
        # We attach it to the last axes in the column
        cbar = fig.colorbar(
            p, ax=axes[:, j], 
            orientation='horizontal', 
            shrink=0.6, pad=0.03, aspect=25
        )
        cbar.set_label(r"Mean OHC ($ZJ$)", fontsize=12)

    fig.suptitle("8-yr-Mean Ocean Heat Content Comparison", fontsize=20, y=1.05)
    plt.show()


def calculate_temperature_map_layer(dataset, var_name, dz, wet_mask, depth_index, time_idx=None):
    """
    Computes a 2D map for a specific variable at a specific time index.
    - 3D vars: Computes a specific depth.
    - 2D vars: Returns surface field masked by Truth.
    """
    # Select Time Step
    # 2. Time Average
    ds_t = dataset.mean(dim='time')
    #ds_t = dataset.isel(time=time_idx)
    field = ds_t[var_name]
    
    # Select Depth Slice
    f_slice = field.isel(lev=depth_index)
    dz_slice = dz.isel(lev=depth_index)
    wet_slice = wet_mask.isel(lev=depth_index)
    
    # Depth Average
    numerator = (f_slice * dz_slice * wet_slice)
    denominator = (dz_slice * wet_slice)
    result = numerator / denominator
        
    return result.compute()


def calculate_temperature_map(dataset, var_name, dz, wet_mask, d_min, d_max, time_idx=None):
    """
    Computes a 2D map for a specific variable at a specific time index.
    - 3D vars: Computes depth-weighted average.
    - 2D vars: Returns surface field masked by Truth.
    """
    # Select Time Step
    # 2. Time Average
    ds_t = dataset.mean(dim='time')
    #ds_t = dataset.isel(time=time_idx)
    field = ds_t[var_name]
    
    # Check if variable is 3D (has 'lev') or 2D (like zos)
    if 'lev' in field.dims:
        # Select Depth Slice
        f_slice = field.sel(lev=slice(d_min, d_max))
        dz_slice = dz.sel(lev=slice(d_min, d_max))
        wet_slice = wet_mask.sel(lev=slice(d_min, d_max))
        
        # Depth Average
        numerator = (f_slice * dz_slice * wet_slice).sum('lev')
        denominator = (dz_slice * wet_slice).sum('lev')
        result = numerator / denominator
    else:
        # 2D Variable (Use Top Layer of wet mask if wet_mask is 3D)
        if 'lev' in wet_mask.dims:
            mask_2d = wet_mask.isel(lev=0)
        else:
            mask_2d = wet_mask
            
        result = field * mask_2d
        
    return result.compute()


def plot_temperature_mean_comparison(ds_truth, emulators_dict,TRUTH_LABEL='Truth',colorbar_fix=False):
    VARIABLE_TO_PLOT = 'thetao'
    
    # Setup Metrics
    print("Preparing Truth Metrics...")
    dz_truth = ds_truth['dz']
    # Determine dimensionality for mask
    if 'lev' in ds_truth[VARIABLE_TO_PLOT].dims:
        wet_mask = (ds_truth[VARIABLE_TO_PLOT].isel(time=0) * 0 + 1).compute()
    else:
        # For 2D vars like zos, check for 2D or 3D mask availability
        # Fallback to simple NaN check on the 2D variable itself
        wet_mask = (ds_truth[VARIABLE_TO_PLOT].isel(time=0) * 0 + 1).compute()

    # 1. Compute Truth Maps (Row 1)
    truth_maps = {}
    print("Calculating Truth Temperature Maps...")
    for Slices in DEPTH_LAYERS:
        key = Slices['title']
        truth_maps[key] = calculate_temperature_map_layer(
            ds_truth, VARIABLE_TO_PLOT, dz_truth, wet_mask, Slices['index']
        ).compute()
    # 2. Compute Emulator Maps
    emu_maps = {}
    for label, emu_data in emulators_dict.items():
        print(f"Calculating Temperature Maps for {label}...")
        ds_emu = emu_data['ds']
        emu_maps[label] = {}
        for Slices in DEPTH_LAYERS:
            key = Slices['title']
            # Note: We use Truth's dz and wet_mask to ensure integration domain consistency
            emu_maps[label][key] = calculate_temperature_map_layer(
                ds_emu, VARIABLE_TO_PLOT,  dz_truth, wet_mask, Slices['index']
            ).compute()

    # --- Plotting Setup ---
    datasets = [TRUTH_LABEL] + list(emulators_dict.keys())
    nrows = len(datasets)
    ncols = len(DEPTH_LAYERS)

    print(nrows, ncols,len(datasets))
    
    fig, axes = plt.subplots(
        nrows, ncols, 
        figsize=(4.5 * ncols, 3 * nrows),
        subplot_kw={'projection': ccrs.Robinson(central_longitude=210)},
        constrained_layout=True
    )
    
    # Ensure axes is 2D array even if 1 row
    if nrows == 1: axes = np.expand_dims(axes, 0)
    if ncols == 1: axes = np.expand_dims(axes, 1)

    cmap = cmocean.cm.thermal
    
    # 3. Iterate Column-by-Column (Depth Slice) to lock colorbars
    for j, Slices in enumerate(DEPTH_LAYERS):
        key = Slices['title']
        
        # Determine Color Scale based on TRUTH (or global max)
        data_truth = truth_maps[key].values
        valid_data = data_truth[~np.isnan(data_truth)]
        
        vmin = np.percentile(valid_data, .5)
        vmax = np.percentile(valid_data, 99.5)

        if colorbar_fix:
            if j==0:
                vmin, vmax = 0, 30
            elif j==1:
                vmin, vmax = 0, 15
            elif j==2:
                vmin, vmax = 0, 8
            elif j==3:
                vmin, vmax = 0, 4

        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        # Iterate Row-by-Row (Datasets)
        for i, ds_name in enumerate(datasets):
            ax = axes[i, j]
            
            # Select Data
            if ds_name == TRUTH_LABEL:
                data_to_plot = truth_maps[key]
                row_label = "Ground Truth"
            else:
                # Map back from display label to dictionary key if needed, 
                # but here ds_name is the label
                data_to_plot = emu_maps[ds_name][key]
                row_label = ds_name
            
            # Plot
            p = data_to_plot.plot.pcolormesh(
                ax=ax, transform=ccrs.PlateCarree(),
                x='x', y='y',
                cmap=cmap, norm=norm,
                add_colorbar=False,
                rasterized=True
            )
            
            # Styling
            ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
            ax.coastlines(zorder=11)
            
            # Titles/Labels
            if i == 0:
                ax.set_title(key, fontsize=16, fontweight='bold')
            else:
                ax.set_title("")
                
            if j == 0:
                # Add Row Label on the left
                ax.text(-0.07, 0.55, row_label, va='center', ha='right',
                        rotation='vertical', transform=ax.transAxes, 
                        fontsize=14, fontweight='bold')

        # Add ONE Colorbar per Column (at the bottom of the column)
        # We attach it to the last axes in the column
        cbar = fig.colorbar(
            p, ax=axes[:, j], 
            orientation='horizontal', 
            shrink=0.6, pad=0.03, aspect=25
        )
        cbar.set_label(r"Mean Temperature ($^\circ$C)", fontsize=12)

    fig.suptitle("8-yr-Mean Temperature Comparison", fontsize=20, y=1.05)
    plt.show()


def calculate_zonal_mean(field_mean, mask, dx, wet):
    """
    Simple Zonal Mean using your exact method.
    Numerator: Sum(Field * Mask * dx)
    Denominator: Sum(Mask * Wet_Truth * dx)
    """
    numerator = (field_mean * mask * dx).sum('x')
    denominator = (mask * wet * dx).sum('x')
    return numerator / denominator

def plot_zonal_mean_comparison(ds_truth, emulators_dict,variable_name, units, masks,colorbar_fix=False):
    """
    Generates a grid of OHC maps:
    Rows: Datasets (Truth, Emulator 1, Emulator 2...)
    Cols: Depth Slices (Upper, Mid, Deep)
    """
    
    # Setup Metrics
    print("Preparing Truth Metrics...")
    truth_field_mean = ds_truth[variable_name].mean(dim='time')
    dx_truth = ds_truth['dx']
    # Determine dimensionality for mask
    wet_mask = (ds_truth[variable_name].isel(time=0, drop=True) * 0 + 1).compute()
    
    # 1. Compute Truth Maps (Row 1)
    truth_profiles = {}
    print("Calculating Truth Zonal Profiles...")
    for basin in BASINS_TO_PLOT:
        truth_profiles[basin] = calculate_zonal_mean(
            truth_field_mean, masks[basin], dx_truth, wet_mask
        ).compute()

    # 2. Compute Emulator Maps
    emu_maps = {}
    for iindex, (label, emu_data) in enumerate(emulators_dict.items()):
        print(f"Calculating Temperature Maps for {label}...")
        ds_emu = emu_data['ds']
        emu_field_mean = ds_emu[variable_name].mean('time')
        emu_maps[label] = {}

        print(f"\nGenerating Zonal Bias Plot for {label}...")
        
        # Weights for RMSE (Mean across X)
        weight_2d = (ds_truth['dy'] * ds_truth['dz']).mean('x').compute()

        for basin in BASINS_TO_PLOT:
            # Calculate Emulator Profile using TRUTH metrics
            emu_maps[label][basin] = calculate_zonal_mean(
                emu_field_mean, masks[basin], dx_truth, wet_mask
            ).compute()

        # --- Plotting ---
        plt.style.use('seaborn-v0_8-talk')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True, constrained_layout=True)
        axes = axes.flatten()

        for i, basin in enumerate(BASINS_TO_PLOT):
            data_truth = truth_profiles[basin].values
            valid_data = data_truth[~np.isnan(data_truth)]
            
            if iindex==0 and i==0:
                vmin = np.percentile(valid_data, .5)
                vmax = np.percentile(valid_data, 98)
            
            if colorbar_fix:
                vmin = 0.54
                vmax = 27.25

            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = cmocean.cm.balance

            ax = axes[i]
            p_mesh = emu_maps[label][basin].plot.contourf(
                ax=ax, cmap=cmap, norm=norm, add_colorbar=False, 
                levels=np.linspace(vmin, vmax, 25), yincrease=False,
                x='y', y='lev' 
            )
            ax.set_title(f"{basin} ({units})", fontsize=16, loc='left')
            ax.set_facecolor('grey')
            ax.set_xlabel(''); ax.set_ylabel('')
        
        axes[0].set_ylabel('Depth (m)'); axes[2].set_ylabel('Depth (m)')
        axes[2].set_xlabel('Latitude index'); axes[3].set_xlabel('Latitude index')
        cbar = fig.colorbar(p_mesh, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label(f"Snapshot ({units})")
        fig.suptitle(f"Time-Averaged Zonal-Mean Temperature ({label})", fontsize=20, y=1.03)
        plt.show()

        # --- Plotting ground truth ---
        plt.style.use('seaborn-v0_8-talk')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True, constrained_layout=True)
        axes = axes.flatten()

        for i, basin in enumerate(BASINS_TO_PLOT):
            data_truth = truth_profiles[basin].values
            valid_data = data_truth[~np.isnan(data_truth)]
            
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

            ax = axes[i]
            p_mesh = truth_profiles[basin].plot.contourf(
                ax=ax, cmap=cmap, norm=norm, add_colorbar=False, 
                levels=np.linspace(vmin, vmax, 25), yincrease=False,
                x='y', y='lev' 
            )
            ax.set_title(f"{basin} ({units})", fontsize=16, loc='left')
            ax.set_facecolor('grey')
            ax.set_xlabel(''); ax.set_ylabel('')
        
        axes[0].set_ylabel('Depth (m)'); axes[2].set_ylabel('Depth (m)')
        axes[2].set_xlabel('Latitude index'); axes[3].set_xlabel('Latitude index')
        cbar = fig.colorbar(p_mesh, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label(f"Snapshot ({units})")
        fig.suptitle(f"Time-Averaged Zonal-Mean Temperature (Ground Truth)", fontsize=20, y=1.03)
        plt.show()

def plot_zonal_mean_snapshot_comparison(ds_truth, emulators_dict, variable_name, units, masks, TIME_INDICES,colorbar_fix=False):
    """
    Generates zonal mean plots for snapshots at specific time indices.
    
    Parameters:
    -----------
    ds_truth : xr.Dataset
        Ground truth dataset
    emulators_dict : dict
        Dictionary of emulator datasets with structure: {label: {'ds': xr.Dataset}}
    variable_name : str
        Name of the variable to plot (e.g., 'thetao')
    units : str
        Units for the variable
    masks : dict
        Basin masks dictionary
    TIME_INDICES : list
        List of time indices to plot (e.g., [0, 40, 80])
    """
    
    # Setup Metrics
    print("Preparing Truth Metrics...")
    dx_truth = ds_truth['dx']
    # Determine dimensionality for mask
    wet_mask = (ds_truth[variable_name].isel(time=0, drop=True) * 0 + 1).compute()
    
    # Process each time index
    for time_idx in TIME_INDICES:
        print(f"\nProcessing time index: {time_idx}")
        
        # Format time for title
        try:
            time_val = ds_truth.time[time_idx].values
            if hasattr(time_val, 'strftime'):
                time_str = time_val.strftime("%Y-%m-%d")
            else:
                time_str = str(time_val)[:10]
        except:
            time_str = f"Index {time_idx}"
        
        try:
            t0_val = ds_truth.time.values[0]
            ti_val = ds_truth.time.values[time_idx]
            delta = ti_val - t0_val
            
            if isinstance(delta, np.timedelta64):
                days_elapsed = float(delta) / (1e9 * 86400)
            else:
                days_elapsed = delta.days + delta.seconds / 86400.0
            
            if days_elapsed < 365:
                day_str = f"t = {int(days_elapsed)} days"
            else:
                years = days_elapsed / 365
                day_str = f"t = {years:.1f} years"
        except:
            day_str = f"Index {time_idx}"
        
        # 1. Compute Truth Profiles at this snapshot
        truth_profiles = {}
        print(f"Calculating Truth Zonal Profiles at time index {time_idx}...")
        truth_field_snapshot = ds_truth[variable_name].isel(time=time_idx)
        
        for basin in BASINS_TO_PLOT:
            truth_profiles[basin] = calculate_zonal_mean(
                truth_field_snapshot, masks[basin], dx_truth, wet_mask
            ).compute()
        
        # 2. Compute All Emulator Profiles
        all_emu_maps = {}
        for label, emu_data in emulators_dict.items():
            print(f"Calculating {variable_name} profiles for {label} at time index {time_idx}...")
            ds_emu = emu_data['ds']
            emu_field_snapshot = ds_emu[variable_name].isel(time=time_idx)
            all_emu_maps[label] = {}
            
            for basin in BASINS_TO_PLOT:
                all_emu_maps[label][basin] = calculate_zonal_mean(
                    emu_field_snapshot, masks[basin], dx_truth, wet_mask
                ).compute()
        
        # 3. Plotting: 4 rows (basins) x (1 + N emulators) cols
        num_emulators = len(emulators_dict)
        num_cols = 1 + num_emulators  # Ground Truth + all emulators
        
        plt.style.use('seaborn-v0_8-talk')
        fig, axes = plt.subplots(4, num_cols, figsize=(7 * num_cols, 18), 
                                sharex=True, sharey=True, constrained_layout=True)
        
        # Ensure axes is 2D
        if num_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Determine colorbar range from truth data
        all_truth_data = np.concatenate([truth_profiles[basin].values.flatten() for basin in BASINS_TO_PLOT])
        valid_data = all_truth_data[~np.isnan(all_truth_data)]
        vmin = np.percentile(valid_data, .5)
        vmax = np.percentile(valid_data, 98)

        if colorbar_fix:
            vmin =0
            vmax =25
        
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cmocean.cm.thermal
        
        for i, basin in enumerate(BASINS_TO_PLOT):
            # Column 0: Ground Truth
            ax_truth = axes[i, 0]
            p_mesh = truth_profiles[basin].plot.contourf(
                ax=ax_truth, cmap=cmap, norm=norm, add_colorbar=False, 
                levels=np.linspace(vmin, vmax, 25), yincrease=False,
                x='y', y='lev'
            )
            ax_truth.set_facecolor('grey')
            ax_truth.set_xlabel(''); ax_truth.set_ylabel('')
            
            # Row labels (basin names) on the left
            ax_truth.set_ylabel(f'{basin}\nDepth (m)', fontsize=12)
            
            # Columns 1+: Emulators
            for j, (label, emu_maps) in enumerate(all_emu_maps.items(), start=1):
                ax_emu = axes[i, j]
                emu_maps[basin].plot.contourf(
                    ax=ax_emu, cmap=cmap, norm=norm, add_colorbar=False, 
                    levels=np.linspace(vmin, vmax, 25), yincrease=False,
                    x='y', y='lev'
                )
                ax_emu.set_facecolor('grey')
                ax_emu.set_xlabel(''); ax_emu.set_ylabel('')
        
        # Column titles
        axes[0, 0].set_title('Ground Truth', fontsize=14, fontweight='bold')
        for j, label in enumerate(emulators_dict.keys(), start=1):
            axes[0, j].set_title(label, fontsize=14, fontweight='bold')
        
        # X-axis labels only on bottom row
        for j in range(num_cols):
            axes[3, j].set_xlabel('Latitude index', fontsize=12)
        
        # Colorbar
        cbar = fig.colorbar(p_mesh, ax=axes, orientation='horizontal', shrink=0.6, pad=0.02, aspect=30)
        cbar.set_label(f"{variable_name} ({units})", fontsize=12)
        
        fig.suptitle(f"Snapshot Zonal-Mean {variable_name}: {time_str} ({day_str})", fontsize=18, y=1.03)
        plt.show()


def plot_ohc_zonal_mean_comparison(ds_truth, emulators_dict,variable_name, units, masks,colorbar_fix=False):
    """
    Generates a grid of OHC maps:
    Rows: Datasets (Truth, Emulator 1, Emulator 2...)
    Cols: Depth Slices (Upper, Mid, Deep)
    """
    variable_name = 'thetao'
    units = 'J/m^3'
    
    # Setup Metrics
    print("Preparing Truth Metrics...")
    truth_field_mean = calculate_ohc_2d(ds_truth).mean('time').compute()
    dx_truth = ds_truth['dx']
    # Determine dimensionality for mask
    wet_mask = (ds_truth[variable_name].isel(time=0, drop=True) * 0 + 1).compute()
    
    # 1. Compute Truth Maps (Row 1)
    truth_profiles = {}
    print("Calculating Truth Zonal Profiles...")
    for basin in BASINS_TO_PLOT:
        truth_profiles[basin] = calculate_zonal_mean_2d(
            truth_field_mean, masks[basin], dx_truth, wet_mask
        ).compute()

    # 2. Compute Emulator Maps
    emu_maps = {}
    for iindex, (label, emu_data) in enumerate(emulators_dict.items()):
        print(f"Calculating OHC Maps for {label}...")
        ds_emu = emu_data['ds']
        emu_field_mean = calculate_ohc_2d(ds_emu).mean('time').compute()
        emu_maps[label] = {}

        print(f"\nGenerating Zonal Mean Plot for {label}...")
        
        for basin in BASINS_TO_PLOT:
            # Calculate Emulator Profile using TRUTH metrics
            emu_maps[label][basin] = calculate_zonal_mean_2d(
                emu_field_mean, masks[basin], dx_truth, wet_mask
            ).compute()

        # --- Plotting ---
        plt.style.use('seaborn-v0_8-talk')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True, constrained_layout=True)
        axes = axes.flatten()

        for i, basin in enumerate(BASINS_TO_PLOT):
            data_truth = truth_profiles[basin].values
            valid_data = data_truth[~np.isnan(data_truth)]
            
            if iindex==0 and i==0:
                vmin = np.percentile(valid_data, .5)
                vmax = np.percentile(valid_data, 98)

            if colorbar_fix:
                vmin = 0 * 10**9
                vmax = 9 * 10**9

            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = cmocean.cm.balance

            ax = axes[i]
            p_mesh = emu_maps[label][basin].plot.contourf(
                ax=ax, cmap=cmap, norm=norm, add_colorbar=False, 
                levels=np.linspace(vmin, vmax, 25), yincrease=False,
                x='y', y='lev' 
            )
            ax.set_title(f"{basin} ({units})", fontsize=16, loc='left')
            ax.set_facecolor('grey')
            ax.set_xlabel(''); ax.set_ylabel('')
        
        axes[0].set_ylabel('Depth (m)'); axes[2].set_ylabel('Depth (m)')
        axes[2].set_xlabel('Latitude index'); axes[3].set_xlabel('Latitude index')
        cbar = fig.colorbar(p_mesh, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label(f"Snapshot ({units})")
        fig.suptitle(f"Time-Averaged Zonal-Mean OHC ({label})", fontsize=20, y=1.03)
        plt.show()

        # --- Plotting ground truth ---
        plt.style.use('seaborn-v0_8-talk')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True, constrained_layout=True)
        axes = axes.flatten()

        for i, basin in enumerate(BASINS_TO_PLOT):
            data_truth = truth_profiles[basin].values
            valid_data = data_truth[~np.isnan(data_truth)]
            
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

            ax = axes[i]
            p_mesh = truth_profiles[basin].plot.contourf(
                ax=ax, cmap=cmap, norm=norm, add_colorbar=False, 
                levels=np.linspace(vmin, vmax, 25), yincrease=False,
                x='y', y='lev' 
            )
            ax.set_title(f"{basin} ({units})", fontsize=16, loc='left')
            ax.set_facecolor('grey')
            ax.set_xlabel(''); ax.set_ylabel('')
        
        axes[0].set_ylabel('Depth (m)'); axes[2].set_ylabel('Depth (m)')
        axes[2].set_xlabel('Latitude index'); axes[3].set_xlabel('Latitude index')
        cbar = fig.colorbar(p_mesh, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label(f"Snapshot ({units})")
        fig.suptitle(f"Time-Averaged Zonal-Mean OHC (Ground Truth)", fontsize=20, y=1.03)
        plt.show()


def plot_zonal_snapshot_comparison(ds_truth, emulators_dict,variable_name, units, masks,colorbar_fix=False):
    """
    Generates a grid of OHC maps:
    Rows: Datasets (Truth, Emulator 1, Emulator 2...)
    Cols: Depth Slices (Upper, Mid, Deep)
    """
    target_step = 0
    
    # Setup Metrics
    print("Preparing Truth Metrics...")
    truth_field_mean = ds_truth[variable_name].isel(time=target_step)
    dx_truth = ds_truth['dx']
    # Determine dimensionality for mask
    wet_mask = (ds_truth[variable_name].isel(time=0) * 0 + 1).compute()
    
    # 1. Compute Truth Maps (Row 1)
    truth_profiles = {}
    print("Calculating Truth Zonal Profiles...")
    for basin in BASINS_TO_PLOT:
        truth_profiles[basin] = calculate_zonal_mean(
            truth_field_mean, masks[basin], dx_truth, wet_mask
        ).compute()

    # 2. Compute Emulator Maps
    emu_maps = {}
    for iindex, (label, emu_data) in enumerate(emulators_dict.items()):
        print(f"Calculating Temperature Maps for {label}...")
        ds_emu = emu_data['ds']
        emu_field_mean = ds_emu[variable_name].isel(time=target_step)
        emu_maps[label] = {}

        print(f"\nGenerating Zonal Bias Plot for {label}...")
        
        # Weights for RMSE (Mean across X)
        weight_2d = (ds_truth['dy'] * ds_truth['dz']).mean('x').compute()

        for basin in BASINS_TO_PLOT:
            # Calculate Emulator Profile using TRUTH metrics
            emu_maps[label][basin] = calculate_zonal_mean(
                emu_field_mean, masks[basin], dx_truth, wet_mask
            ).compute()

        # --- Plotting ---
        plt.style.use('seaborn-v0_8-talk')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True, constrained_layout=True)
        axes = axes.flatten()

        for i, basin in enumerate(BASINS_TO_PLOT):
            data_truth = truth_profiles[basin].values
            valid_data = data_truth[~np.isnan(data_truth)]
            
            if iindex==0 and i==0:
                vmin = np.percentile(valid_data, .5)
                vmax = np.percentile(valid_data, 98)
            
            if colorbar_fix:
                vmin = 0.58
                vmax = 28
            
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = cmocean.cm.balance

            ax = axes[i]
            p_mesh = emu_maps[label][basin].plot.contourf(
                ax=ax, cmap=cmap, norm=norm, add_colorbar=False, 
                levels=np.linspace(vmin, vmax, 25), yincrease=False,
                x='y', y='lev' 
            )
            ax.set_title(f"{basin} ({units})", fontsize=16, loc='left')
            ax.set_facecolor('grey')
            ax.set_xlabel(''); ax.set_ylabel('')
        
        axes[0].set_ylabel('Depth (m)'); axes[2].set_ylabel('Depth (m)')
        axes[2].set_xlabel('Latitude index'); axes[3].set_xlabel('Latitude index')
        cbar = fig.colorbar(p_mesh, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label(f"Snapshot ({units})")
        fig.suptitle(f"Snapshots of Zonal Mean Temperature ({label})", fontsize=20, y=1.03)
        plt.show()

        # --- Plotting ground truth ---
        plt.style.use('seaborn-v0_8-talk')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True, constrained_layout=True)
        axes = axes.flatten()

        for i, basin in enumerate(BASINS_TO_PLOT):
            data_truth = truth_profiles[basin].values
            valid_data = data_truth[~np.isnan(data_truth)]
            
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

            ax = axes[i]
            p_mesh = truth_profiles[basin].plot.contourf(
                ax=ax, cmap=cmap, norm=norm, add_colorbar=False, 
                levels=np.linspace(vmin, vmax, 25), yincrease=False,
                x='y', y='lev' 
            )
            ax.set_title(f"{basin} ({units})", fontsize=16, loc='left')
            ax.set_facecolor('grey')
            ax.set_xlabel(''); ax.set_ylabel('')
        
        axes[0].set_ylabel('Depth (m)'); axes[2].set_ylabel('Depth (m)')
        axes[2].set_xlabel('Latitude index'); axes[3].set_xlabel('Latitude index')
        cbar = fig.colorbar(p_mesh, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label(f"Snapshot ({units})")
        fig.suptitle(f"Snapshots of Zonal Mean Temperature (Ground Truth)", fontsize=20, y=1.03)
        plt.show()


def plot_ohc_zonal_snapshot_comparison(ds_truth, emulators_dict, variable_name, units, masks, TIME_INDICES,colorbar_fix=False):
    """
    Generates zonal mean OHC plots for snapshots at specific time indices.
    
    Parameters:
    -----------
    ds_truth : xr.Dataset
        Ground truth dataset
    emulators_dict : dict
        Dictionary of emulator datasets with structure: {label: {'ds': xr.Dataset}}
    variable_name : str
        Name of the variable to plot (typically 'thetao' for OHC calculation)
    units : str
        Units for the variable
    masks : dict
        Basin masks dictionary
    TIME_INDICES : list
        List of time indices to plot (e.g., [0, 40, 80])
    """
    variable_name = 'thetao'
    units = 'J/m^2'
    
    # Setup Metrics
    print("Preparing Truth Metrics...")
    dx_truth = ds_truth['dx']
    # Determine dimensionality for mask
    wet_mask = (ds_truth[variable_name].isel(time=0) * 0 + 1).compute()
    
    # Process each time index
    for time_idx in TIME_INDICES:
        print(f"\nProcessing time index: {time_idx}")
        
        # Format time for title
        try:
            time_val = ds_truth.time[time_idx].values
            if hasattr(time_val, 'strftime'):
                time_str = time_val.strftime("%Y-%m-%d")
            else:
                time_str = str(time_val)[:10]
        except:
            time_str = f"Index {time_idx}"
        
        try:
            t0_val = ds_truth.time.values[0]
            ti_val = ds_truth.time.values[time_idx]
            delta = ti_val - t0_val
            
            if isinstance(delta, np.timedelta64):
                days_elapsed = float(delta) / (1e9 * 86400)
            else:
                days_elapsed = delta.days + delta.seconds / 86400.0
            
            if days_elapsed < 365:
                day_str = f"t = {int(days_elapsed)} days"
            else:
                years = days_elapsed / 365
                day_str = f"t = {years:.1f} years"
        except:
            day_str = f"Index {time_idx}"
        
        # Calculate OHC at this snapshot
        truth_field_snapshot = calculate_ohc_2d(ds_truth).isel(time=time_idx).compute()
        
        # 1. Compute Truth Profiles
        truth_profiles = {}
        print(f"Calculating Truth Zonal Profiles at time index {time_idx}...")
        for basin in BASINS_TO_PLOT:
            truth_profiles[basin] = calculate_zonal_mean_2d(
                truth_field_snapshot, masks[basin], dx_truth, wet_mask
            ).compute()

        # 2. Compute All Emulator Profiles
        all_emu_maps = {}
        for label, emu_data in emulators_dict.items():
            print(f"Calculating OHC profiles for {label} at time index {time_idx}...")
            ds_emu = emu_data['ds']
            emu_field_snapshot = calculate_ohc_2d(ds_emu).isel(time=time_idx).compute()
            all_emu_maps[label] = {}

            for basin in BASINS_TO_PLOT:
                all_emu_maps[label][basin] = calculate_zonal_mean_2d(
                    emu_field_snapshot, masks[basin], dx_truth, wet_mask
                ).compute()

        # 3. Plotting: 4 rows (basins) x (1 + N emulators) cols
        num_emulators = len(emulators_dict)
        num_cols = 1 + num_emulators  # Ground Truth + all emulators
        
        plt.style.use('seaborn-v0_8-talk')
        fig, axes = plt.subplots(4, num_cols, figsize=(7 * num_cols, 18),
                                sharex=True, sharey=True, constrained_layout=True)

        # Ensure axes is 2D
        if num_cols == 1:
            axes = axes.reshape(-1, 1)

        # Determine colorbar range from truth data
        all_truth_data = np.concatenate([truth_profiles[basin].values.flatten() for basin in BASINS_TO_PLOT])
        valid_data = all_truth_data[~np.isnan(all_truth_data)]
        vmin = np.percentile(valid_data, .5)
        vmax = np.percentile(valid_data, 98)

        if colorbar_fix:
            vmin = 0 * 10**9
            vmax = 9 * 10**9
        
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cmocean.cm.balance

        for i, basin in enumerate(BASINS_TO_PLOT):
            # Column 0: Ground Truth
            ax_truth = axes[i, 0]
            p_mesh = truth_profiles[basin].plot.contourf(
                ax=ax_truth, cmap=cmap, norm=norm, add_colorbar=False, 
                levels=np.linspace(vmin, vmax, 25), yincrease=False,
                x='y', y='lev'
            )
            ax_truth.set_facecolor('grey')
            ax_truth.set_xlabel(''); ax_truth.set_ylabel('')
            
            # Row labels (basin names) on the left
            ax_truth.set_ylabel(f'{basin}\nDepth (m)', fontsize=12)
            
            # Columns 1+: Emulators
            for j, (label, emu_maps) in enumerate(all_emu_maps.items(), start=1):
                ax_emu = axes[i, j]
                emu_maps[basin].plot.contourf(
                    ax=ax_emu, cmap=cmap, norm=norm, add_colorbar=False, 
                    levels=np.linspace(vmin, vmax, 25), yincrease=False,
                    x='y', y='lev'
                )
                ax_emu.set_facecolor('grey')
                ax_emu.set_xlabel(''); ax_emu.set_ylabel('')
        
        # Column titles
        axes[0, 0].set_title('Ground Truth', fontsize=14, fontweight='bold')
        for j, label in enumerate(emulators_dict.keys(), start=1):
            axes[0, j].set_title(label, fontsize=14, fontweight='bold')
        
        # X-axis labels only on bottom row
        for j in range(num_cols):
            axes[3, j].set_xlabel('Latitude index', fontsize=12)
        
        # Colorbar
        cbar = fig.colorbar(p_mesh, ax=axes, orientation='horizontal', shrink=0.6, pad=0.02, aspect=30)
        cbar.set_label(f"OHC ({units})", fontsize=12)
        
        fig.suptitle(f"Snapshot Zonal-Mean OHC: {time_str} ({day_str})", fontsize=18, y=1.03)
        plt.show()

def plot_ohc_bias_comparison(ds_truth, emulators_dict,colorbar_fix=False):
    """
    Generates a grid of OHC maps:
    Rows: Datasets (Truth, Emulator 1, Emulator 2...)
    Cols: Depth Slices (Upper, Mid, Deep)
    """

    units = 'ZJ'
    
    # Setup Metrics
    print("Preparing Truth Metrics...")
    dz_truth = ds_truth['dz']
    # Wet mask from Truth (1 for Ocean, NaN for Land)
    wet_mask = (ds_truth['thetao'].isel(time=0) * 0 + 1).compute()
    area = ds_truth['areacello']
    
    # 1. Compute Truth Maps (Row 1)
    truth_maps = {}
    print("Calculating Truth OHC Maps...")
    for Slices in DEPTH_SLICES:
        key = Slices['title']
        truth_maps[key] = calculate_ohc_map(
            ds_truth, dz_truth, wet_mask, Slices['min'], Slices['max']
        ).compute()

    fixed_limits = {
        'Surface (0-700m)': 2.5*10**(-12),
        'Intermediate (700-2000m)': 2.5*10**(-12),
        'Deep (2000-7000m)': 1*10**(-12)
    }
        
    # 2. Compute Emulator Maps
    emu_maps = {}
    shared_limits = {}
    for idx, (label, emu_data) in enumerate(emulators_dict.items()):
        print(f"Calculating OHC Maps for {label}...")
        ds_emu = emu_data['ds']
        emu_maps[label] = {}
        bias_maps = {}
        rmse_vals = {}
        all_biases_flat = []

        for Slices in DEPTH_SLICES:
            key = Slices['title']
            # Note: We use Truth's dz and wet_mask to ensure integration domain consistency
            emu_maps[label][key] = calculate_ohc_map(
                ds_emu, dz_truth, wet_mask, Slices['min'], Slices['max']
            ).compute()

            map_truth = truth_maps[key]
            map_emu = emu_maps[label][key]

            bias = map_emu - map_truth
            bias_maps[key] = bias

            rmse_vals[key] = rmse_spatial(map_truth, map_emu, area)    

            all_biases_flat.append(bias.values.flatten())

        # Dynamic Color Scaling
        flat_data = np.concatenate(all_biases_flat)
        flat_data = flat_data[~np.isnan(flat_data)]
        
        if len(flat_data) > 0:
            max_abs_bias = np.percentile(np.abs(flat_data), 99.5)
            print(f"  > Bias Range (98th percentile): +/- {max_abs_bias:.3f} {units}")
        else:
            max_abs_bias = 1.0

        if idx==0:
            vmin, vmax = -max_abs_bias, max_abs_bias

        if colorbar_fix:
            vmin = -0.02
            vmax = 0.02
        
        #vmin, vmax = -0.6, 0.6
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cmocean.cm.balance

        # --- Plotting Setup ---
        num_slices = len(DEPTH_SLICES)
        
        fig, axes = plt.subplots(
            1, num_slices, 
            figsize=(6 * num_slices, 5),
            subplot_kw={'projection': ccrs.Robinson(central_longitude=210)},
            constrained_layout=True
        )
        
        # Ensure axes is 2D array even if 1 row
        if num_slices == 1: axes = [axes]

        cmap = cmocean.cm.balance

        for i, Slices in enumerate(DEPTH_SLICES):
            key = Slices['title']
            ax = axes[i]
            bias = bias_maps[key]
        
            # Determine Color Scaling (Lock to First Emulator)
            if idx == 0:
                flat_bias = bias.values.flatten()
                flat_bias = flat_bias[~np.isnan(flat_bias)]
                
                if len(flat_bias) > 0:
                    local_max = np.percentile(np.abs(flat_bias), 99.5)
                else:
                    local_max = 1.0

                print('local_max: ', local_max)
                
                shared_limits[key] = local_max
                print(f"  > {key} Scale locked to: +/- {local_max:.3e} {units}")
            
            limit = shared_limits[key]

            if colorbar_fix:
                limit = fixed_limits[key]

            norm = colors.Normalize(vmin=-limit, vmax=limit)
            
            # Plot with INDIVIDUAL colorbar
            p = bias.plot.pcolormesh(
                ax=ax, transform=ccrs.PlateCarree(),
                x='x', y='y',
                cmap=cmap, norm=norm, 
                add_colorbar=True,
                cbar_kwargs={
                    'orientation': 'horizontal', 
                    'shrink': 0.8, 
                    'pad': 0.08, 
                    'label': f"Bias ({units})"
                },
                rasterized=True
            )
            
            ax.set_title(f"{key}\nRMSE: {rmse_vals[key]:.2e} {units}" + r"", fontsize=14)
            ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
            ax.coastlines(zorder=11)
            ax.gridlines(draw_labels=False, alpha=0.3)

        fig.suptitle(f"8-yr-Mean OHC Bias: {label} vs Truth", fontsize=18, y=1.05)
        plt.show()


def plot_ohc_snapshot_bias_comparison(ds_truth, emulators_dict, time_indices, colorbar_fix=False,area=True):
    """
    Generates OHC snapshot bias comparison plots for each time point.
    For each time point, creates a grid comparing all emulators:
    Rows: Emulators
    Cols: Depth Slices (Surface, Intermediate, Deep)
    Each depth slice has its own colorbar for better visibility.
    
    Args:
        ds_truth: Truth dataset (must have 'time' coordinate)
        emulators_dict: Dictionary of emulator datasets
        time_indices: List of time indices to plot (e.g., [0, 4, 8])
        colorbar_fix: If True, uses fixed colorbar limits
    """
    
    units = 'ZJ'
    
    # Setup Metrics
    print("Preparing Truth Metrics...")
    dz_truth = ds_truth['dz']
    # Wet mask from Truth (1 for Ocean, NaN for Land)
    wet_mask = (ds_truth['thetao'].isel(time=0) * 0 + 1).compute()
    area = ds_truth['areacello']
    
    # Get emulator labels
    emulator_labels = list(emulators_dict.keys())
    num_emulators = len(emulator_labels)
    num_depth_slices = len(DEPTH_SLICES)
    
    # Create one figure for each time point
    for time_idx in time_indices:
        # Get the actual time value from the dataset
        time_value = ds_truth['time'].isel(time=time_idx).values
        
        # Format time string based on type
        if isinstance(time_value, (cftime.datetime, cftime.DatetimeNoLeap, cftime.DatetimeGregorian)):
            time_str = time_value.strftime('%Y-%m-%d')
        elif isinstance(time_value, np.datetime64):
            time_str = str(time_value)[:10]  # Get YYYY-MM-DD part
        else:
            time_str = str(time_value)
        
        print(f"\nCreating plot for time point: {time_str}")
        
        # Calculate bias maps for all emulators at this time point
        all_bias_maps = {}  # key: (emulator_label, depth_slice_title)
        all_rmse_vals = {}  # key: (emulator_label, depth_slice_title)
        shared_limits = {}  # key: depth_slice_title
        
        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            print(f"  Calculating OHC Snapshot Maps for {label}...")
            ds_emu = emu_data['ds']
            
            for slice_index, Slices in enumerate(DEPTH_SLICES):
                key = Slices['title']
                
                # Calculate OHC for truth at this time index
                truth_map = calculate_ohc_snapshot_map(
                    ds_truth, dz_truth, wet_mask, Slices['min'], Slices['max'], time_idx
                ).compute()
                
                # Calculate OHC for emulator at this time index
                emu_map = calculate_ohc_snapshot_map(
                    ds_emu, dz_truth, wet_mask, Slices['min'], Slices['max'], time_idx
                ).compute()
                
                # Calculate bias
                bias = emu_map - truth_map
                all_bias_maps[(label, key)] = bias
                
                # Calculate RMSE
                all_rmse_vals[(label, key)] = rmse_spatial(truth_map, emu_map, area)
                
                # Calculate color limits for this depth slice (on first emulator)
                if idx == 0:
                    flat_bias = bias.values.flatten()
                    flat_bias = flat_bias[~np.isnan(flat_bias)]
                    
                    if len(flat_bias) > 0:
                        local_max = np.percentile(np.abs(flat_bias), 99.5)
                    else:
                        local_max = 1.0
                    
                    shared_limits[key] = local_max
                    print(f"    > {key} Scale locked to: +/- {local_max:.3e} {units}")

                
        cmap = cmocean.cm.balance
        
        # --- Plotting Setup ---
        fig, axes = plt.subplots(
            num_emulators, num_depth_slices,
            figsize=(6 * num_depth_slices, 5 * num_emulators),
            subplot_kw={'projection': ccrs.Robinson(central_longitude=210)},
            constrained_layout=True
        )
        
        # Ensure axes is 2D array
        if num_emulators == 1 and num_depth_slices == 1:
            axes = np.array([[axes]])
        elif num_emulators == 1:
            axes = axes.reshape(1, -1)
        elif num_depth_slices == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each emulator (rows) and depth slice (columns)
        for i, label in enumerate(emulator_labels):
            for j, Slices in enumerate(DEPTH_SLICES):
                key = Slices['title']
                ax = axes[i, j]
                bias = all_bias_maps[(label, key)]
                rmse = all_rmse_vals[(label, key)]
                
                # Use independent color scale for each depth layer
                if colorbar_fix:
                    if j == 0:
                        vmin_global = -6*10**(-12)
                        vmax_global = 6*10**(-12)
                    elif j == 1:
                        vmin_global = -4*10**(-12)
                        vmax_global = 4*10**(-12)
                    elif j == 2:
                        vmin_global = -2*10**(-12)
                        vmax_global = 2*10**(-12)
                    norm = colors.Normalize(vmin=vmin_global, vmax=vmax_global)
                else:
                    limit = shared_limits[key]
                    norm = colors.Normalize(vmin=-limit, vmax=limit)
                
                # Plot with individual horizontal colorbar below each map
                p = bias.plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    x='x', y='y',
                    cmap=cmap, norm=norm,
                    add_colorbar=True,
                    cbar_kwargs={
                        'orientation': 'horizontal',
                        'shrink': 0.8,
                        'pad': 0.08,
                        'label': f"Bias ({units})"
                    },
                    rasterized=True
                )
                
                # Title with depth slice and RMSE
                title = f"{key}\nRMSE: {rmse:.2e} {units}"
                ax.set_title(title, fontsize=14)
                
                # Add map features
                ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
                ax.coastlines(zorder=11)
                ax.gridlines(draw_labels=False, alpha=0.3)
                
                # Add emulator label on the leftmost column
                if j == 0:
                    ax.text(-0.07, 0.5, label, va='center', ha='right',
                           rotation='vertical', transform=ax.transAxes,
                           fontsize=14, fontweight='bold')
        
        fig.suptitle(f"OHC Snapshot Bias at {time_str} (vs Truth)", fontsize=18, y=1.02)
        plt.show()


def plot_temperature_bias_comparison(ds_truth, emulators_dict,colorbar_fix=False):

    VARIABLE_TO_PLOT = 'thetao'  # Options: 'thetao', 'so', 'zos', 'uo', 'vo'
    units = '$^\circ$C'

    # Setup Metrics
    print("Preparing Truth Metrics...")
    dz_truth = ds_truth['dz']
    # Determine dimensionality for mask
    if 'lev' in ds_truth[VARIABLE_TO_PLOT].dims:
        wet_mask = (ds_truth[VARIABLE_TO_PLOT].isel(time=0) * 0 + 1).compute()
    else:
        # For 2D vars like zos, check for 2D or 3D mask availability
        # Fallback to simple NaN check on the 2D variable itself
        wet_mask = (ds_truth[VARIABLE_TO_PLOT].isel(time=0) * 0 + 1).compute()

    # 1. Compute Truth Maps (Row 1)
    truth_maps = {}
    area = ds_truth['areacello']
    print("Calculating Truth Temperature Maps...")
    for Slices in DEPTH_LAYERS:
        key = Slices['title']
        truth_maps[key] = calculate_temperature_map_layer(
            ds_truth, VARIABLE_TO_PLOT, dz_truth, wet_mask, Slices['index']
        ).compute()
    # 2. Compute Emulator Maps
    emu_maps = {}
    shared_limits = {}
    for idx, (label, emu_data) in enumerate(emulators_dict.items()):
        print(f"Calculating Temperature Maps for {label}...")
        bias_maps = {}
        rmse_vals = {}
        all_biases_flat  = []
        ds_emu = emu_data['ds']
        emu_maps[label] = {}
        for Slices in DEPTH_LAYERS:
            key = Slices['title']
            # Note: We use Truth's dz and wet_mask to ensure integration domain consistency
            emu_maps[label][key] = calculate_temperature_map_layer(
                ds_emu, VARIABLE_TO_PLOT,  dz_truth, wet_mask, Slices['index']
            ).compute()
            
            map_truth = truth_maps[key]
            
            map_emu = emu_maps[label][key]

            bias_maps[key] = map_emu - map_truth 
            rmse_vals[key] = rmse_spatial(map_emu, map_truth, area)

            all_biases_flat.append(bias_maps[key].values.flatten())

        # Dynamic Color Scaling
        flat_data = np.concatenate(all_biases_flat)
        flat_data = flat_data[~np.isnan(flat_data)]
        
        if len(flat_data) > 0:
            max_abs_bias = np.percentile(np.abs(flat_data), 99.5)
            print(f"  > Bias Range (98th percentile): +/- {max_abs_bias:.3f} {units}")
        else:
            max_abs_bias = 1.0

        vmin, vmax = -max_abs_bias, max_abs_bias
        #vmin, vmax = -0.6, 0.6
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cmocean.cm.balance


        # --- Plotting ---
        num_slices = len(DEPTH_LAYERS)
        fig, axes = plt.subplots(
            1, num_slices, 
            figsize=(6 * num_slices, 5),
            subplot_kw={'projection': ccrs.Robinson(central_longitude=210)},
            constrained_layout=True
        )
        if num_slices == 1: axes = [axes]
        
        cmap = cmocean.cm.balance
        
        for i, Slices in enumerate(DEPTH_LAYERS):
            key = Slices['title']
            ax = axes[i]
            bias = bias_maps[key]
            
            # Determine Color Scaling (Lock to First Emulator)
            if idx == 0:
                flat_bias = bias.values.flatten()
                flat_bias = flat_bias[~np.isnan(flat_bias)]
                
                if len(flat_bias) > 0:
                    local_max = np.percentile(np.abs(flat_bias), 99)
                else:
                    local_max = 1.0
                
                shared_limits[key] = local_max
                print(f"  > {key} Scale locked to: +/- {local_max:.3e} {units}")
            
            limit = shared_limits[key]

            if colorbar_fix:
                if i==0:
                    limit = 1.2
                elif i==1:
                    limit = 0.7
                elif i==2:
                    limit = 0.18
                elif i==3:
                    limit = 0.1

            norm = colors.Normalize(vmin=-limit, vmax=limit)
            
            # Plot with INDIVIDUAL colorbar
            p = bias.plot.pcolormesh(
                ax=ax, transform=ccrs.PlateCarree(),
                x='x', y='y',
                cmap=cmap, norm=norm, 
                add_colorbar=True,
                cbar_kwargs={
                    'orientation': 'horizontal', 
                    'shrink': 0.8, 
                    'pad': 0.08, 
                    'label': f"Bias ({units})"
                },
                rasterized=True
            )
            
            ax.set_title(f"{key}\nRMSE: {rmse_vals[key]:.2e} {units}" + r"", fontsize=14)
            ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
            ax.coastlines(zorder=11)
            ax.gridlines(draw_labels=False, alpha=0.3)
        
        fig.suptitle(f"8-yr-Mean Temperature Bias: {label} vs Truth", fontsize=18, y=1.05)
    plt.show()


def plot_temperature_snapshot_bias_comparison(ds_truth, emulators_dict, time_indices, colorbar_fix=False):
    """
    Generates temperature snapshot bias comparison plots for each time point.
    For each time point, creates a grid comparing all emulators:
    Rows: Emulators
    Cols: Depth Layers (2.5m, 700m, 2000m, 3000m)
    Each depth layer has its own colorbar for better visibility.
    
    Args:
        ds_truth: Truth dataset (must have 'time' coordinate)
        emulators_dict: Dictionary of emulator datasets
        time_indices: List of time indices to plot (e.g., [0, 4, 8])
        colorbar_fix: If True, uses fixed colorbar limits per depth layer
    """
    
    VARIABLE_TO_PLOT = 'thetao'
    units = '$^\\circ$C'
    
    # Setup Metrics
    print("Preparing Truth Metrics...")
    dz_truth = ds_truth['dz']
    # Determine dimensionality for mask
    if 'lev' in ds_truth[VARIABLE_TO_PLOT].dims:
        wet_mask = (ds_truth[VARIABLE_TO_PLOT].isel(time=0) * 0 + 1).compute()
    else:
        wet_mask = (ds_truth[VARIABLE_TO_PLOT].isel(time=0) * 0 + 1).compute()
    
    area = ds_truth['areacello']
    
    # Get emulator labels
    emulator_labels = list(emulators_dict.keys())
    num_emulators = len(emulator_labels)
    num_depth_layers = len(DEPTH_LAYERS)
    
    # Fixed colorbar limits per depth layer (optional)
    fixed_limits = {
        '2.5m':2.5,
        '700m': 1.5,
        '2000m': 0.3,
        '3000m': 0.2
    }
    
    cmap = cmocean.cm.balance
    
    # Create one figure for each time point
    for time_idx in time_indices:
        # Get the actual time value from the dataset
        time_value = ds_truth['time'].isel(time=time_idx).values
        
        # Format time string based on type
        if isinstance(time_value, (cftime.datetime, cftime.DatetimeNoLeap, cftime.DatetimeGregorian)):
            time_str = time_value.strftime('%Y-%m-%d')
        elif isinstance(time_value, np.datetime64):
            time_str = str(time_value)[:10]  # Get YYYY-MM-DD part
        else:
            time_str = str(time_value)
        
        print(f"\nCreating plot for time point: {time_str}")
        
        # Calculate bias maps for all emulators at this time point
        all_bias_maps = {}  # key: (emulator_label, depth_layer_title)
        all_rmse_vals = {}  # key: (emulator_label, depth_layer_title)
        shared_limits = {}  # key: depth_layer_title
        
        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            print(f"  Calculating Temperature Snapshot Maps for {label}...")
            ds_emu = emu_data['ds']
            
            for layer in DEPTH_LAYERS:
                key = layer['title']
                depth_index = layer['index']
                
                # Calculate temperature for truth at this time index
                truth_map = calculate_snapshot_map_SpecificLayer(
                    ds_truth, VARIABLE_TO_PLOT, dz_truth, wet_mask, depth_index, time_idx
                ).compute()
                
                # Calculate temperature for emulator at this time index
                emu_map = calculate_snapshot_map_SpecificLayer(
                    ds_emu, VARIABLE_TO_PLOT, dz_truth, wet_mask, depth_index, time_idx
                ).compute()
                
                # Calculate bias
                bias = emu_map - truth_map
                all_bias_maps[(label, key)] = bias
                
                # Calculate RMSE
                all_rmse_vals[(label, key)] = rmse_spatial(truth_map, emu_map, area)
                
                # Calculate color limits for this depth layer (on first emulator)
                if idx == 0:
                    flat_bias = bias.values.flatten()
                    flat_bias = flat_bias[~np.isnan(flat_bias)]
                    
                    if len(flat_bias) > 0:
                        local_max = np.percentile(np.abs(flat_bias), 99)
                    else:
                        local_max = 1.0
                    
                    shared_limits[key] = local_max
                    print(f"    > {key} Scale locked to: +/- {local_max:.3e} {units}")

                if colorbar_fix:
                    shared_limits[key] = fixed_limits[key]
        
        # --- Plotting Setup ---
        fig, axes = plt.subplots(
            num_emulators, num_depth_layers,
            figsize=(6 * num_depth_layers, 5 * num_emulators),
            subplot_kw={'projection': ccrs.Robinson(central_longitude=210)},
            constrained_layout=True
        )
        
        # Ensure axes is 2D array
        if num_emulators == 1 and num_depth_layers == 1:
            axes = np.array([[axes]])
        elif num_emulators == 1:
            axes = axes.reshape(1, -1)
        elif num_depth_layers == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each emulator (rows) and depth layer (columns)
        for i, label in enumerate(emulator_labels):
            for j, layer in enumerate(DEPTH_LAYERS):
                key = layer['title']
                ax = axes[i, j]
                bias = all_bias_maps[(label, key)]
                rmse = all_rmse_vals[(label, key)]
                
                # Use independent color scale for each depth layer
                if colorbar_fix and key in fixed_limits:
                    limit = fixed_limits[key]
                    norm = colors.Normalize(vmin=-limit, vmax=limit)
                else:
                    limit = shared_limits[key]
                    norm = colors.Normalize(vmin=-limit, vmax=limit)
                
                # Plot with individual horizontal colorbar below each map
                p = bias.plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    x='x', y='y',
                    cmap=cmap, norm=norm,
                    add_colorbar=True,
                    cbar_kwargs={
                        'orientation': 'horizontal',
                        'shrink': 0.8,
                        'pad': 0.08,
                        'label': f"Bias ({units})"
                    },
                    rasterized=True
                )
                
                # Title with depth layer and RMSE
                title = f"{key}\nRMSE: {rmse:.2e} {units}"
                ax.set_title(title, fontsize=14)
                
                # Add map features
                ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
                ax.coastlines(zorder=11)
                ax.gridlines(draw_labels=False, alpha=0.3)
                
                # Add emulator label on the leftmost column
                if j == 0:
                    ax.text(-0.07, 0.5, label, va='center', ha='right',
                           rotation='vertical', transform=ax.transAxes,
                           fontsize=14, fontweight='bold')
        
        fig.suptitle(f"Temperature Snapshot Bias at {time_str} (vs Truth)", fontsize=18, y=1.02)
        plt.show()


def calculate_snapshot_map_SpecificLayer(dataset, var_name, dz, wet_mask, depth_index, time_idx):
    """
    Computes a 2D map for a specific variable at a specific time index.
    - 3D vars: Computes depth-weighted average.
    - 2D vars: Returns surface field masked by Truth.
    """
    # Select Time Step
    ds_t = dataset.isel(time=time_idx)
    field = ds_t[var_name]
    
    # Select Depth Slice
    f_slice = field.isel(lev=depth_index)
    dz_slice = dz.isel(lev=depth_index)
    wet_slice = wet_mask.isel(lev=depth_index)
    
    # Depth Average
    numerator = (f_slice * dz_slice * wet_slice)
    denominator = (dz_slice * wet_slice)
    result = numerator / denominator
        
    return result.compute()

def plot_snapshot_comparison_SpecificLayer(ds_truth, emulators_dict, var_name, time_indices, depth_config,TRUTH_LABEL='Truth',fix_colorbar=False):
    """
    Generates a grid of snapshot maps:
    Rows: Datasets
    Cols: Time Indices
    """
    config = VAR_CONFIG.get(var_name, {'cmap': 'viridis', 'units': '', 'label': var_name, 'centered': False})
    units = config['units']
    
    print(f"Preparing Snapshot Maps for {config['label']} ({depth_config['title']})...")
    
    # Setup Metrics from Truth
    dz_truth = ds_truth['dz']
    
    # Determine dimensionality for mask
    if 'lev' in ds_truth[var_name].dims:
        wet_mask = (ds_truth[var_name].isel(time=0) * 0 + 1).compute()
    else:
        # For 2D vars like zos, check for 2D or 3D mask availability
        # Fallback to simple NaN check on the 2D variable itself
        wet_mask = (ds_truth[var_name].isel(time=0) * 0 + 1).compute()

    # Pre-Compute Maps
    # Structure: maps[dataset_label][time_index]
    maps_data = {}
    datasets = [TRUTH_LABEL] + list(emulators_dict.keys())
    
    # 1. Truth
    maps_data[TRUTH_LABEL] = {}
    for t in time_indices:
        maps_data[TRUTH_LABEL][t] = calculate_snapshot_map_SpecificLayer(
            ds_truth, var_name, dz_truth, wet_mask, 
            depth_config['index'], t
        )
        
    # 2. Emulators
    for label, emu_data in emulators_dict.items():
        ds_emu = emu_data['ds']
        maps_data[label] = {}
        for t in time_indices:
            # Note: Using Truth's dz and wet_mask for consistency
            maps_data[label][t] = calculate_snapshot_map_SpecificLayer(
                ds_emu, var_name, dz_truth, wet_mask, 
                depth_config['index'], t
            )

    # --- Plotting ---
    nrows = len(datasets)
    ncols = len(time_indices)
    
    fig, axes = plt.subplots(
        nrows, ncols, 
        figsize=(6 * ncols, 4 * nrows),
        subplot_kw={'projection': ccrs.Robinson(central_longitude=210)},
        constrained_layout=True
    )
    
    if nrows == 1: axes = np.expand_dims(axes, 0)
    if ncols == 1: axes = np.expand_dims(axes, 1)

    cmap = config['cmap']
    
    # Iterate Column-by-Column (Time Steps) to lock colorbars
    for j, t_idx in enumerate(time_indices):
        
        # Determine Range based on TRUTH at this timestep
        truth_slice = maps_data[TRUTH_LABEL][t_idx].values
        valid_data = truth_slice[~np.isnan(truth_slice)]
        
        if len(valid_data) > 0:
            if config['centered']:
                # Centered Colorbar (e.g. Velocity, SSH anomaly)
                abs_max = np.percentile(np.abs(valid_data), 99.75)
                vmin, vmax = -abs_max, abs_max
            else:
                # Sequential Colorbar (e.g. Temperature)
                vmin = np.percentile(valid_data, .25)
                vmax = np.percentile(valid_data, 99.75)
        else:
            vmin, vmax = 0, 1

        if fix_colorbar:
            if depth_config['index'] == 0:
                vmin, vmax = -2, 32
            elif depth_config['index'] == 10:
                vmin, vmax = -2, 15
            elif depth_config['index'] == 14:
                vmin, vmax = -2, 10
            elif depth_config['index'] == 15:
                vmin, vmax = -2, 10
            
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        # Retrieve actual time string for title
        try:
            time_val = ds_truth.time[t_idx].values
            # Format nicely if it's datetime/cftime
            if hasattr(time_val, 'strftime'):
                time_str = time_val.strftime("%Y-%m-%d")
            else:
                time_str = str(time_val)[:10]
        except:
            time_str = f"Index {t_idx}"


        try:
            # Extract raw values to handle both numpy and cftime objects
            t0_val = ds_truth.time.values[0]
            ti_val = ds_truth.time.values[t_idx]
            
            delta = ti_val - t0_val
            
            # Convert to days
            if isinstance(delta, np.timedelta64):
                # Convert from nanoseconds (default for xarray/numpy) to days
                days_elapsed = float(delta) / (1e9 * 86400)
            else:
                # Assume datetime.timedelta (cftime)
                days_elapsed = delta.days + delta.seconds / 86400.0
            
            if days_elapsed < 365:
                day_str = f"t = {int(days_elapsed)} days"
            else:
                years = days_elapsed / 365
                day_str = f"t = {years:.1f} years"
        except:
            day_str = f"Index {t_idx}"

        
        # Iterate Row-by-Row (Datasets)
        for i, ds_name in enumerate(datasets):
            ax = axes[i, j]
            data_to_plot = maps_data[ds_name][t_idx]
            
            p = data_to_plot.plot.pcolormesh(
                ax=ax, transform=ccrs.PlateCarree(),
                x='x', y='y',
                cmap=cmap, norm=norm,
                add_colorbar=False,
                rasterized=True
            )
            
            ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
            ax.coastlines(zorder=11)
            
            # Titles
            if i == 0:
                ax.set_title(f"Time: {time_str} ({day_str})", fontsize=16, fontweight='bold')
            else:
                ax.set_title("")
                
            # Row Labels
            if j == 0:
                label_text = "Ground Truth" if ds_name == TRUTH_LABEL else ds_name
                ax.text(-0.07, 0.55, label_text, va='center', ha='right',
                        rotation='vertical', transform=ax.transAxes, 
                        fontsize=14, fontweight='bold')

        # Add Colorbar for this Time Step
        cbar = fig.colorbar(
            p, ax=axes[:, j], 
            orientation='horizontal', 
            shrink=0.6, pad=0.03, aspect=25
        )
        cbar.set_label(f"{config['label']} ({units})", fontsize=12)

    main_title = f"{config['label']} Snapshot"
    if 'lev' in ds_truth[var_name].dims:
        main_title += f" ({depth_config['title']})"
        
    fig.suptitle(main_title, fontsize=20, y=1.05)
    plt.show()

def calculate_ohc_snapshot_map(dataset, dz, wet_mask, min_depth, max_depth, time_index):
    """
    Calculates the Time-Mean OHC map (Vertically Integrated).
    Formula: Sum(rho * cp * thetao * dz) / Scale
    Units: GJ/m^2
    """
    # Select slices
    thetao_slice = dataset['thetao'].sel(lev=slice(min_depth, max_depth))
    dz_slice = dz.sel(lev=slice(min_depth, max_depth))
    wet_slice = wet_mask.sel(lev=slice(min_depth, max_depth))
    
    # 1. Integrate Vertically (Sum over lev)
    # Result is (time, y, x)
    #ohc_integrated = (RHO * CP * thetao_slice * dz_slice *dataset['areacello']* wet_slice).sum('lev')

    ohc_integrated = (RHO * CP * thetao_slice * dz_slice * wet_slice).sum('lev')

    # 2. Time snapshot
    ohc_mean = ohc_integrated.isel(time=time_index)
    
    # 3. Scale
    return ohc_mean / SCALE_FACTOR



def plot_ohc_snapshot_comparison(ds_truth, emulators_dict, var_name, time_indices, depth_config,TRUTH_LABEL='Truth',colorbar_fix=False):
    """
    Generates a grid of snapshot maps:
    Rows: Datasets
    Cols: Time Indices
    """
    config = VAR_CONFIG.get(var_name, {'cmap': 'viridis', 'units': '', 'label': var_name, 'centered': False})
    units = config['units']
    
    print(f"Preparing Snapshot Maps for OHC ({depth_config['title']})...")
    
    # Setup Metrics from Truth
    dz_truth = ds_truth['dz']
    
    # Determine dimensionality for mask
    if 'lev' in ds_truth[var_name].dims:
        wet_mask = (ds_truth[var_name].isel(time=0) * 0 + 1).compute()
    else:
        # For 2D vars like zos, check for 2D or 3D mask availability
        # Fallback to simple NaN check on the 2D variable itself
        wet_mask = (ds_truth[var_name].isel(time=0) * 0 + 1).compute()

    # Pre-Compute Maps
    # Structure: maps[dataset_label][time_index]
    maps_data = {}
    datasets = [TRUTH_LABEL] + list(emulators_dict.keys())
    
    # 1. Truth
    maps_data[TRUTH_LABEL] = {}
    for t in time_indices:
        maps_data[TRUTH_LABEL][t] = calculate_ohc_snapshot_map(
            ds_truth, dz_truth, wet_mask, 
            depth_config['min'], depth_config['max'], t
        )
        
    # 2. Emulators
    for label, emu_data in emulators_dict.items():
        ds_emu = emu_data['ds']
        maps_data[label] = {}
        for t in time_indices:
            # Note: Using Truth's dz and wet_mask for consistency
            maps_data[label][t] = calculate_ohc_snapshot_map(
                ds_emu, dz_truth, wet_mask, 
                depth_config['min'], depth_config['max'], t
            )

    # --- Plotting ---
    nrows = len(datasets)
    ncols = len(time_indices)
    
    fig, axes = plt.subplots(
        nrows, ncols, 
        figsize=(6 * ncols, 4 * nrows),
        subplot_kw={'projection': ccrs.Robinson(central_longitude=210)},
        constrained_layout=True
    )
    
    if nrows == 1: axes = np.expand_dims(axes, 0)
    if ncols == 1: axes = np.expand_dims(axes, 1)

    cmap = config['cmap']
    
    # Iterate Column-by-Column (Time Steps) to lock colorbars
    for j, t_idx in enumerate(time_indices):
        
        # Determine Range based on TRUTH at this timestep
        truth_slice = maps_data[TRUTH_LABEL][t_idx].values
        valid_data = truth_slice[~np.isnan(truth_slice)]
        
        if len(valid_data) > 0:
            if config['centered']:
                # Centered Colorbar (e.g. Velocity, SSH anomaly)
                abs_max = np.percentile(np.abs(valid_data), 99.75)
                vmin, vmax = -abs_max, abs_max
            else:
                # Sequential Colorbar (e.g. Temperature)
                vmin = np.percentile(valid_data, .25)
                vmax = np.percentile(valid_data, 99.75)
        else:
            vmin, vmax = 0, 1

        if colorbar_fix:
            if depth_config['title']=='Surface (0-700m)':
                vmin, vmax = 0, 5*10**(-11)
            elif depth_config['title']=='Intermediate (700-2000m)':
                vmin, vmax = 0, 5*10**(-11)
            elif depth_config['title']=='Deep (2000-7000m)':
                vmin, vmax = 0, 5*10**(-11)

            
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        # Retrieve actual time string for title
        try:
            time_val = ds_truth.time[t_idx].values
            # Format nicely if it's datetime/cftime
            if hasattr(time_val, 'strftime'):
                time_str = time_val.strftime("%Y-%m-%d")
            else:
                time_str = str(time_val)[:10]
        except:
            time_str = f"Index {t_idx}"


        try:
            # Extract raw values to handle both numpy and cftime objects
            t0_val = ds_truth.time.values[0]
            ti_val = ds_truth.time.values[t_idx]
            
            delta = ti_val - t0_val
            
            # Convert to days
            if isinstance(delta, np.timedelta64):
                # Convert from nanoseconds (default for xarray/numpy) to days
                days_elapsed = float(delta) / (1e9 * 86400)
            else:
                # Assume datetime.timedelta (cftime)
                days_elapsed = delta.days + delta.seconds / 86400.0
            
            if days_elapsed < 365:
                day_str = f"t = {int(days_elapsed)} days"
            else:
                years = days_elapsed / 365
                day_str = f"t = {years:.1f} years"
        except:
            day_str = f"Index {t_idx}"

        
        # Iterate Row-by-Row (Datasets)
        for i, ds_name in enumerate(datasets):
            ax = axes[i, j]
            data_to_plot = maps_data[ds_name][t_idx]
            
            p = data_to_plot.plot.pcolormesh(
                ax=ax, transform=ccrs.PlateCarree(),
                x='x', y='y',
                cmap=cmap, norm=norm,
                add_colorbar=False,
                rasterized=True
            )
            
            ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
            ax.coastlines(zorder=11)
            
            # Titles
            if i == 0:
                ax.set_title(f"Time: {time_str} ({day_str})", fontsize=16, fontweight='bold')
            else:
                ax.set_title("")
                
            # Row Labels
            if j == 0:
                label_text = "Ground Truth" if ds_name == TRUTH_LABEL else ds_name
                ax.text(-0.07, 0.55, label_text, va='center', ha='right',
                        rotation='vertical', transform=ax.transAxes, 
                        fontsize=14, fontweight='bold')

        # Add Colorbar for this Time Step
        cbar = fig.colorbar(
            p, ax=axes[:, j], 
            orientation='horizontal', 
            shrink=0.6, pad=0.03, aspect=25
        )
        cbar.set_label(f"OHC (ZJ)", fontsize=12)

    main_title = f"OHC Snapshot"
    if 'lev' in ds_truth[var_name].dims:
        main_title += f" ({depth_config['title']} Avg)"
        
    fig.suptitle(main_title, fontsize=20, y=1.05)
    plt.show()



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

    for iindex, (label, emu_data) in enumerate(emulators_dict.items()):
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

        if iindex==0:
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
        
        plt.show()
        # save_path = os.path.join(output_dir, f"Spatial_Bias_{label.replace(' ', '_')}_{variable_name}.png")
        # plt.savefig(save_path, bbox_inches='tight')
        # plt.close(fig)
        # print(f"Saved: {save_path}")

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
    plt.show()
    # save_path = os.path.join(output_dir, f"Snapshots_{var_name}_{depth_config['min']}_{depth_config['max']}.png")
    # plt.savefig(save_path, bbox_inches='tight')
    # plt.close(fig)
    # print(f"Saved: {save_path}")


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

def plot_mean_timeseries_comparison(ds_truth, emulators_dict, variable_name, var_units, var_label, output_dir,TRUTH_LABEL = 'Truth'):
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
    plt.show()
    # Save Figure
    # save_path = os.path.join(output_dir, f"TimeSeries_Comparison_{var_label}.png")
    # plt.savefig(save_path, bbox_inches='tight')
    # plt.close(fig)
    # print(f"Saved: {save_path}")


def plot_variance_comparison(ds_truth, emulators_dict, variable_name, units, output_dir=None,colorbar_fix=False):
    """
    Generates a grid of Time-Variance maps.
    Rows: Truth, then Emulators.
    Cols: Depth Slices.
    Colorbar: Shared per COLUMN (Depth Slice).
    """
    #print(f"\nGenerating Variance Comparison for {variable_name} (Per-Column Colorbars)...")
    
    # 1. Setup Metrics
    dz_truth = ds_truth['dz']
    wet_mask = (ds_truth[variable_name].isel(time=0) * 0 + 1).compute()
    
    # 2. Calculate Truth Variance (3D)
    #print("Calculating Truth Variance...")
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
        #print(f"Calculating Variance for {label}...")
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
            if j==0:
                vmax = np.percentile(flat_data, 93)
            else:
                vmax = np.percentile(flat_data, 95)
        else:
            vmax = 1.0

        vmin = 0

        if colorbar_fix:
            if variable_name == 'thetao':
                if j==0:
                    vmax = 1.5
                elif j==1:
                    vmax = 0.1
                elif j==2:
                    vmax = 0.006
            elif variable_name == 'so':
                if j==0:
                    vmax = 0.05
                elif j==1:
                    vmax = 2*10**(-3)
                elif j==2:
                    vmax = 2.5*0.00001  
        
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
    plt.show()
    # if output_dir:
    #     import os
    #     save_path = os.path.join(output_dir, f"Variance_Comparison_{variable_name}.png")
    #     plt.savefig(save_path, bbox_inches='tight')
    #     print(f"Saved: {save_path}")
    #     plt.close(fig)
    # else:
    #     plt.show()

def plot_zonal_bias_comparison(ds_truth, emulators_dict, variable_name, units, masks, output_dir,colorbar_fix=False):
    """Generates Zonal Mean Bias plots."""
    truth_field_mean = ds_truth[variable_name].mean('time')
    dx_truth = ds_truth['dx']
    wet_mask = (ds_truth[variable_name].isel(time=0) * 0 + 1).compute()
    
    print(f"Calculating Truth Zonal Profiles for {variable_name}...")
    truth_profiles = {}
    for basin in BASINS_TO_PLOT:
        truth_profiles[basin] = calculate_zonal_mean(
            truth_field_mean, masks[basin], dx_truth, wet_mask
        ).compute()

    for iindex, (label, emu_data) in enumerate(emulators_dict.items()):
        ds_emu = emu_data['ds']
        emu_field_mean = ds_emu[variable_name].mean('time')
        
        print(f"Generating Zonal Bias Plot for {label}...")
        bias_profiles = {}
        rmse_vals = {}
        weight_2d = (ds_truth['dy'] * ds_truth['dz']).mean('x').compute()
        all_biases = []

        for basin in BASINS_TO_PLOT:
            prof_emu = calculate_zonal_mean(
                emu_field_mean, masks[basin], dx_truth, wet_mask
            ).compute()
            prof_truth = truth_profiles[basin]
            
            bias = prof_emu - prof_truth
            bias_profiles[basin] = bias
            all_biases.append(bias.values.flatten())
            
            sq_diff = bias**2
            rmse = np.sqrt((sq_diff * weight_2d).sum(['y','lev']) / weight_2d.sum(['y','lev']))
            rmse_vals[basin] = rmse.item()


        if iindex==0:
            flat_biases = np.concatenate(all_biases)
            flat_biases = flat_biases[~np.isnan(flat_biases)]
            max_abs_bias = np.percentile(np.abs(flat_biases), 98) if len(flat_biases) > 0 else 1.0

        if colorbar_fix:
            max_abs_bias = 0.727
        
        norm = colors.Normalize(vmin=-max_abs_bias, vmax=max_abs_bias)
        cmap = cmocean.cm.balance

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True, constrained_layout=True)
        axes = axes.flatten()

        for i, basin in enumerate(BASINS_TO_PLOT):
            ax = axes[i]
            p_mesh = bias_profiles[basin].plot.contourf(
                ax=ax, cmap=cmap, norm=norm, add_colorbar=False, 
                levels=np.linspace(-max_abs_bias, max_abs_bias, 25), yincrease=False,
                x='y', y='lev',add_labels=False
            )
            ax.set_title(f"{basin} (RMSE: {rmse_vals[basin]:.3f} {units})", fontsize=16, loc='left')
            ax.set_facecolor('grey')
            ax.set_xlabel(''); ax.set_ylabel('')
            axes[0].set_ylabel('Depth (m)'); axes[2].set_ylabel('Depth (m)')
            axes[2].set_xlabel('Latitude index'); axes[3].set_xlabel('Latitude index')
        
        cbar = fig.colorbar(p_mesh, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label(f"Bias ({units})")
        fig.suptitle(f"Zonal Mean Bias: {label} vs Truth ({variable_name})", fontsize=20, y=1.03)
        plt.show()
        # save_path = os.path.join(output_dir, f"Zonal_Bias_{label.replace(' ', '_')}_{variable_name}.png")
        # plt.savefig(save_path, bbox_inches='tight')
        # plt.close(fig)
        # print(f"Saved: {save_path}")

def plot_zonal_bias_ocean_comparison(ds_truth, emulators_dict, variable_name, units, masks, output_dir, colorbar_fix=False):
    """
    Generates a single Zonal Mean Bias plot with all emulators.
    Layout: Rows = Ocean basins (4), Cols = Emulators (N)
    """
    truth_field_mean = ds_truth[variable_name].mean('time')
    dx_truth = ds_truth['dx']
    wet_mask = (ds_truth[variable_name].isel(time=0) * 0 + 1).compute()
    
    print(f"Calculating Truth Zonal Profiles for {variable_name}...")
    truth_profiles = {}
    for basin in BASINS_TO_PLOT:
        truth_profiles[basin] = calculate_zonal_mean(
            truth_field_mean, masks[basin], dx_truth, wet_mask
        ).compute()

    # Compute all emulator profiles and biases
    all_bias_profiles = {}
    all_rmse_vals = {}
    all_biases_flat = []
    
    weight_2d = (ds_truth['dy'] * ds_truth['dz']).mean('x').compute()
    
    for label, emu_data in emulators_dict.items():
        ds_emu = emu_data['ds']
        emu_field_mean = ds_emu[variable_name].mean('time')
        
        print(f"Calculating Zonal Bias for {label}...")
        all_bias_profiles[label] = {}
        all_rmse_vals[label] = {}

        for basin in BASINS_TO_PLOT:
            prof_emu = calculate_zonal_mean(
                emu_field_mean, masks[basin], dx_truth, wet_mask
            ).compute()
            prof_truth = truth_profiles[basin]
            
            bias = prof_emu - prof_truth
            all_bias_profiles[label][basin] = bias
            all_biases_flat.append(bias.values.flatten())
            
            sq_diff = bias**2
            rmse = np.sqrt((sq_diff * weight_2d).sum(['y','lev']) / weight_2d.sum(['y','lev']))
            all_rmse_vals[label][basin] = rmse.item()

    # Determine colorbar range
    flat_biases = np.concatenate(all_biases_flat)
    flat_biases = flat_biases[~np.isnan(flat_biases)]
    max_abs_bias = np.percentile(np.abs(flat_biases), 98) if len(flat_biases) > 0 else 1.0

    if colorbar_fix:
        max_abs_bias = 0.514
    
    norm = colors.Normalize(vmin=-max_abs_bias, vmax=max_abs_bias)
    cmap = cmocean.cm.balance

    # Plotting: 4 rows (basins) x N cols (emulators)
    num_emulators = len(emulators_dict)
    
    plt.style.use('seaborn-v0_8-talk')
    fig, axes = plt.subplots(4, num_emulators, figsize=(7 * num_emulators, 18), 
                            sharex=True, sharey=True, constrained_layout=True)
    
    # Ensure axes is 2D
    if num_emulators == 1:
        axes = axes.reshape(-1, 1)

    for i, basin in enumerate(BASINS_TO_PLOT):
        for j, (label, bias_profiles) in enumerate(all_bias_profiles.items()):
            ax = axes[i, j]
            p_mesh = bias_profiles[basin].plot.contourf(
                ax=ax, cmap=cmap, norm=norm, add_colorbar=False, 
                levels=np.linspace(-max_abs_bias, max_abs_bias, 25), yincrease=False,
                x='y', y='lev', add_labels=False
            )
            ax.set_facecolor('grey')
            ax.set_xlabel(''); ax.set_ylabel('')
            
            # Row labels on the left column
            if j == 0:
                ax.set_ylabel(f'{basin}\nDepth (m)', fontsize=12)
            
            # Add RMSE to title in first row
            if i == 0:
                ax.set_title(f"{label}\n(RMSE: {all_rmse_vals[label][basin]:.3f} {units})", 
                           fontsize=14, fontweight='bold')
            else:
                ax.set_title(f"RMSE: {all_rmse_vals[label][basin]:.3f} {units}", fontsize=11)
    
    # X-axis labels only on bottom row
    for j in range(num_emulators):
        axes[3, j].set_xlabel('Latitude index', fontsize=12)
    
    # Colorbar
    cbar = fig.colorbar(p_mesh, ax=axes, orientation='horizontal', shrink=1, pad=0.02, aspect=30)
    cbar.set_label(f"Bias ({units})", fontsize=12)
    
    fig.suptitle(f"Zonal Mean Bias: All Emulators vs Truth ({variable_name})", fontsize=18, y=1.03)
    plt.show()


def plot_ohc_zonal_bias_ocean_comparison(ds_truth, emulators_dict, masks, colorbar_fix=False):
    """
    Generates a single OHC Zonal Mean Bias plot with all emulators.
    Layout: Rows = Ocean basins (4), Cols = Emulators (N)
    """
    variable_name = 'OHC'
    units = 'J/m^2'
    
    # Setup Truth Metrics
    truth_field_2d = calculate_ohc_2d(ds_truth).mean('time').compute()
    dx_truth = ds_truth['dx']
    wet_mask_2d = (ds_truth['thetao'].isel(time=0, lev=0) * 0 + 1).compute()
    
    # Calculate Truth Zonal Profiles
    truth_profiles = {}
    print("Calculating Truth OHC Zonal Profiles...")
    for basin in BASINS_TO_PLOT:
        truth_profiles[basin] = calculate_zonal_mean_2d(
            truth_field_2d, masks[basin], dx_truth, wet_mask_2d
        ).compute()

    # Compute all emulator profiles and biases
    all_bias_profiles = {}
    all_rmse_vals = {}
    all_biases_flat = []
    
    weight_2d = (ds_truth['dy'] * ds_truth['dz']).mean('x').compute()

    for label, emu_data in emulators_dict.items():
        ds_emu = emu_data['ds']
        emu_field_2d = calculate_ohc_2d(ds_emu).mean('time').compute()
        
        print(f"Calculating OHC Zonal Bias for {label}...")
        all_bias_profiles[label] = {}
        all_rmse_vals[label] = {}

        for basin in BASINS_TO_PLOT:
            prof_emu = calculate_zonal_mean_2d(
                emu_field_2d, masks[basin], dx_truth, wet_mask_2d
            ).compute()
            prof_truth = truth_profiles[basin]
            
            bias = prof_emu - prof_truth
            all_bias_profiles[label][basin] = bias.drop('time')
            all_biases_flat.append(bias.values.flatten())
            
            sq_diff = bias**2
            rmse = np.sqrt((sq_diff * weight_2d).sum(['y','lev']) / weight_2d.sum(['y','lev']))
            all_rmse_vals[label][basin] = rmse.item()

    # Determine colorbar range
    flat_biases = np.concatenate(all_biases_flat)
    flat_biases = flat_biases[~np.isnan(flat_biases)]
    max_abs_bias = np.percentile(np.abs(flat_biases), 98) if len(flat_biases) > 0 else 1.0

    if colorbar_fix:
        max_abs_bias = 2.5*10**8
    
    norm = colors.Normalize(vmin=-max_abs_bias, vmax=max_abs_bias)
    cmap = cmocean.cm.balance

    # Plotting: 4 rows (basins) x N cols (emulators)
    num_emulators = len(emulators_dict)
    
    plt.style.use('seaborn-v0_8-talk')
    fig, axes = plt.subplots(4, num_emulators, figsize=(7 * num_emulators, 18),
                            sharex=True, sharey=True, constrained_layout=True)
    
    # Ensure axes is 2D
    if num_emulators == 1:
        axes = axes.reshape(-1, 1)

    for i, basin in enumerate(BASINS_TO_PLOT):
        for j, (label, bias_profiles) in enumerate(all_bias_profiles.items()):
            ax = axes[i, j]
            p_mesh = bias_profiles[basin].plot.contourf(
                ax=ax, cmap=cmap, norm=norm, add_colorbar=False, 
                levels=np.linspace(-max_abs_bias, max_abs_bias, 25), yincrease=False,
                x='y', y='lev'
            )
            ax.set_facecolor('grey')
            ax.set_xlabel(''); ax.set_ylabel('')
            
            # Row labels on the left column
            if j == 0:
                ax.set_ylabel(f'{basin}\nDepth (m)', fontsize=12)
            
            # Add RMSE to title in first row
            if i == 0:
                ax.set_title(f"{label}\n(RMSE: {all_rmse_vals[label][basin]:.3e} {units})", 
                           fontsize=14, fontweight='bold')
            else:
                ax.set_title(f"RMSE: {all_rmse_vals[label][basin]:.3e} {units}", fontsize=11)
    
    # X-axis labels only on bottom row
    for j in range(num_emulators):
        axes[3, j].set_xlabel('Latitude index', fontsize=12)
    
    # Colorbar
    cbar = fig.colorbar(p_mesh, ax=axes, orientation='horizontal', shrink=1, pad=0.02, aspect=30)
    cbar.set_label(f"Bias ({units})", fontsize=12)
    
    fig.suptitle(f"Zonal Mean Bias of OHC: All Emulators vs Truth", fontsize=18, y=1.03)
    plt.show()


def plot_snapshot_zonal_bias_ocean_comparison(ds_truth, emulators_dict, variable_name, units, masks, TIME_INDICES, output_dir=None, colorbar_fix=False):
    """
    Generates Zonal Mean Bias plots for snapshots at specific time indices.
    Layout: Rows = Ocean basins (4), Cols = Emulators (N)
    
    Parameters:
    -----------
    TIME_INDICES : list
        List of time indices to plot (e.g., [0, 40, 80])
    """
    dx_truth = ds_truth['dx']
    wet_mask = (ds_truth[variable_name].isel(time=0) * 0 + 1).compute()
    weight_2d = (ds_truth['dy'] * ds_truth['dz']).mean('x').compute()
    
    # Process each time index
    for time_idx in TIME_INDICES:
        print(f"\nProcessing time index: {time_idx}")
        
        # Format time for title
        try:
            time_val = ds_truth.time[time_idx].values
            if hasattr(time_val, 'strftime'):
                time_str = time_val.strftime("%Y-%m-%d")
            else:
                time_str = str(time_val)[:10]
        except:
            time_str = f"Index {time_idx}"
        
        try:
            t0_val = ds_truth.time.values[0]
            ti_val = ds_truth.time.values[time_idx]
            delta = ti_val - t0_val
            
            if isinstance(delta, np.timedelta64):
                days_elapsed = float(delta) / (1e9 * 86400)
            else:
                days_elapsed = delta.days + delta.seconds / 86400.0
            
            if days_elapsed < 365:
                day_str = f"t = {int(days_elapsed)} days"
            else:
                years = days_elapsed / 365
                day_str = f"t = {years:.1f} years"
        except:
            day_str = f"Index {time_idx}"
        
        # Calculate Truth profiles at this snapshot
        truth_field_snapshot = ds_truth[variable_name].isel(time=time_idx)
        
        print(f"Calculating Truth Zonal Profiles for {variable_name} at time index {time_idx}...")
        truth_profiles = {}
        for basin in BASINS_TO_PLOT:
            truth_profiles[basin] = calculate_zonal_mean(
                truth_field_snapshot, masks[basin], dx_truth, wet_mask
            ).compute()

        # Compute all emulator profiles and biases
        all_bias_profiles = {}
        all_rmse_vals = {}
        all_biases_flat = []
        
        for label, emu_data in emulators_dict.items():
            ds_emu = emu_data['ds']
            emu_field_snapshot = ds_emu[variable_name].isel(time=time_idx)
            
            print(f"Calculating Zonal Bias for {label} at time index {time_idx}...")
            all_bias_profiles[label] = {}
            all_rmse_vals[label] = {}

            for basin in BASINS_TO_PLOT:
                prof_emu = calculate_zonal_mean(
                    emu_field_snapshot, masks[basin], dx_truth, wet_mask
                ).compute()
                prof_truth = truth_profiles[basin]
                
                bias = prof_emu - prof_truth
                all_bias_profiles[label][basin] = bias
                all_biases_flat.append(bias.values.flatten())
                
                sq_diff = bias**2
                rmse = np.sqrt((sq_diff * weight_2d).sum(['y','lev']) / weight_2d.sum(['y','lev']))
                all_rmse_vals[label][basin] = rmse.item()

        # Determine colorbar range
        flat_biases = np.concatenate(all_biases_flat)
        flat_biases = flat_biases[~np.isnan(flat_biases)]
        max_abs_bias = np.percentile(np.abs(flat_biases), 98) if len(flat_biases) > 0 else 1.0

        if colorbar_fix:
            max_abs_bias = 1.338
            
        norm = colors.Normalize(vmin=-max_abs_bias, vmax=max_abs_bias)
        cmap = cmocean.cm.balance

        # Plotting: 4 rows (basins) x N cols (emulators)
        num_emulators = len(emulators_dict)
        
        plt.style.use('seaborn-v0_8-talk')
        fig, axes = plt.subplots(4, num_emulators, figsize=(7 * num_emulators, 18), 
                                sharex=True, sharey=True, constrained_layout=True)
        
        # Ensure axes is 2D
        if num_emulators == 1:
            axes = axes.reshape(-1, 1)

        for i, basin in enumerate(BASINS_TO_PLOT):
            for j, (label, bias_profiles) in enumerate(all_bias_profiles.items()):
                ax = axes[i, j]
                p_mesh = bias_profiles[basin].plot.contourf(
                    ax=ax, cmap=cmap, norm=norm, add_colorbar=False, 
                    levels=np.linspace(-max_abs_bias, max_abs_bias, 25), yincrease=False,
                    x='y', y='lev', add_labels=False
                )
                ax.set_facecolor('grey')
                ax.set_xlabel(''); ax.set_ylabel('')
                
                # Row labels on the left column
                if j == 0:
                    ax.set_ylabel(f'{basin}\nDepth (m)', fontsize=12)
                
                # Add RMSE to title in first row
                if i == 0:
                    ax.set_title(f"{label}\n(RMSE: {all_rmse_vals[label][basin]:.3f} {units})", 
                               fontsize=14, fontweight='bold')
                else:
                    ax.set_title(f"RMSE: {all_rmse_vals[label][basin]:.3f} {units}", fontsize=11)
        
        # X-axis labels only on bottom row
        for j in range(num_emulators):
            axes[3, j].set_xlabel('Latitude index', fontsize=12)
        
        # Colorbar
        cbar = fig.colorbar(p_mesh, ax=axes, orientation='horizontal', shrink=1, pad=0.02, aspect=30)
        cbar.set_label(f"Bias ({units})", fontsize=12)
        
        fig.suptitle(f"Snapshot Zonal Mean Bias: {time_str} ({day_str}) - {variable_name}", fontsize=18, y=1.03)
        plt.show()


def plot_snapshot_ohc_zonal_bias_ocean_comparison(ds_truth, emulators_dict, masks, TIME_INDICES, colorbar_fix=False):
    """
    Generates OHC Zonal Mean Bias plots for snapshots at specific time indices.
    Layout: Rows = Ocean basins (4), Cols = Emulators (N)
    
    Parameters:
    -----------
    TIME_INDICES : list
        List of time indices to plot (e.g., [0, 40, 80])
    """
    variable_name = 'OHC'
    units = 'J/m^2'
    
    dx_truth = ds_truth['dx']
    wet_mask_2d = (ds_truth['thetao'].isel(time=0, lev=0) * 0 + 1).compute()
    weight_2d = (ds_truth['dy'] * ds_truth['dz']).mean('x').compute()
    
    # Process each time index
    for time_idx in TIME_INDICES:
        print(f"\nProcessing time index: {time_idx}")
        
        # Format time for title
        try:
            time_val = ds_truth.time[time_idx].values
            if hasattr(time_val, 'strftime'):
                time_str = time_val.strftime("%Y-%m-%d")
            else:
                time_str = str(time_val)[:10]
        except:
            time_str = f"Index {time_idx}"
        
        try:
            t0_val = ds_truth.time.values[0]
            ti_val = ds_truth.time.values[time_idx]
            delta = ti_val - t0_val
            
            if isinstance(delta, np.timedelta64):
                days_elapsed = float(delta) / (1e9 * 86400)
            else:
                days_elapsed = delta.days + delta.seconds / 86400.0
            
            if days_elapsed < 365:
                day_str = f"t = {int(days_elapsed)} days"
            else:
                years = days_elapsed / 365
                day_str = f"t = {years:.1f} years"
        except:
            day_str = f"Index {time_idx}"
        
        # Calculate OHC at this snapshot
        truth_field_snapshot = calculate_ohc_2d(ds_truth).isel(time=time_idx).compute()
        
        # Calculate Truth Zonal Profiles
        truth_profiles = {}
        print(f"Calculating Truth OHC Zonal Profiles at time index {time_idx}...")
        for basin in BASINS_TO_PLOT:
            truth_profiles[basin] = calculate_zonal_mean_2d(
                truth_field_snapshot, masks[basin], dx_truth, wet_mask_2d
            ).compute()

        # Compute all emulator profiles and biases
        all_bias_profiles = {}
        all_rmse_vals = {}
        all_biases_flat = []

        for label, emu_data in emulators_dict.items():
            ds_emu = emu_data['ds']
            emu_field_snapshot = calculate_ohc_2d(ds_emu).isel(time=time_idx).compute()
            
            print(f"Calculating OHC Zonal Bias for {label} at time index {time_idx}...")
            all_bias_profiles[label] = {}
            all_rmse_vals[label] = {}

            for basin in BASINS_TO_PLOT:
                prof_emu = calculate_zonal_mean_2d(
                    emu_field_snapshot, masks[basin], dx_truth, wet_mask_2d
                ).compute()
                prof_truth = truth_profiles[basin]
                
                bias = prof_emu - prof_truth
                # Time dimension already removed by isel, no need to drop
                all_bias_profiles[label][basin] = bias
                all_biases_flat.append(bias.values.flatten())
                
                sq_diff = bias**2
                rmse = np.sqrt((sq_diff * weight_2d).sum(['y','lev']) / weight_2d.sum(['y','lev']))
                all_rmse_vals[label][basin] = rmse.item()

        # Determine colorbar range
        flat_biases = np.concatenate(all_biases_flat)
        flat_biases = flat_biases[~np.isnan(flat_biases)]
        max_abs_bias = np.percentile(np.abs(flat_biases), 98) if len(flat_biases) > 0 else 1.0

        if colorbar_fix:
            max_abs_bias = 4.2*10**8
        
        norm = colors.Normalize(vmin=-max_abs_bias, vmax=max_abs_bias)
        cmap = cmocean.cm.balance

        # Plotting: 4 rows (basins) x N cols (emulators)
        num_emulators = len(emulators_dict)
        
        plt.style.use('seaborn-v0_8-talk')
        fig, axes = plt.subplots(4, num_emulators, figsize=(7 * num_emulators, 18),
                                sharex=True, sharey=True, constrained_layout=True)
        
        # Ensure axes is 2D
        if num_emulators == 1:
            axes = axes.reshape(-1, 1)

        for i, basin in enumerate(BASINS_TO_PLOT):
            for j, (label, bias_profiles) in enumerate(all_bias_profiles.items()):
                ax = axes[i, j]
                p_mesh = bias_profiles[basin].plot.contourf(
                    ax=ax, cmap=cmap, norm=norm, add_colorbar=False, 
                    levels=np.linspace(-max_abs_bias, max_abs_bias, 25), yincrease=False,
                    x='y', y='lev'
                )
                ax.set_facecolor('grey')
                ax.set_xlabel(''); ax.set_ylabel('')
                
                # Row labels on the left column
                if j == 0:
                    ax.set_ylabel(f'{basin}\nDepth (m)', fontsize=12)
                
                # Add RMSE to title in first row
                if i == 0:
                    ax.set_title(f"{label}\n(RMSE: {all_rmse_vals[label][basin]:.3e} {units})", 
                               fontsize=14, fontweight='bold')
                else:
                    ax.set_title(f"RMSE: {all_rmse_vals[label][basin]:.3e} {units}", fontsize=11)
        
        # X-axis labels only on bottom row
        for j in range(num_emulators):
            axes[3, j].set_xlabel('Latitude index', fontsize=12)
        
        # Colorbar
        cbar = fig.colorbar(p_mesh, ax=axes, orientation='horizontal', shrink=1, pad=0.02, aspect=30)
        cbar.set_label(f"Bias ({units})", fontsize=12)
        
        fig.suptitle(f"Snapshot Zonal Mean Bias of OHC: {time_str} ({day_str})", fontsize=18, y=1.03)
        plt.show()


def plot_isotropic_spectrum_comparison(
    dataset_truth, 
    emulators_to_analyze, 
    var_to_eval='thetao', 
    lev_idx=0,
    time_window=100,
    lon_slice=slice(180, 243),
    lat_slice=slice(-40, 35),
    target_wavenumbers=[0.01, 0.02],
    TRUTH_LABEL='OM4',
    show_ratios=True,
    colorbar_fix=False
):
    """
    Plots isotropic power spectrum comparison for temperature anomalies between truth and emulators.
    
    Parameters:
    -----------
    dataset_truth : xr.Dataset
        Ground truth dataset
    emulators_to_analyze : dict
        Dictionary of emulators with structure:
        {
            'label': {
                'ds': xr.Dataset,
                'color': str (matplotlib color)
            }
        }
    var_to_eval : str, default='thetao'
        Variable name to evaluate
    lev_idx : int, default=0
        Level index for depth slice
    time_window : int, default=100
        Number of time steps to use
    lon_slice : slice, default=slice(180, 243)
        Longitude slice for analysis
    lat_slice : slice, default=slice(-40, 35)
        Latitude slice for analysis
    target_wavenumbers : list, default=[0.01, 0.02]
        Wavenumbers at which to calculate and annotate ratios
    TRUTH_LABEL : str, default='OM4'
        Label for the truth dataset
    colorbar_fix : bool, default=False
        Not used in this function, kept for consistency with other functions
    
    Returns:
    --------
    None (displays plot)
    """
    
    print(f"\nGenerating Isotropic Power Spectrum Comparison for {var_to_eval}...")
    
    # Calculate mean grid spacing
    dx_mean = float(dataset_truth.dx.sel({'x': lon_slice, 'y': lat_slice}).mean().values)
    dy_mean = float(dataset_truth.dy.sel({'x': lon_slice, 'y': lat_slice}).mean().values)
    
    def get_spectrum(ds_target, var_name, lev_i, t_window, x_sl, y_sl):
        """Helper function to compute isotropic spectrum for a dataset."""
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
    
    # Calculate reference spectrum
    print(f"Calculating spectrum for {TRUTH_LABEL}...")
    k_ref, spec_ref = get_spectrum(
        dataset_truth, var_to_eval, lev_idx, time_window, lon_slice, lat_slice
    )
    
    # Process all emulators
    emulator_spectra = {}
    for label, emu_data in emulators_to_analyze.items():
        print(f"Calculating spectrum for {label}...")
        k, spec = get_spectrum(
            emu_data['ds'], var_to_eval, lev_idx, time_window, lon_slice, lat_slice
        )
        emulator_spectra[label] = {'k': k, 'spec': spec}
    
    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot truth
    ax.loglog(
        k_ref, spec_ref.mean(0), '-', 
        label=f'{TRUTH_LABEL} (Temperature Anomalies)', 
        linewidth=2, color='black'
    )
    
    # Plot emulators
    for label, data in emulator_spectra.items():
        color = emulators_to_analyze[label]['color']
        ax.loglog(
            data['k'], data['spec'].mean(0), '-', 
            label=f'{label} (Temperature Anomalies)', 
            color=color
        )
    
    # Add ratio annotations at target wavenumbers (optional)
    if show_ratios:
        for target_k in target_wavenumbers:
            idx_ref = np.argmin(np.abs(k_ref - target_k))
            actual_k = k_ref[idx_ref]
            spec_ref_value = spec_ref.mean(0)[idx_ref]
            
            print(f"\nRatios at wavenumber {actual_k:.4f}:")
            print(f"{TRUTH_LABEL}: {spec_ref_value:.2e}")
            
            for i, (label, data) in enumerate(emulator_spectra.items()):
                idx_emu = np.argmin(np.abs(data['k'] - target_k))
                spec_emu_value = data['spec'].mean(0)[idx_emu]
                ratio = spec_emu_value / spec_ref_value
                
                print(f"{label}: {spec_emu_value:.2e}, Ratio={ratio:.3f}")
                
                color = emulators_to_analyze[label]['color']
                x_pos = actual_k * 1.1
                y_pos = spec_emu_value * (1.5 ** i)
                
                ax.text(
                    x_pos, y_pos, 
                    f'{label}/{TRUTH_LABEL}={ratio:.3f}', 
                    fontsize=8, color=color,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color)
                )
            
            # Add vertical line at target wavenumber
            ax.axvline(x=actual_k, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    
    # Format axes - use LogLocator for better tick placement
    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    
    # Show ticks at 10^n and intermediate values (2*10^n, 5*10^n) for better coverage
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0]))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(labelOnlyBase=False))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs='all'))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())  # Disable minor tick labels
    
    ax.set_xlabel(r'Wavenumber $\kappa$ (km$^{-1}$)')
    ax.set_ylabel('Power Spectral Density')
    
    # Get depth value and spatial boundaries for title
    depth_value = dataset_truth.lev.values[lev_idx]
    
    # Format longitude range
    lon_start = lon_slice.start if lon_slice.start is not None else 0
    lon_stop = lon_slice.stop if lon_slice.stop is not None else 360
    if lon_stop > 180:
        lon_stop_display = f"{lon_stop - 360}°W" if lon_stop > 180 else f"{lon_stop}°E"
    else:
        lon_stop_display = f"{lon_stop}°E"
    lon_start_display = f"{lon_start}°E"
    
    # Format latitude range
    lat_start = lat_slice.start if lat_slice.start is not None else -90
    lat_stop = lat_slice.stop if lat_slice.stop is not None else 90
    lat_start_display = f"{abs(lat_start)}°S" if lat_start < 0 else f"{lat_start}°N"
    lat_stop_display = f"{abs(lat_stop)}°S" if lat_stop < 0 else f"{lat_stop}°N"
    
    ax.set_title(f'Isotropic Power Spectrum of {var_to_eval} ({depth_value}m)\n' + 
                 f'Region: {lon_start_display}-{lon_stop_display}, {lat_start_display}-{lat_stop_display}',
                 fontsize=11)
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.show()


def plot_ke_spectrum_comparison(
    dataset_truth, 
    emulators_to_analyze,
    var_u='uo',
    var_v='vo',
    lev_idx=0,
    time_window=100,
    lon_slice=slice(180, 243),
    lat_slice=slice(-40, 35),
    target_wavenumbers=[0.01, 0.02],
    TRUTH_LABEL='OM4',
    show_ratios=True,
    colorbar_fix=False
):
    """
    Plots kinetic energy spectrum comparison between truth and emulators.
    
    Parameters:
    -----------
    dataset_truth : xr.Dataset
        Ground truth dataset
    emulators_to_analyze : dict
        Dictionary of emulators with structure:
        {
            'label': {
                'ds': xr.Dataset,
                'color': str (matplotlib color)
            }
        }
    var_u : str, default='uo'
        U-velocity variable name
    var_v : str, default='vo'
        V-velocity variable name
    lev_idx : int, default=0
        Level index for depth slice
    time_window : int, default=100
        Number of time steps to use
    lon_slice : slice, default=slice(180, 243)
        Longitude slice for analysis
    lat_slice : slice, default=slice(-40, 35)
        Latitude slice for analysis
    target_wavenumbers : list, default=[0.01, 0.02]
        Wavenumbers at which to calculate and annotate ratios
    TRUTH_LABEL : str, default='OM4'
        Label for the truth dataset
    colorbar_fix : bool, default=False
        Not used in this function, kept for consistency with other functions
    
    Returns:
    --------
    None (displays plot)
    """
    
    #print(f"\nGenerating Kinetic Energy Spectrum Comparison...")
    
    # Calculate mean grid spacing
    dx_mean = float(dataset_truth.dx.sel({'x': lon_slice, 'y': lat_slice}).mean().values)
    dy_mean = float(dataset_truth.dy.sel({'x': lon_slice, 'y': lat_slice}).mean().values)
    
    def get_ke_spectrum(ds_target, lev_i, t_window, x_sl, y_sl):
        """Helper function to compute KE spectrum for a dataset."""
        def process_var(var_name):
            data = ds_target[var_name].isel(lev=lev_i, time=slice(None, t_window))
            data = data.transpose('time', ...).sel({'x': x_sl, 'y': y_sl})
            
            # Subtract time-mean to get anomalies
            data_anom = data - data.mean(dim='time')
            return torch.as_tensor(data_anom.fillna(0).values)
        
        u_tensor = process_var(var_u)
        v_tensor = process_var(var_v)
        
        k_cent, spec_u = compute_isotropic_spectrum_torch(
            u_tensor, dx=dx_mean, dy=dy_mean, n_factor=2, 
            detrend='linear', window='hann'
        )
        _, spec_v = compute_isotropic_spectrum_torch(
            v_tensor, dx=dx_mean, dy=dy_mean, n_factor=2, 
            detrend='linear', window='hann'
        )
        
        # Kinetic energy spectrum is 0.5 * (u² + v²)
        spec_ke = 0.5 * (spec_u + spec_v)
        
        k_km = k_cent.cpu().numpy() * 1000 * 2 * np.pi
        return k_km, spec_ke.cpu().numpy()
    
    # Calculate reference KE spectrum
    #print(f"Calculating KE spectrum for {TRUTH_LABEL}...")
    k_ref, spec_ref = get_ke_spectrum(
        dataset_truth, lev_idx, time_window, lon_slice, lat_slice
    )
    
    # Process all emulators
    ke_spectra = {}
    for label, emu_data in emulators_to_analyze.items():
        #print(f"Calculating KE spectrum for {label}...")
        k, spec = get_ke_spectrum(
            emu_data['ds'], lev_idx, time_window, lon_slice, lat_slice
        )
        ke_spectra[label] = {'k': k, 'spec': spec}
    
    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot truth
    ax.loglog(
        k_ref, spec_ref.mean(0), '-', 
        label=f'{TRUTH_LABEL} (KE Anomalies)', 
        linewidth=2, color='black'
    )
    
    # Plot emulators
    for label, data in ke_spectra.items():
        color = emulators_to_analyze[label]['color']
        ax.loglog(
            data['k'], data['spec'].mean(0), '-', 
            label=f'{label} (KE Anomalies)', 
            color=color
        )
    
    # Add ratio annotations at target wavenumbers (optional)
    if show_ratios:
        for target_k in target_wavenumbers:
            idx_ref = np.argmin(np.abs(k_ref - target_k))
            actual_k = k_ref[idx_ref]
            spec_ref_value = spec_ref.mean(0)[idx_ref]
            
            #print(f"\nRatios at wavenumber {actual_k:.4f}:")
            #print(f"{TRUTH_LABEL} PSD = {spec_ref_value:.2e}")
            
            for i, (label, data) in enumerate(ke_spectra.items()):
                idx_emu = np.argmin(np.abs(data['k'] - target_k))
                spec_emu_value = data['spec'].mean(0)[idx_emu]
                ratio = spec_emu_value / spec_ref_value
                
                #print(f"{label} PSD = {spec_emu_value:.2e}, Ratio = {ratio:.3f}")
                
                color = emulators_to_analyze[label]['color']
                x_pos = actual_k * 1.1
                y_pos = spec_emu_value * (1.5 ** i)
                
                ax.text(
                    x_pos, y_pos, 
                    f'{label}/{TRUTH_LABEL}={ratio:.3f}', 
                    fontsize=8, color=color,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color)
                )
            
            # Add vertical line at target wavenumber
            ax.axvline(x=actual_k, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    
    # Format axes - use LogLocator for dynamic tick placement
    from matplotlib.ticker import LogLocator, LogFormatter
    
    # Set up dynamic log-scale ticks
    # subs parameter controls intermediate ticks (e.g., [1.0, 2.0, 5.0] gives ticks at 10^n, 2*10^n, 5*10^n)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=8))
    ax.xaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=False))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())  # Disable minor tick labels to prevent overlap
    
    ax.set_xlabel(r'Wavenumber $\kappa$ (km$^{-1}$)')
    ax.set_ylabel('Kinetic Energy Density')
    
    # Get depth value and spatial boundaries for title
    depth_value = dataset_truth.lev.values[lev_idx]
    
    # Format longitude range
    lon_start = lon_slice.start if lon_slice.start is not None else 0
    lon_stop = lon_slice.stop if lon_slice.stop is not None else 360
    if lon_stop > 180:
        lon_stop_display = f"{lon_stop - 360}°W" if lon_stop > 180 else f"{lon_stop}°E"
    else:
        lon_stop_display = f"{lon_stop}°E"
    lon_start_display = f"{lon_start}°E"
    
    # Format latitude range
    lat_start = lat_slice.start if lat_slice.start is not None else -90
    lat_stop = lat_slice.stop if lat_slice.stop is not None else 90
    lat_start_display = f"{abs(lat_start)}°S" if lat_start < 0 else f"{lat_start}°N"
    lat_stop_display = f"{abs(lat_stop)}°S" if lat_stop < 0 else f"{lat_stop}°N"
    
    ax.set_title(f'Kinetic Energy Spectrum (Anomalies) at {depth_value}m\n' +
                 f'Region: {lon_start_display}-{lon_stop_display}, {lat_start_display}-{lat_stop_display}',
                 fontsize=11)
    
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.show()


def plot_temporal_spectrum_comparison(
    dataset_truth, 
    emulators_to_analyze, 
    var_to_eval='thetao',
    lev_idx=0,
    lon_slice=slice(180, 243),
    lat_slice=slice(-40, 35),
    target_frequencies=None,
    TRUTH_LABEL='OM4',
    show_ratios=True,
    colorbar_fix=False
):
    """
    Plots temporal (time-domain) power spectrum comparison for a variable between truth and emulators.
    
    Parameters:
    -----------
    dataset_truth : xr.Dataset
        Ground truth dataset
    emulators_to_analyze : dict
        Dictionary of emulators with structure:
        {
            'label': {
                'ds': xr.Dataset,
                'color': str (matplotlib color)
            }
        }
    var_to_eval : str, default='thetao'
        Variable name to evaluate
    lev_idx : int, default=0
        Level index for depth slice
    lon_slice : slice, default=slice(180, 243)
        Longitude slice for spatial averaging
    lat_slice : slice, default=slice(-40, 35)
        Latitude slice for spatial averaging
    target_frequencies : list, optional
        Frequencies (in cycles/year) at which to calculate and annotate ratios
        If None, defaults to [1.0, 4.0] representing annual and seasonal cycles
    TRUTH_LABEL : str, default='OM4'
        Label for the truth dataset
    show_ratios : bool, default=True
        Whether to show ratio annotations at target frequencies
    colorbar_fix : bool, default=False
        Not used in this function, kept for consistency with other functions
    
    Returns:
    --------
    None (displays plot)
    """
    
    if target_frequencies is None:
        target_frequencies = [1.0, 4.0]  # Annual and seasonal cycles
    
    #print(f"\nGenerating Temporal Power Spectrum Comparison for {var_to_eval}...")
    
    def get_temporal_spectrum(ds_target, var_name, lev_i, x_sl, y_sl):
        """Helper function to compute temporal spectrum at each spatial point, then average."""
        from scipy import signal
        
        # Extract data at specified depth and region
        data = ds_target[var_name].isel(lev=lev_i).sel({'x': x_sl, 'y': y_sl})
        
        # Get dimensions
        n_time = data.sizes['time']
        
        # Get time step for frequency calculation
        try:
            time_vals = ds_target.time.values
            if len(time_vals) > 1:
                delta_t = (time_vals[1] - time_vals[0])
                if isinstance(delta_t, np.timedelta64):
                    dt_days = float(delta_t) / np.timedelta64(1, 'D')
                else:
                    dt_days = delta_t.days + delta_t.seconds / 86400.0
            else:
                dt_days = 1.0
        except:
            dt_days = 1.0
        
        # Convert to years for sampling frequency
        dt_years = dt_days / 365.25
        fs = 1.0 / dt_years  # Sampling frequency in cycles/year
        
        # Convert to numpy array (time, y, x)
        data_np = data.values
        
        # Replace NaN with 0
        data_np = np.nan_to_num(data_np, nan=0.0)
        
        # Apply linear detrending along time axis for each spatial point
        # This removes both mean and linear trend (a*t + b)
        data_detrended = signal.detrend(data_np, axis=0, type='linear')

        # Remove seasonal cycle (monthly climatology)
        steps_per_year = int(round(365.25 / dt_days))
        n_full = (n_time // steps_per_year) * steps_per_year
        if n_full >= steps_per_year:
            climatology = data_detrended[:n_full].reshape(-1, steps_per_year, *data_detrended.shape[1:]).mean(axis=0)
            seasonal = np.tile(climatology, (n_time // steps_per_year + 1, 1, 1))[:n_time]
            data_detrended = data_detrended - seasonal

        # Apply Hanning window along time axis
        window = signal.windows.hann(n_time)
        data_windowed = data_detrended * window[:, np.newaxis, np.newaxis]

        # Compute window correction factor
        window_correction = np.mean(window**2)
        
        # Compute FFT along time axis for each spatial point
        fft_result = np.fft.rfft(data_windowed, axis=0)
        
        # Compute power spectral density at each spatial point
        # For one-sided spectrum (rfft), multiply by 2 to account for negative frequencies
        # PSD = 2 * |FFT|^2 / (fs * N * window_correction)
        power_spectrum = np.abs(fft_result)**2
        psd_spatial = 2.0 * power_spectrum / (fs * n_time * window_correction)
        
        # DC component (frequency = 0) should not be multiplied by 2
        psd_spatial[0, :, :] = psd_spatial[0, :, :] / 2.0
        
        # If n_time is even, Nyquist frequency should not be multiplied by 2
        if n_time % 2 == 0:
            psd_spatial[-1, :, :] = psd_spatial[-1, :, :] / 2.0
        
        # Average PSD over all spatial points
        psd = np.nanmean(psd_spatial, axis=(1, 2))
        
        # Get frequency array
        freqs = np.fft.rfftfreq(n_time, d=dt_years)
        
        return freqs, psd
    
    # Calculate reference spectrum
    #print(f"Calculating temporal spectrum for {TRUTH_LABEL}...")
    freq_ref, psd_ref = get_temporal_spectrum(
        dataset_truth, var_to_eval, lev_idx, lon_slice, lat_slice
    )
    
    # Process all emulators
    emulator_spectra = {}
    for label, emu_data in emulators_to_analyze.items():
        print(f"Calculating temporal spectrum for {label}...")
        freq, psd = get_temporal_spectrum(
            emu_data['ds'], var_to_eval, lev_idx, lon_slice, lat_slice
        )
        emulator_spectra[label] = {'freq': freq, 'psd': psd}
    
    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot truth
    ax.loglog(
        freq_ref[1:], psd_ref[1:], '-',  # Skip DC component
        label=f'{TRUTH_LABEL} ({var_to_eval} Anomalies)', 
        linewidth=2, color='black'
    )
    
    # Plot emulators
    for label, data in emulator_spectra.items():
        color = emulators_to_analyze[label]['color']
        ax.loglog(
            data['freq'][1:], data['psd'][1:], '-',  # Skip DC component
            label=f'{label} ({var_to_eval} Anomalies)', 
            color=color
        )
    
    # Add ratio annotations at target frequencies (optional)
    if show_ratios and len(target_frequencies) > 0:
        for target_f in target_frequencies:
            idx_ref = np.argmin(np.abs(freq_ref - target_f))
            actual_f = freq_ref[idx_ref]
            psd_ref_value = psd_ref[idx_ref]
            
            # Calculate period for display
            period_years = 1.0 / actual_f if actual_f > 0 else np.inf
            
            #print(f"\nRatios at frequency {actual_f:.4f} cycles/year (period: {period_years:.2f} years):")
            #print(f"{TRUTH_LABEL}: {psd_ref_value:.2e}")
            
            for i, (label, data) in enumerate(emulator_spectra.items()):
                idx_emu = np.argmin(np.abs(data['freq'] - target_f))
                psd_emu_value = data['psd'][idx_emu]
                ratio = psd_emu_value / psd_ref_value if psd_ref_value > 0 else 0
                
                #print(f"{label}: {psd_emu_value:.2e}, Ratio={ratio:.3f}")
                
                color = emulators_to_analyze[label]['color']
                x_pos = actual_f * 1.2
                y_pos = psd_emu_value * (1.5 ** i)
                
                ax.text(
                    x_pos, y_pos, 
                    f'{label}/{TRUTH_LABEL}={ratio:.3f}', 
                    fontsize=8, color=color,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color)
                )
            
            # Add vertical line at target frequency
            ax.axvline(x=actual_f, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    
    # Format axes
    ax.set_xlabel('Frequency (cycles/year)')
    ax.set_ylabel('Power Spectral Density')
    
    # Get depth value and spatial boundaries for title
    depth_value = dataset_truth.lev.values[lev_idx]
    
    # Format longitude range
    lon_start = lon_slice.start if lon_slice.start is not None else 0
    lon_stop = lon_slice.stop if lon_slice.stop is not None else 360
    if lon_stop > 180:
        lon_stop_display = f"{lon_stop - 360}°W" if lon_stop > 180 else f"{lon_stop}°E"
    else:
        lon_stop_display = f"{lon_stop}°E"
    lon_start_display = f"{lon_start}°E"
    
    # Format latitude range
    lat_start = lat_slice.start if lat_slice.start is not None else -90
    lat_stop = lat_slice.stop if lat_slice.stop is not None else 90
    lat_start_display = f"{abs(lat_start)}°S" if lat_start < 0 else f"{lat_start}°N"
    lat_stop_display = f"{abs(lat_stop)}°S" if lat_stop < 0 else f"{lat_stop}°N"
    
    ax.set_title(f'Temporal Power Spectrum of {var_to_eval} at {depth_value}m\n' +
                 f'Region: {lon_start_display}-{lon_stop_display}, {lat_start_display}-{lat_stop_display}',
                 fontsize=11)
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.show()


def plot_ke_temporal_spectrum_comparison(
    dataset_truth, 
    emulators_to_analyze,
    var_u='uo',
    var_v='vo',
    lev_idx=0,
    lon_slice=slice(180, 243),
    lat_slice=slice(-40, 35),
    target_frequencies=None,
    TRUTH_LABEL='OM4',
    show_ratios=True,
    colorbar_fix=False
):
    """
    Plots temporal (time-domain) kinetic energy spectrum comparison between truth and emulators.
    
    Parameters:
    -----------
    dataset_truth : xr.Dataset
        Ground truth dataset
    emulators_to_analyze : dict
        Dictionary of emulators with structure:
        {
            'label': {
                'ds': xr.Dataset,
                'color': str (matplotlib color)
            }
        }
    var_u : str, default='uo'
        U-velocity variable name
    var_v : str, default='vo'
        V-velocity variable name
    lev_idx : int, default=0
        Level index for depth slice
    lon_slice : slice, default=slice(180, 243)
        Longitude slice for spatial averaging
    lat_slice : slice, default=slice(-40, 35)
        Latitude slice for spatial averaging
    target_frequencies : list, optional
        Frequencies (in cycles/year) at which to calculate and annotate ratios
        If None, defaults to [1.0, 4.0] representing annual and seasonal cycles
    TRUTH_LABEL : str, default='OM4'
        Label for the truth dataset
    show_ratios : bool, default=True
        Whether to show ratio annotations at target frequencies
    colorbar_fix : bool, default=False
        Not used in this function, kept for consistency with other functions
    
    Returns:
    --------
    None (displays plot)
    """
    
    if target_frequencies is None:
        target_frequencies = [1.0, 4.0]  # Annual and seasonal cycles
    
    print(f"\nGenerating Temporal KE Power Spectrum Comparison...")
    
    def get_ke_temporal_spectrum(ds_target, lev_i, x_sl, y_sl):
        """Helper function to compute temporal KE spectrum at each spatial point, then average.
        Correctly computes: 0.5 * (|FFT(u)|^2 + |FFT(v)|^2)
        """
        from scipy import signal
        
        # Extract velocity components at specified depth and region
        u_data = ds_target[var_u].isel(lev=lev_i).sel({'x': x_sl, 'y': y_sl})
        v_data = ds_target[var_v].isel(lev=lev_i).sel({'x': x_sl, 'y': y_sl})
        
        # Get dimensions
        n_time = u_data.sizes['time']
        
        # Get time step for frequency calculation
        try:
            time_vals = ds_target.time.values
            if len(time_vals) > 1:
                delta_t = (time_vals[1] - time_vals[0])
                if isinstance(delta_t, np.timedelta64):
                    dt_days = float(delta_t) / np.timedelta64(1, 'D')
                else:
                    dt_days = delta_t.days + delta_t.seconds / 86400.0
            else:
                dt_days = 1.0
        except:
            dt_days = 1.0
        
        # Convert to years for sampling frequency
        dt_years = dt_days / 365.25
        fs = 1.0 / dt_years  # Sampling frequency in cycles/year
        
        # Convert to numpy arrays (time, y, x)
        u_np = np.nan_to_num(u_data.values, nan=0.0)
        v_np = np.nan_to_num(v_data.values, nan=0.0)
        
        # Apply linear detrending along time axis for each spatial point
        u_detrended = signal.detrend(u_np, axis=0, type='linear')
        v_detrended = signal.detrend(v_np, axis=0, type='linear')

        # Remove seasonal cycle (monthly climatology)
        steps_per_year = int(round(365.25 / dt_days))
        n_full = (n_time // steps_per_year) * steps_per_year
        if n_full >= steps_per_year:
            clim_u = u_detrended[:n_full].reshape(-1, steps_per_year, *u_detrended.shape[1:]).mean(axis=0)
            clim_v = v_detrended[:n_full].reshape(-1, steps_per_year, *v_detrended.shape[1:]).mean(axis=0)
            seasonal_u = np.tile(clim_u, (n_time // steps_per_year + 1, 1, 1))[:n_time]
            seasonal_v = np.tile(clim_v, (n_time // steps_per_year + 1, 1, 1))[:n_time]
            u_detrended = u_detrended - seasonal_u
            v_detrended = v_detrended - seasonal_v

        # Apply Hanning window along time axis
        window = signal.windows.hann(n_time)
        u_windowed = u_detrended * window[:, np.newaxis, np.newaxis]
        v_windowed = v_detrended * window[:, np.newaxis, np.newaxis]

        # Compute window correction factor
        window_correction = np.mean(window**2)

        # Compute FFT along time axis for each spatial point
        fft_u = np.fft.rfft(u_windowed, axis=0)
        fft_v = np.fft.rfft(v_windowed, axis=0)

        # Compute power spectrum for u and v at each spatial point
        power_u = np.abs(fft_u)**2
        power_v = np.abs(fft_v)**2

        # Compute KE power spectrum: 0.5 * (|FFT(u)|^2 + |FFT(v)|^2)
        power_ke = 0.5 * (power_u + power_v)

        # Convert to PSD for one-sided spectrum (rfft)
        # Multiply by 2 to account for negative frequencies
        psd_spatial = 2.0 * power_ke / (fs * n_time * window_correction)
        
        # DC component (frequency = 0) should not be multiplied by 2
        psd_spatial[0, :, :] = psd_spatial[0, :, :] / 2.0
        
        # If n_time is even, Nyquist frequency should not be multiplied by 2
        if n_time % 2 == 0:
            psd_spatial[-1, :, :] = psd_spatial[-1, :, :] / 2.0
        
        # Average PSD over all spatial points
        psd = np.nanmean(psd_spatial, axis=(1, 2))
        
        # Get frequency array
        freqs = np.fft.rfftfreq(n_time, d=dt_years)
        
        return freqs, psd
    
    # Calculate reference spectrum
    #print(f"Calculating temporal KE spectrum for {TRUTH_LABEL}...")
    freq_ref, psd_ref = get_ke_temporal_spectrum(
        dataset_truth, lev_idx, lon_slice, lat_slice
    )
    
    # Process all emulators
    ke_spectra = {}
    for label, emu_data in emulators_to_analyze.items():
        #print(f"Calculating temporal KE spectrum for {label}...")
        freq, psd = get_ke_temporal_spectrum(
            emu_data['ds'], lev_idx, lon_slice, lat_slice
        )
        ke_spectra[label] = {'freq': freq, 'psd': psd}
    
    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot truth
    ax.loglog(
        freq_ref[1:], psd_ref[1:], '-',  # Skip DC component
        label=f'{TRUTH_LABEL} (KE Anomalies)', 
        linewidth=2, color='black'
    )
    
    # Plot emulators
    for label, data in ke_spectra.items():
        color = emulators_to_analyze[label]['color']
        ax.loglog(
            data['freq'][1:], data['psd'][1:], '-',  # Skip DC component
            label=f'{label} (KE Anomalies)', 
            color=color
        )
    
    # Add ratio annotations at target frequencies (optional)
    if show_ratios and len(target_frequencies) > 0:
        for target_f in target_frequencies:
            idx_ref = np.argmin(np.abs(freq_ref - target_f))
            actual_f = freq_ref[idx_ref]
            psd_ref_value = psd_ref[idx_ref]
            
            period_years = 1.0 / actual_f if actual_f > 0 else np.inf
            
            #print(f"\nRatios at frequency {actual_f:.4f} cycles/year (period: {period_years:.2f} years):")
            #print(f"{TRUTH_LABEL} PSD = {psd_ref_value:.2e}")
            
            for i, (label, data) in enumerate(ke_spectra.items()):
                idx_emu = np.argmin(np.abs(data['freq'] - target_f))
                psd_emu_value = data['psd'][idx_emu]
                ratio = psd_emu_value / psd_ref_value if psd_ref_value > 0 else 0
                
                print(f"{label} PSD = {psd_emu_value:.2e}, Ratio = {ratio:.3f}")
                
                color = emulators_to_analyze[label]['color']
                x_pos = actual_f * 1.2
                y_pos = psd_emu_value * (1.5 ** i)
                
                ax.text(
                    x_pos, y_pos, 
                    f'{label}/{TRUTH_LABEL}={ratio:.3f}', 
                    fontsize=8, color=color,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color)
                )
            
            # Add vertical line at target frequency
            ax.axvline(x=actual_f, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    
    # Format axes
    ax.set_xlabel('Frequency (cycles/year)')
    ax.set_ylabel('Kinetic Energy Power Spectral Density')
    
    # Get depth value and spatial boundaries for title
    depth_value = dataset_truth.lev.values[lev_idx]
    
    # Format longitude range
    lon_start = lon_slice.start if lon_slice.start is not None else 0
    lon_stop = lon_slice.stop if lon_slice.stop is not None else 360
    if lon_stop > 180:
        lon_stop_display = f"{lon_stop - 360}°W" if lon_stop > 180 else f"{lon_stop}°E"
    else:
        lon_stop_display = f"{lon_stop}°E"
    lon_start_display = f"{lon_start}°E"
    
    # Format latitude range
    lat_start = lat_slice.start if lat_slice.start is not None else -90
    lat_stop = lat_slice.stop if lat_slice.stop is not None else 90
    lat_start_display = f"{abs(lat_start)}°S" if lat_start < 0 else f"{lat_start}°N"
    lat_stop_display = f"{abs(lat_stop)}°S" if lat_stop < 0 else f"{lat_stop}°N"
    
    ax.set_title(f'Temporal KE Power Spectrum at {depth_value}m\n' +
                 f'Region: {lon_start_display}-{lon_stop_display}, {lat_start_display}-{lat_stop_display}',
                 fontsize=11)
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# "_together" versions: Each emulator entry contains 'truth' and 'truth_label'
# emulators_dict format: {label: {'ds': ds, 'color': color, 'truth': ds_truth, 'truth_label': label}}
# =============================================================================

def plot_mean_timeseries_comparison_together(emulators_dict, variable_name, var_units, var_label, output_dir):
    """
    Time series comparison where each emulator uses its corresponding truth.
    emulators_dict: dict with 'ds', 'color', 'truth', 'truth_label' for each emulator
    """
    print(f"Generating Time Series for {var_label} (Multi-Resolution)...")
    num_plots = len(DEPTH_SLICES)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3.5 * num_plots), sharex=True, constrained_layout=True)
    if num_plots == 1: axes = [axes]
    
    for i, Slices in enumerate(DEPTH_SLICES):
        ax = axes[i]
        depth_slice = slice(Slices['min'], Slices['max'])
        stats_text = []
        
        for label, emu_data in emulators_dict.items():
            ds_truth = emu_data['truth']
            ds_emu = emu_data['ds']
            color = emu_data['color']
            truth_label = emu_data.get('truth_label', 'Truth')
            
            area = ds_truth['areacello']
            wet_mask = xr.where(np.isnan(ds_truth[variable_name].isel(time=0)), 0.0, 1.0)
            
            field_truth = ds_truth[variable_name].sel(lev=depth_slice)
            dz_truth = ds_truth['dz'].sel(lev=depth_slice)
            vol_weight_truth = dz_truth * area
            ts_truth = (field_truth * vol_weight_truth).sum(['x', 'y', 'lev']).compute() / \
                       (vol_weight_truth * wet_mask.sel(lev=depth_slice)).sum(['x', 'y', 'lev']).compute()
            
            # Convert cftime to numerical year values for matplotlib
            time_truth = np.array([t.year + (t.dayofyr - 1) / 365.25 for t in ts_truth.time.values])
            ax.plot(time_truth, ts_truth.values, label=f'{truth_label}', color=color, lw=2, linestyle='--', alpha=0.7)
            
            field_emu = ds_emu[variable_name].sel(lev=depth_slice)
            dz_emu = ds_emu['dz'].sel(lev=depth_slice)
            vol_weight_emu = dz_emu * area
            ts_emu = (field_emu * vol_weight_emu).sum(['x', 'y', 'lev']).compute() / \
                     (vol_weight_emu * wet_mask.sel(lev=depth_slice)).sum(['x', 'y', 'lev']).compute()
            
            min_len = min(len(ts_truth), len(ts_emu))
            r_val, _ = pearsonr(ts_truth.values[:min_len], ts_emu.values[:min_len])
            stats_text.append(f"{label} R²={r_val**2:.2f}")
            
            # Convert cftime to numerical year values for matplotlib
            time_emu = np.array([t.year + (t.dayofyr - 1) / 365.25 for t in ts_emu.time.values])
            ax.plot(time_emu, ts_emu.values, label=label, color=color, lw=2)
        
        ax.set_title(f"{Slices['title']}\n{' | '.join(stats_text)}", fontsize=12, loc='left')
        ax.set_ylabel(f"Global Mean {var_label} [{var_units}]", fontsize=10)
        ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))

    axes[-1].set_xlabel("Year", fontsize=12)
    fig.suptitle(f"Global Mean {var_label} Time Series (Multi-Resolution)", fontsize=14, y=1.02)
    plt.show()


def plot_variance_comparison_together(emulators_dict, variable_name, units, output_dir=None, colorbar_fix=False):
    """Variance comparison where each emulator uses its corresponding truth.
    Also plots the ground truth variance from the first emulator as the top row.
    """
    print(f"\nGenerating Variance Comparison for {variable_name} (Multi-Resolution)...")
    
    datasets = list(emulators_dict.keys())
    nrows = len(datasets) + 1  # +1 for ground truth row
    ncols = len(DEPTH_SLICES)
    
    # Get first emulator's truth data for the ground truth row
    first_label = datasets[0]
    first_emu_data = emulators_dict[first_label]
    ds_truth_first = first_emu_data['truth']
    truth_label = first_emu_data.get('truth_label', 'Truth')
    dz_truth_first = ds_truth_first['dz']
    wet_mask_first = (ds_truth_first[variable_name].isel(time=0) * 0 + 1).compute()
    truth_var_3d = ds_truth_first[variable_name].var('time')
    
    # Compute ground truth variance for each depth slice
    plot_data_truth = {}
    for Slices in DEPTH_SLICES:
        plot_data_truth[Slices['title']] = calculate_vertical_mean_simple(
            truth_var_3d, dz_truth_first, wet_mask_first, Slices['min'], Slices['max']
        ).compute()
    
    # Compute emulator variance
    plot_data = {}
    for label, emu_data in emulators_dict.items():
        ds_truth = emu_data['truth']
        dz_truth = ds_truth['dz']
        wet_mask = (ds_truth[variable_name].isel(time=0) * 0 + 1).compute()
        emu_var_3d = emu_data['ds'][variable_name].var('time')
        
        plot_data[label] = {}
        for Slices in DEPTH_SLICES:
            plot_data[label][Slices['title']] = calculate_vertical_mean_simple(
                emu_var_3d, dz_truth, wet_mask, Slices['min'], Slices['max']
            ).compute()
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows),
        subplot_kw={'projection': ccrs.Robinson(central_longitude=210)}, 
        gridspec_kw={'hspace': 0.1, 'wspace': 0.05})
    if nrows == 1: axes = np.expand_dims(axes, 0)
    if ncols == 1: axes = np.expand_dims(axes, 1)

    for j, Slices in enumerate(DEPTH_SLICES):
        key = Slices['title']
        # Include truth data in flat_data for normalization
        all_data = [plot_data_truth[key].values.flatten()] + [plot_data[ds][key].values.flatten() for ds in datasets]
        flat_data = np.concatenate(all_data)
        flat_data = flat_data[~np.isnan(flat_data)]
        vmax = np.percentile(flat_data, 95) if len(flat_data) > 0 else 1.0
        norm = colors.Normalize(vmin=0, vmax=vmax)
        
        # Row 0: Ground truth
        ax = axes[0, j]
        p_mesh = plot_data_truth[key].plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(), x='x', y='y',
            cmap=cmocean.cm.thermal, norm=norm, add_colorbar=False, rasterized=True)
        ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
        ax.coastlines(zorder=11)
        if j == 0: ax.text(-0.07, 0.55, truth_label, va='center', ha='right', rotation='vertical', 
                           transform=ax.transAxes, fontsize=14, fontweight='bold')
        ax.set_title(key, fontsize=14, fontweight='bold')
        
        # Rows 1+: Emulators
        for i, ds_name in enumerate(datasets):
            ax = axes[i + 1, j]  # offset by 1 for truth row
            p_mesh = plot_data[ds_name][key].plot.pcolormesh(
                ax=ax, transform=ccrs.PlateCarree(), x='x', y='y',
                cmap=cmocean.cm.thermal, norm=norm, add_colorbar=False, rasterized=True)
            ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
            ax.coastlines(zorder=11)
            if j == 0: ax.text(-0.07, 0.55, ds_name, va='center', ha='right', rotation='vertical', 
                               transform=ax.transAxes, fontsize=14, fontweight='bold')
            ax.set_title("")  # Title only on first row

        fig.colorbar(p_mesh, ax=axes[:, j], orientation='horizontal', shrink=0.7, pad=0.03)

    fig.suptitle(f"Temporal Variance: {variable_name} (Multi-Resolution)", fontsize=18, y=0.96)
    plt.show()


def plot_ohc_zonal_bias_comparison_together(emulators_dict, masks_list, colorbar_fix=False):
    """OHC Zonal Bias where each emulator uses its corresponding truth and masks."""
    units = 'J/m^2'
    
    for idx, (label, emu_data) in enumerate(emulators_dict.items()):
        ds_truth = emu_data['truth']
        masks = masks_list[idx]
        
        truth_field_2d = calculate_ohc_2d(ds_truth).mean('time').compute()
        emu_field_2d = calculate_ohc_2d(emu_data['ds']).mean('time').compute()
        dx_truth = ds_truth['dx']
        wet_mask_2d = (ds_truth['thetao'].isel(time=0, lev=0) * 0 + 1).compute()
        weight_2d = (ds_truth['dy'] * ds_truth['dz']).mean('x').compute()
        
        bias_profiles, rmse_vals, all_biases = {}, {}, []
        for basin in BASINS_TO_PLOT:
            prof_truth = calculate_zonal_mean_2d(truth_field_2d, masks[basin], dx_truth, wet_mask_2d).compute()
            prof_emu = calculate_zonal_mean_2d(emu_field_2d, masks[basin], dx_truth, wet_mask_2d).compute()
            bias = prof_emu - prof_truth
            bias_profiles[basin] = bias.drop('time')
            all_biases.append(bias.values.flatten())
            rmse_vals[basin] = np.sqrt(((bias**2) * weight_2d).sum(['y','lev']) / weight_2d.sum(['y','lev'])).item()

        flat_biases = np.concatenate(all_biases)
        max_abs = np.percentile(np.abs(flat_biases[~np.isnan(flat_biases)]), 98)
        if colorbar_fix: max_abs = 2.5e8
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True, constrained_layout=True)
        for i, basin in enumerate(BASINS_TO_PLOT):
            ax = axes.flatten()[i]
            p = bias_profiles[basin].plot.contourf(ax=ax, cmap=cmocean.cm.balance,
                norm=colors.Normalize(vmin=-max_abs, vmax=max_abs), add_colorbar=False,
                levels=np.linspace(-max_abs, max_abs, 25), yincrease=False, x='y', y='lev')
            ax.set_title(f"{basin} (RMSE: {rmse_vals[basin]:.3e} {units})", fontsize=16, loc='left')
            ax.set_facecolor('grey')
        fig.colorbar(p, ax=axes, shrink=0.8).set_label(f"Bias ({units})")
        fig.suptitle(f"Zonal Mean OHC Bias: {label} vs Truth", fontsize=20, y=1.03)
        plt.show()


def plot_zonal_bias_comparison_together(emulators_dict, variable_name, units, masks_list, output_dir, colorbar_fix=False):
    """Zonal Bias where each emulator uses its corresponding truth and masks."""
    for idx, (label, emu_data) in enumerate(emulators_dict.items()):
        ds_truth = emu_data['truth']
        masks = masks_list[idx]
        
        truth_field = ds_truth[variable_name].mean('time')
        emu_field = emu_data['ds'][variable_name].mean('time')
        dx_truth = ds_truth['dx']
        wet_mask = (ds_truth[variable_name].isel(time=0) * 0 + 1).compute()
        weight_2d = (ds_truth['dy'] * ds_truth['dz']).mean('x').compute()
        
        bias_profiles, rmse_vals, all_biases = {}, {}, []
        for basin in BASINS_TO_PLOT:
            prof_truth = calculate_zonal_mean(truth_field, masks[basin], dx_truth, wet_mask).compute()
            prof_emu = calculate_zonal_mean(emu_field, masks[basin], dx_truth, wet_mask).compute()
            bias = prof_emu - prof_truth
            bias_profiles[basin] = bias
            all_biases.append(bias.values.flatten())
            rmse_vals[basin] = np.sqrt(((bias**2) * weight_2d).sum(['y','lev']) / weight_2d.sum(['y','lev'])).item()

        flat_biases = np.concatenate(all_biases)
        max_abs = np.percentile(np.abs(flat_biases[~np.isnan(flat_biases)]), 98)
        if colorbar_fix: max_abs = 0.727
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True, constrained_layout=True)
        for i, basin in enumerate(BASINS_TO_PLOT):
            ax = axes.flatten()[i]
            p = bias_profiles[basin].plot.contourf(ax=ax, cmap=cmocean.cm.balance,
                norm=colors.Normalize(vmin=-max_abs, vmax=max_abs), add_colorbar=False,
                levels=np.linspace(-max_abs, max_abs, 25), yincrease=False, x='y', y='lev', add_labels=False)
            ax.set_title(f"{basin} (RMSE: {rmse_vals[basin]:.3f} {units})", fontsize=16, loc='left')
            ax.set_facecolor('grey')
        fig.colorbar(p, ax=axes, shrink=0.8).set_label(f"Bias ({units})")
        fig.suptitle(f"Zonal Mean Bias: {label} vs Truth ({variable_name})", fontsize=20, y=1.03)
        plt.show()


def plot_ohc_zonal_bias_ocean_comparison_together(emulators_dict, masks_list, colorbar_fix=False):
    """OHC Zonal Bias (Ocean layout) where each emulator uses its corresponding truth."""
    units = 'J/m^2'
    num_emulators = len(emulators_dict)
    
    all_bias_profiles = {}
    all_rmse_vals = {}
    all_biases_flat = []
    
    for idx, (label, emu_data) in enumerate(emulators_dict.items()):
        ds_truth = emu_data['truth']
        masks = masks_list[idx]
        
        truth_2d = calculate_ohc_2d(ds_truth).mean('time').compute()
        emu_2d = calculate_ohc_2d(emu_data['ds']).mean('time').compute()
        dx = ds_truth['dx']
        wet = (ds_truth['thetao'].isel(time=0, lev=0) * 0 + 1).compute()
        weight = (ds_truth['dy'] * ds_truth['dz']).mean('x').compute()
        
        all_bias_profiles[label] = {}
        all_rmse_vals[label] = {}
        for basin in BASINS_TO_PLOT:
            t_prof = calculate_zonal_mean_2d(truth_2d, masks[basin], dx, wet).compute()
            e_prof = calculate_zonal_mean_2d(emu_2d, masks[basin], dx, wet).compute()
            bias = e_prof - t_prof
            all_bias_profiles[label][basin] = bias.drop('time')
            all_biases_flat.append(bias.values.flatten())
            all_rmse_vals[label][basin] = np.sqrt(((bias**2)*weight).sum(['y','lev'])/weight.sum(['y','lev'])).item()

    flat = np.concatenate(all_biases_flat)
    max_abs = np.percentile(np.abs(flat[~np.isnan(flat)]), 98)
    if colorbar_fix: max_abs = 2.5e8

    fig, axes = plt.subplots(4, num_emulators, figsize=(7*num_emulators, 18), sharex=True, sharey=True, constrained_layout=True)
    if num_emulators == 1: axes = axes.reshape(-1, 1)
    
    for i, basin in enumerate(BASINS_TO_PLOT):
        for j, (label, bp) in enumerate(all_bias_profiles.items()):
            ax = axes[i, j]
            p = bp[basin].plot.contourf(ax=ax, cmap=cmocean.cm.balance,
                norm=colors.Normalize(vmin=-max_abs, vmax=max_abs), add_colorbar=False,
                levels=np.linspace(-max_abs, max_abs, 25), yincrease=False, x='y', y='lev')
            ax.set_facecolor('grey'); ax.set_xlabel(''); ax.set_ylabel('')
            if j == 0: ax.set_ylabel(f'{basin}\nDepth (m)', fontsize=12)
            ax.set_title(f"{label}\n(RMSE: {all_rmse_vals[label][basin]:.3e} {units})" if i == 0 else f"RMSE: {all_rmse_vals[label][basin]:.3e}", fontsize=11)
    for j in range(num_emulators): axes[3, j].set_xlabel('Latitude index', fontsize=12)
    fig.colorbar(p, ax=axes, orientation='horizontal', shrink=1, pad=0.02)
    fig.suptitle(f"Time-Averaged OHC Zonal Mean Bias: All Emulators (Multi-Resolution)", fontsize=18, y=1.03)
    plt.show()


def plot_zonal_bias_ocean_comparison_together(emulators_dict, variable_name, units, masks_list, output_dir, colorbar_fix=False):
    """Zonal Bias (Ocean layout) where each emulator uses its corresponding truth."""
    num_emulators = len(emulators_dict)
    
    all_bias_profiles = {}
    all_rmse_vals = {}
    all_biases_flat = []
    
    for idx, (label, emu_data) in enumerate(emulators_dict.items()):
        ds_truth = emu_data['truth']
        masks = masks_list[idx]
        
        truth_field = ds_truth[variable_name].mean('time')
        emu_field = emu_data['ds'][variable_name].mean('time')
        dx = ds_truth['dx']
        wet = (ds_truth[variable_name].isel(time=0) * 0 + 1).compute()
        weight = (ds_truth['dy'] * ds_truth['dz']).mean('x').compute()
        
        all_bias_profiles[label] = {}
        all_rmse_vals[label] = {}
        for basin in BASINS_TO_PLOT:
            t_prof = calculate_zonal_mean(truth_field, masks[basin], dx, wet).compute()
            e_prof = calculate_zonal_mean(emu_field, masks[basin], dx, wet).compute()
            bias = e_prof - t_prof
            all_bias_profiles[label][basin] = bias
            all_biases_flat.append(bias.values.flatten())
            all_rmse_vals[label][basin] = np.sqrt(((bias**2)*weight).sum(['y','lev'])/weight.sum(['y','lev'])).item()

    flat = np.concatenate(all_biases_flat)
    max_abs = np.percentile(np.abs(flat[~np.isnan(flat)]), 98)
    if colorbar_fix: max_abs = 0.514

    fig, axes = plt.subplots(4, num_emulators, figsize=(7*num_emulators, 18), sharex=True, sharey=True, constrained_layout=True)
    if num_emulators == 1: axes = axes.reshape(-1, 1)
    
    for i, basin in enumerate(BASINS_TO_PLOT):
        for j, (label, bp) in enumerate(all_bias_profiles.items()):
            ax = axes[i, j]
            p = bp[basin].plot.contourf(ax=ax, cmap=cmocean.cm.balance,
                norm=colors.Normalize(vmin=-max_abs, vmax=max_abs), add_colorbar=False,
                levels=np.linspace(-max_abs, max_abs, 25), yincrease=False, x='y', y='lev', add_labels=False)
            ax.set_facecolor('grey'); ax.set_xlabel(''); ax.set_ylabel('')
            if j == 0: ax.set_ylabel(f'{basin}\nDepth (m)', fontsize=12)
            ax.set_title(f"{label}\n(RMSE: {all_rmse_vals[label][basin]:.3f} {units})" if i == 0 else f"RMSE: {all_rmse_vals[label][basin]:.3f}", fontsize=11)
    for j in range(num_emulators): axes[3, j].set_xlabel('Latitude index', fontsize=12)
    fig.colorbar(p, ax=axes, orientation='horizontal', shrink=1, pad=0.02)
    fig.suptitle(f"Time-Averaged Zonal Mean Bias: All Emulators ({variable_name}, Multi-Resolution)", fontsize=18, y=1.03)
    plt.show()


def plot_snapshot_zonal_bias_ocean_comparison_together(emulators_dict, variable_name, units, masks_list, TIME_INDICES, output_dir=None, colorbar_fix=False):
    """Snapshot Zonal Bias (Ocean layout) where each emulator uses its corresponding truth."""
    num_emulators = len(emulators_dict)
    
    # Get first emulator's truth for time conversion
    first_label = list(emulators_dict.keys())[0]
    first_ds_truth = emulators_dict[first_label]['truth']
    
    for time_idx in TIME_INDICES:
        # Get actual date from time coordinate
        time_val = first_ds_truth.time.isel(time=time_idx).values
        if hasattr(time_val, 'year'):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]  # Fallback: take first 10 chars (YYYY-MM-DD)
        
        all_bias_profiles = {}
        all_rmse_vals = {}
        all_biases_flat = []
        
        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            ds_truth = emu_data['truth']
            masks = masks_list[idx]
            
            truth_snap = ds_truth[variable_name].isel(time=time_idx)
            emu_snap = emu_data['ds'][variable_name].isel(time=time_idx)
            dx = ds_truth['dx']
            wet = (ds_truth[variable_name].isel(time=0) * 0 + 1).compute()
            weight = (ds_truth['dy'] * ds_truth['dz']).mean('x').compute()
            
            all_bias_profiles[label] = {}
            all_rmse_vals[label] = {}
            for basin in BASINS_TO_PLOT:
                t_prof = calculate_zonal_mean(truth_snap, masks[basin], dx, wet).compute()
                e_prof = calculate_zonal_mean(emu_snap, masks[basin], dx, wet).compute()
                bias = e_prof - t_prof
                all_bias_profiles[label][basin] = bias
                all_biases_flat.append(bias.values.flatten())
                all_rmse_vals[label][basin] = np.sqrt(((bias**2)*weight).sum(['y','lev'])/weight.sum(['y','lev'])).item()

        flat = np.concatenate(all_biases_flat)
        max_abs = np.percentile(np.abs(flat[~np.isnan(flat)]), 98)
        if colorbar_fix: max_abs = 0.6

        fig, axes = plt.subplots(4, num_emulators, figsize=(7*num_emulators, 18), sharex=True, sharey=True, constrained_layout=True)
        if num_emulators == 1: axes = axes.reshape(-1, 1)
        
        for i, basin in enumerate(BASINS_TO_PLOT):
            for j, (label, bp) in enumerate(all_bias_profiles.items()):
                ax = axes[i, j]
                p = bp[basin].plot.contourf(ax=ax, cmap=cmocean.cm.balance,
                    norm=colors.Normalize(vmin=-max_abs, vmax=max_abs), add_colorbar=False,
                    levels=np.linspace(-max_abs, max_abs, 25), yincrease=False, x='y', y='lev', add_labels=False)
                ax.set_facecolor('grey'); ax.set_xlabel(''); ax.set_ylabel('')
                if j == 0: ax.set_ylabel(f'{basin}\nDepth (m)', fontsize=12)
                ax.set_title(f"{label}\n(RMSE: {all_rmse_vals[label][basin]:.3f} {units})" if i == 0 else f"RMSE: {all_rmse_vals[label][basin]:.3f}", fontsize=11)
        for j in range(num_emulators): axes[3, j].set_xlabel('Latitude index', fontsize=12)
        fig.colorbar(p, ax=axes, orientation='horizontal', shrink=1, pad=0.02)
        fig.suptitle(f"Snapshot Zonal Bias at {time_str} ({variable_name}, Multi-Resolution)", fontsize=18, y=1.03)
        plt.show()


def plot_snapshot_ohc_zonal_bias_ocean_comparison_together(emulators_dict, masks_list, TIME_INDICES, colorbar_fix=False):
    """Snapshot OHC Zonal Bias (Ocean layout) where each emulator uses its corresponding truth."""
    units = 'J/m^2'
    num_emulators = len(emulators_dict)
    
    # Get first emulator's truth for time conversion
    first_label = list(emulators_dict.keys())[0]
    first_ds_truth = emulators_dict[first_label]['truth']
    
    for time_idx in TIME_INDICES:
        # Get actual date from time coordinate
        time_val = first_ds_truth.time.isel(time=time_idx).values
        if hasattr(time_val, 'year'):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]  # Fallback: take first 10 chars (YYYY-MM-DD)
        
        all_bias_profiles = {}
        all_rmse_vals = {}
        all_biases_flat = []
        
        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            ds_truth = emu_data['truth']
            masks = masks_list[idx]
            
            truth_snap = calculate_ohc_2d(ds_truth).isel(time=time_idx).compute()
            emu_snap = calculate_ohc_2d(emu_data['ds']).isel(time=time_idx).compute()
            dx = ds_truth['dx']
            wet = (ds_truth['thetao'].isel(time=0, lev=0) * 0 + 1).compute()
            weight = (ds_truth['dy'] * ds_truth['dz']).mean('x').compute()
            
            all_bias_profiles[label] = {}
            all_rmse_vals[label] = {}
            for basin in BASINS_TO_PLOT:
                t_prof = calculate_zonal_mean_2d(truth_snap, masks[basin], dx, wet).compute()
                e_prof = calculate_zonal_mean_2d(emu_snap, masks[basin], dx, wet).compute()
                bias = e_prof - t_prof
                all_bias_profiles[label][basin] = bias
                all_biases_flat.append(bias.values.flatten())
                all_rmse_vals[label][basin] = np.sqrt(((bias**2)*weight).sum(['y','lev'])/weight.sum(['y','lev'])).item()

        flat = np.concatenate(all_biases_flat)
        max_abs = np.percentile(np.abs(flat[~np.isnan(flat)]), 98)
        if colorbar_fix: max_abs = 4.2e8

        fig, axes = plt.subplots(4, num_emulators, figsize=(7*num_emulators, 18), sharex=True, sharey=True, constrained_layout=True)
        if num_emulators == 1: axes = axes.reshape(-1, 1)
        
        for i, basin in enumerate(BASINS_TO_PLOT):
            for j, (label, bp) in enumerate(all_bias_profiles.items()):
                ax = axes[i, j]
                p = bp[basin].plot.contourf(ax=ax, cmap=cmocean.cm.balance,
                    norm=colors.Normalize(vmin=-max_abs, vmax=max_abs), add_colorbar=False,
                    levels=np.linspace(-max_abs, max_abs, 25), yincrease=False, x='y', y='lev')
                ax.set_facecolor('grey'); ax.set_xlabel(''); ax.set_ylabel('')
                if j == 0: ax.set_ylabel(f'{basin}\nDepth (m)', fontsize=12)
                ax.set_title(f"{label}\n(RMSE: {all_rmse_vals[label][basin]:.3e} {units})" if i == 0 else f"RMSE: {all_rmse_vals[label][basin]:.3e}", fontsize=11)
        for j in range(num_emulators): axes[3, j].set_xlabel('Latitude index', fontsize=12)
        fig.colorbar(p, ax=axes, orientation='horizontal', shrink=1, pad=0.02)
        fig.suptitle(f"Snapshot OHC Zonal Bias at {time_str} (Multi-Resolution)", fontsize=18, y=1.03)
        plt.show()


def plot_ohc_bias_comparison_together(emulators_dict, colorbar_fix=False):
    """OHC Bias comparison (Multi-Resolution) in one figure."""
    print(f"\nGenerating OHC Bias Comparison (Multi-Resolution)...")
    units = 'ZJ'
    
    datasets = list(emulators_dict.keys())
    nrows = len(datasets)
    ncols = len(DEPTH_SLICES)
    
    all_biases = {}
    all_rmses = {}
    
    for label, emu_data in emulators_dict.items():
        ds_truth = emu_data['truth']
        dz = ds_truth['dz']
        wet = (ds_truth['thetao'].isel(time=0) * 0 + 1).compute()
        area = ds_truth['areacello']
        ds_emu = emu_data['ds']
        
        all_biases[label] = {}
        all_rmses[label] = {}
        
        for Slices in DEPTH_SLICES:
            key = Slices['title']
            truth_map = calculate_ohc_map(ds_truth, dz, wet, Slices['min'], Slices['max']).compute()
            emu_map = calculate_ohc_map(ds_emu, dz, wet, Slices['min'], Slices['max']).compute()
            all_biases[label][key] = emu_map - truth_map
            all_rmses[label][key] = rmse_spatial(truth_map, emu_map, area)
            
    limits = {}
    for Slices in DEPTH_SLICES:
        key = Slices['title']
        flat_list = [all_biases[label][key].values.flatten() for label in datasets]
        flat = np.concatenate(flat_list)
        flat = flat[~np.isnan(flat)]
        limit = np.percentile(np.abs(flat), 99) if len(flat) > 0 else 1.0
        
        if colorbar_fix:
            limit = {'Surface (0-700m)': 3e-12, 'Intermediate (700-2000m)': 2.5e-12, 'Deep (2000-7000m)': 2e-12}.get(key, limit)
        limits[key] = limit

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows),
        subplot_kw={'projection': ccrs.Robinson(central_longitude=210)}, gridspec_kw={'hspace': 0.15, 'wspace': 0.05})
    if nrows == 1: axes = np.expand_dims(axes, 0)
    if ncols == 1: axes = np.expand_dims(axes, 1)
    
    for j, Slices in enumerate(DEPTH_SLICES):
        key = Slices['title']
        limit = limits[key]
        norm = colors.Normalize(vmin=-limit, vmax=limit)
        
        for i, label in enumerate(datasets):
            ax = axes[i, j]
            p = all_biases[label][key].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='x', y='y',
                cmap=cmocean.cm.balance, norm=norm, add_colorbar=False, rasterized=True, add_labels=False)
            ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
            ax.coastlines(zorder=11)
            
            if i == 0: 
                ax.set_title(key, fontsize=14, fontweight='bold', y=1.12)
            
            rmse_str = f"RMSE: {all_rmses[label][key]:.2e} {units}"
            ax.text(0.5, 1.02, rmse_str, transform=ax.transAxes, fontsize=10, ha='center', va='bottom')
            
            if j == 0: ax.text(-0.07, 0.55, label, va='center', ha='right', rotation='vertical',
                              transform=ax.transAxes, fontsize=12, fontweight='bold')
        
        fig.colorbar(p, ax=axes[:, j], orientation='horizontal', shrink=0.7, pad=0.03, label=f"Bias ({units})")
        
    fig.suptitle(f"Time-Averaged OHC Bias Comparison (Multi-Resolution)", fontsize=18, y=0.98)
    plt.show()


def plot_temperature_bias_comparison_together(emulators_dict, colorbar_fix=False):
    """Temperature Bias comparison (Multi-Resolution) in one figure."""
    print(f"\nGenerating Temperature Bias Comparison (Multi-Resolution)...")
    VARIABLE_TO_PLOT = 'thetao'
    units = '$^\\circ$C'
    
    datasets = list(emulators_dict.keys())
    nrows = len(datasets)
    ncols = len(DEPTH_LAYERS)
    
    all_biases = {}
    all_rmses = {}
    
    for label, emu_data in emulators_dict.items():
        ds_truth = emu_data['truth']
        dz = ds_truth['dz']
        wet = (ds_truth[VARIABLE_TO_PLOT].isel(time=0) * 0 + 1).compute()
        area = ds_truth['areacello']
        ds_emu = emu_data['ds']
        
        all_biases[label] = {}
        all_rmses[label] = {}
        
        for Slices in DEPTH_LAYERS:
            key = Slices['title']
            idx_layer = Slices['index']
            truth_map = calculate_temperature_map_layer(ds_truth, VARIABLE_TO_PLOT, dz, wet, idx_layer).compute()
            emu_map = calculate_temperature_map_layer(ds_emu, VARIABLE_TO_PLOT, dz, wet, idx_layer).compute()
            all_biases[label][key] = emu_map - truth_map
            all_rmses[label][key] = rmse_spatial(truth_map, emu_map, area)

    limits = {}
    for i, Slices in enumerate(DEPTH_LAYERS):
        key = Slices['title']
        flat_list = [all_biases[label][key].values.flatten() for label in datasets]
        flat = np.concatenate(flat_list)
        flat = flat[~np.isnan(flat)]
        limit = np.percentile(np.abs(flat), 99) if len(flat) > 0 else 1.0
        
        if colorbar_fix:
            limit = {0:2, 1:0.7, 2:0.2, 3:0.1}.get(i, limit)
        limits[key] = limit

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows),
        subplot_kw={'projection': ccrs.Robinson(central_longitude=210)}, gridspec_kw={'hspace': 0.15, 'wspace': 0.05})
    if nrows == 1: axes = np.expand_dims(axes, 0)
    if ncols == 1: axes = np.expand_dims(axes, 1)
    
    for j, Slices in enumerate(DEPTH_LAYERS):
        key = Slices['title']
        limit = limits[key]
        norm = colors.Normalize(vmin=-limit, vmax=limit)
        
        for i, label in enumerate(datasets):
            ax = axes[i, j]
            p = all_biases[label][key].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='x', y='y',
                cmap=cmocean.cm.balance, norm=norm, add_colorbar=False, rasterized=True, add_labels=False)
            ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
            ax.coastlines(zorder=11)
            
            if i == 0: 
                ax.set_title(key, fontsize=14, fontweight='bold', y=1.12)
            
            rmse_str = f"RMSE: {all_rmses[label][key]:.2e} {units}"
            ax.text(0.5, 1.02, rmse_str, transform=ax.transAxes, fontsize=10, ha='center', va='bottom')
            
            if j == 0: ax.text(-0.07, 0.55, label, va='center', ha='right', rotation='vertical',
                              transform=ax.transAxes, fontsize=12, fontweight='bold')
        
        fig.colorbar(p, ax=axes[:, j], orientation='horizontal', shrink=0.7, pad=0.03, label=f"Bias ({units})")
        
    fig.suptitle(f"Time-Averaged Temperature Bias Comparison (Multi-Resolution)", fontsize=18, y=0.98)
    plt.show()


def plot_ohc_snapshot_bias_comparison_together(emulators_dict, time_indices, colorbar_fix=False, area=True):
    """OHC Snapshot Bias where each emulator uses its corresponding truth."""
    units = 'ZJ'
    num_emulators = len(emulators_dict)
    
    # Get first emulator's truth for time conversion
    first_label = list(emulators_dict.keys())[0]
    first_ds_truth = emulators_dict[first_label]['truth']
    
    for time_idx in time_indices:
        # Get actual date from time coordinate
        time_val = first_ds_truth.time.isel(time=time_idx).values
        if hasattr(time_val, 'year'):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]  # Fallback: take first 10 chars (YYYY-MM-DD)
        
        all_bias_maps = {}
        all_rmse_vals = {}
        shared_limits = {}
        
        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            ds_truth = emu_data['truth']
            dz = ds_truth['dz']
            wet = (ds_truth['thetao'].isel(time=0) * 0 + 1).compute()
            area_weight = ds_truth['areacello']
            
            for j, Slices in enumerate(DEPTH_SLICES):
                key = Slices['title']
                truth_map = calculate_ohc_snapshot_map(ds_truth, dz, wet, Slices['min'], Slices['max'], time_idx).compute()
                emu_map = calculate_ohc_snapshot_map(emu_data['ds'], dz, wet, Slices['min'], Slices['max'], time_idx).compute()
                bias = emu_map - truth_map
                all_bias_maps[(label, key)] = bias
                all_rmse_vals[(label, key)] = rmse_spatial(truth_map, emu_map, area_weight)
                
                if idx == 0:
                    flat = bias.values.flatten()
                    flat = flat[~np.isnan(flat)]
                    shared_limits[key] = np.percentile(np.abs(flat), 99.5) if len(flat) > 0 else 1.0

        emulator_labels = list(emulators_dict.keys())
        fig, axes = plt.subplots(num_emulators, len(DEPTH_SLICES), figsize=(6*len(DEPTH_SLICES), 3.5*num_emulators),
            subplot_kw={'projection': ccrs.Robinson(central_longitude=210)}, gridspec_kw={'hspace': 0.15, 'wspace': 0.05})
        if num_emulators == 1: axes = axes.reshape(1, -1)
        if len(DEPTH_SLICES) == 1: axes = axes.reshape(-1, 1)
        
        for j, Slices in enumerate(DEPTH_SLICES):
            key = Slices['title']
            limit = shared_limits[key]
            if colorbar_fix: limit = {0:3e-12, 1:2.5e-12, 2:2e-12}.get(j, limit)
            norm = colors.Normalize(vmin=-limit, vmax=limit)
            
            for i, label in enumerate(emulator_labels):
                ax = axes[i, j]
                bias = all_bias_maps[(label, key)]
                
                p = bias.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='x', y='y',
                    cmap=cmocean.cm.balance, norm=norm, add_colorbar=False, rasterized=True, add_labels=False)
                
                if i == 0: 
                    ax.set_title(key, fontsize=14, fontweight='bold', y=1.12)
                
                rmse_str = f"RMSE: {all_rmse_vals[(label, key)]:.2e} {units}"
                ax.text(0.5, 1.02, rmse_str, transform=ax.transAxes, fontsize=10, ha='center', va='bottom')

                ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
                ax.coastlines(zorder=11)
                if j == 0: ax.text(-0.07, 0.55, label, transform=ax.transAxes, fontsize=12, va='center', ha='right', rotation='vertical', fontweight='bold')

            fig.colorbar(p, ax=axes[:, j], orientation='horizontal', shrink=0.7, pad=0.03, label=f"Bias ({units})")
        
        fig.suptitle(f"OHC Snapshot Bias at {time_str} (Multi-Resolution)", fontsize=18, y=0.98)
        plt.show()


def plot_temperature_snapshot_bias_comparison_together(emulators_dict, time_indices, colorbar_fix=False):
    """Temperature Snapshot Bias where each emulator uses its corresponding truth."""
    VARIABLE_TO_PLOT = 'thetao'
    units = '$^\\circ$C'
    num_emulators = len(emulators_dict)
    
    # Get first emulator's truth for time conversion
    first_label = list(emulators_dict.keys())[0]
    first_ds_truth = emulators_dict[first_label]['truth']
    
    for time_idx in time_indices:
        # Get actual date from time coordinate
        time_val = first_ds_truth.time.isel(time=time_idx).values
        if hasattr(time_val, 'year'):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]  # Fallback: take first 10 chars (YYYY-MM-DD)
        
        all_bias_maps = {}
        all_rmse_vals = {}
        shared_limits = {}
        
        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            ds_truth = emu_data['truth']
            dz = ds_truth['dz']
            wet = (ds_truth[VARIABLE_TO_PLOT].isel(time=0) * 0 + 1).compute()
            area = ds_truth['areacello']
            
            for layer in DEPTH_LAYERS:
                key = layer['title']
                truth_map = calculate_snapshot_map_SpecificLayer(ds_truth, VARIABLE_TO_PLOT, dz, wet, layer['index'], time_idx).compute()
                emu_map = calculate_snapshot_map_SpecificLayer(emu_data['ds'], VARIABLE_TO_PLOT, dz, wet, layer['index'], time_idx).compute()
                bias = emu_map - truth_map
                all_bias_maps[(label, key)] = bias
                all_rmse_vals[(label, key)] = rmse_spatial(truth_map, emu_map, area)
                
                if idx == 0:
                    flat = bias.values.flatten()
                    flat = flat[~np.isnan(flat)]
                    shared_limits[key] = np.percentile(np.abs(flat), 99) if len(flat) > 0 else 1.0
                    
        emulator_labels = list(emulators_dict.keys())
        fig, axes = plt.subplots(num_emulators, len(DEPTH_LAYERS), figsize=(6*len(DEPTH_LAYERS), 3.5*num_emulators),
            subplot_kw={'projection': ccrs.Robinson(central_longitude=210)}, gridspec_kw={'hspace': 0.15, 'wspace': 0.05})
        if num_emulators == 1: axes = axes.reshape(1, -1)
        if len(DEPTH_LAYERS) == 1: axes = axes.reshape(-1, 1)
        
        fixed_limits = {'2.5m':2, '700m':0.7, '2000m':0.2, '3000m':0.1}
        
        for j, layer in enumerate(DEPTH_LAYERS):
            key = layer['title']
            limit = fixed_limits.get(key, shared_limits[key]) if colorbar_fix else shared_limits[key]
            norm = colors.Normalize(vmin=-limit, vmax=limit)
            
            for i, label in enumerate(emulator_labels):
                ax = axes[i, j]
                bias = all_bias_maps[(label, key)]
                
                p = bias.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='x', y='y',
                    cmap=cmocean.cm.balance, norm=norm, add_colorbar=False, rasterized=True, add_labels=False)
                
                if i == 0: 
                    ax.set_title(key, fontsize=14, fontweight='bold', y=1.12)
                
                rmse_str = f"RMSE: {all_rmse_vals[(label, key)]:.2e} {units}"
                ax.text(0.5, 1.02, rmse_str, transform=ax.transAxes, fontsize=10, ha='center', va='bottom')

                ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
                ax.coastlines(zorder=11)
                if j == 0: ax.text(-0.07, 0.55, label, transform=ax.transAxes, fontsize=12, va='center', ha='right', rotation='vertical', fontweight='bold')
        
            fig.colorbar(p, ax=axes[:, j], orientation='horizontal', shrink=0.7, pad=0.03, label=f"Bias ({units})")
        
        fig.suptitle(f"Temperature Snapshot Bias at {time_str} (Multi-Resolution)", fontsize=18, y=0.98)
        plt.show()


# ============================================================================
# MULTI-RESOLUTION _TOGETHER FUNCTIONS
# ============================================================================

def plot_ohc_mean_comparison_together(emulators_dict, variable_name=None, units=None, colorbar_fix=False):
    """OHC Mean Comparison: Ground truth (first emulator) + all emulators on one figure.
    Rows: Truth, Emulator1, Emulator2, ...
    Cols: Depth Slices
    """
    print("\nGenerating OHC Mean Comparison (Multi-Resolution)...")
    
    # Get first emulator's truth
    first_label = list(emulators_dict.keys())[0]
    ds_truth = emulators_dict[first_label]['truth']
    truth_label = emulators_dict[first_label].get('truth_label', 'Truth')
    dz_truth = ds_truth['dz']
    wet_mask = (ds_truth['thetao'].isel(time=0) * 0 + 1).compute()
    
    # Compute truth maps
    truth_maps = {}
    for Slices in DEPTH_SLICES:
        key = Slices['title']
        truth_maps[key] = calculate_ohc_map(ds_truth, dz_truth, wet_mask, Slices['min'], Slices['max']).compute()
    
    # Compute emulator maps
    emu_maps = {}
    for label, emu_data in emulators_dict.items():
        ds_emu = emu_data['ds']
        ds_emu_truth = emu_data['truth']
        dz_emu = ds_emu_truth['dz']
        wet_emu = (ds_emu_truth['thetao'].isel(time=0) * 0 + 1).compute()
        emu_maps[label] = {}
        for Slices in DEPTH_SLICES:
            key = Slices['title']
            emu_maps[label][key] = calculate_ohc_map(ds_emu, dz_emu, wet_emu, Slices['min'], Slices['max']).compute()
    
    # Plotting
    datasets = [truth_label] + list(emulators_dict.keys())
    nrows = len(datasets)
    ncols = len(DEPTH_SLICES)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows),
        subplot_kw={'projection': ccrs.Robinson(central_longitude=210)}, gridspec_kw={'hspace': 0.15, 'wspace': 0.05})
    if nrows == 1: axes = np.expand_dims(axes, 0)
    if ncols == 1: axes = np.expand_dims(axes, 1)
    
    for j, Slices in enumerate(DEPTH_SLICES):
        key = Slices['title']
        data_truth = truth_maps[key].values
        valid_data = data_truth[~np.isnan(data_truth)]
        vmin = np.percentile(valid_data, .5)
        vmax = np.percentile(valid_data, 99.5)
        
        if colorbar_fix:
            if key == 'Surface (0-700m)': vmin, vmax = 0, 5e-11
            elif key == 'Intermediate (700-2000m)': vmin, vmax = 0, 5e-11
            elif key == 'Deep (2000-7000m)': vmin, vmax = 0, 4e-11
        
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        for i, ds_name in enumerate(datasets):
            ax = axes[i, j]
            if ds_name == truth_label:
                data_to_plot = truth_maps[key]
            else:
                data_to_plot = emu_maps[ds_name][key]
            
            p = data_to_plot.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='x', y='y',
                cmap=cmocean.cm.thermal, norm=norm, add_colorbar=False, rasterized=True)
            ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
            ax.coastlines(zorder=11)
            if i == 0: ax.set_title(key, fontsize=14, fontweight='bold')
            else: ax.set_title("")
            if j == 0: ax.text(-0.07, 0.55, ds_name, va='center', ha='right', rotation='vertical',
                              transform=ax.transAxes, fontsize=12, fontweight='bold')
        
        # Colorbar per column
        fig.colorbar(p, ax=axes[:, j], orientation='horizontal', shrink=0.7, pad=0.03)
    
    fig.suptitle("8-yr-Mean OHC Comparison (Multi-Resolution)", fontsize=18, y=0.96)
    plt.show()


def plot_temperature_mean_comparison_together(emulators_dict, colorbar_fix=False):
    """Temperature Mean Comparison: Ground truth (first emulator) + all emulators on one figure."""
    print("\nGenerating Temperature Mean Comparison (Multi-Resolution)...")
    VARIABLE_TO_PLOT = 'thetao'
    
    first_label = list(emulators_dict.keys())[0]
    ds_truth = emulators_dict[first_label]['truth']
    truth_label = emulators_dict[first_label].get('truth_label', 'Truth')
    dz_truth = ds_truth['dz']
    wet_mask = (ds_truth[VARIABLE_TO_PLOT].isel(time=0) * 0 + 1).compute()
    
    # Compute truth maps
    truth_maps = {}
    for Slices in DEPTH_LAYERS:
        key = Slices['title']
        truth_maps[key] = calculate_temperature_map_layer(ds_truth, VARIABLE_TO_PLOT, dz_truth, wet_mask, Slices['index']).compute()
    
    # Compute emulator maps
    emu_maps = {}
    for label, emu_data in emulators_dict.items():
        ds_emu = emu_data['ds']
        ds_emu_truth = emu_data['truth']
        dz_emu = ds_emu_truth['dz']
        wet_emu = (ds_emu_truth[VARIABLE_TO_PLOT].isel(time=0) * 0 + 1).compute()
        emu_maps[label] = {}
        for Slices in DEPTH_LAYERS:
            key = Slices['title']
            emu_maps[label][key] = calculate_temperature_map_layer(ds_emu, VARIABLE_TO_PLOT, dz_emu, wet_emu, Slices['index']).compute()
    
    # Plotting
    datasets = [truth_label] + list(emulators_dict.keys())
    nrows = len(datasets)
    ncols = len(DEPTH_LAYERS)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows),
        subplot_kw={'projection': ccrs.Robinson(central_longitude=210)}, gridspec_kw={'hspace': 0.15, 'wspace': 0.05})
    if nrows == 1: axes = np.expand_dims(axes, 0)
    if ncols == 1: axes = np.expand_dims(axes, 1)
    
    for j, Slices in enumerate(DEPTH_LAYERS):
        key = Slices['title']
        data_truth = truth_maps[key].values
        valid_data = data_truth[~np.isnan(data_truth)]
        vmin = np.percentile(valid_data, .5)
        vmax = np.percentile(valid_data, 99.5)
        
        if colorbar_fix:
            fixed = {0: (0, 30), 1: (0, 15), 2: (0, 8), 3: (0, 4)}
            vmin, vmax = fixed.get(j, (vmin, vmax))
        
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        for i, ds_name in enumerate(datasets):
            ax = axes[i, j]
            if ds_name == truth_label:
                data_to_plot = truth_maps[key]
            else:
                data_to_plot = emu_maps[ds_name][key]
            
            p = data_to_plot.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='x', y='y',
                cmap=cmocean.cm.thermal, norm=norm, add_colorbar=False, rasterized=True)
            ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
            ax.coastlines(zorder=11)
            if i == 0: ax.set_title(key, fontsize=14, fontweight='bold')
            else: ax.set_title("")
            if j == 0: ax.text(-0.07, 0.55, ds_name, va='center', ha='right', rotation='vertical',
                              transform=ax.transAxes, fontsize=12, fontweight='bold')
        
        fig.colorbar(p, ax=axes[:, j], orientation='horizontal', shrink=0.7, pad=0.03)
    
    fig.suptitle("8-yr-Mean Temperature Comparison (Multi-Resolution)", fontsize=18, y=0.96)
    plt.show()


def plot_zonal_mean_comparison_together(emulators_dict, variable_name, units, masks_list, colorbar_fix=False):
    """Zonal Mean Comparison: Ground truth (first emulator) + all emulators on one figure.
    Layout: 4 basins x (1 truth + N emulators) columns
    """
    print(f"\nGenerating Zonal Mean Comparison for {variable_name} (Multi-Resolution)...")
    
    first_label = list(emulators_dict.keys())[0]
    ds_truth = emulators_dict[first_label]['truth']
    truth_label = emulators_dict[first_label].get('truth_label', 'Truth')
    masks_truth = masks_list[0]
    dx_truth = ds_truth['dx']
    wet_mask = (ds_truth[variable_name].isel(time=0, drop=True) * 0 + 1).compute()
    
    # Compute truth profiles
    truth_field_mean = ds_truth[variable_name].mean(dim='time')
    truth_profiles = {}
    for basin in BASINS_TO_PLOT:
        truth_profiles[basin] = calculate_zonal_mean(truth_field_mean, masks_truth[basin], dx_truth, wet_mask).compute()
    
    # Compute emulator profiles
    emu_profiles = {}
    for idx, (label, emu_data) in enumerate(emulators_dict.items()):
        ds_emu = emu_data['ds']
        ds_emu_truth = emu_data['truth']
        masks_emu = masks_list[idx]
        dx_emu = ds_emu_truth['dx']
        wet_emu = (ds_emu_truth[variable_name].isel(time=0, drop=True) * 0 + 1).compute()
        emu_field_mean = ds_emu[variable_name].mean('time')
        emu_profiles[label] = {}
        for basin in BASINS_TO_PLOT:
            emu_profiles[label][basin] = calculate_zonal_mean(emu_field_mean, masks_emu[basin], dx_emu, wet_emu).compute()
    
    # Plotting
    num_cols = 1 + len(emulators_dict)
    fig, axes = plt.subplots(4, num_cols, figsize=(6*num_cols, 14), sharex=True, sharey=True,
                             gridspec_kw={'hspace': 0.12, 'wspace': 0.08})
    if num_cols == 1: axes = axes.reshape(-1, 1)
    
    # Determine colorbar range
    all_data = np.concatenate([truth_profiles[basin].values.flatten() for basin in BASINS_TO_PLOT])
    valid_data = all_data[~np.isnan(all_data)]
    vmin = np.percentile(valid_data, .5)
    vmax = np.percentile(valid_data, 98)
    if colorbar_fix: vmin, vmax = 0.54, 27.25
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cmocean.cm.thermal
    
    for i, basin in enumerate(BASINS_TO_PLOT):
        # Column 0: Truth
        ax = axes[i, 0]
        p = truth_profiles[basin].plot.contourf(ax=ax, cmap=cmap, norm=norm, add_colorbar=False,
            levels=np.linspace(vmin, vmax, 25), yincrease=False, x='y', y='lev', add_labels=False)
        ax.set_facecolor('grey')
        ax.set_ylabel(f'{basin}\nDepth (m)', fontsize=11)
        ax.set_xlabel('')
        if i == 0: ax.set_title(truth_label, fontsize=12, fontweight='bold')
        
        # Columns 1+: Emulators
        for j, (label, profiles) in enumerate(emu_profiles.items(), start=1):
            ax = axes[i, j]
            profiles[basin].plot.contourf(ax=ax, cmap=cmap, norm=norm, add_colorbar=False,
                levels=np.linspace(vmin, vmax, 25), yincrease=False, x='y', y='lev', add_labels=False)
            ax.set_facecolor('grey')
            ax.set_ylabel(''); ax.set_xlabel('')
            if i == 0: ax.set_title(label, fontsize=12, fontweight='bold')
    
    for j in range(num_cols): axes[3, j].set_xlabel('Latitude index', fontsize=11)
    fig.colorbar(p, ax=axes, orientation='horizontal', shrink=0.6, pad=0.08)
    fig.suptitle(f"Time-Averaged Zonal Mean {variable_name} (Multi-Resolution)", fontsize=16, y=0.935)
    plt.show()


def plot_ohc_zonal_mean_comparison_together(emulators_dict, masks_list, variable_name=None, units=None, colorbar_fix=False):
    """OHC Zonal Mean Comparison: Ground truth (first emulator) + all emulators."""
    print("\nGenerating OHC Zonal Mean Comparison (Multi-Resolution)...")
    units = 'J/m^3'
    
    first_label = list(emulators_dict.keys())[0]
    ds_truth = emulators_dict[first_label]['truth']
    truth_label = emulators_dict[first_label].get('truth_label', 'Truth')
    masks_truth = masks_list[0]
    dx_truth = ds_truth['dx']
    wet_mask = (ds_truth['thetao'].isel(time=0, drop=True) * 0 + 1).compute()
    
    truth_field_mean = calculate_ohc_2d(ds_truth).mean('time').compute()
    truth_profiles = {}
    for basin in BASINS_TO_PLOT:
        truth_profiles[basin] = calculate_zonal_mean_2d(truth_field_mean, masks_truth[basin], dx_truth, wet_mask).compute()
    
    emu_profiles = {}
    for idx, (label, emu_data) in enumerate(emulators_dict.items()):
        ds_emu = emu_data['ds']
        ds_emu_truth = emu_data['truth']
        masks_emu = masks_list[idx]
        dx_emu = ds_emu_truth['dx']
        wet_emu = (ds_emu_truth['thetao'].isel(time=0, drop=True) * 0 + 1).compute()
        emu_field_mean = calculate_ohc_2d(ds_emu).mean('time').compute()
        emu_profiles[label] = {}
        for basin in BASINS_TO_PLOT:
            emu_profiles[label][basin] = calculate_zonal_mean_2d(emu_field_mean, masks_emu[basin], dx_emu, wet_emu).compute()
    
    num_cols = 1 + len(emulators_dict)
    fig, axes = plt.subplots(4, num_cols, figsize=(6*num_cols, 14), sharex=True, sharey=True,
                             gridspec_kw={'hspace': 0.12, 'wspace': 0.08})
    if num_cols == 1: axes = axes.reshape(-1, 1)
    
    all_data = np.concatenate([truth_profiles[basin].values.flatten() for basin in BASINS_TO_PLOT])
    valid_data = all_data[~np.isnan(all_data)]
    vmin = np.percentile(valid_data, .5)
    vmax = np.percentile(valid_data, 98)
    if colorbar_fix: vmin, vmax = 0, 16e9
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cmocean.cm.thermal
    
    for i, basin in enumerate(BASINS_TO_PLOT):
        ax = axes[i, 0]
        p = truth_profiles[basin].plot.contourf(ax=ax, cmap=cmap, norm=norm, add_colorbar=False,
            levels=np.linspace(vmin, vmax, 25), yincrease=False, x='y', y='lev', add_labels=False)
        ax.set_facecolor('grey')
        ax.set_ylabel(f'{basin}\nDepth (m)', fontsize=11)
        ax.set_xlabel('')
        if i == 0: ax.set_title(truth_label, fontsize=12, fontweight='bold')
        
        for j, (label, profiles) in enumerate(emu_profiles.items(), start=1):
            ax = axes[i, j]
            profiles[basin].plot.contourf(ax=ax, cmap=cmap, norm=norm, add_colorbar=False,
                levels=np.linspace(vmin, vmax, 25), yincrease=False, x='y', y='lev', add_labels=False)
            ax.set_facecolor('grey')
            ax.set_ylabel(''); ax.set_xlabel('')
            if i == 0: ax.set_title(label, fontsize=12, fontweight='bold')
    
    for j in range(num_cols): axes[3, j].set_xlabel('Latitude index', fontsize=11)
    fig.colorbar(p, ax=axes, orientation='horizontal', shrink=0.6, pad=0.08)
    fig.suptitle(f"Time-Averaged Zonal Mean OHC ({units}, Multi-Resolution)", fontsize=16, y=0.935)
    plt.show()


def plot_zonal_snapshot_comparison_together(emulators_dict, variable_name, units, masks_list, time_indices=None, colorbar_fix=False):
    """Zonal Snapshot Comparison: Ground truth (first emulator) + all emulators."""
    print(f"\nGenerating Zonal Snapshot Comparison for {variable_name} (Multi-Resolution)...")
    if time_indices is None:
        time_indices = [0]
    
    first_label = list(emulators_dict.keys())[0]
    ds_truth = emulators_dict[first_label]['truth']
    truth_label = emulators_dict[first_label].get('truth_label', 'Truth')
    masks_truth = masks_list[0]
    dx_truth = ds_truth['dx']
    wet_mask = (ds_truth[variable_name].isel(time=0) * 0 + 1).compute()
    
    for time_idx in time_indices:
        time_val = ds_truth.time.isel(time=time_idx).values
        if hasattr(time_val, 'year'):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]

        truth_field = ds_truth[variable_name].isel(time=time_idx)
        truth_profiles = {}
        for basin in BASINS_TO_PLOT:
            truth_profiles[basin] = calculate_zonal_mean(truth_field, masks_truth[basin], dx_truth, wet_mask).compute()
        
        emu_profiles = {}
        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            ds_emu = emu_data['ds']
            ds_emu_truth = emu_data['truth']
            masks_emu = masks_list[idx]
            dx_emu = ds_emu_truth['dx']
            wet_emu = (ds_emu_truth[variable_name].isel(time=0) * 0 + 1).compute()
            emu_field = ds_emu[variable_name].isel(time=time_idx)
            emu_profiles[label] = {}
            for basin in BASINS_TO_PLOT:
                emu_profiles[label][basin] = calculate_zonal_mean(emu_field, masks_emu[basin], dx_emu, wet_emu).compute()
        
        num_cols = 1 + len(emulators_dict)
        fig, axes = plt.subplots(4, num_cols, figsize=(6*num_cols, 14), sharex=True, sharey=True,
                                 gridspec_kw={'hspace': 0.12, 'wspace': 0.08})
        if num_cols == 1: axes = axes.reshape(-1, 1)
        
        all_data = np.concatenate([truth_profiles[basin].values.flatten() for basin in BASINS_TO_PLOT])
        valid_data = all_data[~np.isnan(all_data)]
        vmin = np.percentile(valid_data, .5)
        vmax = np.percentile(valid_data, 98)
        if colorbar_fix: vmin, vmax = 0, 25
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cmocean.cm.thermal
        
        for i, basin in enumerate(BASINS_TO_PLOT):
            ax = axes[i, 0]
            p = truth_profiles[basin].plot.contourf(ax=ax, cmap=cmap, norm=norm, add_colorbar=False,
                levels=np.linspace(vmin, vmax, 25), yincrease=False, x='y', y='lev', add_labels=False)
            ax.set_facecolor('grey')
            ax.set_ylabel(f'{basin}\nDepth (m)', fontsize=11)
            ax.set_xlabel('')
            if i == 0: ax.set_title(truth_label, fontsize=12, fontweight='bold')
            
            for j, (label, profiles) in enumerate(emu_profiles.items(), start=1):
                ax = axes[i, j]
                profiles[basin].plot.contourf(ax=ax, cmap=cmap, norm=norm, add_colorbar=False,
                    levels=np.linspace(vmin, vmax, 25), yincrease=False, x='y', y='lev', add_labels=False)
                ax.set_facecolor('grey')
                ax.set_ylabel(''); ax.set_xlabel('')
                if i == 0: ax.set_title(label, fontsize=12, fontweight='bold')
        
        for j in range(num_cols): axes[3, j].set_xlabel('Latitude index', fontsize=11)
        fig.colorbar(p, ax=axes, orientation='horizontal', shrink=0.6, pad=0.08)
        fig.suptitle(f"Snapshot Zonal Mean {variable_name} at {time_str} (Multi-Resolution)", fontsize=16, y=0.935)
        plt.show()


def plot_ohc_zonal_snapshot_comparison_together(emulators_dict, masks_list, TIME_INDICES, variable_name=None, units=None, colorbar_fix=False):
    """OHC Zonal Snapshot Comparison: Ground truth (first emulator) + all emulators."""
    print("\nGenerating OHC Zonal Snapshot Comparison (Multi-Resolution)...")
    units = 'J/m^2'
    
    first_label = list(emulators_dict.keys())[0]
    ds_truth = emulators_dict[first_label]['truth']
    truth_label = emulators_dict[first_label].get('truth_label', 'Truth')
    
    for time_idx in TIME_INDICES:
        # Get time string
        time_val = ds_truth.time.isel(time=time_idx).values
        if hasattr(time_val, 'year'):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]
        
        masks_truth = masks_list[0]
        dx_truth = ds_truth['dx']
        wet_mask = (ds_truth['thetao'].isel(time=0) * 0 + 1).compute()
        
        truth_field = calculate_ohc_2d(ds_truth).isel(time=time_idx).compute()
        truth_profiles = {}
        for basin in BASINS_TO_PLOT:
            truth_profiles[basin] = calculate_zonal_mean_2d(truth_field, masks_truth[basin], dx_truth, wet_mask).compute()
        
        emu_profiles = {}
        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            ds_emu = emu_data['ds']
            ds_emu_truth = emu_data['truth']
            masks_emu = masks_list[idx]
            dx_emu = ds_emu_truth['dx']
            wet_emu = (ds_emu_truth['thetao'].isel(time=0) * 0 + 1).compute()
            emu_field = calculate_ohc_2d(ds_emu).isel(time=time_idx).compute()
            emu_profiles[label] = {}
            for basin in BASINS_TO_PLOT:
                emu_profiles[label][basin] = calculate_zonal_mean_2d(emu_field, masks_emu[basin], dx_emu, wet_emu).compute()
        
        num_cols = 1 + len(emulators_dict)
        fig, axes = plt.subplots(4, num_cols, figsize=(6*num_cols, 14), sharex=True, sharey=True,
                                 gridspec_kw={'hspace': 0.12, 'wspace': 0.08})
        if num_cols == 1: axes = axes.reshape(-1, 1)
        
        all_data = np.concatenate([truth_profiles[basin].values.flatten() for basin in BASINS_TO_PLOT])
        valid_data = all_data[~np.isnan(all_data)]
        vmin = np.percentile(valid_data, .5)
        vmax = np.percentile(valid_data, 98)
        if colorbar_fix: vmin, vmax = 0, 16e9
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cmocean.cm.thermal
        
        for i, basin in enumerate(BASINS_TO_PLOT):
            ax = axes[i, 0]
            p = truth_profiles[basin].plot.contourf(ax=ax, cmap=cmap, norm=norm, add_colorbar=False,
                levels=np.linspace(vmin, vmax, 25), yincrease=False, x='y', y='lev', add_labels=False)
            ax.set_facecolor('grey')
            ax.set_ylabel(f'{basin}\nDepth (m)', fontsize=11)
            ax.set_xlabel('')
            if i == 0: ax.set_title(truth_label, fontsize=12, fontweight='bold')
            
            for j, (label, profiles) in enumerate(emu_profiles.items(), start=1):
                ax = axes[i, j]
                profiles[basin].plot.contourf(ax=ax, cmap=cmap, norm=norm, add_colorbar=False,
                    levels=np.linspace(vmin, vmax, 25), yincrease=False, x='y', y='lev', add_labels=False)
                ax.set_facecolor('grey')
                ax.set_ylabel(''); ax.set_xlabel('')
                if i == 0: ax.set_title(label, fontsize=12, fontweight='bold')
        
        for j in range(num_cols): axes[3, j].set_xlabel('Latitude index', fontsize=11)
        fig.colorbar(p, ax=axes, orientation='horizontal', shrink=0.6, pad=0.08)
        fig.suptitle(f"Snapshot Zonal Mean OHC at {time_str} ({units}, Multi-Resolution)", fontsize=16, y=0.935)
        plt.show()


def plot_zonal_mean_snapshot_comparison_together(emulators_dict, variable_name, units, masks_list, TIME_INDICES, colorbar_fix=False):
    """Zonal Mean Snapshot Comparison at specific time indices."""
    print(f"\nGenerating Zonal Mean Snapshot Comparison for {variable_name} (Multi-Resolution)...")
    
    first_label = list(emulators_dict.keys())[0]
    ds_truth = emulators_dict[first_label]['truth']
    truth_label = emulators_dict[first_label].get('truth_label', 'Truth')
    
    for time_idx in TIME_INDICES:
        time_val = ds_truth.time.isel(time=time_idx).values
        if hasattr(time_val, 'year'):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]
        
        masks_truth = masks_list[0]
        dx_truth = ds_truth['dx']
        wet_mask = (ds_truth[variable_name].isel(time=0, drop=True) * 0 + 1).compute()
        
        truth_field = ds_truth[variable_name].isel(time=time_idx)
        truth_profiles = {}
        for basin in BASINS_TO_PLOT:
            truth_profiles[basin] = calculate_zonal_mean(truth_field, masks_truth[basin], dx_truth, wet_mask).compute()
        
        emu_profiles = {}
        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            ds_emu = emu_data['ds']
            ds_emu_truth = emu_data['truth']
            masks_emu = masks_list[idx]
            dx_emu = ds_emu_truth['dx']
            wet_emu = (ds_emu_truth[variable_name].isel(time=0, drop=True) * 0 + 1).compute()
            emu_field = ds_emu[variable_name].isel(time=time_idx)
            emu_profiles[label] = {}
            for basin in BASINS_TO_PLOT:
                emu_profiles[label][basin] = calculate_zonal_mean(emu_field, masks_emu[basin], dx_emu, wet_emu).compute()
        
        num_cols = 1 + len(emulators_dict)
        fig, axes = plt.subplots(4, num_cols, figsize=(6*num_cols, 14), sharex=True, sharey=True,
                                 gridspec_kw={'hspace': 0.12, 'wspace': 0.08})
        if num_cols == 1: axes = axes.reshape(-1, 1)
        
        all_data = np.concatenate([truth_profiles[basin].values.flatten() for basin in BASINS_TO_PLOT])
        valid_data = all_data[~np.isnan(all_data)]
        vmin = np.percentile(valid_data, .5)
        vmax = np.percentile(valid_data, 98)
        if colorbar_fix: vmin, vmax = 0, 25
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cmocean.cm.thermal
        
        for i, basin in enumerate(BASINS_TO_PLOT):
            ax = axes[i, 0]
            p = truth_profiles[basin].plot.contourf(ax=ax, cmap=cmap, norm=norm, add_colorbar=False,
                levels=np.linspace(vmin, vmax, 25), yincrease=False, x='y', y='lev', add_labels=False)
            ax.set_facecolor('grey')
            ax.set_ylabel(f'{basin}\nDepth (m)', fontsize=11)
            ax.set_xlabel('')
            if i == 0: ax.set_title(truth_label, fontsize=12, fontweight='bold')
            
            for j, (label, profiles) in enumerate(emu_profiles.items(), start=1):
                ax = axes[i, j]
                profiles[basin].plot.contourf(ax=ax, cmap=cmap, norm=norm, add_colorbar=False,
                    levels=np.linspace(vmin, vmax, 25), yincrease=False, x='y', y='lev', add_labels=False)
                ax.set_facecolor('grey')
                ax.set_ylabel(''); ax.set_xlabel('')
                if i == 0: ax.set_title(label, fontsize=12, fontweight='bold')
        
        for j in range(num_cols): axes[3, j].set_xlabel('Latitude index', fontsize=11)
        fig.colorbar(p, ax=axes, orientation='horizontal', shrink=0.6, pad=0.08)
        fig.suptitle(f"Snapshot Zonal Mean {variable_name} at {time_str} (Multi-Resolution)", fontsize=16, y=0.935)
        plt.show()


def plot_snapshot_comparison_SpecificLayer_together(emulators_dict, var_name, time_indices, depth_config, fix_colorbar=False):
    """Snapshot Comparison at Specific Layer: Ground truth (first emulator) + all emulators."""
    config = VAR_CONFIG.get(var_name, {'cmap': 'viridis', 'units': '', 'label': var_name, 'centered': False})
    print(f"\nGenerating Snapshot Comparison for {config['label']} at {depth_config['title']} (Multi-Resolution)...")
    
    first_label = list(emulators_dict.keys())[0]
    ds_truth = emulators_dict[first_label]['truth']
    truth_label = emulators_dict[first_label].get('truth_label', 'Truth')
    dz_truth = ds_truth['dz']
    wet_mask = (ds_truth[var_name].isel(time=0) * 0 + 1).compute()
    
    for time_idx in time_indices:
        time_val = ds_truth.time.isel(time=time_idx).values
        if hasattr(time_val, 'year'):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]
        
        # Compute truth map
        truth_map = calculate_snapshot_map_SpecificLayer(ds_truth, var_name, dz_truth, wet_mask, depth_config['index'], time_idx).compute()
        
        # Compute emulator maps
        emu_maps = {}
        for label, emu_data in emulators_dict.items():
            ds_emu = emu_data['ds']
            ds_emu_truth = emu_data['truth']
            dz_emu = ds_emu_truth['dz']
            wet_emu = (ds_emu_truth[var_name].isel(time=0) * 0 + 1).compute()
            emu_maps[label] = calculate_snapshot_map_SpecificLayer(ds_emu, var_name, dz_emu, wet_emu, depth_config['index'], time_idx).compute()
        
        # Plotting
        datasets = [truth_label] + list(emulators_dict.keys())
        nrows = len(datasets)
        
        fig, axes = plt.subplots(nrows, 1, figsize=(10, 4*nrows),
            subplot_kw={'projection': ccrs.Robinson(central_longitude=210)}, gridspec_kw={'hspace': 0.15})
        if nrows == 1: axes = [axes]
        
        # Determine colorbar range
        valid_data = truth_map.values[~np.isnan(truth_map.values)]
        if config['centered']:
            abs_max = np.percentile(np.abs(valid_data), 99.75)
            vmin, vmax = -abs_max, abs_max
        else:
            vmin = np.percentile(valid_data, .25)
            vmax = np.percentile(valid_data, 99.75)
        
        if fix_colorbar:
            fixed = {0: (-2, 32), 10: (-2, 15), 14: (-2, 10), 15: (-2, 10)}
            vmin, vmax = fixed.get(depth_config['index'], (vmin, vmax))
        
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        for i, ds_name in enumerate(datasets):
            ax = axes[i]
            if ds_name == truth_label:
                data_to_plot = truth_map
            else:
                data_to_plot = emu_maps[ds_name]
            
            p = data_to_plot.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='x', y='y',
                cmap=config['cmap'], norm=norm, add_colorbar=False, rasterized=True)
            ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
            ax.coastlines(zorder=11)
            ax.text(-0.05, 0.5, ds_name, va='center', ha='right', rotation='vertical',
                   transform=ax.transAxes, fontsize=12, fontweight='bold')
        
        fig.colorbar(p, ax=axes, orientation='vertical', shrink=0.8, pad=0.02)
        fig.suptitle(f"{config['label']} Snapshot at {depth_config['title']}, {time_str} (Multi-Resolution)", fontsize=16, y=0.96)
        plt.show()


def plot_ohc_snapshot_comparison_together(emulators_dict, time_indices, depth_config, variable_name=None, units=None, colorbar_fix=False):
    """OHC Snapshot Comparison: Ground truth (first emulator) + all emulators."""
    print(f"\nGenerating OHC Snapshot Comparison at {depth_config['title']} (Multi-Resolution)...")
    
    first_label = list(emulators_dict.keys())[0]
    ds_truth = emulators_dict[first_label]['truth']
    truth_label = emulators_dict[first_label].get('truth_label', 'Truth')
    dz_truth = ds_truth['dz']
    wet_mask = (ds_truth['thetao'].isel(time=0) * 0 + 1).compute()
    
    for time_idx in time_indices:
        time_val = ds_truth.time.isel(time=time_idx).values
        if hasattr(time_val, 'year'):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]
        
        # Compute truth map
        truth_map = calculate_ohc_snapshot_map(ds_truth, dz_truth, wet_mask, depth_config['min'], depth_config['max'], time_idx).compute()
        
        # Compute emulator maps
        emu_maps = {}
        for label, emu_data in emulators_dict.items():
            ds_emu = emu_data['ds']
            ds_emu_truth = emu_data['truth']
            dz_emu = ds_emu_truth['dz']
            wet_emu = (ds_emu_truth['thetao'].isel(time=0) * 0 + 1).compute()
            emu_maps[label] = calculate_ohc_snapshot_map(ds_emu, dz_emu, wet_emu, depth_config['min'], depth_config['max'], time_idx).compute()
        
        # Plotting
        datasets = [truth_label] + list(emulators_dict.keys())
        nrows = len(datasets)
        
        fig, axes = plt.subplots(nrows, 1, figsize=(10, 4*nrows),
            subplot_kw={'projection': ccrs.Robinson(central_longitude=210)}, gridspec_kw={'hspace': 0.15})
        if nrows == 1: axes = [axes]
        
        valid_data = truth_map.values[~np.isnan(truth_map.values)]
        vmin = np.percentile(valid_data, .25)
        vmax = np.percentile(valid_data, 99.75)
        
        if colorbar_fix:
            if depth_config['title'] == 'Surface (0-700m)': vmin, vmax = 0, 5e-11
            elif depth_config['title'] == 'Intermediate (700-2000m)': vmin, vmax = 0, 5e-11
            elif depth_config['title'] == 'Deep (2000-7000m)': vmin, vmax = 0, 5e-11
        
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        for i, ds_name in enumerate(datasets):
            ax = axes[i]
            if ds_name == truth_label:
                data_to_plot = truth_map
            else:
                data_to_plot = emu_maps[ds_name]
            
            p = data_to_plot.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='x', y='y',
                cmap=cmocean.cm.thermal, norm=norm, add_colorbar=False, rasterized=True)
            ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
            ax.coastlines(zorder=11)
            ax.text(-0.05, 0.5, ds_name, va='center', ha='right', rotation='vertical',
                   transform=ax.transAxes, fontsize=12, fontweight='bold')
        
        fig.colorbar(p, ax=axes, orientation='vertical', shrink=0.8, pad=0.02)
        fig.suptitle(f"OHC Snapshot at {depth_config['title']}, {time_str} (Multi-Resolution)", fontsize=16, y=0.96)
        plt.show()


def plot_snapshot_comparison_multidepth_together(emulators_dict, var_name, time_indices, depth_configs, fix_colorbar=False):
    """
    Snapshot Comparison at Multiple Depths (Columns) for specified Time Indices (Separate Figures).
    Rows: Emulators
    Cols: Depths
    """
    config = VAR_CONFIG.get(var_name, {'cmap': 'viridis', 'units': '', 'label': var_name, 'centered': False})
    print(f"\nGenerating Snapshot Comparison for {config['label']} (Multi-Resolution, Multi-Depth)...")
    
    first_label = list(emulators_dict.keys())[0]
    ds_truth = emulators_dict[first_label]['truth']
    truth_label = emulators_dict[first_label].get('truth_label', 'Truth')
    dz_truth = ds_truth['dz']
    wet_mask = (ds_truth[var_name].isel(time=0) * 0 + 1).compute()
    
    datasets = [truth_label] + list(emulators_dict.keys())
    nrows = len(datasets)
    ncols = len(depth_configs)
    
    for time_idx in time_indices:
        time_val = ds_truth.time.isel(time=time_idx).values
        if hasattr(time_val, 'year'):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]

        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows),
            subplot_kw={'projection': ccrs.Robinson(central_longitude=210)}, gridspec_kw={'hspace': 0.15, 'wspace': 0.05})
        if nrows == 1: axes = np.expand_dims(axes, 0)
        if ncols == 1: axes = np.expand_dims(axes, 1)

        # Loop Columns (Depths)
        for j, d_conf in enumerate(depth_configs):
            key = d_conf['title']
            d_idx = d_conf['index']
            
            # Truth Map
            truth_map = calculate_snapshot_map_SpecificLayer(ds_truth, var_name, dz_truth, wet_mask, d_idx, time_idx).compute()
            
            # Emulator Maps
            emu_maps = {}
            for label, emu_data in emulators_dict.items():
                ds_emu = emu_data['ds']
                ds_emu_truth = emu_data['truth']
                dz_emu = ds_emu_truth['dz']
                wet_emu = (ds_emu_truth[var_name].isel(time=0) * 0 + 1).compute()
                emu_maps[label] = calculate_snapshot_map_SpecificLayer(ds_emu, var_name, dz_emu, wet_emu, d_idx, time_idx).compute()
            
            valid_data = truth_map.values[~np.isnan(truth_map.values)]
            if len(valid_data) > 0:
                if config['centered']:
                    abs_max = np.percentile(np.abs(valid_data), 99.75)
                    vmin, vmax = -abs_max, abs_max
                else:
                    vmin = np.percentile(valid_data, .25)
                    vmax = np.percentile(valid_data, 99.75)
            else:
                vmin, vmax = 0, 1
            
            if fix_colorbar:
                fixed = {0: (-2, 32), 10: (-2, 15), 14: (-2, 10), 15: (-2, 10)}
                vmin, vmax = fixed.get(d_idx, (vmin, vmax))
            
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            
            # Plot Rows (Datasets)
            for i, ds_name in enumerate(datasets):
                ax = axes[i, j]
                if ds_name == truth_label:
                    data_to_plot = truth_map
                else:
                    data_to_plot = emu_maps[ds_name]
                
                p = data_to_plot.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='x', y='y',
                    cmap=config['cmap'], norm=norm, add_colorbar=False, rasterized=True)
                ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
                ax.coastlines(zorder=11)
                
                if i == 0: ax.set_title(key, fontsize=14, fontweight='bold')
                else: ax.set_title("")
                
                if j == 0: ax.text(-0.07, 0.55, ds_name, va='center', ha='right', rotation='vertical',
                                  transform=ax.transAxes, fontsize=12, fontweight='bold')
            
            fig.colorbar(p, ax=axes[:, j], orientation='horizontal', shrink=0.7, pad=0.03)

        fig.suptitle(f"{config['label']} Snapshot at {time_str} (Multi-Resolution)", fontsize=18, y=0.96)
        plt.show()


def plot_ohc_snapshot_comparison_multidepth_together(emulators_dict, time_indices, depth_configs, variable_name=None, units=None, colorbar_fix=False):
    """
    OHC Snapshot Comparison at Multiple Depths (Columns) for specified Time Indices (Separate Figures).
    Rows: Emulators
    Cols: Depth Slices
    """
    print(f"\nGenerating OHC Snapshot Comparison (Multi-Resolution, Multi-Depth)...")
    
    first_label = list(emulators_dict.keys())[0]
    ds_truth = emulators_dict[first_label]['truth']
    truth_label = emulators_dict[first_label].get('truth_label', 'Truth')
    dz_truth = ds_truth['dz']
    # OHC mask uses thetao
    wet_mask = (ds_truth['thetao'].isel(time=0) * 0 + 1).compute()
    
    datasets = [truth_label] + list(emulators_dict.keys())
    nrows = len(datasets)
    ncols = len(depth_configs)
    
    for time_idx in time_indices:
        time_val = ds_truth.time.isel(time=time_idx).values
        if hasattr(time_val, 'year'):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]

        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows),
            subplot_kw={'projection': ccrs.Robinson(central_longitude=210)}, gridspec_kw={'hspace': 0.15, 'wspace': 0.05})
        if nrows == 1: axes = np.expand_dims(axes, 0)
        if ncols == 1: axes = np.expand_dims(axes, 1)

        # Loop Columns (Depths)
        for j, d_conf in enumerate(depth_configs):
            key = d_conf['title']
            d_min = d_conf['min']
            d_max = d_conf['max']
            
            # Truth Map
            truth_map = calculate_ohc_snapshot_map(ds_truth, dz_truth, wet_mask, d_min, d_max, time_idx).compute()
            
            # Emulator Maps
            emu_maps = {}
            for label, emu_data in emulators_dict.items():
                ds_emu = emu_data['ds']
                ds_emu_truth = emu_data['truth']
                dz_emu = ds_emu_truth['dz']
                wet_emu = (ds_emu_truth['thetao'].isel(time=0) * 0 + 1).compute()
                emu_maps[label] = calculate_ohc_snapshot_map(ds_emu, dz_emu, wet_emu, d_min, d_max, time_idx).compute()
            
            valid_data = truth_map.values[~np.isnan(truth_map.values)]
            if len(valid_data) > 0:
                vmin = np.percentile(valid_data, .25)
                vmax = np.percentile(valid_data, 99.75)
            else:
                vmin, vmax = 0, 1
            
            if colorbar_fix:
                if key == 'Surface (0-700m)': vmin, vmax = 0, 5e-11
                elif key == 'Intermediate (700-2000m)': vmin, vmax = 0, 5e-11
                elif key == 'Deep (2000-7000m)': vmin, vmax = 0, 5e-11
            
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            
            # Plot Rows (Datasets)
            for i, ds_name in enumerate(datasets):
                ax = axes[i, j]
                if ds_name == truth_label:
                    data_to_plot = truth_map
                else:
                    data_to_plot = emu_maps[ds_name]
                
                p = data_to_plot.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='x', y='y',
                    cmap=cmocean.cm.thermal, norm=norm, add_colorbar=False, rasterized=True)
                ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black', facecolor='lightgray')
                ax.coastlines(zorder=11)
                
                if i == 0: ax.set_title(key, fontsize=14, fontweight='bold')
                else: ax.set_title("")
                
                if j == 0: ax.text(-0.07, 0.55, ds_name, va='center', ha='right', rotation='vertical',
                                  transform=ax.transAxes, fontsize=12, fontweight='bold')
            
            fig.colorbar(p, ax=axes[:, j], orientation='horizontal', shrink=0.7, pad=0.03)

        fig.suptitle(f"OHC Snapshot at {time_str} (Multi-Resolution)", fontsize=18, y=0.96)
        plt.show()


# =============================================================================
# SPECTRUM _TOGETHER FUNCTIONS (Multi-Resolution)
# Each emulator has its own corresponding ground truth.
# emulators_dict format:
#   {label: {'ds': ds_emu, 'color': color, 'truth': ds_truth, 'truth_label': str}}
# =============================================================================

def plot_isotropic_spectrum_comparison_together(
    emulators_dict,
    var_to_eval='thetao',
    lev_idx=0,
    time_window=100,
    lon_slice=slice(180, 243),
    lat_slice=slice(-40, 35),
    target_wavenumbers=[0.01, 0.02],
    show_ratios=True,
    colorbar_fix=False
):
    """
    Plots isotropic power spectrum comparison where each emulator uses its own corresponding truth.

    Parameters:
    -----------
    emulators_dict : dict
        Dictionary of emulators with structure:
        {
            'label': {
                'ds': xr.Dataset,          # emulator dataset
                'color': str,              # matplotlib color
                'truth': xr.Dataset,       # corresponding ground truth dataset
                'truth_label': str         # label for the truth dataset
            }
        }
    var_to_eval : str, default='thetao'
        Variable name to evaluate
    lev_idx : int, default=0
        Level index for depth slice
    time_window : int, default=100
        Number of time steps to use
    lon_slice : slice, default=slice(180, 243)
        Longitude slice for analysis
    lat_slice : slice, default=slice(-40, 35)
        Latitude slice for analysis
    target_wavenumbers : list, default=[0.01, 0.02]
        Wavenumbers at which to calculate and annotate ratios
    show_ratios : bool, default=True
        Whether to show ratio annotations at target wavenumbers
    colorbar_fix : bool, default=False
        Not used, kept for consistency

    Returns:
    --------
    None (displays plot)
    """

    #print(f"\nGenerating Isotropic Power Spectrum Comparison for {var_to_eval} (Multi-Resolution)...")

    def get_spectrum(ds_target, var_name, lev_i, t_window, x_sl, y_sl):
        """Helper function to compute isotropic spectrum for a dataset."""
        dx_mean = float(ds_target.dx.sel({'x': x_sl, 'y': y_sl}).mean().values)
        dy_mean = float(ds_target.dy.sel({'x': x_sl, 'y': y_sl}).mean().values)

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

    # Compute spectra for each truth-emulator pair
    all_spectra = {}  # {label: {'k_truth': ..., 'spec_truth': ..., 'k_emu': ..., 'spec_emu': ...}}
    for label, emu_data in emulators_dict.items():
        ds_truth = emu_data['truth']
        ds_emu = emu_data['ds']
        truth_label = emu_data.get('truth_label', 'Truth')

        #print(f"Calculating spectrum for {truth_label} (truth of {label})...")
        k_truth, spec_truth = get_spectrum(ds_truth, var_to_eval, lev_idx, time_window, lon_slice, lat_slice)

        #print(f"Calculating spectrum for {label}...")
        k_emu, spec_emu = get_spectrum(ds_emu, var_to_eval, lev_idx, time_window, lon_slice, lat_slice)

        all_spectra[label] = {
            'k_truth': k_truth, 'spec_truth': spec_truth,
            'k_emu': k_emu, 'spec_emu': spec_emu,
            'truth_label': truth_label,
        }

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))

    plotted_truths = set()
    for label, data in all_spectra.items():
        color = emulators_dict[label]['color']
        truth_label = data['truth_label']

        # Plot truth (dashed) — skip if already plotted for this truth_label
        if truth_label not in plotted_truths:
            ax.loglog(
                data['k_truth'], data['spec_truth'].mean(0), '--',
                label=f'{truth_label} ({var_to_eval} Anomalies)',
                linewidth=2, color=color, alpha=0.7
            )
            plotted_truths.add(truth_label)
        # Plot emulator (solid)
        ax.loglog(
            data['k_emu'], data['spec_emu'].mean(0), '-',
            label=f'{label} ({var_to_eval} Anomalies)',
            linewidth=2, color=color
        )

    # Add ratio annotations at target wavenumbers
    if show_ratios:
        for target_k in target_wavenumbers:
            for i, (label, data) in enumerate(all_spectra.items()):
                idx_truth = np.argmin(np.abs(data['k_truth'] - target_k))
                idx_emu = np.argmin(np.abs(data['k_emu'] - target_k))
                actual_k = data['k_truth'][idx_truth]
                spec_truth_val = data['spec_truth'].mean(0)[idx_truth]
                spec_emu_val = data['spec_emu'].mean(0)[idx_emu]
                ratio = spec_emu_val / spec_truth_val if spec_truth_val > 0 else 0

                print(f"{label} vs {data['truth_label']} at k={actual_k:.4f}: ratio={ratio:.3f}")

                color = emulators_dict[label]['color']
                x_pos = actual_k * 1.1
                y_pos = spec_emu_val * (1.5 ** i)

                ax.text(
                    x_pos, y_pos,
                    f'{label}/{data["truth_label"]}={ratio:.3f}',
                    fontsize=8, color=color,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color)
                )

            ax.axvline(x=actual_k, color='blue', linestyle='--', alpha=0.5, linewidth=1)

    # Format axes
    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0]))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(labelOnlyBase=False))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs='all'))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    ax.set_xlabel(r'Wavenumber $\kappa$ (km$^{-1}$)')
    ax.set_ylabel('Power Spectral Density')

    # Get depth value from first emulator's truth
    first_label = list(emulators_dict.keys())[0]
    depth_value = emulators_dict[first_label]['truth'].lev.values[lev_idx]

    lon_start = lon_slice.start if lon_slice.start is not None else 0
    lon_stop = lon_slice.stop if lon_slice.stop is not None else 360
    lon_stop_display = f"{lon_stop - 360}°W" if lon_stop > 180 else f"{lon_stop}°E"
    lon_start_display = f"{lon_start}°E"
    lat_start = lat_slice.start if lat_slice.start is not None else -90
    lat_stop = lat_slice.stop if lat_slice.stop is not None else 90
    lat_start_display = f"{abs(lat_start)}°S" if lat_start < 0 else f"{lat_start}°N"
    lat_stop_display = f"{abs(lat_stop)}°S" if lat_stop < 0 else f"{lat_stop}°N"

    ax.set_title(f'Isotropic Power Spectrum of {var_to_eval} ({depth_value}m)\n' +
                 f'Region: {lon_start_display}-{lon_stop_display}, {lat_start_display}-{lat_stop_display}',
                 fontsize=11)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_ke_spectrum_comparison_together(
    emulators_dict,
    var_u='uo',
    var_v='vo',
    lev_idx=0,
    time_window=100,
    lon_slice=slice(180, 243),
    lat_slice=slice(-40, 35),
    target_wavenumbers=[0.01, 0.02],
    show_ratios=True,
    colorbar_fix=False
):
    """
    Plots kinetic energy spectrum comparison where each emulator uses its own corresponding truth.

    Parameters:
    -----------
    emulators_dict : dict
        Dictionary of emulators with structure:
        {
            'label': {
                'ds': xr.Dataset,          # emulator dataset
                'color': str,              # matplotlib color
                'truth': xr.Dataset,       # corresponding ground truth dataset
                'truth_label': str         # label for the truth dataset
            }
        }
    var_u : str, default='uo'
        U-velocity variable name
    var_v : str, default='vo'
        V-velocity variable name
    lev_idx : int, default=0
        Level index for depth slice
    time_window : int, default=100
        Number of time steps to use
    lon_slice : slice, default=slice(180, 243)
        Longitude slice for analysis
    lat_slice : slice, default=slice(-40, 35)
        Latitude slice for analysis
    target_wavenumbers : list, default=[0.01, 0.02]
        Wavenumbers at which to calculate and annotate ratios
    show_ratios : bool, default=True
        Whether to show ratio annotations at target wavenumbers
    colorbar_fix : bool, default=False
        Not used, kept for consistency

    Returns:
    --------
    None (displays plot)
    """

    #print(f"\nGenerating Kinetic Energy Spectrum Comparison (Multi-Resolution)...")

    def get_ke_spectrum(ds_target, lev_i, t_window, x_sl, y_sl):
        """Helper function to compute KE spectrum for a dataset."""
        dx_mean = float(ds_target.dx.sel({'x': x_sl, 'y': y_sl}).mean().values)
        dy_mean = float(ds_target.dy.sel({'x': x_sl, 'y': y_sl}).mean().values)

        def process_var(var_name):
            data = ds_target[var_name].isel(lev=lev_i, time=slice(None, t_window))
            data = data.transpose('time', ...).sel({'x': x_sl, 'y': y_sl})
            data_anom = data - data.mean(dim='time')
            return torch.as_tensor(data_anom.fillna(0).values)

        u_tensor = process_var(var_u)
        v_tensor = process_var(var_v)

        k_cent, spec_u = compute_isotropic_spectrum_torch(
            u_tensor, dx=dx_mean, dy=dy_mean, n_factor=2,
            detrend='linear', window='hann'
        )
        _, spec_v = compute_isotropic_spectrum_torch(
            v_tensor, dx=dx_mean, dy=dy_mean, n_factor=2,
            detrend='linear', window='hann'
        )

        spec_ke = 0.5 * (spec_u + spec_v)
        k_km = k_cent.cpu().numpy() * 1000 * 2 * np.pi
        return k_km, spec_ke.cpu().numpy()

    # Compute spectra for each truth-emulator pair
    all_spectra = {}
    for label, emu_data in emulators_dict.items():
        ds_truth = emu_data['truth']
        ds_emu = emu_data['ds']
        truth_label = emu_data.get('truth_label', 'Truth')

        #print(f"Calculating KE spectrum for {truth_label} (truth of {label})...")
        k_truth, spec_truth = get_ke_spectrum(ds_truth, lev_idx, time_window, lon_slice, lat_slice)

        #print(f"Calculating KE spectrum for {label}...")
        k_emu, spec_emu = get_ke_spectrum(ds_emu, lev_idx, time_window, lon_slice, lat_slice)

        all_spectra[label] = {
            'k_truth': k_truth, 'spec_truth': spec_truth,
            'k_emu': k_emu, 'spec_emu': spec_emu,
            'truth_label': truth_label,
        }

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))

    plotted_truths = set()
    for label, data in all_spectra.items():
        color = emulators_dict[label]['color']
        truth_label = data['truth_label']

        # Plot truth (dashed) — skip if already plotted for this truth_label
        if truth_label not in plotted_truths:
            ax.loglog(
                data['k_truth'], data['spec_truth'].mean(0), '--',
                label=f'{truth_label} (KE Anomalies)',
                linewidth=2, color=color, alpha=0.7
            )
            plotted_truths.add(truth_label)
        # Plot emulator (solid)
        ax.loglog(
            data['k_emu'], data['spec_emu'].mean(0), '-',
            label=f'{label} (KE Anomalies)',
            linewidth=2, color=color
        )

    # Add ratio annotations at target wavenumbers
    if show_ratios:
        for target_k in target_wavenumbers:
            for i, (label, data) in enumerate(all_spectra.items()):
                idx_truth = np.argmin(np.abs(data['k_truth'] - target_k))
                idx_emu = np.argmin(np.abs(data['k_emu'] - target_k))
                actual_k = data['k_truth'][idx_truth]
                spec_truth_val = data['spec_truth'].mean(0)[idx_truth]
                spec_emu_val = data['spec_emu'].mean(0)[idx_emu]
                ratio = spec_emu_val / spec_truth_val if spec_truth_val > 0 else 0

                #print(f"{label} vs {data['truth_label']} at k={actual_k:.4f}: ratio={ratio:.3f}")

                color = emulators_dict[label]['color']
                x_pos = actual_k * 1.1
                y_pos = spec_emu_val * (1.5 ** i)

                ax.text(
                    x_pos, y_pos,
                    f'{label}/{data["truth_label"]}={ratio:.3f}',
                    fontsize=8, color=color,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color)
                )

            ax.axvline(x=actual_k, color='blue', linestyle='--', alpha=0.5, linewidth=1)

    # Format axes
    from matplotlib.ticker import LogLocator, LogFormatter
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=8))
    ax.xaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=False))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    ax.set_xlabel(r'Wavenumber $\kappa$ (km$^{-1}$)')
    ax.set_ylabel('Kinetic Energy Density')

    first_label = list(emulators_dict.keys())[0]
    depth_value = emulators_dict[first_label]['truth'].lev.values[lev_idx]

    lon_start = lon_slice.start if lon_slice.start is not None else 0
    lon_stop = lon_slice.stop if lon_slice.stop is not None else 360
    lon_stop_display = f"{lon_stop - 360}°W" if lon_stop > 180 else f"{lon_stop}°E"
    lon_start_display = f"{lon_start}°E"
    lat_start = lat_slice.start if lat_slice.start is not None else -90
    lat_stop = lat_slice.stop if lat_slice.stop is not None else 90
    lat_start_display = f"{abs(lat_start)}°S" if lat_start < 0 else f"{lat_start}°N"
    lat_stop_display = f"{abs(lat_stop)}°S" if lat_stop < 0 else f"{lat_stop}°N"

    ax.set_title(f'Kinetic Energy Spectrum (Anomalies) at {depth_value}m\n' +
                 f'Region: {lon_start_display}-{lon_stop_display}, {lat_start_display}-{lat_stop_display}',
                 fontsize=11)

    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_temporal_spectrum_comparison_together(
    emulators_dict,
    var_to_eval='thetao',
    lev_idx=0,
    lon_slice=slice(180, 243),
    lat_slice=slice(-40, 35),
    target_frequencies=None,
    show_ratios=True,
    colorbar_fix=False
):
    """
    Plots temporal power spectrum comparison where each emulator uses its own corresponding truth.

    Parameters:
    -----------
    emulators_dict : dict
        Dictionary of emulators with structure:
        {
            'label': {
                'ds': xr.Dataset,          # emulator dataset
                'color': str,              # matplotlib color
                'truth': xr.Dataset,       # corresponding ground truth dataset
                'truth_label': str         # label for the truth dataset
            }
        }
    var_to_eval : str, default='thetao'
        Variable name to evaluate
    lev_idx : int, default=0
        Level index for depth slice
    lon_slice : slice, default=slice(180, 243)
        Longitude slice for spatial averaging
    lat_slice : slice, default=slice(-40, 35)
        Latitude slice for spatial averaging
    target_frequencies : list, optional
        Frequencies (in cycles/year) at which to calculate and annotate ratios.
        If None, defaults to [1.0, 4.0] (annual and seasonal cycles)
    show_ratios : bool, default=True
        Whether to show ratio annotations at target frequencies
    colorbar_fix : bool, default=False
        Not used, kept for consistency

    Returns:
    --------
    None (displays plot)
    """

    if target_frequencies is None:
        target_frequencies = [1.0, 4.0]

    #print(f"\nGenerating Temporal Power Spectrum Comparison for {var_to_eval} (Multi-Resolution)...")

    def get_temporal_spectrum(ds_target, var_name, lev_i, x_sl, y_sl):
        """Helper function to compute temporal spectrum at each spatial point, then average."""
        from scipy import signal

        data = ds_target[var_name].isel(lev=lev_i).sel({'x': x_sl, 'y': y_sl})
        n_time = data.sizes['time']

        try:
            time_vals = ds_target.time.values
            if len(time_vals) > 1:
                delta_t = (time_vals[1] - time_vals[0])
                if isinstance(delta_t, np.timedelta64):
                    dt_days = float(delta_t) / np.timedelta64(1, 'D')
                else:
                    dt_days = delta_t.days + delta_t.seconds / 86400.0
            else:
                dt_days = 1.0
        except:
            dt_days = 1.0

        dt_years = dt_days / 365.25
        fs = 1.0 / dt_years

        data_np = np.nan_to_num(data.values, nan=0.0)
        data_detrended = signal.detrend(data_np, axis=0, type='linear')

        # Remove seasonal cycle
        steps_per_year = int(round(365.25 / dt_days))
        n_full = (n_time // steps_per_year) * steps_per_year
        if n_full >= steps_per_year:
            climatology = data_detrended[:n_full].reshape(-1, steps_per_year, *data_detrended.shape[1:]).mean(axis=0)
            seasonal = np.tile(climatology, (n_time // steps_per_year + 1, 1, 1))[:n_time]
            data_detrended = data_detrended - seasonal

        window = signal.windows.hann(n_time)
        data_windowed = data_detrended * window[:, np.newaxis, np.newaxis]
        window_correction = np.mean(window**2)

        fft_result = np.fft.rfft(data_windowed, axis=0)
        power_spectrum = np.abs(fft_result)**2
        psd_spatial = 2.0 * power_spectrum / (fs * n_time * window_correction)
        psd_spatial[0, :, :] = psd_spatial[0, :, :] / 2.0
        if n_time % 2 == 0:
            psd_spatial[-1, :, :] = psd_spatial[-1, :, :] / 2.0

        psd = np.nanmean(psd_spatial, axis=(1, 2))
        freqs = np.fft.rfftfreq(n_time, d=dt_years)

        return freqs, psd

    # Compute spectra for each truth-emulator pair
    all_spectra = {}
    for label, emu_data in emulators_dict.items():
        ds_truth = emu_data['truth']
        ds_emu = emu_data['ds']
        truth_label = emu_data.get('truth_label', 'Truth')

        #print(f"Calculating temporal spectrum for {truth_label} (truth of {label})...")
        freq_truth, psd_truth = get_temporal_spectrum(ds_truth, var_to_eval, lev_idx, lon_slice, lat_slice)

        #print(f"Calculating temporal spectrum for {label}...")
        freq_emu, psd_emu = get_temporal_spectrum(ds_emu, var_to_eval, lev_idx, lon_slice, lat_slice)

        all_spectra[label] = {
            'freq_truth': freq_truth, 'psd_truth': psd_truth,
            'freq_emu': freq_emu, 'psd_emu': psd_emu,
            'truth_label': truth_label,
        }

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))

    plotted_truths = set()
    for label, data in all_spectra.items():
        color = emulators_dict[label]['color']
        truth_label = data['truth_label']

        # Plot truth (dashed), skip DC component — skip if already plotted for this truth_label
        if truth_label not in plotted_truths:
            ax.loglog(
                data['freq_truth'][1:], data['psd_truth'][1:], '--',
                label=f'{truth_label} ({var_to_eval} Anomalies)',
                linewidth=2, color=color, alpha=0.7
            )
            plotted_truths.add(truth_label)
        # Plot emulator (solid), skip DC component
        ax.loglog(
            data['freq_emu'][1:], data['psd_emu'][1:], '-',
            label=f'{label} ({var_to_eval} Anomalies)',
            linewidth=2, color=color
        )

    # Add ratio annotations at target frequencies
    if show_ratios and len(target_frequencies) > 0:
        for target_f in target_frequencies:
            for i, (label, data) in enumerate(all_spectra.items()):
                idx_truth = np.argmin(np.abs(data['freq_truth'] - target_f))
                idx_emu = np.argmin(np.abs(data['freq_emu'] - target_f))
                actual_f = data['freq_truth'][idx_truth]
                psd_truth_val = data['psd_truth'][idx_truth]
                psd_emu_val = data['psd_emu'][idx_emu]
                ratio = psd_emu_val / psd_truth_val if psd_truth_val > 0 else 0

                period_years = 1.0 / actual_f if actual_f > 0 else np.inf
                print(f"{label} vs {data['truth_label']} at f={actual_f:.4f} cy/yr (period {period_years:.2f}yr): ratio={ratio:.3f}")

                color = emulators_dict[label]['color']
                x_pos = actual_f * 1.2
                y_pos = psd_emu_val * (1.5 ** i)

                ax.text(
                    x_pos, y_pos,
                    f'{label}/{data["truth_label"]}={ratio:.3f}',
                    fontsize=8, color=color,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color)
                )

            ax.axvline(x=actual_f, color='blue', linestyle='--', alpha=0.5, linewidth=1)

    # Format axes
    ax.set_xlabel('Frequency (cycles/year)')
    ax.set_ylabel('Power Spectral Density')

    first_label = list(emulators_dict.keys())[0]
    depth_value = emulators_dict[first_label]['truth'].lev.values[lev_idx]

    lon_start = lon_slice.start if lon_slice.start is not None else 0
    lon_stop = lon_slice.stop if lon_slice.stop is not None else 360
    lon_stop_display = f"{lon_stop - 360}°W" if lon_stop > 180 else f"{lon_stop}°E"
    lon_start_display = f"{lon_start}°E"
    lat_start = lat_slice.start if lat_slice.start is not None else -90
    lat_stop = lat_slice.stop if lat_slice.stop is not None else 90
    lat_start_display = f"{abs(lat_start)}°S" if lat_start < 0 else f"{lat_start}°N"
    lat_stop_display = f"{abs(lat_stop)}°S" if lat_stop < 0 else f"{lat_stop}°N"

    ax.set_title(f'Temporal Power Spectrum of {var_to_eval} at {depth_value}m\n' +
                 f'Region: {lon_start_display}-{lon_stop_display}, {lat_start_display}-{lat_stop_display}',
                 fontsize=11)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_ke_temporal_spectrum_comparison_together(
    emulators_dict,
    var_u='uo',
    var_v='vo',
    lev_idx=0,
    lon_slice=slice(180, 243),
    lat_slice=slice(-40, 35),
    target_frequencies=None,
    show_ratios=True,
    colorbar_fix=False
):
    """
    Plots temporal kinetic energy spectrum comparison where each emulator uses its own corresponding truth.

    Parameters:
    -----------
    emulators_dict : dict
        Dictionary of emulators with structure:
        {
            'label': {
                'ds': xr.Dataset,          # emulator dataset
                'color': str,              # matplotlib color
                'truth': xr.Dataset,       # corresponding ground truth dataset
                'truth_label': str         # label for the truth dataset
            }
        }
    var_u : str, default='uo'
        U-velocity variable name
    var_v : str, default='vo'
        V-velocity variable name
    lev_idx : int, default=0
        Level index for depth slice
    lon_slice : slice, default=slice(180, 243)
        Longitude slice for spatial averaging
    lat_slice : slice, default=slice(-40, 35)
        Latitude slice for spatial averaging
    target_frequencies : list, optional
        Frequencies (in cycles/year) at which to calculate and annotate ratios.
        If None, defaults to [1.0, 4.0] (annual and seasonal cycles)
    show_ratios : bool, default=True
        Whether to show ratio annotations at target frequencies
    colorbar_fix : bool, default=False
        Not used, kept for consistency

    Returns:
    --------
    None (displays plot)
    """

    if target_frequencies is None:
        target_frequencies = [1.0, 4.0]

    #print(f"\nGenerating Temporal KE Power Spectrum Comparison (Multi-Resolution)...")

    def get_ke_temporal_spectrum(ds_target, lev_i, x_sl, y_sl):
        """Helper function to compute temporal KE spectrum at each spatial point, then average."""
        from scipy import signal

        u_data = ds_target[var_u].isel(lev=lev_i).sel({'x': x_sl, 'y': y_sl})
        v_data = ds_target[var_v].isel(lev=lev_i).sel({'x': x_sl, 'y': y_sl})

        n_time = u_data.sizes['time']

        try:
            time_vals = ds_target.time.values
            if len(time_vals) > 1:
                delta_t = (time_vals[1] - time_vals[0])
                if isinstance(delta_t, np.timedelta64):
                    dt_days = float(delta_t) / np.timedelta64(1, 'D')
                else:
                    dt_days = delta_t.days + delta_t.seconds / 86400.0
            else:
                dt_days = 1.0
        except:
            dt_days = 1.0

        dt_years = dt_days / 365.25
        fs = 1.0 / dt_years

        u_np = np.nan_to_num(u_data.values, nan=0.0)
        v_np = np.nan_to_num(v_data.values, nan=0.0)

        u_detrended = signal.detrend(u_np, axis=0, type='linear')
        v_detrended = signal.detrend(v_np, axis=0, type='linear')

        # Remove seasonal cycle
        steps_per_year = int(round(365.25 / dt_days))
        n_full = (n_time // steps_per_year) * steps_per_year
        if n_full >= steps_per_year:
            clim_u = u_detrended[:n_full].reshape(-1, steps_per_year, *u_detrended.shape[1:]).mean(axis=0)
            clim_v = v_detrended[:n_full].reshape(-1, steps_per_year, *v_detrended.shape[1:]).mean(axis=0)
            seasonal_u = np.tile(clim_u, (n_time // steps_per_year + 1, 1, 1))[:n_time]
            seasonal_v = np.tile(clim_v, (n_time // steps_per_year + 1, 1, 1))[:n_time]
            u_detrended = u_detrended - seasonal_u
            v_detrended = v_detrended - seasonal_v

        window = signal.windows.hann(n_time)
        u_windowed = u_detrended * window[:, np.newaxis, np.newaxis]
        v_windowed = v_detrended * window[:, np.newaxis, np.newaxis]
        window_correction = np.mean(window**2)

        fft_u = np.fft.rfft(u_windowed, axis=0)
        fft_v = np.fft.rfft(v_windowed, axis=0)

        power_u = np.abs(fft_u)**2
        power_v = np.abs(fft_v)**2
        power_ke = 0.5 * (power_u + power_v)

        psd_spatial = 2.0 * power_ke / (fs * n_time * window_correction)
        psd_spatial[0, :, :] = psd_spatial[0, :, :] / 2.0
        if n_time % 2 == 0:
            psd_spatial[-1, :, :] = psd_spatial[-1, :, :] / 2.0

        psd = np.nanmean(psd_spatial, axis=(1, 2))
        freqs = np.fft.rfftfreq(n_time, d=dt_years)

        return freqs, psd

    # Compute spectra for each truth-emulator pair
    all_spectra = {}
    for label, emu_data in emulators_dict.items():
        ds_truth = emu_data['truth']
        ds_emu = emu_data['ds']
        truth_label = emu_data.get('truth_label', 'Truth')

        #print(f"Calculating temporal KE spectrum for {truth_label} (truth of {label})...")
        freq_truth, psd_truth = get_ke_temporal_spectrum(ds_truth, lev_idx, lon_slice, lat_slice)

        #print(f"Calculating temporal KE spectrum for {label}...")
        freq_emu, psd_emu = get_ke_temporal_spectrum(ds_emu, lev_idx, lon_slice, lat_slice)

        all_spectra[label] = {
            'freq_truth': freq_truth, 'psd_truth': psd_truth,
            'freq_emu': freq_emu, 'psd_emu': psd_emu,
            'truth_label': truth_label,
        }

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))

    plotted_truths = set()
    for label, data in all_spectra.items():
        color = emulators_dict[label]['color']
        truth_label = data['truth_label']

        # Plot truth (dashed), skip DC component — skip if already plotted for this truth_label
        if truth_label not in plotted_truths:
            ax.loglog(
                data['freq_truth'][1:], data['psd_truth'][1:], '--',
                label=f'{truth_label} (KE Anomalies)',
                linewidth=2, color=color, alpha=0.7
            )
            plotted_truths.add(truth_label)
        # Plot emulator (solid), skip DC component
        ax.loglog(
            data['freq_emu'][1:], data['psd_emu'][1:], '-',
            label=f'{label} (KE Anomalies)',
            linewidth=2, color=color
        )

    # Add ratio annotations at target frequencies
    if show_ratios and len(target_frequencies) > 0:
        for target_f in target_frequencies:
            for i, (label, data) in enumerate(all_spectra.items()):
                idx_truth = np.argmin(np.abs(data['freq_truth'] - target_f))
                idx_emu = np.argmin(np.abs(data['freq_emu'] - target_f))
                actual_f = data['freq_truth'][idx_truth]
                psd_truth_val = data['psd_truth'][idx_truth]
                psd_emu_val = data['psd_emu'][idx_emu]
                ratio = psd_emu_val / psd_truth_val if psd_truth_val > 0 else 0

                period_years = 1.0 / actual_f if actual_f > 0 else np.inf
                print(f"{label} vs {data['truth_label']} at f={actual_f:.4f} cy/yr (period {period_years:.2f}yr): ratio={ratio:.3f}")

                color = emulators_dict[label]['color']
                x_pos = actual_f * 1.2
                y_pos = psd_emu_val * (1.5 ** i)

                ax.text(
                    x_pos, y_pos,
                    f'{label}/{data["truth_label"]}={ratio:.3f}',
                    fontsize=8, color=color,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color)
                )

            ax.axvline(x=actual_f, color='blue', linestyle='--', alpha=0.5, linewidth=1)

    # Format axes
    ax.set_xlabel('Frequency (cycles/year)')
    ax.set_ylabel('Kinetic Energy Power Spectral Density')

    first_label = list(emulators_dict.keys())[0]
    depth_value = emulators_dict[first_label]['truth'].lev.values[lev_idx]

    lon_start = lon_slice.start if lon_slice.start is not None else 0
    lon_stop = lon_slice.stop if lon_slice.stop is not None else 360
    lon_stop_display = f"{lon_stop - 360}°W" if lon_stop > 180 else f"{lon_stop}°E"
    lon_start_display = f"{lon_start}°E"
    lat_start = lat_slice.start if lat_slice.start is not None else -90
    lat_stop = lat_slice.stop if lat_slice.stop is not None else 90
    lat_start_display = f"{abs(lat_start)}°S" if lat_start < 0 else f"{lat_start}°N"
    lat_stop_display = f"{abs(lat_stop)}°S" if lat_stop < 0 else f"{lat_stop}°N"

    ax.set_title(f'Temporal KE Power Spectrum at {depth_value}m\n' +
                 f'Region: {lon_start_display}-{lon_stop_display}, {lat_start_display}-{lat_stop_display}',
                 fontsize=11)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

