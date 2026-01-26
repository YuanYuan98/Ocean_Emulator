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
    {'title':'3000m','index':15}
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
            max_abs_bias = 2*10**8
        
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
    
    return numerator / denominator


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
    ohc_integrated = (RHO * CP * thetao_slice * dz_slice *dataset['areacello']* wet_slice).sum('lev')
    
    # 2. Time Average
    ohc_mean = ohc_integrated.mean('time')
    
    # 3. Scale
    return ohc_mean / SCALE_FACTOR

def plot_ohc_mean_comparison(ds_truth, emulators_dict,TRUTH_LABEL='Truth'):
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


def plot_temperature_mean_comparison(ds_truth, emulators_dict,TRUTH_LABEL='Truth'):
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

def plot_zonal_mean_comparison(ds_truth, emulators_dict,variable_name, units, masks):
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

def plot_zonal_mean_snapshot_comparison(ds_truth, emulators_dict, variable_name, units, masks, TIME_INDICES):
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


def plot_ohc_zonal_mean_comparison(ds_truth, emulators_dict,variable_name, units, masks):
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


def plot_zonal_snapshot_comparison(ds_truth, emulators_dict,variable_name, units, masks):
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


def plot_ohc_zonal_snapshot_comparison(ds_truth, emulators_dict, variable_name, units, masks, TIME_INDICES):
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
            #norm = colors.Normalize(vmin=-limit, vmax=limit)
            
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


def plot_ohc_snapshot_bias_comparison(ds_truth, emulators_dict, time_indices, colorbar_fix=False):
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
            
            for Slices in DEPTH_SLICES:
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
        
        # Override with fixed limits if requested
        if colorbar_fix:
            vmin_global = -0.02
            vmax_global = 0.02
        
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
        '2.5m': 1.2,
        '700m': 0.7,
        '2000m': 0.18,
        '3000m': 0.1
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
            if DEPTH_CONFIG['index'] == 0:
                vmin, vmax = -2, 32
            elif DEPTH_CONFIG['index'] == 10:
                vmin, vmax = -2, 15
            elif DEPTH_CONFIG['index'] == 14:
                vmin, vmax = -2, 6
            elif DEPTH_CONFIG['index'] == 15:
                vmin, vmax = -2, 6
            
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
    ohc_integrated = (RHO * CP * thetao_slice * dz_slice *dataset['areacello']* wet_slice).sum('lev')
    
    # 2. Time snapshot
    ohc_mean = ohc_integrated.isel(time=time_index)
    
    # 3. Scale
    return ohc_mean / SCALE_FACTOR



def plot_ohc_snapshot_comparison(ds_truth, emulators_dict, var_name, time_indices, depth_config,TRUTH_LABEL='Truth'):
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
            max_abs_bias = 0.514
        
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
    cbar = fig.colorbar(p_mesh, ax=axes, orientation='horizontal', shrink=0.6, pad=0.02, aspect=30)
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
        max_abs_bias = 2*10**8
    
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
    cbar = fig.colorbar(p_mesh, ax=axes, orientation='horizontal', shrink=0.6, pad=0.02, aspect=30)
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
        cbar = fig.colorbar(p_mesh, ax=axes, orientation='horizontal', shrink=0.6, pad=0.02, aspect=30)
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
            max_abs_bias = 2*10**8
        
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
        cbar = fig.colorbar(p_mesh, ax=axes, orientation='horizontal', shrink=0.6, pad=0.02, aspect=30)
        cbar.set_label(f"Bias ({units})", fontsize=12)
        
        fig.suptitle(f"Snapshot Zonal Mean Bias of OHC: {time_str} ({day_str})", fontsize=18, y=1.03)
        plt.show()
