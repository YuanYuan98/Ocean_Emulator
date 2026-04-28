"""
Observation-based evaluation driver for Samudra rollouts.

This script mirrors the existing notebook-exported evaluation drivers, but it
compares emulator rollouts against observational products instead of OM4 truth.

Implemented evaluations
-----------------------
1. Surface mean EKE map over 2022 vs DUACS geostrophic velocity.
2. Regional surface isotropic temperature spectra vs OISST.
3. Regional surface temporal KE spectra vs DUACS.
4. Global monthly OHC time series for upper / middle / deep ocean vs ARGO-IAP.

Notes
-----
- Observation file layouts vary a lot. The loaders below try several common
  coordinate and variable aliases, but the top-level OBS paths may still need
  adjustment to your local files.
- `FOMO.txt` lists the ARGO-IAP path as the DUACS directory. That looks like a
  typo, so ARGO-IAP is configured separately here.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
from scipy import signal


parser = argparse.ArgumentParser(description="Observation-based evaluation")
parser.add_argument("--year", type=int, default=2022, help="Target evaluation year for map/time-series products")
parser.add_argument(
    "--aligned-obs-root",
    type=Path,
    default=Path("/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/obs_data/nativegrid"),
    help="Root directory containing observation products prepared on their native grid",
)
parser.add_argument(
    "--raw-obs-root",
    type=Path,
    default=Path("/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/obs_data"),
    help="Root directory containing the raw observation products",
)
args = parser.parse_args()


OCEAN_FIELD_NAME_PREFIXES = {
    "thetao": ["thetao_"],
    "so": ["so_"],
    "uo": ["uo_"],
    "vo": ["vo_"],
    "zos": ["zos"],
}

DEPTH_SLICES = [
    {"min": 0, "max": 700, "title": "Surface (0-700m)"},
    {"min": 700, "max": 2000, "title": "Intermediate (700-2000m)"},
    {"min": 2000, "max": 7000, "title": "Deep (2000-7000m)"},
]

OHC_DEPTH_SLICES = DEPTH_SLICES[:2]

SPECTRUM_REGIONS = [
    ("Gulf Stream", slice(300, 320), slice(25, 45)),
    ("Kuroshio", slice(150, 170), slice(25, 45)),
    ("Agulhas", slice(40, 60), slice(-50, -30)),
    ("Malvinas", slice(311, 331), slice(-51, -31)),
]

RHO = 1035
CP = 3850


def concat_levels(ds: xr.Dataset, prefix: str) -> xr.DataArray:
    levels = []
    names = []
    level_pattern = re.compile(rf"{prefix}(\d+)$")
    for name in ds.data_vars:
        match = level_pattern.search(name)
        if match:
            levels.append(int(match.group(1)))
            names.append(name)
    names = sorted(names, key=lambda name: levels[names.index(name)])
    da = xr.concat([ds[name] for name in names], dim="lev")
    da.name = prefix.rstrip("_")
    return da


def _detrend_linear_torch(data: torch.Tensor) -> torch.Tensor:
    bsz, channels, height, width = data.shape
    y_coords = torch.linspace(-1, 1, height, device=data.device, dtype=data.dtype)
    x_coords = torch.linspace(-1, 1, width, device=data.device, dtype=data.dtype)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    design = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.ones_like(grid_x).flatten()], dim=1)

    flat = data.reshape(bsz * channels, height * width)
    coeffs, _, _, _ = torch.linalg.lstsq(design, flat.T)
    plane = (design @ coeffs.permute(1, 0).unsqueeze(-1)).reshape(bsz * channels, height, width)
    detrended = data.reshape(bsz * channels, height, width) - plane
    return detrended.reshape(bsz, channels, height, width)


def compute_isotropic_spectrum_torch(
    data: torch.Tensor,
    dx: float = 1.0,
    dy: float = 1.0,
    num_bins: int | None = None,
    n_factor: int = 4,
    remove_mean: bool = True,
    detrend: str | None = None,
    window: str = "Hann",
    truncate: bool = True,
    cutoff_before_bins: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dim = data.dim()
    if orig_dim == 2:
        data = data.reshape(1, 1, *data.shape)
    elif orig_dim == 3:
        data = data.unsqueeze(1)
    elif orig_dim != 4:
        raise ValueError("Input data must be 2D, 3D, or 4D (B, C, H, W)")

    bsz, channels, height, width = data.shape
    merged = bsz * channels
    lx = width * dx
    ly = height * dy

    if num_bins is None:
        num_bins = min(height, width) // n_factor

    if detrend == "linear":
        data = _detrend_linear_torch(data)
    elif detrend == "constant" or remove_mean:
        data = data - torch.mean(data, dim=(-2, -1), keepdim=True)

    if window and window.lower() == "hann":
        win_y = torch.hann_window(height, device=data.device, dtype=data.dtype).unsqueeze(1)
        win_x = torch.hann_window(width, device=data.device, dtype=data.dtype).unsqueeze(0)
        win_2d = (win_y * win_x).reshape(1, 1, height, width)
        window_correction = torch.mean(win_2d**2).item()
        data = data * win_2d
    else:
        window_correction = 1.0

    fft_2d = torch.fft.rfft2(data, norm="forward")
    psd_2d = (torch.abs(fft_2d) ** 2 / window_correction) * (lx * ly)

    k_x = torch.fft.rfftfreq(width, d=dx, device=data.device, dtype=data.dtype)
    k_y = torch.fft.fftfreq(height, d=dy, device=data.device, dtype=data.dtype)
    k_y_grid, k_x_grid = torch.meshgrid(k_y, k_x, indexing="ij")
    k_mag = torch.sqrt(k_x_grid**2 + k_y_grid**2)

    k_x_nyq = 1.0 / (2.0 * dx)
    k_y_nyq = 1.0 / (2.0 * dy)
    k_max_domain = k_mag.max()
    if truncate and cutoff_before_bins:
        k_max = min(k_max_domain, min(k_x_nyq, k_y_nyq))
    else:
        k_max = k_max_domain

    k_bins = torch.linspace(0, k_max, num_bins + 1, device=data.device, dtype=data.dtype)
    if truncate and not cutoff_before_bins:
        k_bins = k_bins[k_bins < min(k_x_nyq, k_y_nyq)]
        num_bins = k_bins.numel() - 1
    k_centers = (k_bins[:-1] + k_bins[1:]) / 2

    k_mag_flat = k_mag.flatten()
    bin_indices = torch.bucketize(k_mag_flat, k_bins[1:-1], right=True)
    psd_flat = psd_2d.reshape(merged, k_mag_flat.shape[0])
    binned_sum = torch.zeros(merged, num_bins, device=data.device, dtype=data.dtype)
    binned_sum.scatter_add_(dim=1, index=bin_indices.expand(merged, -1), src=psd_flat)
    binned_counts = torch.bincount(bin_indices, minlength=num_bins).to(data.dtype)
    iso = binned_sum / torch.clamp(binned_counts, min=1).unsqueeze(0)
    iso = iso * k_centers.unsqueeze(0)

    if orig_dim == 2:
        iso = iso.reshape(num_bins)
    elif orig_dim == 3:
        iso = iso.reshape(bsz, num_bins)
    else:
        iso = iso.reshape(bsz, channels, num_bins)
    return k_centers, iso


def process_dataset(ds_input: xr.Dataset, truth_levs: np.ndarray, dz_arr: np.ndarray, grid_metrics_dict: dict) -> xr.Dataset:
    ds_stacked = xr.Dataset(
        coords={
            "time": (["time"], ds_input.time.values),
            "y": (["y"], ds_input.y.values),
            "x": (["x"], ds_input.x.values),
            "lev": (["lev"], truth_levs),
            "dz": (["lev"], dz_arr),
        }
    )

    for var in ["uo", "vo", "thetao", "so"]:
        prefixes = OCEAN_FIELD_NAME_PREFIXES.get(var, [])
        if not prefixes:
            continue
        try:
            data_processed = concat_levels(ds_input, prefixes[0]).squeeze().transpose("time", "y", "x", "lev").data
            ds_stacked[var] = (["time", "y", "x", "lev"], data_processed)
        except Exception:
            pass

    if "zos" in ds_input:
        ds_stacked["zos"] = (["time", "y", "x"], ds_input["zos"].squeeze().data)

    if "vo" in ds_stacked:
        ds_stacked["dz"] = (ds_stacked["dz"] * (ds_stacked["vo"].isel(time=0) * 0 + 1)).compute()

    for metric, data in grid_metrics_dict.items():
        ds_stacked[metric] = data
        ds_stacked = ds_stacked.set_coords(metric)

    return ds_stacked


def save_current_fig(name: str, output_dir: str, dpi: int = 200) -> None:
    path = os.path.join(output_dir, f"{name}.pdf")
    fig = plt.gcf()
    fig.savefig(path, format="pdf", dpi=dpi)
    plt.close("all")
    print(f"  -> Saved: {path}")


def _legend_truth_first(handles: list, labels: list[str], truth_labels: set[str]) -> tuple[list, list[str], int]:
    truth_h, truth_l, other_h, other_l = [], [], [], []
    for handle, label in zip(handles, labels):
        if label in truth_labels:
            truth_h.append(handle)
            truth_l.append(label)
        else:
            other_h.append(handle)
            other_l.append(label)
    return truth_h + other_h, truth_l + other_l, max(1, len(truth_h) + len(other_h))


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = "Figures_observations"
PDF_DIR = os.path.join(OUTPUT_DIR, "pdf_figures")
MAP_DIR = os.path.join(PDF_DIR, "maps")
SPECTRUM_DIR = os.path.join(PDF_DIR, "spectrum")
TIMESERIES_DIR = os.path.join(PDF_DIR, "timeseries")
METRICS_DIR = os.path.join(PDF_DIR, "metrics")

for directory in [OUTPUT_DIR, PDF_DIR, MAP_DIR, SPECTRUM_DIR, TIMESERIES_DIR, METRICS_DIR]:
    os.makedirs(directory, exist_ok=True)


EMULATOR_CONFIGS = [
    (
        "/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/checkpoints/2025-02-08-samudra_om4_quarterdeg_cap_20_eval_fix/predictions.zarr",
        "Samudra 2 (1/4$^\\circ$)",
        "tab:blue",
    ),
    (
        "/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/quarterdeg/OM4.zarr",
        "OM4 (1/4$^\\circ$)",
        "tab:olive",
    ),
]


OBS_CONFIG = {
    "duacs": args.aligned_obs_root / "duacs.zarr",
    "oisst": args.aligned_obs_root / "oisst.zarr",
    "argo_iap": args.aligned_obs_root / "argo-iap.zarr",
}


DEPTH_LEVELS = np.array(
    [
        0,
        5,
        15,
        30,
        50,
        80,
        130,
        200,
        300,
        450,
        650,
        900,
        1200,
        1600,
        2100,
        2700,
        3500,
        4500,
        5500,
        6750,
    ],
    dtype=float,
)
LEV_CENTERS = np.array(
    [
        2.5,
        10,
        22.5,
        40,
        65,
        105,
        165,
        250,
        375,
        550,
        775,
        1050,
        1400,
        1850,
        2400,
        3100,
        4000,
        5000,
        6000,
    ]
)
DZ = DEPTH_LEVELS[1:] - DEPTH_LEVELS[:-1]


def _find_coord_name(ds: xr.Dataset | xr.DataArray, candidates: list[str], required: bool = True) -> str | None:
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    if required:
        raise KeyError(f"Could not find coordinate in {candidates}. Available: {list(ds.coords)} / {list(ds.dims)}")
    return None


def _find_var_name(ds: xr.Dataset, candidates: list[str]) -> str:
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError(f"Could not find variable in {candidates}. Available variables: {list(ds.data_vars)}")


def _maybe_open_dataset(path: Path) -> xr.Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Observation path does not exist: {path}")
    if path.is_dir() and path.suffix == ".zarr":
        return xr.open_zarr(path)
    if path.is_file() and path.suffix in {".zarr"}:
        return xr.open_zarr(path)
    if path.is_file() and path.suffix in {".nc", ".nc4"}:
        return xr.open_dataset(path)
    if path.is_dir():
        files = sorted([p for p in path.iterdir() if p.suffix in {".nc", ".nc4"}])
        if not files:
            zarrs = sorted([p for p in path.iterdir() if p.suffix == ".zarr"])
            if len(zarrs) == 1:
                return xr.open_zarr(zarrs[0])
            raise FileNotFoundError(f"No readable observation files under {path}")
        return xr.open_mfdataset(files, combine="by_coords")
    raise ValueError(f"Unsupported observation path: {path}")


def _coerce_time_coord(ds: xr.Dataset, time_name: str) -> xr.Dataset:
    time = ds[time_name]
    if np.issubdtype(time.dtype, np.datetime64):
        return ds

    values = np.asarray(time.values)
    if values.ndim != 1:
        raise ValueError(f"Expected 1D time coordinate, got shape {values.shape}")

    if np.issubdtype(values.dtype, np.number):
        numeric = values.astype(float)
        if np.all(np.isfinite(numeric)) and np.all((numeric >= 100001) & (numeric <= 999912)):
            year = (numeric // 100).astype(int)
            month = np.round(numeric - year * 100).astype(int)
            timestamps = pd.to_datetime(
                [f"{yy:04d}-{mm:02d}-01" for yy, mm in zip(year, month, strict=True)]
            )
            return ds.assign_coords({time_name: timestamps})

    timestamps = pd.to_datetime(values)
    return ds.assign_coords({time_name: timestamps})


def _coerce_datetime_values(values: np.ndarray) -> pd.DatetimeIndex:
    flat = np.asarray(values)
    if np.issubdtype(flat.dtype, np.datetime64):
        return pd.to_datetime(flat)
    if flat.size == 0:
        return pd.DatetimeIndex([])

    first = flat.flat[0]
    if hasattr(first, "year") and hasattr(first, "month") and hasattr(first, "day"):
        return pd.to_datetime(
            [
                f"{val.year:04d}-{val.month:02d}-{val.day:02d} "
                f"{getattr(val, 'hour', 0):02d}:{getattr(val, 'minute', 0):02d}:{getattr(val, 'second', 0):02d}"
                for val in flat
            ]
        )
    return pd.to_datetime(flat.astype(str))


def _normalize_lon(da: xr.Dataset | xr.DataArray, lon_name: str) -> xr.Dataset | xr.DataArray:
    lon = da[lon_name]
    if np.nanmin(lon.values) < 0:
        da = da.assign_coords({lon_name: (lon % 360)}).sortby(lon_name)
    return da


def _standardize_latlon_time(ds: xr.Dataset) -> xr.Dataset:
    lat_name = _find_coord_name(ds, ["lat", "latitude", "y", "nav_lat"])
    lon_name = _find_coord_name(ds, ["lon", "longitude", "x", "nav_lon"])
    time_name = _find_coord_name(ds, ["time", "valid_time"])
    ds = _coerce_time_coord(ds, time_name)
    ds = ds.rename({lat_name: "lat", lon_name: "lon", time_name: "time"})
    ds = _normalize_lon(ds, "lon")
    return ds.sortby("lat").sortby("lon")


def _drop_singleton_nonspatial_dims(ds: xr.Dataset) -> xr.Dataset:
    squeeze_dims = [
        dim
        for dim, size in ds.sizes.items()
        if size == 1 and dim not in {"time", "lat", "lon", "depth", "depth_0", "depth_std", "lev", "DEPTH", "z_t"}
    ]
    return ds.squeeze(squeeze_dims, drop=True) if squeeze_dims else ds


def _prepare_emulator_dataset(path: str) -> xr.Dataset:
    raw = xr.open_zarr(path)
    rename_map = {}
    if "lat" in raw.dims and "y" not in raw.dims:
        rename_map["lat"] = "y"
    if "lon" in raw.dims and "x" not in raw.dims:
        rename_map["lon"] = "x"
    if rename_map:
        raw = raw.rename(rename_map)
    raw = raw.assign_coords(time=_coerce_datetime_values(raw.time.values))
    dx = _estimate_dx(raw["x"].values, raw["y"].values)
    dy = _estimate_dy(raw["y"].values, raw["x"].values)
    area = _estimate_area(raw["x"].values, raw["y"].values)
    grid_metrics = {
        "dx": (["y", "x"], dx),
        "dy": (["y", "x"], dy),
        "areacello": (["y", "x"], area),
    }
    ds = process_dataset(raw, LEV_CENTERS, DZ, grid_metrics)
    return ds.assign_coords({"x": raw["x"].values % 360, "y": raw["y"].values}).sortby("x").sortby("y")


def _estimate_dx(lon_vals: np.ndarray, lat_vals: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon_vals, dtype=float)
    lat = np.asarray(lat_vals, dtype=float)
    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError("Expected 1D lat/lon coordinates for emulator grid")
    dlon = np.gradient(np.deg2rad(lon))
    lat2d = np.deg2rad(lat)[:, None]
    dx = 6.371e6 * np.cos(lat2d) * dlon[None, :]
    return np.abs(dx)


def _estimate_dy(lat_vals: np.ndarray, lon_vals: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon_vals, dtype=float)
    lat = np.asarray(lat_vals, dtype=float)
    dlat = np.gradient(np.deg2rad(lat))
    dy = 6.371e6 * dlat[:, None] * np.ones((lat.size, lon.size))
    return np.abs(dy)


def _estimate_area(lon_vals: np.ndarray, lat_vals: np.ndarray) -> np.ndarray:
    return _estimate_dx(lon_vals, lat_vals) * _estimate_dy(lat_vals, lon_vals)


def _infer_obs_grid_metrics(ds: xr.Dataset) -> xr.Dataset:
    area = _estimate_area(ds["lon"].values, ds["lat"].values)
    dx = _estimate_dx(ds["lon"].values, ds["lat"].values)
    dy = _estimate_dy(ds["lat"].values, ds["lon"].values)
    return ds.assign(
        area=(["lat", "lon"], area),
        dx=(["lat", "lon"], dx),
        dy=(["lat", "lon"], dy),
    )


def _temporal_overlap(sim: xr.Dataset, obs: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    start = max(pd.Timestamp(sim.time.values[0]), pd.Timestamp(obs.time.values[0]))
    end = min(pd.Timestamp(sim.time.values[-1]), pd.Timestamp(obs.time.values[-1]))
    if start > end:
        raise ValueError("No temporal overlap between emulator and observation datasets")
    return sim.sel(time=slice(start, end)), obs.sel(time=slice(start, end))


def _year_slice(ds: xr.Dataset | xr.DataArray, year: int) -> xr.Dataset | xr.DataArray:
    return ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))


def _surface_eke_from_uv(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    u_anom = u - u.mean("time")
    v_anom = v - v.mean("time")
    return 0.5 * (u_anom ** 2 + v_anom ** 2).mean("time")


def _apply_equatorial_mask(
    da: xr.DataArray,
    lat_dim: str = "lat",
    equatorial_mask_deg: float = 5.0,
) -> xr.DataArray:
    lat_vals = np.asarray(da[lat_dim].values, dtype=float)
    eq_mask = np.abs(lat_vals) >= equatorial_mask_deg
    eq_mask_da = xr.DataArray(eq_mask, dims=[lat_dim], coords={lat_dim: da[lat_dim]})
    return da.where(eq_mask_da)


def _geostrophic_velocity_from_zos(
    zos: xr.DataArray,
    lat_dim: str = "y",
    lon_dim: str = "x",
    equatorial_mask_deg: float = 5.0,
) -> tuple[xr.DataArray, xr.DataArray]:
    g = 9.81
    omega = 7.2921e-5
    R = 6.371e6
    deg_per_m = 180.0 / (np.pi * R)

    lat_da = zos[lat_dim]
    f = 2.0 * omega * np.sin(np.deg2rad(lat_da))
    coslat = np.cos(np.deg2rad(lat_da))

    deta_dlat = zos.differentiate(lat_dim)
    deta_dlon = zos.differentiate(lon_dim)
    deta_dy = deta_dlat * deg_per_m
    deta_dx = deta_dlon * deg_per_m / coslat

    u_g = -g / f * deta_dy
    v_g = g / f * deta_dx

    u_g = _apply_equatorial_mask(u_g, lat_dim=lat_dim, equatorial_mask_deg=equatorial_mask_deg)
    v_g = _apply_equatorial_mask(v_g, lat_dim=lat_dim, equatorial_mask_deg=equatorial_mask_deg)
    return u_g, v_g


def _compute_isotropic_spectrum_from_surface(
    field: xr.DataArray,
    dx_2d: xr.DataArray,
    dy_2d: xr.DataArray,
    lon_slice: slice,
    lat_slice: slice,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> tuple[np.ndarray, np.ndarray]:
    squeeze_dims = [dim for dim, size in field.sizes.items() if size == 1 and dim not in {"time", lat_dim, lon_dim}]
    if squeeze_dims:
        field = field.squeeze(squeeze_dims, drop=True)
    data = field.sel({lon_dim: lon_slice, lat_dim: lat_slice}).transpose("time", lat_dim, lon_dim)
    dx_mean = float(dx_2d.sel({lon_dim: lon_slice, lat_dim: lat_slice}).mean().values)
    dy_mean = float(dy_2d.sel({lon_dim: lon_slice, lat_dim: lat_slice}).mean().values)
    anomaly = data - data.mean("time")
    k_cent, spec = compute_isotropic_spectrum_torch(
        torch.as_tensor(anomaly.fillna(0.0).values),
        dx=dx_mean,
        dy=dy_mean,
        n_factor=2,
        detrend="linear",
        window="hann",
        cutoff_before_bins=False,
    )
    return k_cent.cpu().numpy() * 1000 * 2 * np.pi, spec.cpu().numpy().mean(0)


def _compute_temporal_ke_spectrum_from_surface(
    u: xr.DataArray,
    v: xr.DataArray,
    lon_slice: slice,
    lat_slice: slice,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> tuple[np.ndarray, np.ndarray]:
    u_reg = u.sel({lon_dim: lon_slice, lat_dim: lat_slice})
    v_reg = v.sel({lon_dim: lon_slice, lat_dim: lat_slice})
    ke = 0.5 * (u_reg ** 2 + v_reg ** 2)
    delta_days = float((pd.Timestamp(ke.time.values[1]) - pd.Timestamp(ke.time.values[0])).days)
    dt_years = delta_days / 365.25
    ke_values = ke.fillna(0.0).transpose("time", lat_dim, lon_dim).values
    nperseg = min(256, ke_values.shape[0])
    freqs, psd = signal.welch(
        ke_values,
        fs=1.0 / dt_years,
        axis=0,
        detrend="linear",
        window="hann",
        scaling="density",
        nperseg=nperseg,
    )
    psd_mean = np.nanmean(psd, axis=(1, 2))
    valid = freqs > 0
    return freqs[valid], psd_mean[valid]


def _rmse_between_curves(x_ref: np.ndarray, y_ref: np.ndarray, x_other: np.ndarray, y_other: np.ndarray) -> float:
    y_other_on_ref = np.interp(x_ref, x_other, y_other)
    valid = np.isfinite(y_ref) & np.isfinite(y_other_on_ref)
    if valid.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_ref[valid] - y_other_on_ref[valid]) ** 2)))


def _find_depth_name(ds: xr.Dataset | xr.DataArray) -> str:
    return _find_coord_name(ds, ["lev", "depth", "depth_0", "depth_std", "DEPTH", "z_t", "zlev"])


def _global_ohc_timeseries_from_temperature(
    ds: xr.Dataset,
    temp_var: str,
    area_name: str = "area",
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
) -> xr.DataArray:
    depth_name = _find_depth_name(ds)
    depth_vals = ds[depth_name].values.astype(float)
    dz = np.gradient(depth_vals)
    thetao = ds[temp_var]
    area = ds[area_name]
    out = {}
    for depth_slice in depth_slices:
        mask = (depth_vals >= depth_slice["min"]) & (depth_vals < depth_slice["max"])
        thetao_layer = thetao.isel({depth_name: mask})
        dz_layer = xr.DataArray(dz[mask], dims=[depth_name], coords={depth_name: thetao_layer[depth_name]})
        ohc = (thetao_layer * dz_layer * area * RHO * CP).sum([depth_name, "lat", "lon"])
        out[depth_slice["title"]] = ohc
    return xr.Dataset(out).to_array("depth_slice").transpose("time", "depth_slice")


def _global_ohc_timeseries_from_precomputed(
    ds: xr.Dataset,
    area_name: str = "area",
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
) -> xr.DataArray:
    candidate_sets = [
        {
            "Surface (0-700m)": ["ohc_upper", "ohc_700", "ohc0_700"],
            "Intermediate (700-2000m)": ["ohc_middle", "ohc_700_2000", "ohc700_2000"],
            "Deep (2000-7000m)": ["ohc_deep", "ohc_2000_bottom", "ohc2000_bottom"],
        }
    ]
    for mapping in candidate_sets:
        try:
            arrays = {}
            for title, aliases in mapping.items():
                arrays[title] = ds[_find_var_name(ds, aliases)]
            stacked = xr.Dataset(arrays).to_array("depth_slice").transpose("time", "depth_slice", ...)
            if {"lat", "lon"}.issubset(stacked.dims):
                stacked = (stacked * ds[area_name]).sum(["lat", "lon"])
            titles = [depth_slice["title"] for depth_slice in depth_slices]
            return stacked.sel(depth_slice=titles).transpose("time", "depth_slice")
        except KeyError:
            continue
    raise KeyError("Could not find ARGO-IAP OHC variables or temperature field")


def _compute_sim_ohc_timeseries(
    ds_sim: xr.Dataset,
    year: int,
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
) -> xr.DataArray:
    ds_year = _year_slice(ds_sim, year)
    area = xr.DataArray(
        ds_year["areacello"].values,
        dims=["y", "x"],
        coords={"y": ds_year["y"], "x": ds_year["x"]},
    )
    wet = xr.where(np.isfinite(ds_year["thetao"].isel(time=0)), 1.0, 0.0)
    out = {}
    for depth_slice in depth_slices:
        thetao_layer = ds_year["thetao"].sel(lev=slice(depth_slice["min"], depth_slice["max"]))
        dz_layer = ds_year["dz"].sel(lev=slice(depth_slice["min"], depth_slice["max"]))
        wet_layer = wet.sel(lev=slice(depth_slice["min"], depth_slice["max"]))
        ohc = (thetao_layer * dz_layer * wet_layer * area * RHO * CP).sum(["lev", "y", "x"])
        monthly = ohc.resample(time="MS").mean()
        out[depth_slice["title"]] = monthly
    return xr.Dataset(out).to_array("depth_slice").transpose("time", "depth_slice")


def _compute_obs_ohc_timeseries(
    ds_obs: xr.Dataset,
    year: int,
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
) -> xr.DataArray:
    ds_obs = _year_slice(ds_obs, year)
    if "area" not in ds_obs:
        ds_obs = _infer_obs_grid_metrics(ds_obs)
    try:
        temp_var = _find_var_name(ds_obs, ["thetao", "temperature", "temp", "T"])
        out = _global_ohc_timeseries_from_temperature(ds_obs, temp_var, depth_slices=depth_slices)
    except KeyError:
        out = _global_ohc_timeseries_from_precomputed(ds_obs, depth_slices=depth_slices)
    return out.resample(time="MS").mean()


def _area_weighted_mean(field: xr.DataArray, area: xr.DataArray) -> float:
    mask = np.isfinite(field)
    weighted_sum = (field.where(mask) * area).sum()
    weight_sum = area.where(mask).sum()
    if float(weight_sum.values) == 0.0:
        return float("nan")
    return float((weighted_sum / weight_sum).values)


def _plot_eke_map_comparison(duacs: xr.Dataset, emulators: dict[str, dict], year: int) -> pd.DataFrame:
    u_obs_name = _find_var_name(duacs, ["ugos", "uo", "u", "surface_geostrophic_eastward_sea_water_velocity"])
    v_obs_name = _find_var_name(duacs, ["vgos", "vo", "v", "surface_geostrophic_northward_sea_water_velocity"])
    obs_year = _year_slice(duacs, year)
    obs_u = _apply_equatorial_mask(obs_year[u_obs_name], lat_dim="lat")
    obs_v = _apply_equatorial_mask(obs_year[v_obs_name], lat_dim="lat")
    obs_eke = _surface_eke_from_uv(obs_u, obs_v)
    obs_area = obs_year["area"]
    obs_mean = _area_weighted_mean(obs_eke, obs_area)

    fig, axes = plt.subplots(
        1,
        len(emulators) + 1,
        figsize=(5.2 * (len(emulators) + 1), 4.4),
        subplot_kw={"projection": ccrs.Robinson(central_longitude=210)},
    )
    axes = np.atleast_1d(axes)

    vmax = np.nanpercentile(obs_eke.values, 99)
    norm = colors.LogNorm(vmin=max(vmax / 1e4, 1e-8), vmax=vmax)

    rows = []
    panels = [("DUACS", obs_eke, obs_mean)]
    for label, emu in emulators.items():
        sim_year, _ = _temporal_overlap(_year_slice(emu["ds"], year), obs_year)
        sim_ug, sim_vg = _geostrophic_velocity_from_zos(sim_year["zos"], lat_dim="y", lon_dim="x")
        sim_eke = _surface_eke_from_uv(sim_ug, sim_vg)
        sim_mean = _area_weighted_mean(sim_eke, sim_year["areacello"])
        rows.append({"model": label, "metric": "global_mean_geostrophic_eke", "value": sim_mean})
        rows.append({"model": label, "metric": "global_mean_geostrophic_eke_bias", "value": sim_mean - obs_mean})
        sim_eke_plot = sim_eke.rename({"y": "lat", "x": "lon"})
        panels.append((label, sim_eke_plot, sim_mean))

    for ax, (title, field, mean_val) in zip(axes, panels):
        mesh = field.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            x="lon",
            y="lat",
            cmap=cmocean.cm.thermal,
            norm=norm,
            add_colorbar=False,
        )
        ax.add_feature(cfeature.LAND, zorder=10, edgecolor="black", facecolor="lightgray")
        ax.coastlines(zorder=11)
        ax.set_title(f"{title}  mean={mean_val:.2e}", fontsize=13)

    cbar = fig.colorbar(
        mesh,
        ax=axes,
        orientation="horizontal",
        shrink=0.58,
        aspect=36,
        fraction=0.04,
        pad=0.14,
        label="Geostrophic EKE (m$^2$/s$^2$)",
    )
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Geostrophic EKE (m$^2$/s$^2$)", fontsize=12)
    fig.suptitle(f"Surface Geostrophic EKE vs DUACS ({year}; |lat| > 5$^\\circ$)", y=0.935, fontsize=15)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.18, wspace=0.03)
    return pd.DataFrame(rows)


def _gaussian_lowpass_field(
    da: xr.DataArray,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    sigma_deg: float = 0.5,
) -> xr.DataArray:
    from scipy.ndimage import gaussian_filter

    lat_vals = np.asarray(da[lat_dim].values, dtype=float)
    lon_vals = np.asarray(da[lon_dim].values, dtype=float)
    dlat = abs(float(np.median(np.diff(lat_vals))))
    dlon = abs(float(np.median(np.diff(lon_vals))))
    sigma_lat = sigma_deg / max(dlat, 1e-9)
    sigma_lon = sigma_deg / max(dlon, 1e-9)

    dims = list(da.dims)
    lat_ax = dims.index(lat_dim)
    lon_ax = dims.index(lon_dim)
    spatial_axes = (lat_ax, lon_ax)

    raw = np.asarray(da.values, dtype=np.float32)
    valid_mask = np.isfinite(raw)
    filled = np.where(valid_mask, raw, 0.0).astype(np.float32)
    weights = valid_mask.astype(np.float32)

    if "time" in dims:
        time_ax = dims.index("time")
        result = np.empty_like(filled)
        moved_filled = np.moveaxis(filled, time_ax, 0)
        moved_weights = np.moveaxis(weights, time_ax, 0)
        moved_result = np.empty_like(moved_filled)
        new_lat_ax = spatial_axes[0] - (1 if time_ax < spatial_axes[0] else 0)
        new_lon_ax = spatial_axes[1] - (1 if time_ax < spatial_axes[1] else 0)
        per_step_sigma = [0.0] * (moved_filled.ndim - 1)
        per_step_sigma[new_lat_ax] = sigma_lat
        per_step_sigma[new_lon_ax] = sigma_lon
        for t in range(moved_filled.shape[0]):
            smoothed = gaussian_filter(moved_filled[t], sigma=per_step_sigma, mode="reflect")
            norm = gaussian_filter(moved_weights[t], sigma=per_step_sigma, mode="reflect")
            with np.errstate(invalid="ignore", divide="ignore"):
                moved_result[t] = np.where(norm > 1e-6, smoothed / norm, np.nan)
        result = np.moveaxis(moved_result, 0, time_ax)
    else:
        sigma_vec = [0.0] * len(dims)
        sigma_vec[lat_ax] = sigma_lat
        sigma_vec[lon_ax] = sigma_lon
        smoothed = gaussian_filter(filled, sigma=sigma_vec, mode="reflect")
        norm = gaussian_filter(weights, sigma=sigma_vec, mode="reflect")
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(norm > 1e-6, smoothed / norm, np.nan)

    result = np.where(valid_mask, result, np.nan)
    return xr.DataArray(result, dims=da.dims, coords=da.coords, attrs=da.attrs)


def _plot_eke_map_lowpass_comparison(
    duacs: xr.Dataset,
    emulators: dict[str, dict],
    year: int,
    sigma_deg: float = 0.5,
) -> pd.DataFrame:
    u_obs_name = _find_var_name(duacs, ["ugos", "uo", "u", "surface_geostrophic_eastward_sea_water_velocity"])
    v_obs_name = _find_var_name(duacs, ["vgos", "vo", "v", "surface_geostrophic_northward_sea_water_velocity"])
    obs_year = _year_slice(duacs, year)
    obs_u = _apply_equatorial_mask(obs_year[u_obs_name], lat_dim="lat")
    obs_v = _apply_equatorial_mask(obs_year[v_obs_name], lat_dim="lat")
    obs_u_lp = _gaussian_lowpass_field(obs_u, lat_dim="lat", lon_dim="lon", sigma_deg=sigma_deg)
    obs_v_lp = _gaussian_lowpass_field(obs_v, lat_dim="lat", lon_dim="lon", sigma_deg=sigma_deg)
    obs_eke = _surface_eke_from_uv(obs_u_lp, obs_v_lp)
    obs_area = obs_year["area"]
    obs_mean = _area_weighted_mean(obs_eke, obs_area)

    fig, axes = plt.subplots(
        1,
        len(emulators) + 1,
        figsize=(5.2 * (len(emulators) + 1), 4.4),
        subplot_kw={"projection": ccrs.Robinson(central_longitude=210)},
    )
    axes = np.atleast_1d(axes)

    vmax = np.nanpercentile(obs_eke.values, 99)
    norm = colors.LogNorm(vmin=max(vmax / 1e4, 1e-8), vmax=vmax)

    rows = []
    panels = [("DUACS", obs_eke, obs_mean)]
    for label, emu in emulators.items():
        sim_year, _ = _temporal_overlap(_year_slice(emu["ds"], year), obs_year)
        sim_ug, sim_vg = _geostrophic_velocity_from_zos(sim_year["zos"], lat_dim="y", lon_dim="x")
        sim_ug_lp = _gaussian_lowpass_field(sim_ug, lat_dim="y", lon_dim="x", sigma_deg=sigma_deg)
        sim_vg_lp = _gaussian_lowpass_field(sim_vg, lat_dim="y", lon_dim="x", sigma_deg=sigma_deg)
        sim_eke = _surface_eke_from_uv(sim_ug_lp, sim_vg_lp)
        sim_mean = _area_weighted_mean(sim_eke, sim_year["areacello"])
        rows.append({"model": label, "metric": "global_mean_geostrophic_eke_lowpass", "value": sim_mean})
        rows.append({"model": label, "metric": "global_mean_geostrophic_eke_lowpass_bias", "value": sim_mean - obs_mean})
        sim_eke_plot = sim_eke.rename({"y": "lat", "x": "lon"})
        panels.append((label, sim_eke_plot, sim_mean))

    for ax, (title, field, mean_val) in zip(axes, panels):
        mesh = field.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            x="lon",
            y="lat",
            cmap=cmocean.cm.thermal,
            norm=norm,
            add_colorbar=False,
        )
        ax.add_feature(cfeature.LAND, zorder=10, edgecolor="black", facecolor="lightgray")
        ax.coastlines(zorder=11)
        ax.set_title(f"{title}  mean={mean_val:.2e}", fontsize=13)

    cbar = fig.colorbar(
        mesh,
        ax=axes,
        orientation="horizontal",
        shrink=0.58,
        aspect=36,
        fraction=0.04,
        pad=0.14,
        label="Low-pass Geostrophic EKE (m$^2$/s$^2$)",
    )
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Low-pass Geostrophic EKE (m$^2$/s$^2$)", fontsize=12)
    fig.suptitle(
        f"Low-pass ($\\sigma$={sigma_deg}$^\\circ$ Gaussian) Geostrophic EKE vs DUACS "
        f"({year}; |lat| > 5$^\\circ$)",
        y=0.935,
        fontsize=14,
    )
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.18, wspace=0.03)
    return pd.DataFrame(rows)


def _plot_temperature_spectra(oisst: xr.Dataset, emulators: dict[str, dict]) -> pd.DataFrame:
    sst_name = _find_var_name(oisst, ["sst", "analysed_sst", "thetao", "tos"])
    rows = []
    n_regions = len(SPECTRUM_REGIONS)
    nrows = 2
    ncols = (n_regions + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows))
    axes = np.ravel(axes)

    for idx, (region_name, lon_slice, lat_slice) in enumerate(SPECTRUM_REGIONS):
        ax = axes[idx]
        k_obs, spec_obs = _compute_isotropic_spectrum_from_surface(
            oisst[sst_name], oisst["dx"], oisst["dy"], lon_slice, lat_slice
        )
        ax.loglog(k_obs, spec_obs, "--", color="black", linewidth=2, label="OISST")
        for label, emu in emulators.items():
            sim, _ = _temporal_overlap(emu["ds"], oisst)
            sim_sst = sim["thetao"].isel(lev=0)
            k_sim, spec_sim = _compute_isotropic_spectrum_from_surface(
                sim_sst, sim["dx"], sim["dy"], lon_slice, lat_slice,
                lat_dim="y", lon_dim="x",
            )
            rmse = _rmse_between_curves(k_obs, spec_obs, k_sim, spec_sim)
            rows.append({"model": label, "region": region_name, "metric": "surface_temp_isotropic_spectrum_rmse", "value": rmse})
            ax.loglog(k_sim, spec_sim, "-", color=emu["color"], linewidth=2, label=label)
        ax.set_title(region_name)
        ax.set_xlabel(r"Wavenumber $\kappa$ (km$^{-1}$)")
        ax.set_ylabel("PSD")
        ax.grid(True, which="both", alpha=0.3)

    handles, labels, ncol = _legend_truth_first(*axes[0].get_legend_handles_labels(), truth_labels={"OISST"})
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=11)
    fig.suptitle("Regional Surface Temperature Spectra vs OISST", y=0.955, fontsize=15)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.14, wspace=0.25, hspace=0.32)
    return pd.DataFrame(rows)


def _plot_temporal_ke_spectra(duacs: xr.Dataset, emulators: dict[str, dict]) -> pd.DataFrame:
    u_name = _find_var_name(duacs, ["ugos", "uo", "u", "surface_geostrophic_eastward_sea_water_velocity"])
    v_name = _find_var_name(duacs, ["vgos", "vo", "v", "surface_geostrophic_northward_sea_water_velocity"])

    obs_times = pd.DatetimeIndex(duacs.time.values)
    obs_span_years = (obs_times.max() - obs_times.min()).days / 365.25
    if obs_span_years < 3:
        print(
            f"  WARNING: temporal KE spectrum uses only {obs_span_years:.1f} years of data; "
            "Welch PSD has very low spectral degrees of freedom and curves below "
            "~1 cycle/year are not statistically reliable."
        )

    rows = []
    n_regions = len(SPECTRUM_REGIONS)
    nrows = 2
    ncols = (n_regions + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows))
    axes = np.ravel(axes)

    for idx, (region_name, lon_slice, lat_slice) in enumerate(SPECTRUM_REGIONS):
        ax = axes[idx]
        freq_obs, psd_obs = _compute_temporal_ke_spectrum_from_surface(duacs[u_name], duacs[v_name], lon_slice, lat_slice)
        ax.loglog(freq_obs, psd_obs, "--", color="black", linewidth=2, label="DUACS")
        for label, emu in emulators.items():
            sim, _ = _temporal_overlap(emu["ds"], duacs)
            sim_ug, sim_vg = _geostrophic_velocity_from_zos(sim["zos"], lat_dim="y", lon_dim="x")
            freq_sim, psd_sim = _compute_temporal_ke_spectrum_from_surface(
                sim_ug, sim_vg, lon_slice, lat_slice, lat_dim="y", lon_dim="x",
            )
            rmse = _rmse_between_curves(freq_obs, psd_obs, freq_sim, psd_sim)
            rows.append({"model": label, "region": region_name, "metric": "surface_geostrophic_ke_temporal_spectrum_rmse", "value": rmse})
            ax.loglog(freq_sim, psd_sim, "-", color=emu["color"], linewidth=2, label=label)
        ax.set_title(region_name)
        ax.set_xlabel("Frequency (cycles/year)")
        ax.set_ylabel("PSD")
        ax.grid(True, which="both", alpha=0.3)

    handles, labels, ncol = _legend_truth_first(*axes[0].get_legend_handles_labels(), truth_labels={"DUACS"})
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=11)
    fig.suptitle("Regional Surface Geostrophic KE Temporal Spectra vs DUACS", y=0.955, fontsize=15)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.14, wspace=0.25, hspace=0.32)
    return pd.DataFrame(rows)


def _plot_ohc_timeseries(argo_iap: xr.Dataset, emulators: dict[str, dict], year: int) -> pd.DataFrame:
    obs_ohc = _compute_obs_ohc_timeseries(argo_iap, year)
    fig, axes = plt.subplots(len(OHC_DEPTH_SLICES), 1, figsize=(11.5, 3.8 * len(OHC_DEPTH_SLICES)), sharex=True)
    axes = np.atleast_1d(axes)
    rows = []

    for depth_idx, depth_slice in enumerate(OHC_DEPTH_SLICES):
        ax = axes[depth_idx]
        depth_title = depth_slice["title"]
        obs_series = obs_ohc.sel(depth_slice=depth_title).to_series()
        ax.plot(obs_series.index, obs_series.values, "--", color="black", linewidth=2, label="ARGO-IAP")

        for label, emu in emulators.items():
            sim_ohc = _compute_sim_ohc_timeseries(emu["ds"], year)
            sim_series = sim_ohc.sel(depth_slice=depth_title).to_series()
            common_index = obs_series.index.intersection(sim_series.index)
            if common_index.empty:
                rmse = np.nan
            else:
                rmse = float(np.sqrt(np.mean((sim_series.loc[common_index] - obs_series.loc[common_index]) ** 2)))
            rows.append({"model": label, "depth": depth_title, "metric": "monthly_ohc_rmse", "value": rmse})
            ax.plot(sim_series.index, sim_series.values, "-", color=emu["color"], linewidth=2, label=label)

        ax.set_title(f"{depth_title}", fontsize=14)
        ax.set_ylabel("OHC (J)", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)

    handles, labels, ncol = _legend_truth_first(*axes[0].get_legend_handles_labels(), truth_labels={"ARGO-IAP"})
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, -0.015),
        frameon=False,
        fontsize=12,
    )
    axes[-1].set_xlabel("Time", fontsize=13)
    fig.suptitle(f"Monthly Global OHC vs ARGO-IAP ({year})", y=0.965, fontsize=16)
    fig.tight_layout(rect=[0, 0.04, 1, 0.955])
    return pd.DataFrame(rows)


def _plot_ohc_anomaly_timeseries(argo_iap: xr.Dataset, emulators: dict[str, dict], year: int) -> pd.DataFrame:
    obs_ohc = _compute_obs_ohc_timeseries(argo_iap, year)
    fig, axes = plt.subplots(len(OHC_DEPTH_SLICES), 1, figsize=(11.5, 3.8 * len(OHC_DEPTH_SLICES)), sharex=True)
    axes = np.atleast_1d(axes)
    rows = []

    for depth_idx, depth_slice in enumerate(OHC_DEPTH_SLICES):
        ax = axes[depth_idx]
        depth_title = depth_slice["title"]
        obs_series = obs_ohc.sel(depth_slice=depth_title).to_series()
        obs_anom = obs_series - obs_series.mean()
        ax.plot(obs_anom.index, obs_anom.values, "--", color="black", linewidth=2, label="ARGO-IAP")

        for label, emu in emulators.items():
            sim_ohc = _compute_sim_ohc_timeseries(emu["ds"], year)
            sim_series = sim_ohc.sel(depth_slice=depth_title).to_series()
            sim_anom = sim_series - sim_series.mean()
            common_index = obs_anom.index.intersection(sim_anom.index)
            if common_index.empty:
                rmse = np.nan
            else:
                rmse = float(np.sqrt(np.mean((sim_anom.loc[common_index] - obs_anom.loc[common_index]) ** 2)))
            rows.append({"model": label, "depth": depth_title, "metric": "monthly_ohc_anomaly_rmse", "value": rmse})
            ax.plot(sim_anom.index, sim_anom.values, "-", color=emu["color"], linewidth=2, label=label)

        ax.set_title(f"{depth_title}", fontsize=14)
        ax.set_ylabel("Anomaly (J)", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)

    handles, labels, ncol = _legend_truth_first(*axes[0].get_legend_handles_labels(), truth_labels={"ARGO-IAP"})
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, -0.015),
        frameon=False,
        fontsize=12,
    )
    axes[-1].set_xlabel("Time", fontsize=13)
    fig.suptitle(f"Monthly OHC Anomaly vs ARGO-IAP ({year})", y=0.965, fontsize=16)
    fig.tight_layout(rect=[0, 0.04, 1, 0.955])
    return pd.DataFrame(rows)


def main() -> None:
    print("Loading emulator rollouts...")
    emulators = {}
    for path, label, color in EMULATOR_CONFIGS:
        print(f" - {label}: {path}")
        emulators[label] = {"ds": _prepare_emulator_dataset(path), "color": color}

    print("Loading observations...")
    duacs = _infer_obs_grid_metrics(_drop_singleton_nonspatial_dims(_standardize_latlon_time(_maybe_open_dataset(OBS_CONFIG["duacs"]))))
    oisst = _infer_obs_grid_metrics(_drop_singleton_nonspatial_dims(_standardize_latlon_time(_maybe_open_dataset(OBS_CONFIG["oisst"]))))
    argo_iap = _drop_singleton_nonspatial_dims(_standardize_latlon_time(_maybe_open_dataset(OBS_CONFIG["argo_iap"])))
    if {"lat", "lon"}.issubset(argo_iap.dims) and "area" not in argo_iap:
        argo_iap = _infer_obs_grid_metrics(argo_iap)

    print("Evaluating EKE maps vs DUACS...")
    eke_metrics = _plot_eke_map_comparison(duacs, emulators, args.year)
    save_current_fig(f"eke_mean_vs_duacs_{args.year}", MAP_DIR)

    print("Evaluating low-pass-filtered EKE maps vs DUACS...")
    eke_lp_metrics = _plot_eke_map_lowpass_comparison(duacs, emulators, args.year)
    save_current_fig(f"eke_mean_lowpass_vs_duacs_{args.year}", MAP_DIR)

    print("Evaluating isotropic temperature spectra vs OISST...")
    temp_spec_metrics = _plot_temperature_spectra(oisst, emulators)
    save_current_fig("temperature_isotropic_spectra_vs_oisst", SPECTRUM_DIR)

    print("Evaluating temporal KE spectra vs DUACS...")
    ke_spec_metrics = _plot_temporal_ke_spectra(duacs, emulators)
    save_current_fig("ke_temporal_spectra_vs_duacs", SPECTRUM_DIR)

    print("Evaluating layered monthly OHC vs ARGO-IAP...")
    ohc_metrics = _plot_ohc_timeseries(argo_iap, emulators, args.year)
    save_current_fig(f"ohc_monthly_vs_argo_iap_{args.year}", TIMESERIES_DIR)

    print("Evaluating monthly OHC anomaly vs ARGO-IAP...")
    ohc_anomaly_metrics = _plot_ohc_anomaly_timeseries(argo_iap, emulators, args.year)
    save_current_fig(f"ohc_anomaly_vs_argo_iap_{args.year}", TIMESERIES_DIR)

    metrics = pd.concat([eke_metrics, eke_lp_metrics, temp_spec_metrics, ke_spec_metrics, ohc_metrics, ohc_anomaly_metrics], ignore_index=True)
    metrics_path = os.path.join(METRICS_DIR, f"observation_metrics_{args.year}.csv")
    metrics.to_csv(metrics_path, index=False)
    print(f"  -> Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
