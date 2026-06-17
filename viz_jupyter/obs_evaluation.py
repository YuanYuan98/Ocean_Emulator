"""
Observation-based evaluation driver for Samudra rollouts.

This script mirrors the existing notebook-exported evaluation drivers, but it
compares emulator rollouts against observational products instead of OM4 truth.

Implemented evaluations
-----------------------
1. Surface mean EKE map over 2022 vs DUACS geostrophic velocity.
2. Regional surface isotropic temperature spectra vs OISST.
3. Regional surface temporal KE spectra vs raw and 5-day-averaged DUACS.
4. Annual-mean temperature and salinity structure vs ARGO-IAP.
5. Global monthly OHC time series for upper / middle / deep ocean vs ARGO-IAP.

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
    "--ohc-start-date",
    default=None,
    help="Optional start date for OHC diagnostics. Defaults to the common ARGO/model overlap.",
)
parser.add_argument(
    "--ohc-end-date",
    default=None,
    help="Optional end date for OHC diagnostics. Defaults to the common ARGO/model overlap.",
)
parser.add_argument(
    "--obs-climatology-start-date",
    default=None,
    help="Optional start date for ARGO-IAP monthly climatologies. Defaults to the full ARGO-IAP record.",
)
parser.add_argument(
    "--obs-climatology-end-date",
    default=None,
    help="Optional end date for ARGO-IAP monthly climatologies. Defaults to the full ARGO-IAP record.",
)
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
parser.add_argument(
    "--duacs-velocity-kind",
    choices=["absolute", "anomaly"],
    default="absolute",
    help=(
        "DUACS geostrophic velocity field to use. 'absolute' uses ugos/vgos "
        "from absolute dynamic topography; 'anomaly' uses ugosa/vgosa from SLA."
    ),
)
parser.add_argument(
    "--plot-deseasoned-anomaly",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Write monthly-climatology-removed anomaly time-series figures. "
        "Enabled by default; pass --no-plot-deseasoned-anomaly to skip."
    ),
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

DUACS_RAW_ZARR = (
    "duacs/"
    "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_multi-vars_"
    "179.94W-179.94E_89.94S-89.94N_2022-01-01-2023-01-01.zarr"
)

DUACS_VELOCITY_COMPONENTS = {
    "absolute": {
        "u": ["ugos", "surface_geostrophic_eastward_sea_water_velocity"],
        "v": ["vgos", "surface_geostrophic_northward_sea_water_velocity"],
        "ssh": "adt",
        "ssh_label": "absolute dynamic topography",
        "label": "absolute geostrophic velocity",
    },
    "anomaly": {
        "u": ["ugosa"],
        "v": ["vgosa"],
        "ssh": "sla",
        "ssh_label": "sea-level anomaly",
        "label": "geostrophic velocity anomaly",
    },
}

ARGO_STRUCTURE_VARIABLES = [
    {
        "model_var": "thetao",
        "obs_aliases": ["thetao", "temp", "temperature", "T"],
        "metric_prefix": "annual_mean_temperature",
        "timeseries_metric": "monthly_temperature_layer_mean_rmse",
        "filename": "temperature",
        "title": "Temperature",
        "units": "$^\\circ$C",
    },
    {
        "model_var": "so",
        "obs_aliases": ["so", "salinity", "salt", "sal", "psal", "S"],
        "metric_prefix": "annual_mean_salinity",
        "timeseries_metric": "monthly_salinity_layer_mean_rmse",
        "filename": "salinity",
        "title": "Salinity",
        "units": "psu",
    },
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
    fig.savefig(path, format="pdf", dpi=dpi, bbox_inches="tight", pad_inches=0.08)
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


def _figure_title_with_context(title: str, period_label: str | None = None, extra: str | None = None) -> str:
    parts = [title]
    context = []
    if period_label is not None:
        context.append(period_label)
    if extra is not None:
        context.append(extra)
    if context:
        parts.append("; ".join(context))
    return "\n".join(parts)


def _finish_timeseries_figure(
    fig,
    axes,
    title: str,
    period_label: str,
    xlabel: str,
    extra: str | None = None,
    top: float | None = None,
) -> None:
    axes = np.atleast_1d(axes)
    axes[-1].set_xlabel(xlabel, fontsize=13)
    title_text = _figure_title_with_context(title, period_label, extra)
    if top is None:
        top = 0.955 if extra else 0.965
    fig.suptitle(
        title_text,
        y=0.94,
        fontsize=15,
        linespacing=1.15,
    )
    fig.tight_layout(rect=[0.025, 0.09, 0.99, top])


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = "Figures_observations"
PDF_DIR = os.path.join(OUTPUT_DIR, "pdf_figures")
MAP_DIR = os.path.join(PDF_DIR, "maps")
SPECTRUM_DIR = os.path.join(PDF_DIR, "spectrum")
TIMESERIES_DIR = os.path.join(PDF_DIR, "timeseries")
STRUCTURE_DIR = os.path.join(PDF_DIR, "structure")
METRICS_DIR = os.path.join(PDF_DIR, "metrics")

for directory in [OUTPUT_DIR, PDF_DIR, MAP_DIR, SPECTRUM_DIR, TIMESERIES_DIR, STRUCTURE_DIR, METRICS_DIR]:
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
    "duacs_raw": args.raw_obs_root / DUACS_RAW_ZARR,
    "oisst": args.aligned_obs_root / "oisst.zarr",
    "oisst_daily": args.aligned_obs_root / "oisst_daily.zarr",
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


def _find_var_name_optional(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in ds.data_vars:
            return name
    return None


def _duacs_velocity_names(ds: xr.Dataset, velocity_kind: str) -> tuple[str, str]:
    config = DUACS_VELOCITY_COMPONENTS[velocity_kind]
    return _find_var_name(ds, config["u"]), _find_var_name(ds, config["v"])


def _duacs_velocity_label(velocity_kind: str) -> str:
    return DUACS_VELOCITY_COMPONENTS[velocity_kind]["label"]


def _missing_argo_structure_variables(ds: xr.Dataset) -> list[str]:
    return [
        spec["title"]
        for spec in ARGO_STRUCTURE_VARIABLES
        if _find_var_name_optional(ds, spec["obs_aliases"]) is None
    ]


def _require_argo_structure_variables(ds: xr.Dataset) -> None:
    missing = _missing_argo_structure_variables(ds)
    if missing:
        raise KeyError(
            "ARGO-IAP temperature/salinity structure diagnostic requires all requested variables. "
            f"Missing: {missing}. Available variables: {list(ds.data_vars)}. "
            "Add the missing ARGO salinity product to the prepared argo-iap.zarr before running this diagnostic."
        )


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


def _native_model_wet_mask(ds: xr.Dataset) -> xr.DataArray | None:
    if "thetao" not in ds:
        return None
    return np.isfinite(ds["thetao"].isel(time=0)).transpose("y", "x", "lev")


def _surface_wet_mask(wet_mask: xr.DataArray | None) -> xr.DataArray | None:
    if wet_mask is None:
        return None
    if "lev" in wet_mask.dims:
        return wet_mask.isel(lev=0)
    return wet_mask


def _same_native_grid(a: xr.Dataset, b: xr.DataArray) -> bool:
    return (
        "y" in b.dims
        and "x" in b.dims
        and a.sizes.get("y") == b.sizes.get("y")
        and a.sizes.get("x") == b.sizes.get("x")
        and np.allclose(a["y"].values, b["y"].values)
        and np.allclose(a["x"].values, b["x"].values)
    )


def _attach_common_model_wet_masks(emulators: dict[str, dict]) -> None:
    candidates = []
    for label, emu in emulators.items():
        mask = _native_model_wet_mask(emu["ds"])
        if mask is None:
            continue
        surface_fraction = float(_surface_wet_mask(mask).mean().compute().values)
        candidates.append((surface_fraction, label, mask))

    if not candidates:
        return

    _, reference_label, reference_mask = min(candidates, key=lambda item: item[0])
    print(f"Using {reference_label} finite ocean mask for model-native diagnostics.")
    for _, label, own_mask in candidates:
        ds = emulators[label]["ds"]
        if _same_native_grid(ds, reference_mask):
            emulators[label]["wet_mask"] = reference_mask
            emulators[label]["wet_mask_source"] = reference_label
        else:
            emulators[label]["wet_mask"] = own_mask
            emulators[label]["wet_mask_source"] = label


def _apply_native_wet_mask(field: xr.DataArray, wet_mask: xr.DataArray | None) -> xr.DataArray:
    if wet_mask is None:
        return field
    mask = wet_mask
    if "lev" not in field.dims and "lev" in mask.dims:
        mask = mask.isel(lev=0)
    return field.where(mask)


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


def _depth_thickness_from_centers(depth_vals: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_vals, dtype=float)
    if depth.ndim != 1 or depth.size < 2:
        raise ValueError("Expected at least two 1D depth centers")
    edges = np.empty(depth.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (depth[:-1] + depth[1:])
    edges[0] = max(0.0, depth[0] - 0.5 * (depth[1] - depth[0]))
    edges[-1] = depth[-1] + 0.5 * (depth[-1] - depth[-2])
    return np.diff(edges)


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


def _expand_numeric_slice(selection: slice, margin: float) -> slice:
    start = selection.start - margin if selection.start is not None else None
    stop = selection.stop + margin if selection.stop is not None else None
    return slice(start, stop, selection.step)


def _date_or_none(value: str | None) -> pd.Timestamp | None:
    return pd.Timestamp(value) if value else None


def _time_bounds(ds: xr.Dataset | xr.DataArray) -> tuple[pd.Timestamp, pd.Timestamp]:
    times = pd.DatetimeIndex(ds.time.values)
    if times.empty:
        raise ValueError("Dataset has an empty time coordinate")
    return pd.Timestamp(times.min()), pd.Timestamp(times.max())


def _resolve_ohc_period(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    start_date: str | None,
    end_date: str | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    starts = [_time_bounds(argo_iap)[0]]
    ends = [_time_bounds(argo_iap)[1]]
    for emu in emulators.values():
        start, end = _time_bounds(emu["ds"])
        starts.append(start)
        ends.append(end)

    start = _date_or_none(start_date) or max(starts)
    end = _date_or_none(end_date) or min(ends)
    if start > end:
        raise ValueError(f"No OHC temporal overlap for requested period: start={start}, end={end}")
    return start, end


def _resolve_obs_climatology_period(
    argo_iap: xr.Dataset,
    start_date: str | None,
    end_date: str | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    obs_start, obs_end = _time_bounds(argo_iap)
    start = _date_or_none(start_date) or obs_start
    end = _date_or_none(end_date) or obs_end
    if start > end:
        raise ValueError(f"No ARGO-IAP climatology period for requested dates: start={start}, end={end}")
    return start, end


def _period_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    if start.year == end.year and start.month == 1 and end.month == 12:
        return str(start.year)
    return f"{start:%Y-%m} to {end:%Y-%m}"


def _period_slug(start: pd.Timestamp, end: pd.Timestamp) -> str:
    if start.year == end.year and start.month == 1 and end.month == 12:
        return str(start.year)
    return f"{start:%Y%m}_{end:%Y%m}"


def _series_monthly_climatology(series: pd.Series) -> pd.Series:
    return series.groupby(series.index.month).mean()


def _series_without_given_monthly_climatology(series: pd.Series, climatology: pd.Series) -> pd.Series:
    seasonal = pd.Series(series.index.month, index=series.index).map(climatology)
    return series - seasonal


def _series_without_monthly_climatology(series: pd.Series) -> pd.Series:
    climatology = _series_monthly_climatology(series)
    return _series_without_given_monthly_climatology(series, climatology)


def _series_without_linear_trend(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    finite = np.isfinite(values.values)
    if finite.sum() < 2:
        return values - values.mean()

    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x[finite], values.values[finite], 1)
    trend = pd.Series(slope * x + intercept, index=values.index)
    return values - trend


def _has_repeated_calendar_months(series: pd.Series) -> bool:
    counts = series.groupby(series.index.month).size()
    return bool(len(counts) == 12 and (counts >= 2).all())


def _surface_eke_from_uv(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    u_anom = u - u.mean("time")
    v_anom = v - v.mean("time")
    return 0.5 * (u_anom ** 2 + v_anom ** 2).mean("time")


def _time_mean_velocity_anomaly(u: xr.DataArray, v: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    return u - u.mean("time"), v - v.mean("time")


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
    ssta = _calendar_day_climatology_anomaly(data)
    valid_space = np.isfinite(ssta).any("time")
    ssta = ssta.where(valid_space)
    k_cent, spec = compute_isotropic_spectrum_torch(
        torch.as_tensor(ssta.fillna(0.0).values),
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
    valid_space = np.isfinite(ke).any("time")
    delta_days = float((pd.Timestamp(ke.time.values[1]) - pd.Timestamp(ke.time.values[0])).days)
    dt_years = delta_days / 365.25
    ke_values = ke.where(valid_space).transpose("time", lat_dim, lon_dim).values
    valid_flat = np.asarray(valid_space.values).reshape(-1)
    ke_values = ke_values.reshape(ke_values.shape[0], -1)[:, valid_flat]
    if ke_values.shape[1] == 0:
        return np.array([]), np.array([])
    col_mean = np.nanmean(ke_values, axis=0)
    ke_values = np.where(np.isfinite(ke_values), ke_values, col_mean[None, :])
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
    psd_mean = np.nanmean(psd, axis=1)
    valid = freqs > 0
    return freqs[valid], psd_mean[valid]


def _calendar_day_climatology_anomaly(field: xr.DataArray, time_dim: str = "time") -> xr.DataArray:
    times = _coerce_datetime_values(field[time_dim].values)
    calendar_day = xr.DataArray(
        np.asarray([f"{timestamp.month:02d}-{timestamp.day:02d}" for timestamp in times]),
        dims=[time_dim],
        coords={time_dim: field[time_dim]},
        name="calendar_day",
    )
    with_calendar_day = field.assign_coords(calendar_day=calendar_day)
    climatology = with_calendar_day.groupby("calendar_day").mean(time_dim, skipna=True)
    anomaly = with_calendar_day.groupby("calendar_day") - climatology
    return anomaly.drop_vars("calendar_day")


def _rmse_between_curves(x_ref: np.ndarray, y_ref: np.ndarray, x_other: np.ndarray, y_other: np.ndarray) -> float:
    x_ref = np.asarray(x_ref, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    x_other = np.asarray(x_other, dtype=float)
    y_other = np.asarray(y_other, dtype=float)

    ref_valid = np.isfinite(x_ref) & np.isfinite(y_ref)
    other_valid = np.isfinite(x_other) & np.isfinite(y_other)
    if ref_valid.sum() == 0 or other_valid.sum() < 2:
        return np.nan

    order = np.argsort(x_other[other_valid])
    x_other_valid = x_other[other_valid][order]
    y_other_valid = y_other[other_valid][order]
    domain = ref_valid & (x_ref >= x_other_valid[0]) & (x_ref <= x_other_valid[-1])
    if domain.sum() == 0:
        return np.nan
    y_other_on_ref = np.interp(x_ref[domain], x_other_valid, y_other_valid)
    valid = np.isfinite(y_ref[domain]) & np.isfinite(y_other_on_ref)
    if valid.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_ref[domain][valid] - y_other_on_ref[valid]) ** 2)))


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
    start: pd.Timestamp,
    end: pd.Timestamp,
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
    wet_mask: xr.DataArray | None = None,
) -> xr.DataArray:
    ds_year = ds_sim.sel(time=slice(start, end))
    if ds_year.sizes.get("time", 0) == 0:
        raise ValueError(f"No model OHC samples in requested period {start} to {end}")
    area = xr.DataArray(
        ds_year["areacello"].values,
        dims=["y", "x"],
        coords={"y": ds_year["y"], "x": ds_year["x"]},
    )
    wet = wet_mask if wet_mask is not None else np.isfinite(ds_year["thetao"].isel(time=0))
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
    start: pd.Timestamp,
    end: pd.Timestamp,
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
) -> xr.DataArray:
    ds_obs = ds_obs.sel(time=slice(start, end))
    if ds_obs.sizes.get("time", 0) == 0:
        raise ValueError(f"No ARGO-IAP OHC samples in requested period {start} to {end}")
    if "area" not in ds_obs:
        ds_obs = _infer_obs_grid_metrics(ds_obs)
    try:
        temp_var = _find_var_name(ds_obs, ["thetao", "temperature", "temp", "T"])
        out = _global_ohc_timeseries_from_temperature(ds_obs, temp_var, depth_slices=depth_slices)
    except KeyError:
        out = _global_ohc_timeseries_from_precomputed(ds_obs, depth_slices=depth_slices)
    return out.resample(time="MS").mean()


def _collect_ohc_timeseries(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    obs_ohc = _compute_obs_ohc_timeseries(argo_iap, start, end)
    sim_ohc = {
        label: _compute_sim_ohc_timeseries(emu["ds"], start, end, wet_mask=emu.get("wet_mask"))
        for label, emu in emulators.items()
    }
    return obs_ohc, sim_ohc


def _layer_weighted_mean_from_field(
    field: xr.DataArray,
    area: xr.DataArray,
    dz: xr.DataArray,
    depth_name: str,
    horizontal_dims: list[str],
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
) -> xr.DataArray:
    out = {}
    for depth_slice in depth_slices:
        field_layer = field.sel({depth_name: slice(depth_slice["min"], depth_slice["max"])})
        dz_layer = dz.sel({depth_name: field_layer[depth_name]})
        weights = area * dz_layer
        mask = np.isfinite(field_layer)
        weighted_sum = (field_layer.where(mask) * weights).sum([depth_name, *horizontal_dims], skipna=True)
        weight_sum = weights.where(mask).sum([depth_name, *horizontal_dims], skipna=True)
        out[depth_slice["title"]] = weighted_sum / weight_sum
    return xr.Dataset(out).to_array("depth_slice").transpose("time", "depth_slice")


def _compute_obs_layer_mean_timeseries(
    ds_obs: xr.Dataset,
    spec: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
) -> xr.DataArray:
    ds_obs = ds_obs.sel(time=slice(start, end))
    if ds_obs.sizes.get("time", 0) == 0:
        raise ValueError(f"No ARGO-IAP samples in requested period {start} to {end}")
    if "area" not in ds_obs:
        ds_obs = _infer_obs_grid_metrics(ds_obs)

    obs_var = _find_var_name(ds_obs, spec["obs_aliases"])
    field = ds_obs[obs_var]
    depth_name = _find_depth_name(field)
    if depth_name != "depth":
        field = field.rename({depth_name: "depth"})
        depth_name = "depth"
    field = field.transpose("time", "lat", "lon", depth_name)
    dz = xr.DataArray(
        _depth_thickness_from_centers(field[depth_name].values),
        dims=[depth_name],
        coords={depth_name: field[depth_name]},
    )
    out = _layer_weighted_mean_from_field(
        field,
        ds_obs["area"],
        dz,
        depth_name,
        ["lat", "lon"],
        depth_slices=depth_slices,
    )
    return out.resample(time="MS").mean()


def _compute_sim_layer_mean_timeseries(
    ds_sim: xr.Dataset,
    spec: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
    wet_mask: xr.DataArray | None = None,
) -> xr.DataArray:
    ds_year = ds_sim.sel(time=slice(start, end))
    if ds_year.sizes.get("time", 0) == 0:
        raise ValueError(f"No model samples in requested period {start} to {end}")
    model_var = spec["model_var"]
    if model_var not in ds_year:
        raise KeyError(f"Model dataset does not contain required variable {model_var!r}")

    area = xr.DataArray(
        ds_year["areacello"].values,
        dims=["y", "x"],
        coords={"y": ds_year["y"], "x": ds_year["x"]},
    )
    field = _apply_native_wet_mask(ds_year[model_var], wet_mask)
    out = _layer_weighted_mean_from_field(
        field,
        area,
        ds_year["dz"],
        "lev",
        ["y", "x"],
        depth_slices=depth_slices,
    )
    return out.resample(time="MS").mean()


def _collect_layer_mean_timeseries(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    spec: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    obs_series = _compute_obs_layer_mean_timeseries(argo_iap, spec, start, end)
    sim_series = {
        label: _compute_sim_layer_mean_timeseries(emu["ds"], spec, start, end, wet_mask=emu.get("wet_mask"))
        for label, emu in emulators.items()
    }
    return obs_series, sim_series


def _annual_mean_obs_structure_field(ds_obs: xr.Dataset, spec: dict, year: int) -> xr.DataArray:
    obs_var = _find_var_name(ds_obs, spec["obs_aliases"])
    da = _year_slice(ds_obs[obs_var], year)
    depth_name = _find_depth_name(da)
    rename = {}
    if depth_name != "depth":
        rename[depth_name] = "depth"
    if rename:
        da = da.rename(rename)
    da = da.transpose("time", "lat", "lon", "depth")
    return da.mean("time", skipna=True).sortby("lat").sortby("lon").sortby("depth")


def _annual_mean_sim_on_obs_grid(
    ds_sim: xr.Dataset,
    spec: dict,
    obs_field: xr.DataArray,
    year: int,
    wet_mask: xr.DataArray | None = None,
) -> xr.DataArray:
    model_var = spec["model_var"]
    if model_var not in ds_sim:
        raise KeyError(f"Model dataset does not contain required variable {model_var!r}")

    sim = _year_slice(ds_sim[model_var], year).mean("time", skipna=True)
    sim = _apply_native_wet_mask(sim, wet_mask)
    rename = {}
    if "y" in sim.dims:
        rename["y"] = "lat"
    if "x" in sim.dims:
        rename["x"] = "lon"
    if "lev" in sim.dims:
        rename["lev"] = "depth"
    if rename:
        sim = sim.rename(rename)
    sim = sim.transpose("lat", "lon", "depth")
    sim = _normalize_lon(sim, "lon").sortby("lat").sortby("lon").sortby("depth")

    obs_depth = obs_field["depth"]
    depth_mask = (obs_depth >= float(sim["depth"].min())) & (obs_depth <= float(sim["depth"].max()))
    target_depth = obs_depth.where(depth_mask, drop=True)
    if target_depth.size == 0:
        raise ValueError(
            f"No common depth levels for {model_var!r}: "
            f"model={float(sim['depth'].min()):.1f}-{float(sim['depth'].max()):.1f}, "
            f"obs={float(obs_depth.min()):.1f}-{float(obs_depth.max()):.1f}"
        )

    return sim.interp(
        lat=obs_field["lat"],
        lon=obs_field["lon"],
        depth=target_depth,
        kwargs={"fill_value": np.nan},
    )


def _weighted_spatial_vertical_rmse(bias: xr.DataArray, area: xr.DataArray) -> float:
    dz = xr.DataArray(
        _depth_thickness_from_centers(bias["depth"].values),
        dims=["depth"],
        coords={"depth": bias["depth"]},
    )
    weights = area * dz
    mask = np.isfinite(bias)
    weighted_sum = ((bias ** 2).where(mask) * weights).sum(skipna=True)
    weight_sum = weights.where(mask).sum(skipna=True)
    if float(weight_sum.values) == 0.0:
        return float("nan")
    return float(np.sqrt((weighted_sum / weight_sum).values))


def _area_weighted_zonal_mean(da: xr.DataArray, area: xr.DataArray) -> xr.DataArray:
    mask = np.isfinite(da)
    weighted_sum = (da.where(mask) * area).sum("lon", skipna=True)
    weight_sum = area.where(mask).sum("lon", skipna=True)
    return (weighted_sum / weight_sum).transpose("depth", "lat")


def _area_weighted_mean(field: xr.DataArray, area: xr.DataArray) -> float:
    mask = np.isfinite(field)
    weighted_sum = (field.where(mask) * area).sum()
    weight_sum = area.where(mask).sum()
    if float(weight_sum.values) == 0.0:
        return float("nan")
    return float((weighted_sum / weight_sum).values)


def _area_weighted_map_rmse(bias: xr.DataArray, area: xr.DataArray) -> float:
    mean_square = _area_weighted_mean(bias ** 2, area)
    if not np.isfinite(mean_square):
        return float("nan")
    return float(np.sqrt(mean_square))


def _robust_color_limits(fields: list[xr.DataArray], lower: float = 2.0, upper: float = 98.0) -> tuple[float, float]:
    chunks = []
    for field in fields:
        values = np.asarray(field.values)
        finite = values[np.isfinite(values)]
        if finite.size:
            chunks.append(finite)
    if not chunks:
        return 0.0, 1.0

    values = np.concatenate(chunks)
    vmin, vmax = np.nanpercentile(values, [lower, upper])
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0
    if vmin == vmax:
        delta = max(abs(vmin) * 0.05, 1.0)
        return float(vmin - delta), float(vmax + delta)
    return float(vmin), float(vmax)


def _argo_variable_cmap(spec: dict):
    if spec["filename"] == "salinity":
        return cmocean.cm.haline
    return cmocean.cm.thermal


def _common_annual_fields_on_argo_grid(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    spec: dict,
    year: int,
) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    obs_field = _annual_mean_obs_structure_field(argo_iap, spec, year)
    sim_fields = {
        label: _annual_mean_sim_on_obs_grid(emu["ds"], spec, obs_field, year, emu.get("wet_mask"))
        for label, emu in emulators.items()
    }

    depth_min = float(obs_field["depth"].min())
    depth_max = float(obs_field["depth"].max())
    for sim_field in sim_fields.values():
        depth_min = max(depth_min, float(sim_field["depth"].min()))
        depth_max = min(depth_max, float(sim_field["depth"].max()))

    common_depth = obs_field["depth"].where(
        (obs_field["depth"] >= depth_min) & (obs_field["depth"] <= depth_max),
        drop=True,
    )
    if common_depth.size == 0:
        raise ValueError(f"No common ARGO/model depth levels for {spec['title']}")

    obs_common = obs_field.sel(depth=common_depth)
    sim_common = {
        label: sim_field.interp(depth=common_depth, kwargs={"fill_value": np.nan}).where(np.isfinite(obs_common))
        for label, sim_field in sim_fields.items()
    }
    return obs_common, sim_common


def _depth_weighted_layer_maps_from_field(
    field: xr.DataArray,
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
) -> dict[str, xr.DataArray]:
    dz = xr.DataArray(
        _depth_thickness_from_centers(field["depth"].values),
        dims=["depth"],
        coords={"depth": field["depth"]},
    )
    out = {}
    for depth_slice in depth_slices:
        field_layer = field.sel(depth=slice(depth_slice["min"], depth_slice["max"]))
        dz_layer = dz.sel(depth=field_layer["depth"])
        mask = np.isfinite(field_layer)
        weighted_sum = (field_layer.where(mask) * dz_layer).sum("depth", skipna=True)
        weight_sum = dz_layer.where(mask).sum("depth", skipna=True)
        out[depth_slice["title"]] = weighted_sum / weight_sum
    return out


def _ohc_per_area_layer_maps_from_temperature_field(
    temp_field: xr.DataArray,
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
) -> dict[str, xr.DataArray]:
    dz = xr.DataArray(
        _depth_thickness_from_centers(temp_field["depth"].values),
        dims=["depth"],
        coords={"depth": temp_field["depth"]},
    )
    out = {}
    for depth_slice in depth_slices:
        temp_layer = temp_field.sel(depth=slice(depth_slice["min"], depth_slice["max"]))
        dz_layer = dz.sel(depth=temp_layer["depth"])
        valid = np.isfinite(temp_layer).any("depth")
        ohc = (temp_layer * dz_layer * RHO * CP).sum("depth", skipna=True).where(valid)
        out[depth_slice["title"]] = ohc
    return out


def _add_map_features(ax) -> None:
    ax.set_global()
    ax.add_feature(cfeature.LAND, zorder=10, edgecolor="black", linewidth=0.4, facecolor="lightgray")
    ax.coastlines(zorder=11, linewidth=0.5)


def _plot_layer_map_comparison(
    obs_maps: dict[str, xr.DataArray],
    sim_maps: dict[str, dict[str, xr.DataArray]],
    obs_area: xr.DataArray,
    emulators: dict[str, dict],
    title: str,
    units: str,
    metric_prefix: str,
    cmap,
    year: int,
) -> pd.DataFrame:
    layer_titles = [depth_slice["title"] for depth_slice in OHC_DEPTH_SLICES]
    nrows = len(layer_titles)
    ncols = len(emulators) + 1
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 3.6 * nrows),
        squeeze=False,
        subplot_kw={"projection": ccrs.Robinson(central_longitude=210)},
    )

    obs_fields = [obs_maps[layer_title] for layer_title in layer_titles]
    vmin, vmax = _robust_color_limits(obs_fields)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    rows = []
    mesh = None
    for row_idx, layer_title in enumerate(layer_titles):
        obs_map = obs_maps[layer_title]
        panels = [("ARGO-IAP", obs_map, _area_weighted_mean(obs_map, obs_area))]
        for label, model_maps in sim_maps.items():
            sim_map = model_maps[layer_title]
            bias = sim_map - obs_map
            rows.append(
                {
                    "model": label,
                    "depth": layer_title,
                    "metric": f"{metric_prefix}_layer_map_rmse",
                    "value": _area_weighted_map_rmse(bias, obs_area),
                }
            )
            rows.append(
                {
                    "model": label,
                    "depth": layer_title,
                    "metric": f"{metric_prefix}_layer_map_mean_bias",
                    "value": _area_weighted_mean(bias, obs_area),
                }
            )
            panels.append((label, sim_map, _area_weighted_mean(sim_map, obs_area)))

        for col_idx, (source, field, mean_val) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            mesh = field.plot.pcolormesh(
                ax=ax,
                transform=ccrs.PlateCarree(),
                x="lon",
                y="lat",
                cmap=cmap,
                norm=norm,
                add_colorbar=False,
                rasterized=True,
            )
            _add_map_features(ax)
            ax.set_title(f"{source}\nmean={mean_val:.3g}", fontsize=11)
            if col_idx == 0:
                ax.text(
                    -0.08,
                    0.5,
                    layer_title,
                    ha="right",
                    va="center",
                    rotation=90,
                    transform=ax.transAxes,
                    fontsize=12,
                )

    cbar = fig.colorbar(
        mesh,
        ax=axes.ravel(),
        orientation="horizontal",
        shrink=0.62,
        aspect=42,
        fraction=0.045,
        pad=0.08,
        extend="both",
    )
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label(f"{title} ({units})", fontsize=12)
    fig.suptitle(f"Annual-Mean {title} Maps vs ARGO-IAP ({year})", y=0.965, fontsize=15)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.155, wspace=0.03, hspace=0.16)
    return pd.DataFrame(rows)


def _plot_argo_variable_layer_maps(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    year: int,
    spec: dict,
) -> pd.DataFrame:
    if "area" not in argo_iap:
        argo_iap = _infer_obs_grid_metrics(argo_iap)
    obs_field, sim_fields = _common_annual_fields_on_argo_grid(argo_iap, emulators, spec, year)
    obs_maps = _depth_weighted_layer_maps_from_field(obs_field)
    sim_maps = {
        label: _depth_weighted_layer_maps_from_field(sim_field)
        for label, sim_field in sim_fields.items()
    }
    return _plot_layer_map_comparison(
        obs_maps,
        sim_maps,
        argo_iap["area"],
        emulators,
        f"Layer-Mean {spec['title']}",
        spec["units"],
        spec["metric_prefix"],
        _argo_variable_cmap(spec),
        year,
    )


def _plot_argo_ohc_layer_maps(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    year: int,
) -> pd.DataFrame:
    temp_spec = ARGO_STRUCTURE_VARIABLES[0]
    if "area" not in argo_iap:
        argo_iap = _infer_obs_grid_metrics(argo_iap)
    obs_temp, sim_temps = _common_annual_fields_on_argo_grid(argo_iap, emulators, temp_spec, year)
    obs_maps = _ohc_per_area_layer_maps_from_temperature_field(obs_temp)
    sim_maps = {
        label: _ohc_per_area_layer_maps_from_temperature_field(sim_temp)
        for label, sim_temp in sim_temps.items()
    }
    return _plot_layer_map_comparison(
        obs_maps,
        sim_maps,
        argo_iap["area"],
        emulators,
        "Layer-Integrated OHC per Unit Area",
        "J m$^{-2}$",
        "annual_mean_ohc_per_area",
        cmocean.cm.thermal,
        year,
    )


def _plot_eke_map_comparison(
    duacs: xr.Dataset,
    emulators: dict[str, dict],
    year: int,
    velocity_kind: str = "absolute",
) -> pd.DataFrame:
    u_obs_name, v_obs_name = _duacs_velocity_names(duacs, velocity_kind)
    velocity_label = _duacs_velocity_label(velocity_kind)
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
        sim_zos = _apply_native_wet_mask(sim_year["zos"], emu.get("wet_mask"))
        sim_ug, sim_vg = _geostrophic_velocity_from_zos(sim_zos, lat_dim="y", lon_dim="x")
        sim_eke = _surface_eke_from_uv(sim_ug, sim_vg)
        sim_eke = _apply_native_wet_mask(sim_eke, emu.get("wet_mask"))
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
            rasterized=True,
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
        pad=0.115,
        label="Geostrophic EKE (m$^2$/s$^2$)",
    )
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Geostrophic EKE (m$^2$/s$^2$)", fontsize=12)
    fig.suptitle(
        f"Surface Geostrophic EKE vs DUACS {velocity_label} ({year}; |lat| > 5$^\\circ$)",
        y=0.935,
        fontsize=15,
    )
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.17, wspace=0.03)
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
    velocity_kind: str = "absolute",
    sigma_deg: float = 0.5,
) -> pd.DataFrame:
    u_obs_name, v_obs_name = _duacs_velocity_names(duacs, velocity_kind)
    velocity_label = _duacs_velocity_label(velocity_kind)
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
        sim_zos = _apply_native_wet_mask(sim_year["zos"], emu.get("wet_mask"))
        sim_ug, sim_vg = _geostrophic_velocity_from_zos(sim_zos, lat_dim="y", lon_dim="x")
        sim_ug_lp = _gaussian_lowpass_field(sim_ug, lat_dim="y", lon_dim="x", sigma_deg=sigma_deg)
        sim_vg_lp = _gaussian_lowpass_field(sim_vg, lat_dim="y", lon_dim="x", sigma_deg=sigma_deg)
        sim_eke = _surface_eke_from_uv(sim_ug_lp, sim_vg_lp)
        sim_eke = _apply_native_wet_mask(sim_eke, emu.get("wet_mask"))
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
            rasterized=True,
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
        pad=0.115,
        label="Low-pass Geostrophic EKE (m$^2$/s$^2$)",
    )
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Low-pass Geostrophic EKE (m$^2$/s$^2$)", fontsize=12)
    fig.suptitle(
        f"Low-pass ($\\sigma$={sigma_deg}$^\\circ$ Gaussian) Geostrophic EKE vs DUACS "
        f"{velocity_label} ({year}; |lat| > 5$^\\circ$)",
        y=0.935,
        fontsize=14,
    )
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.17, wspace=0.03)
    return pd.DataFrame(rows)


def _weighted_spatial_depth_rmse(bias: xr.DataArray, area: xr.DataArray) -> float:
    weights = area
    mask = np.isfinite(bias)
    weighted_sum = ((bias ** 2).where(mask) * weights).sum(skipna=True)
    weight_sum = weights.where(mask).sum(skipna=True)
    if float(weight_sum.values) == 0.0:
        return float("nan")
    return float(np.sqrt((weighted_sum / weight_sum).values))


def _plot_zonal_mean_panels(
    panels: list[tuple[str, xr.DataArray]],
    title: str,
    units: str,
    cmap,
    year: int,
) -> pd.DataFrame:
    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(4.9 * ncols, 4.9), squeeze=False, sharex=True, sharey=True)
    axes = axes[0]

    vmin, vmax = _robust_color_limits([panels[0][1]])
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    mesh = None
    for col_idx, (label, field) in enumerate(panels):
        ax = axes[col_idx]
        mesh = ax.pcolormesh(
            field["lat"].values,
            field["depth"].values,
            field.values,
            cmap=cmap,
            norm=norm,
            shading="auto",
            rasterized=True,
        )
        ax.invert_yaxis()
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Latitude")
        if col_idx == 0:
            ax.set_ylabel("Depth (m)")
        ax.grid(True, alpha=0.25)

    fig.subplots_adjust(left=0.06, right=0.87, top=0.82, bottom=0.14, wspace=0.20)
    cbar_ax = fig.add_axes([0.885, 0.18, 0.014, 0.55])
    cbar = fig.colorbar(mesh, cax=cbar_ax, orientation="vertical", extend="both")
    cbar.set_label(f"{title} ({units})")
    fig.suptitle(f"Annual-Mean ARGO-IAP {title} Zonal Mean ({year})", y=0.955, fontsize=15)
    return pd.DataFrame()


def _plot_argo_variable_zonal_mean(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    year: int,
    spec: dict,
) -> pd.DataFrame:
    if "area" not in argo_iap:
        argo_iap = _infer_obs_grid_metrics(argo_iap)
    obs_field, sim_fields = _common_annual_fields_on_argo_grid(argo_iap, emulators, spec, year)
    obs_area = argo_iap["area"]

    panels = [("ARGO-IAP", _area_weighted_zonal_mean(obs_field, obs_area))]
    for label, sim_field in sim_fields.items():
        panels.append((label, _area_weighted_zonal_mean(sim_field, obs_area)))

    return _plot_zonal_mean_panels(
        panels,
        spec["title"],
        spec["units"],
        _argo_variable_cmap(spec),
        year,
    )


def _plot_argo_ohc_zonal_mean(argo_iap: xr.Dataset, emulators: dict[str, dict], year: int) -> pd.DataFrame:
    temp_spec = ARGO_STRUCTURE_VARIABLES[0]
    if "area" not in argo_iap:
        argo_iap = _infer_obs_grid_metrics(argo_iap)
    obs_temp, sim_temps = _common_annual_fields_on_argo_grid(argo_iap, emulators, temp_spec, year)
    obs_area = argo_iap["area"]

    dz = xr.DataArray(
        _depth_thickness_from_centers(obs_temp["depth"].values),
        dims=["depth"],
        coords={"depth": obs_temp["depth"]},
    )
    obs_ohc = obs_temp * dz * RHO * CP
    panels = [("ARGO-IAP", _area_weighted_zonal_mean(obs_ohc, obs_area))]
    for label, sim_temp in sim_temps.items():
        sim_ohc = sim_temp * dz * RHO * CP
        panels.append((label, _area_weighted_zonal_mean(sim_ohc, obs_area)))

    return _plot_zonal_mean_panels(
        panels,
        "OHC per-layer",
        "J m$^{-2}$",
        cmocean.cm.thermal,
        year,
    )


def _plot_argo_variable_zonal_mean_bias(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    year: int,
    spec: dict,
) -> pd.DataFrame:
    if _find_var_name_optional(argo_iap, spec["obs_aliases"]) is None:
        raise KeyError(
            f"ARGO-IAP {spec['title']} diagnostic requires one of {spec['obs_aliases']}. "
            f"Available variables: {list(argo_iap.data_vars)}"
        )

    if "area" not in argo_iap:
        argo_iap = _infer_obs_grid_metrics(argo_iap)
    obs_area = argo_iap["area"]

    rows = []
    nmodels = len(emulators)
    fig, axes = plt.subplots(1, nmodels, figsize=(5.9 * nmodels, 4.9), squeeze=False, sharex=True)
    axes = axes[0]

    obs_field = _annual_mean_obs_structure_field(argo_iap, spec, year)
    row_results = []
    vmax_candidates = []

    for label, emu in emulators.items():
        sim_field = _annual_mean_sim_on_obs_grid(emu["ds"], spec, obs_field, year, emu.get("wet_mask"))
        obs_common = obs_field.sel(depth=sim_field["depth"])
        bias = sim_field - obs_common
        rmse = _weighted_spatial_vertical_rmse(bias, obs_area)
        zonal_bias = _area_weighted_zonal_mean(bias, obs_area)
        row_results.append((label, zonal_bias, rmse))
        finite_abs = np.abs(zonal_bias.values[np.isfinite(zonal_bias.values)])
        if finite_abs.size:
            vmax_candidates.append(float(np.nanpercentile(finite_abs, 98)))
        rows.append(
            {
                "model": label,
                "metric": f"{spec['metric_prefix']}_spatial_vertical_rmse",
                "value": rmse,
                "depth": f"{float(sim_field['depth'].min()):.0f}-{float(sim_field['depth'].max()):.0f}m",
            }
        )

    vmax = max(vmax_candidates) if vmax_candidates else 1.0
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    for col_idx, (label, zonal_bias, rmse) in enumerate(row_results):
        ax = axes[col_idx]
        mesh = ax.pcolormesh(
            zonal_bias["lat"].values,
            zonal_bias["depth"].values,
            zonal_bias.values,
            cmap=cmocean.cm.balance,
            vmin=-vmax,
            vmax=vmax,
            shading="auto",
            rasterized=True,
        )
        ax.invert_yaxis()
        ax.set_title(f"{label}\nRMSE={rmse:.3g} {spec['units']}", fontsize=11)
        ax.set_xlabel("Latitude")
        if col_idx == 0:
            ax.set_ylabel("Depth (m)")
        ax.grid(True, alpha=0.25)

    fig.subplots_adjust(left=0.07, right=0.855, top=0.80, bottom=0.14, wspace=0.26)
    cbar_ax = fig.add_axes([0.87, 0.18, 0.018, 0.55])
    cbar = fig.colorbar(mesh, cax=cbar_ax, orientation="vertical")
    cbar.set_label(f"{spec['title']} bias ({spec['units']})")

    fig.suptitle(
        f"Annual-Mean ARGO-IAP {spec['title']} Zonal-Mean Bias ({year})",
        y=0.955,
        fontsize=15,
    )
    return pd.DataFrame(rows)


def _plot_argo_ohc_zonal_mean_bias(argo_iap: xr.Dataset, emulators: dict[str, dict], year: int) -> pd.DataFrame:
    temp_spec = ARGO_STRUCTURE_VARIABLES[0]
    if _find_var_name_optional(argo_iap, temp_spec["obs_aliases"]) is None:
        raise KeyError(
            f"ARGO-IAP OHC structure diagnostic requires temperature. Available variables: {list(argo_iap.data_vars)}"
        )

    if "area" not in argo_iap:
        argo_iap = _infer_obs_grid_metrics(argo_iap)
    obs_area = argo_iap["area"]
    obs_temp = _annual_mean_obs_structure_field(argo_iap, temp_spec, year)

    rows = []
    nmodels = len(emulators)
    fig, axes = plt.subplots(1, nmodels, figsize=(5.9 * nmodels, 4.9), squeeze=False, sharex=True)
    axes = axes[0]
    row_results = []
    vmax_candidates = []

    for label, emu in emulators.items():
        sim_temp = _annual_mean_sim_on_obs_grid(emu["ds"], temp_spec, obs_temp, year, emu.get("wet_mask"))
        obs_common = obs_temp.sel(depth=sim_temp["depth"])
        dz = xr.DataArray(
            _depth_thickness_from_centers(sim_temp["depth"].values),
            dims=["depth"],
            coords={"depth": sim_temp["depth"]},
        )
        bias = (sim_temp - obs_common) * dz * RHO * CP
        rmse = _weighted_spatial_depth_rmse(bias, obs_area)
        zonal_bias = _area_weighted_zonal_mean(bias, obs_area)
        row_results.append((label, zonal_bias, rmse))
        finite_abs = np.abs(zonal_bias.values[np.isfinite(zonal_bias.values)])
        if finite_abs.size:
            vmax_candidates.append(float(np.nanpercentile(finite_abs, 98)))
        rows.append(
            {
                "model": label,
                "metric": "annual_mean_ohc_per_area_spatial_vertical_rmse",
                "value": rmse,
                "depth": f"{float(sim_temp['depth'].min()):.0f}-{float(sim_temp['depth'].max()):.0f}m",
            }
        )

    vmax = max(vmax_candidates) if vmax_candidates else 1.0
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    for col_idx, (label, zonal_bias, rmse) in enumerate(row_results):
        ax = axes[col_idx]
        mesh = ax.pcolormesh(
            zonal_bias["lat"].values,
            zonal_bias["depth"].values,
            zonal_bias.values,
            cmap=cmocean.cm.balance,
            vmin=-vmax,
            vmax=vmax,
            shading="auto",
            rasterized=True,
        )
        ax.invert_yaxis()
        ax.set_title(f"{label}\nRMSE={rmse:.3g} J m$^{{-2}}$", fontsize=11)
        ax.set_xlabel("Latitude")
        if col_idx == 0:
            ax.set_ylabel("Depth (m)")
        ax.grid(True, alpha=0.25)

    fig.subplots_adjust(left=0.07, right=0.855, top=0.80, bottom=0.14, wspace=0.26)
    cbar_ax = fig.add_axes([0.87, 0.18, 0.018, 0.55])
    cbar = fig.colorbar(mesh, cax=cbar_ax, orientation="vertical")
    cbar.set_label("OHC per-layer bias (J m$^{-2}$)")
    fig.suptitle(f"Annual-Mean ARGO-IAP OHC Zonal-Mean Bias ({year})", y=0.955, fontsize=15)
    return pd.DataFrame(rows)


def _plot_temperature_spectra(
    oisst_5day: xr.Dataset,
    emulators: dict[str, dict],
    oisst_daily: xr.Dataset | None = None,
) -> pd.DataFrame:
    sst_5day_name = _find_var_name(oisst_5day, ["sst", "analysed_sst", "thetao", "tos"])
    sst_daily_name = (
        _find_var_name(oisst_daily, ["sst", "analysed_sst", "thetao", "tos"])
        if oisst_daily is not None
        else None
    )
    rows = []
    n_regions = len(SPECTRUM_REGIONS)
    nrows = 2
    ncols = (n_regions + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows))
    axes = np.ravel(axes)

    for idx, (region_name, lon_slice, lat_slice) in enumerate(SPECTRUM_REGIONS):
        ax = axes[idx]
        k_5day, spec_5day = _compute_isotropic_spectrum_from_surface(
            oisst_5day[sst_5day_name], oisst_5day["dx"], oisst_5day["dy"], lon_slice, lat_slice
        )
        if oisst_daily is not None and sst_daily_name is not None:
            k_daily, spec_daily = _compute_isotropic_spectrum_from_surface(
                oisst_daily[sst_daily_name], oisst_daily["dx"], oisst_daily["dy"], lon_slice, lat_slice
            )
            ax.loglog(k_daily, spec_daily, ":", color="0.35", linewidth=2, label="OISST raw daily")
        else:
            k_daily, spec_daily = None, None
        ax.loglog(k_5day, spec_5day, "--", color="black", linewidth=2, label="OISST 5-day avg")
        for label, emu in emulators.items():
            sim, _ = _temporal_overlap(emu["ds"], oisst_5day)
            sim_sst = _apply_native_wet_mask(sim["thetao"].isel(lev=0), emu.get("wet_mask"))
            k_sim, spec_sim = _compute_isotropic_spectrum_from_surface(
                sim_sst, sim["dx"], sim["dy"], lon_slice, lat_slice,
                lat_dim="y", lon_dim="x",
            )
            rmse_5day = _rmse_between_curves(k_5day, spec_5day, k_sim, spec_sim)
            rows.append(
                {
                    "model": label,
                    "region": region_name,
                    "metric": "surface_ssta_isotropic_spectrum_rmse_vs_oisst_5day_avg",
                    "value": rmse_5day,
                }
            )
            if k_daily is not None and spec_daily is not None:
                rmse_daily = _rmse_between_curves(k_daily, spec_daily, k_sim, spec_sim)
                rows.append(
                    {
                        "model": label,
                        "region": region_name,
                        "metric": "surface_ssta_isotropic_spectrum_rmse_vs_oisst_raw_daily",
                        "value": rmse_daily,
                    }
                )
            ax.loglog(k_sim, spec_sim, "-", color=emu["color"], linewidth=2, label=label)
        ax.set_title(region_name)
        ax.set_xlabel(r"Wavenumber $\kappa$ (km$^{-1}$)")
        ax.set_ylabel("PSD")
        ax.grid(True, which="both", alpha=0.3)

    handles, labels, ncol = _legend_truth_first(
        *axes[0].get_legend_handles_labels(),
        truth_labels={"OISST raw daily", "OISST 5-day avg"},
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, -0.005),
        frameon=False,
        fontsize=11,
    )
    fig.suptitle(
        "Regional Surface Temperature SSTA Spectra vs OISST\n2014-2022 calendar-day climatology removed",
        y=0.975,
        fontsize=15,
        linespacing=1.15,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.15, wspace=0.25, hspace=0.34)
    return pd.DataFrame(rows)


def _plot_temporal_ke_spectra(
    duacs_5day: xr.Dataset,
    duacs_raw: xr.Dataset,
    emulators: dict[str, dict],
    velocity_kind: str = "absolute",
) -> pd.DataFrame:
    u_5day_name, v_5day_name = _duacs_velocity_names(duacs_5day, velocity_kind)
    u_raw_name, v_raw_name = _duacs_velocity_names(duacs_raw, velocity_kind)
    velocity_label = _duacs_velocity_label(velocity_kind)

    obs_times = pd.DatetimeIndex(duacs_5day.time.values)
    obs_span_years = (obs_times.max() - obs_times.min()).days / 365.25
    if obs_span_years < 3:
        print(
            f"  WARNING: 5-day temporal KE spectrum uses only {obs_span_years:.1f} years of data; "
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
        freq_raw, psd_raw = _compute_temporal_ke_spectrum_from_surface(
            duacs_raw[u_raw_name], duacs_raw[v_raw_name], lon_slice, lat_slice
        )
        freq_5day, psd_5day = _compute_temporal_ke_spectrum_from_surface(
            duacs_5day[u_5day_name], duacs_5day[v_5day_name], lon_slice, lat_slice
        )
        ax.loglog(freq_raw, psd_raw, ":", color="0.35", linewidth=2, label="DUACS raw daily")
        ax.loglog(freq_5day, psd_5day, "--", color="black", linewidth=2, label="DUACS 5-day avg")
        for label, emu in emulators.items():
            sim, _ = _temporal_overlap(emu["ds"], duacs_5day)
            sim_region = sim.sel(
                x=_expand_numeric_slice(lon_slice, 1.0),
                y=_expand_numeric_slice(lat_slice, 1.0),
            )
            sim_zos = _apply_native_wet_mask(sim_region["zos"], emu.get("wet_mask"))
            sim_ug, sim_vg = _geostrophic_velocity_from_zos(sim_zos, lat_dim="y", lon_dim="x")
            if velocity_kind == "anomaly":
                sim_ug, sim_vg = _time_mean_velocity_anomaly(sim_ug, sim_vg)
            freq_sim, psd_sim = _compute_temporal_ke_spectrum_from_surface(
                sim_ug, sim_vg, lon_slice, lat_slice, lat_dim="y", lon_dim="x",
            )
            rmse_raw = _rmse_between_curves(freq_raw, psd_raw, freq_sim, psd_sim)
            rmse_5day = _rmse_between_curves(freq_5day, psd_5day, freq_sim, psd_sim)
            rows.append(
                {
                    "model": label,
                    "region": region_name,
                    "metric": "surface_geostrophic_ke_temporal_spectrum_rmse_vs_duacs_raw_daily",
                    "value": rmse_raw,
                }
            )
            rows.append(
                {
                    "model": label,
                    "region": region_name,
                    "metric": "surface_geostrophic_ke_temporal_spectrum_rmse_vs_duacs_5day_avg",
                    "value": rmse_5day,
                }
            )
            ax.loglog(freq_sim, psd_sim, "-", color=emu["color"], linewidth=2, label=label)
        ax.set_title(region_name)
        ax.set_xlabel("Frequency (cycles/year)")
        ax.set_ylabel("PSD")
        ax.grid(True, which="both", alpha=0.3)

    handles, labels, ncol = _legend_truth_first(
        *axes[0].get_legend_handles_labels(),
        truth_labels={"DUACS raw daily", "DUACS 5-day avg"},
    )
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=11)
    fig.suptitle(
        _figure_title_with_context(
            "Regional Surface Geostrophic KE Temporal Spectra vs DUACS",
            extra=f"{velocity_label}; raw daily and 5-day averaged DUACS",
        ),
        y=0.975,
        fontsize=14,
        linespacing=1.15,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.15, wspace=0.25, hspace=0.34)
    return pd.DataFrame(rows)


def _rmse_between_series(ref: pd.Series, other: pd.Series) -> float:
    common_index = ref.index.intersection(other.index)
    if common_index.empty:
        return np.nan
    diff = other.loc[common_index] - ref.loc[common_index]
    diff = diff[np.isfinite(diff)]
    if diff.empty:
        return np.nan
    return float(np.sqrt(np.mean(diff**2)))


def _plot_ohc_timeseries(
    obs_ohc: xr.DataArray,
    sim_ohc: dict[str, xr.DataArray],
    emulators: dict[str, dict],
    period_label: str,
) -> pd.DataFrame:
    fig, axes = plt.subplots(len(OHC_DEPTH_SLICES), 1, figsize=(11.5, 3.8 * len(OHC_DEPTH_SLICES)), sharex=True)
    axes = np.atleast_1d(axes)
    rows = []

    for depth_idx, depth_slice in enumerate(OHC_DEPTH_SLICES):
        ax = axes[depth_idx]
        depth_title = depth_slice["title"]
        obs_series = obs_ohc.sel(depth_slice=depth_title).to_series()
        ax.plot(obs_series.index, obs_series.values, "--", color="black", linewidth=2, label="ARGO-IAP")

        for label, sim in sim_ohc.items():
            sim_series = sim.sel(depth_slice=depth_title).to_series()
            rmse = _rmse_between_series(obs_series, sim_series)
            rows.append({"model": label, "depth": depth_title, "metric": "monthly_ohc_rmse", "value": rmse})
            ax.plot(sim_series.index, sim_series.values, "-", color=emulators[label]["color"], linewidth=2, label=label)

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
        bbox_to_anchor=(0.5, 0.04),
        frameon=False,
        fontsize=12,
    )
    _finish_timeseries_figure(fig, axes, "Monthly Global OHC vs ARGO-IAP", period_label, "Time")
    return pd.DataFrame(rows)


def _plot_ohc_anomaly_timeseries(
    obs_ohc: xr.DataArray,
    sim_ohc: dict[str, xr.DataArray],
    emulators: dict[str, dict],
    period_label: str,
) -> pd.DataFrame:
    fig, axes = plt.subplots(len(OHC_DEPTH_SLICES), 1, figsize=(11.5, 3.8 * len(OHC_DEPTH_SLICES)), sharex=True)
    axes = np.atleast_1d(axes)
    rows = []

    for depth_idx, depth_slice in enumerate(OHC_DEPTH_SLICES):
        ax = axes[depth_idx]
        depth_title = depth_slice["title"]
        obs_series = obs_ohc.sel(depth_slice=depth_title).to_series()
        obs_anom = obs_series - obs_series.mean()
        ax.plot(obs_anom.index, obs_anom.values, "--", color="black", linewidth=2, label="ARGO-IAP")

        for label, sim in sim_ohc.items():
            sim_series = sim.sel(depth_slice=depth_title).to_series()
            sim_anom = sim_series - sim_series.mean()
            rmse = _rmse_between_series(obs_anom, sim_anom)
            rows.append({"model": label, "depth": depth_title, "metric": "monthly_ohc_anomaly_rmse", "value": rmse})
            ax.plot(sim_anom.index, sim_anom.values, "-", color=emulators[label]["color"], linewidth=2, label=label)

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
        bbox_to_anchor=(0.5, 0.04),
        frameon=False,
        fontsize=12,
    )
    _finish_timeseries_figure(fig, axes, "Monthly OHC Mean-Removed Anomaly vs ARGO-IAP", period_label, "Time")
    return pd.DataFrame(rows)


def _plot_ohc_monthly_climatology(
    obs_ohc: xr.DataArray,
    sim_ohc: dict[str, xr.DataArray],
    emulators: dict[str, dict],
    period_label: str,
    obs_climatology_ohc: xr.DataArray | None = None,
    climatology_label: str | None = None,
) -> pd.DataFrame:
    fig, axes = plt.subplots(len(OHC_DEPTH_SLICES), 1, figsize=(11.5, 3.8 * len(OHC_DEPTH_SLICES)), sharex=True)
    axes = np.atleast_1d(axes)
    rows = []

    for depth_idx, depth_slice in enumerate(OHC_DEPTH_SLICES):
        ax = axes[depth_idx]
        depth_title = depth_slice["title"]
        obs_clim_source = obs_climatology_ohc if obs_climatology_ohc is not None else obs_ohc
        obs_series = obs_clim_source.sel(depth_slice=depth_title).to_series()
        obs_clim = _series_monthly_climatology(obs_series)
        ax.plot(obs_clim.index, obs_clim.values, "--o", color="black", linewidth=2, markersize=4, label="ARGO-IAP")

        for label, sim in sim_ohc.items():
            sim_series = sim.sel(depth_slice=depth_title).to_series()
            sim_clim = _series_monthly_climatology(sim_series)
            rmse = _rmse_between_series(obs_clim, sim_clim)
            rows.append({"model": label, "depth": depth_title, "metric": "monthly_ohc_climatology_rmse", "value": rmse})
            ax.plot(
                sim_clim.index,
                sim_clim.values,
                "-o",
                color=emulators[label]["color"],
                linewidth=2,
                markersize=4,
                label=label,
            )

        ax.set_title(f"{depth_title}", fontsize=14)
        ax.set_ylabel("OHC (J)", fontsize=13)
        ax.set_xticks(np.arange(1, 13))
        ax.set_xlim(1, 12)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)

    handles, labels, ncol = _legend_truth_first(*axes[0].get_legend_handles_labels(), truth_labels={"ARGO-IAP"})
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, 0.04),
        frameon=False,
        fontsize=12,
    )
    extra = f"ARGO climatology {climatology_label}" if climatology_label is not None else None
    _finish_timeseries_figure(
        fig,
        axes,
        "Monthly OHC Climatology vs ARGO-IAP",
        period_label,
        "Calendar month",
        extra=extra,
    )
    return pd.DataFrame(rows)


def _plot_ohc_deseasoned_anomaly_timeseries(
    obs_ohc: xr.DataArray,
    sim_ohc: dict[str, xr.DataArray],
    emulators: dict[str, dict],
    period_label: str,
    obs_climatology_ohc: xr.DataArray | None = None,
    climatology_label: str | None = None,
) -> pd.DataFrame:
    fig, axes = plt.subplots(len(OHC_DEPTH_SLICES), 1, figsize=(11.5, 3.8 * len(OHC_DEPTH_SLICES)), sharex=True)
    axes = np.atleast_1d(axes)
    rows = []

    for depth_idx, depth_slice in enumerate(OHC_DEPTH_SLICES):
        ax = axes[depth_idx]
        depth_title = depth_slice["title"]
        obs_series = obs_ohc.sel(depth_slice=depth_title).to_series()
        obs_clim_source = obs_climatology_ohc if obs_climatology_ohc is not None else obs_ohc
        obs_clim_series = obs_clim_source.sel(depth_slice=depth_title).to_series()
        obs_has_clim = _has_repeated_calendar_months(obs_clim_series)
        if obs_has_clim:
            obs_clim = _series_monthly_climatology(obs_clim_series)
            obs_anom = _series_without_given_monthly_climatology(obs_series, obs_clim)
            ax.plot(obs_anom.index, obs_anom.values, "--", color="black", linewidth=2, label="ARGO-IAP")
        else:
            ax.text(
                0.5,
                0.5,
                "Need at least two complete calendar years\nfor deseasonalized monthly anomalies",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )

        for label, sim in sim_ohc.items():
            sim_series = sim.sel(depth_slice=depth_title).to_series()
            if obs_has_clim:
                sim_clim = _series_monthly_climatology(sim_series)
                sim_anom = _series_without_given_monthly_climatology(sim_series, sim_clim)
                rmse = _rmse_between_series(obs_anom, sim_anom)
                ax.plot(sim_anom.index, sim_anom.values, "-", color=emulators[label]["color"], linewidth=2, label=label)
            else:
                rmse = np.nan
            rows.append(
                {
                    "model": label,
                    "depth": depth_title,
                    "metric": "monthly_ohc_deseasoned_anomaly_rmse",
                    "value": rmse,
                }
            )

        ax.set_title(f"{depth_title}", fontsize=14)
        ax.set_ylabel("Deseasonalized anomaly (J)", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        handles, labels, ncol = _legend_truth_first(handles, labels, truth_labels={"ARGO-IAP"})
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=ncol,
            bbox_to_anchor=(0.5, 0.04),
            frameon=False,
            fontsize=12,
        )
    extra = (
        f"climatologies: ARGO {climatology_label}; models {period_label}"
        if climatology_label is not None
        else f"climatologies: each series {period_label}"
    )
    _finish_timeseries_figure(
        fig,
        axes,
        "Monthly OHC Deseasonalized Anomaly vs ARGO-IAP",
        period_label,
        "Time",
        extra=extra,
    )
    return pd.DataFrame(rows)


def _plot_ohc_detrended_anomaly_timeseries(
    obs_ohc: xr.DataArray,
    sim_ohc: dict[str, xr.DataArray],
    emulators: dict[str, dict],
    period_label: str,
) -> pd.DataFrame:
    fig, axes = plt.subplots(len(OHC_DEPTH_SLICES), 1, figsize=(11.5, 3.8 * len(OHC_DEPTH_SLICES)), sharex=True)
    axes = np.atleast_1d(axes)
    rows = []

    for depth_idx, depth_slice in enumerate(OHC_DEPTH_SLICES):
        ax = axes[depth_idx]
        depth_title = depth_slice["title"]
        obs_series = obs_ohc.sel(depth_slice=depth_title).to_series()
        obs_anom = _series_without_linear_trend(obs_series)
        ax.plot(obs_anom.index, obs_anom.values, "--", color="black", linewidth=2, label="ARGO-IAP")

        for label, sim in sim_ohc.items():
            sim_series = sim.sel(depth_slice=depth_title).to_series()
            sim_anom = _series_without_linear_trend(sim_series)
            rmse = _rmse_between_series(obs_anom, sim_anom)
            rows.append({"model": label, "depth": depth_title, "metric": "monthly_ohc_detrended_anomaly_rmse", "value": rmse})
            ax.plot(sim_anom.index, sim_anom.values, "-", color=emulators[label]["color"], linewidth=2, label=label)

        ax.set_title(f"{depth_title}", fontsize=14)
        ax.set_ylabel("Detrended anomaly (J)", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)

    handles, labels, ncol = _legend_truth_first(*axes[0].get_legend_handles_labels(), truth_labels={"ARGO-IAP"})
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, 0.04),
        frameon=False,
        fontsize=12,
    )
    _finish_timeseries_figure(fig, axes, "Monthly OHC Detrended Anomaly vs ARGO-IAP", period_label, "Time")
    return pd.DataFrame(rows)


def _plot_ohc_detrended_deseasonalized_anomaly_timeseries(
    obs_ohc: xr.DataArray,
    sim_ohc: dict[str, xr.DataArray],
    emulators: dict[str, dict],
    period_label: str,
    obs_climatology_ohc: xr.DataArray | None = None,
    climatology_label: str | None = None,
) -> pd.DataFrame:
    fig, axes = plt.subplots(len(OHC_DEPTH_SLICES), 1, figsize=(11.5, 3.8 * len(OHC_DEPTH_SLICES)), sharex=True)
    axes = np.atleast_1d(axes)
    rows = []

    for depth_idx, depth_slice in enumerate(OHC_DEPTH_SLICES):
        ax = axes[depth_idx]
        depth_title = depth_slice["title"]
        obs_series = obs_ohc.sel(depth_slice=depth_title).to_series()
        obs_clim_source = obs_climatology_ohc if obs_climatology_ohc is not None else obs_ohc
        obs_clim_series = obs_clim_source.sel(depth_slice=depth_title).to_series()
        obs_has_clim = _has_repeated_calendar_months(obs_clim_series)
        if obs_has_clim:
            obs_clim = _series_monthly_climatology(obs_clim_series)
            obs_deseasonalized = _series_without_given_monthly_climatology(obs_series, obs_clim)
            obs_anom = _series_without_linear_trend(obs_deseasonalized)
            ax.plot(obs_anom.index, obs_anom.values, "--", color="black", linewidth=2, label="ARGO-IAP")
        else:
            ax.text(
                0.5,
                0.5,
                "Need at least two complete calendar years\nfor detrended deseasonalized monthly anomalies",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )

        for label, sim in sim_ohc.items():
            sim_series = sim.sel(depth_slice=depth_title).to_series()
            if obs_has_clim:
                sim_clim = _series_monthly_climatology(sim_series)
                sim_deseasonalized = _series_without_given_monthly_climatology(sim_series, sim_clim)
                sim_anom = _series_without_linear_trend(sim_deseasonalized)
                rmse = _rmse_between_series(obs_anom, sim_anom)
                ax.plot(sim_anom.index, sim_anom.values, "-", color=emulators[label]["color"], linewidth=2, label=label)
            else:
                rmse = np.nan
            rows.append(
                {
                    "model": label,
                    "depth": depth_title,
                    "metric": "monthly_ohc_detrended_deseasonalized_anomaly_rmse",
                    "value": rmse,
                }
            )

        ax.set_title(f"{depth_title}", fontsize=14)
        ax.set_ylabel("Detrended DS anomaly (J)", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        handles, labels, ncol = _legend_truth_first(handles, labels, truth_labels={"ARGO-IAP"})
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=ncol,
            bbox_to_anchor=(0.5, 0.04),
            frameon=False,
            fontsize=12,
        )
    extra = (
        f"climatologies: ARGO {climatology_label}; models {period_label}"
        if climatology_label is not None
        else f"climatologies: each series {period_label}"
    )
    _finish_timeseries_figure(
        fig,
        axes,
        "Monthly OHC Detrended Deseasonalized Anomaly vs ARGO-IAP",
        period_label,
        "Time",
        extra=extra,
    )
    return pd.DataFrame(rows)


def _plot_layer_mean_timeseries(
    obs_series_by_layer: xr.DataArray,
    sim_series_by_layer: dict[str, xr.DataArray],
    emulators: dict[str, dict],
    spec: dict,
    period_label: str,
) -> pd.DataFrame:
    fig, axes = plt.subplots(len(OHC_DEPTH_SLICES), 1, figsize=(11.5, 3.8 * len(OHC_DEPTH_SLICES)), sharex=True)
    axes = np.atleast_1d(axes)
    rows = []

    for depth_idx, depth_slice in enumerate(OHC_DEPTH_SLICES):
        ax = axes[depth_idx]
        depth_title = depth_slice["title"]
        obs_series = obs_series_by_layer.sel(depth_slice=depth_title).to_series()
        ax.plot(obs_series.index, obs_series.values, "--", color="black", linewidth=2, label="ARGO-IAP")

        for label, sim in sim_series_by_layer.items():
            sim_series = sim.sel(depth_slice=depth_title).to_series()
            rmse = _rmse_between_series(obs_series, sim_series)
            rows.append(
                {
                    "model": label,
                    "depth": depth_title,
                    "metric": spec["timeseries_metric"],
                    "value": rmse,
                }
            )
            ax.plot(sim_series.index, sim_series.values, "-", color=emulators[label]["color"], linewidth=2, label=label)

        ax.set_title(depth_title, fontsize=14)
        ax.set_ylabel(f"{spec['title']} ({spec['units']})", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)

    handles, labels, ncol = _legend_truth_first(*axes[0].get_legend_handles_labels(), truth_labels={"ARGO-IAP"})
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, 0.04),
        frameon=False,
        fontsize=12,
    )
    _finish_timeseries_figure(
        fig,
        axes,
        f"Monthly Layer-Mean {spec['title']} vs ARGO-IAP",
        period_label,
        "Time",
    )
    return pd.DataFrame(rows)


def _layer_mean_metric_name(spec: dict, suffix: str) -> str:
    base = spec["timeseries_metric"].removesuffix("_rmse")
    return f"{base}_{suffix}_rmse"


def _plot_layer_mean_anomaly_timeseries(
    obs_series_by_layer: xr.DataArray,
    sim_series_by_layer: dict[str, xr.DataArray],
    emulators: dict[str, dict],
    spec: dict,
    period_label: str,
) -> pd.DataFrame:
    fig, axes = plt.subplots(len(OHC_DEPTH_SLICES), 1, figsize=(11.5, 3.8 * len(OHC_DEPTH_SLICES)), sharex=True)
    axes = np.atleast_1d(axes)
    rows = []

    for depth_idx, depth_slice in enumerate(OHC_DEPTH_SLICES):
        ax = axes[depth_idx]
        depth_title = depth_slice["title"]
        obs_series = obs_series_by_layer.sel(depth_slice=depth_title).to_series()
        obs_anom = obs_series - obs_series.mean()
        ax.plot(obs_anom.index, obs_anom.values, "--", color="black", linewidth=2, label="ARGO-IAP")

        for label, sim in sim_series_by_layer.items():
            sim_series = sim.sel(depth_slice=depth_title).to_series()
            sim_anom = sim_series - sim_series.mean()
            rmse = _rmse_between_series(obs_anom, sim_anom)
            rows.append(
                {
                    "model": label,
                    "depth": depth_title,
                    "metric": _layer_mean_metric_name(spec, "anomaly"),
                    "value": rmse,
                }
            )
            ax.plot(sim_anom.index, sim_anom.values, "-", color=emulators[label]["color"], linewidth=2, label=label)

        ax.set_title(depth_title, fontsize=14)
        ax.set_ylabel(f"Anomaly ({spec['units']})", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)

    handles, labels, ncol = _legend_truth_first(*axes[0].get_legend_handles_labels(), truth_labels={"ARGO-IAP"})
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, 0.04),
        frameon=False,
        fontsize=12,
    )
    _finish_timeseries_figure(
        fig,
        axes,
        f"Monthly Layer-Mean {spec['title']} Mean-Removed Anomaly vs ARGO-IAP",
        period_label,
        "Time",
    )
    return pd.DataFrame(rows)


def _plot_layer_mean_monthly_climatology(
    obs_series_by_layer: xr.DataArray,
    sim_series_by_layer: dict[str, xr.DataArray],
    emulators: dict[str, dict],
    spec: dict,
    period_label: str,
    obs_climatology_by_layer: xr.DataArray | None = None,
    climatology_label: str | None = None,
) -> pd.DataFrame:
    fig, axes = plt.subplots(len(OHC_DEPTH_SLICES), 1, figsize=(11.5, 3.8 * len(OHC_DEPTH_SLICES)), sharex=True)
    axes = np.atleast_1d(axes)
    rows = []

    for depth_idx, depth_slice in enumerate(OHC_DEPTH_SLICES):
        ax = axes[depth_idx]
        depth_title = depth_slice["title"]
        obs_clim_source = obs_climatology_by_layer if obs_climatology_by_layer is not None else obs_series_by_layer
        obs_series = obs_clim_source.sel(depth_slice=depth_title).to_series()
        obs_clim = _series_monthly_climatology(obs_series)
        ax.plot(obs_clim.index, obs_clim.values, "--o", color="black", linewidth=2, markersize=4, label="ARGO-IAP")

        for label, sim in sim_series_by_layer.items():
            sim_series = sim.sel(depth_slice=depth_title).to_series()
            sim_clim = _series_monthly_climatology(sim_series)
            rmse = _rmse_between_series(obs_clim, sim_clim)
            rows.append(
                {
                    "model": label,
                    "depth": depth_title,
                    "metric": _layer_mean_metric_name(spec, "climatology"),
                    "value": rmse,
                }
            )
            ax.plot(
                sim_clim.index,
                sim_clim.values,
                "-o",
                color=emulators[label]["color"],
                linewidth=2,
                markersize=4,
                label=label,
            )

        ax.set_title(depth_title, fontsize=14)
        ax.set_ylabel(f"{spec['title']} ({spec['units']})", fontsize=13)
        ax.set_xticks(np.arange(1, 13))
        ax.set_xlim(1, 12)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)

    handles, labels, ncol = _legend_truth_first(*axes[0].get_legend_handles_labels(), truth_labels={"ARGO-IAP"})
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, 0.04),
        frameon=False,
        fontsize=12,
    )
    extra = f"ARGO climatology {climatology_label}" if climatology_label is not None else None
    _finish_timeseries_figure(
        fig,
        axes,
        f"Monthly Layer-Mean {spec['title']} Climatology vs ARGO-IAP",
        period_label,
        "Calendar month",
        extra=extra,
    )
    return pd.DataFrame(rows)


def _plot_layer_mean_deseasoned_anomaly_timeseries(
    obs_series_by_layer: xr.DataArray,
    sim_series_by_layer: dict[str, xr.DataArray],
    emulators: dict[str, dict],
    spec: dict,
    period_label: str,
    obs_climatology_by_layer: xr.DataArray | None = None,
    climatology_label: str | None = None,
) -> pd.DataFrame:
    fig, axes = plt.subplots(len(OHC_DEPTH_SLICES), 1, figsize=(11.5, 3.8 * len(OHC_DEPTH_SLICES)), sharex=True)
    axes = np.atleast_1d(axes)
    rows = []

    for depth_idx, depth_slice in enumerate(OHC_DEPTH_SLICES):
        ax = axes[depth_idx]
        depth_title = depth_slice["title"]
        obs_series = obs_series_by_layer.sel(depth_slice=depth_title).to_series()
        obs_clim_source = obs_climatology_by_layer if obs_climatology_by_layer is not None else obs_series_by_layer
        obs_clim_series = obs_clim_source.sel(depth_slice=depth_title).to_series()
        obs_has_clim = _has_repeated_calendar_months(obs_clim_series)
        if obs_has_clim:
            obs_clim = _series_monthly_climatology(obs_clim_series)
            obs_anom = _series_without_given_monthly_climatology(obs_series, obs_clim)
            ax.plot(obs_anom.index, obs_anom.values, "--", color="black", linewidth=2, label="ARGO-IAP")
        else:
            ax.text(
                0.5,
                0.5,
                "Need at least two complete calendar years\nfor deseasonalized monthly anomalies",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )

        for label, sim in sim_series_by_layer.items():
            sim_series = sim.sel(depth_slice=depth_title).to_series()
            if obs_has_clim:
                sim_clim = _series_monthly_climatology(sim_series)
                sim_anom = _series_without_given_monthly_climatology(sim_series, sim_clim)
                rmse = _rmse_between_series(obs_anom, sim_anom)
                ax.plot(sim_anom.index, sim_anom.values, "-", color=emulators[label]["color"], linewidth=2, label=label)
            else:
                rmse = np.nan
            rows.append(
                {
                    "model": label,
                    "depth": depth_title,
                    "metric": _layer_mean_metric_name(spec, "deseasoned_anomaly"),
                    "value": rmse,
                }
            )

        ax.set_title(depth_title, fontsize=14)
        ax.set_ylabel(f"Deseasonalized anomaly ({spec['units']})", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        handles, labels, ncol = _legend_truth_first(handles, labels, truth_labels={"ARGO-IAP"})
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=ncol,
            bbox_to_anchor=(0.5, 0.04),
            frameon=False,
            fontsize=12,
        )
    extra = (
        f"climatologies: ARGO {climatology_label}; models {period_label}"
        if climatology_label is not None
        else f"climatologies: each series {period_label}"
    )
    _finish_timeseries_figure(
        fig,
        axes,
        f"Monthly Layer-Mean {spec['title']} Deseasonalized Anomaly vs ARGO-IAP",
        period_label,
        "Time",
        extra=extra,
    )
    return pd.DataFrame(rows)


def _plot_layer_mean_detrended_anomaly_timeseries(
    obs_series_by_layer: xr.DataArray,
    sim_series_by_layer: dict[str, xr.DataArray],
    emulators: dict[str, dict],
    spec: dict,
    period_label: str,
) -> pd.DataFrame:
    fig, axes = plt.subplots(len(OHC_DEPTH_SLICES), 1, figsize=(11.5, 3.8 * len(OHC_DEPTH_SLICES)), sharex=True)
    axes = np.atleast_1d(axes)
    rows = []

    for depth_idx, depth_slice in enumerate(OHC_DEPTH_SLICES):
        ax = axes[depth_idx]
        depth_title = depth_slice["title"]
        obs_series = obs_series_by_layer.sel(depth_slice=depth_title).to_series()
        obs_anom = _series_without_linear_trend(obs_series)
        ax.plot(obs_anom.index, obs_anom.values, "--", color="black", linewidth=2, label="ARGO-IAP")

        for label, sim in sim_series_by_layer.items():
            sim_series = sim.sel(depth_slice=depth_title).to_series()
            sim_anom = _series_without_linear_trend(sim_series)
            rmse = _rmse_between_series(obs_anom, sim_anom)
            rows.append(
                {
                    "model": label,
                    "depth": depth_title,
                    "metric": _layer_mean_metric_name(spec, "detrended_anomaly"),
                    "value": rmse,
                }
            )
            ax.plot(sim_anom.index, sim_anom.values, "-", color=emulators[label]["color"], linewidth=2, label=label)

        ax.set_title(depth_title, fontsize=14)
        ax.set_ylabel(f"Detrended anomaly ({spec['units']})", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)

    handles, labels, ncol = _legend_truth_first(*axes[0].get_legend_handles_labels(), truth_labels={"ARGO-IAP"})
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, 0.04),
        frameon=False,
        fontsize=12,
    )
    _finish_timeseries_figure(
        fig,
        axes,
        f"Monthly Layer-Mean {spec['title']} Detrended Anomaly vs ARGO-IAP",
        period_label,
        "Time",
    )
    return pd.DataFrame(rows)


def _plot_layer_mean_detrended_deseasonalized_anomaly_timeseries(
    obs_series_by_layer: xr.DataArray,
    sim_series_by_layer: dict[str, xr.DataArray],
    emulators: dict[str, dict],
    spec: dict,
    period_label: str,
    obs_climatology_by_layer: xr.DataArray | None = None,
    climatology_label: str | None = None,
) -> pd.DataFrame:
    fig, axes = plt.subplots(len(OHC_DEPTH_SLICES), 1, figsize=(11.5, 3.8 * len(OHC_DEPTH_SLICES)), sharex=True)
    axes = np.atleast_1d(axes)
    rows = []

    for depth_idx, depth_slice in enumerate(OHC_DEPTH_SLICES):
        ax = axes[depth_idx]
        depth_title = depth_slice["title"]
        obs_series = obs_series_by_layer.sel(depth_slice=depth_title).to_series()
        obs_clim_source = obs_climatology_by_layer if obs_climatology_by_layer is not None else obs_series_by_layer
        obs_clim_series = obs_clim_source.sel(depth_slice=depth_title).to_series()
        obs_has_clim = _has_repeated_calendar_months(obs_clim_series)
        if obs_has_clim:
            obs_clim = _series_monthly_climatology(obs_clim_series)
            obs_deseasonalized = _series_without_given_monthly_climatology(obs_series, obs_clim)
            obs_anom = _series_without_linear_trend(obs_deseasonalized)
            ax.plot(obs_anom.index, obs_anom.values, "--", color="black", linewidth=2, label="ARGO-IAP")
        else:
            ax.text(
                0.5,
                0.5,
                "Need at least two complete calendar years\nfor detrended deseasonalized monthly anomalies",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )

        for label, sim in sim_series_by_layer.items():
            sim_series = sim.sel(depth_slice=depth_title).to_series()
            if obs_has_clim:
                sim_clim = _series_monthly_climatology(sim_series)
                sim_deseasonalized = _series_without_given_monthly_climatology(sim_series, sim_clim)
                sim_anom = _series_without_linear_trend(sim_deseasonalized)
                rmse = _rmse_between_series(obs_anom, sim_anom)
                ax.plot(sim_anom.index, sim_anom.values, "-", color=emulators[label]["color"], linewidth=2, label=label)
            else:
                rmse = np.nan
            rows.append(
                {
                    "model": label,
                    "depth": depth_title,
                    "metric": _layer_mean_metric_name(spec, "detrended_deseasonalized_anomaly"),
                    "value": rmse,
                }
            )

        ax.set_title(depth_title, fontsize=14)
        ax.set_ylabel(f"Detrended DS anomaly ({spec['units']})", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        handles, labels, ncol = _legend_truth_first(handles, labels, truth_labels={"ARGO-IAP"})
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=ncol,
            bbox_to_anchor=(0.5, 0.04),
            frameon=False,
            fontsize=12,
        )
    extra = (
        f"climatologies: ARGO {climatology_label}; models {period_label}"
        if climatology_label is not None
        else f"climatologies: each series {period_label}"
    )
    _finish_timeseries_figure(
        fig,
        axes,
        f"Monthly Layer-Mean {spec['title']} Detrended Deseasonalized Anomaly vs ARGO-IAP",
        period_label,
        "Time",
        extra=extra,
    )
    return pd.DataFrame(rows)


def main() -> None:
    print("Loading observations...")
    duacs = _infer_obs_grid_metrics(
        _drop_singleton_nonspatial_dims(
            _standardize_latlon_time(_maybe_open_dataset(OBS_CONFIG["duacs"]))
        )
    )
    duacs_raw = _year_slice(
        _infer_obs_grid_metrics(
            _drop_singleton_nonspatial_dims(
                _standardize_latlon_time(_maybe_open_dataset(OBS_CONFIG["duacs_raw"]))
            )
        ),
        args.year,
    )
    oisst = _infer_obs_grid_metrics(
        _drop_singleton_nonspatial_dims(
            _standardize_latlon_time(_maybe_open_dataset(OBS_CONFIG["oisst"]))
        )
    )
    if OBS_CONFIG["oisst_daily"].exists():
        oisst_daily = _infer_obs_grid_metrics(
            _drop_singleton_nonspatial_dims(
                _standardize_latlon_time(_maybe_open_dataset(OBS_CONFIG["oisst_daily"]))
            )
        )
    else:
        print(f"  WARNING: daily OISST path not found, skipping raw-daily SST spectra: {OBS_CONFIG['oisst_daily']}")
        oisst_daily = None
    argo_iap = _drop_singleton_nonspatial_dims(
        _standardize_latlon_time(_maybe_open_dataset(OBS_CONFIG["argo_iap"]))
    )
    if {"lat", "lon"}.issubset(argo_iap.dims) and "area" not in argo_iap:
        argo_iap = _infer_obs_grid_metrics(argo_iap)

    print("Loading emulator rollouts...")
    emulators = {}
    for path, label, color in EMULATOR_CONFIGS:
        print(f" - {label}: {path}")
        emulators[label] = {"ds": _prepare_emulator_dataset(path), "color": color}
    _attach_common_model_wet_masks(emulators)

    duacs_config = DUACS_VELOCITY_COMPONENTS[args.duacs_velocity_kind]
    print(
        "Using DUACS "
        f"{duacs_config['label']} from {duacs_config['ssh']} ({duacs_config['ssh_label']})."
    )

    print("Evaluating EKE maps vs DUACS...")
    eke_metrics = _plot_eke_map_comparison(
        duacs,
        emulators,
        args.year,
        velocity_kind=args.duacs_velocity_kind,
    )
    save_current_fig(f"eke_mean_vs_duacs_{args.year}", MAP_DIR)

    print("Evaluating low-pass-filtered EKE maps vs DUACS...")
    eke_lp_metrics = _plot_eke_map_lowpass_comparison(
        duacs,
        emulators,
        args.year,
        velocity_kind=args.duacs_velocity_kind,
    )
    save_current_fig(f"eke_mean_lowpass_vs_duacs_{args.year}", MAP_DIR)

    argo_map_metrics = []
    for spec in ARGO_STRUCTURE_VARIABLES:
        print(f"Evaluating annual-mean layer-map {spec['title']} vs ARGO-IAP...")
        argo_map_metrics.append(_plot_argo_variable_layer_maps(argo_iap, emulators, args.year, spec))
        save_current_fig(f"{spec['filename']}_layer_mean_maps_vs_argo_iap_{args.year}", MAP_DIR)

    print("Evaluating annual-mean layer-map OHC vs ARGO-IAP...")
    argo_map_metrics.append(_plot_argo_ohc_layer_maps(argo_iap, emulators, args.year))
    save_current_fig(f"ohc_layer_integrated_maps_vs_argo_iap_{args.year}", MAP_DIR)

    print("Evaluating isotropic temperature spectra vs OISST...")
    temp_spec_metrics = _plot_temperature_spectra(oisst, emulators, oisst_daily=oisst_daily)
    save_current_fig("temperature_isotropic_spectra_vs_oisst", SPECTRUM_DIR)

    print("Evaluating temporal KE spectra vs raw and 5-day DUACS...")
    ke_spec_metrics = _plot_temporal_ke_spectra(
        duacs,
        duacs_raw,
        emulators,
        velocity_kind=args.duacs_velocity_kind,
    )
    save_current_fig("ke_temporal_spectra_vs_duacs", SPECTRUM_DIR)

    ohc_start, ohc_end = _resolve_ohc_period(
        argo_iap,
        emulators,
        args.ohc_start_date,
        args.ohc_end_date,
    )
    ohc_label = _period_label(ohc_start, ohc_end)
    ohc_slug = _period_slug(ohc_start, ohc_end)
    obs_clim_start, obs_clim_end = _resolve_obs_climatology_period(
        argo_iap,
        args.obs_climatology_start_date,
        args.obs_climatology_end_date,
    )
    obs_clim_label = _period_label(obs_clim_start, obs_clim_end)
    print(f"Using ARGO-IAP monthly climatologies over {obs_clim_label}.")
    print(f"Evaluating layered monthly OHC vs ARGO-IAP over {ohc_label}...")
    obs_ohc, sim_ohc = _collect_ohc_timeseries(argo_iap, emulators, ohc_start, ohc_end)
    obs_ohc_climatology_source = _compute_obs_ohc_timeseries(argo_iap, obs_clim_start, obs_clim_end)
    ohc_metrics = _plot_ohc_timeseries(obs_ohc, sim_ohc, emulators, ohc_label)
    save_current_fig(f"ohc_monthly_vs_argo_iap_{ohc_slug}", TIMESERIES_DIR)

    print("Evaluating monthly OHC mean-removed anomaly vs ARGO-IAP...")
    ohc_anomaly_metrics = _plot_ohc_anomaly_timeseries(obs_ohc, sim_ohc, emulators, ohc_label)
    save_current_fig(f"ohc_anomaly_vs_argo_iap_{ohc_slug}", TIMESERIES_DIR)

    print("Evaluating monthly OHC detrended anomaly vs ARGO-IAP...")
    ohc_detrended_metrics = _plot_ohc_detrended_anomaly_timeseries(obs_ohc, sim_ohc, emulators, ohc_label)
    save_current_fig(f"ohc_detrended_anomaly_vs_argo_iap_{ohc_slug}", TIMESERIES_DIR)

    print("Evaluating monthly OHC climatology vs ARGO-IAP...")
    ohc_climatology_metrics = _plot_ohc_monthly_climatology(
        obs_ohc,
        sim_ohc,
        emulators,
        ohc_label,
        obs_climatology_ohc=obs_ohc_climatology_source,
        climatology_label=obs_clim_label,
    )
    save_current_fig(f"ohc_monthly_climatology_vs_argo_iap_{ohc_slug}", TIMESERIES_DIR)

    ohc_deseasoned_metrics = pd.DataFrame()
    ohc_detrended_deseasonalized_metrics = pd.DataFrame()
    if args.plot_deseasoned_anomaly:
        print("Evaluating monthly OHC deseasoned anomaly vs ARGO-IAP...")
        ohc_deseasoned_metrics = _plot_ohc_deseasoned_anomaly_timeseries(
            obs_ohc,
            sim_ohc,
            emulators,
            ohc_label,
            obs_climatology_ohc=obs_ohc_climatology_source,
            climatology_label=obs_clim_label,
        )
        save_current_fig(f"ohc_deseasoned_anomaly_vs_argo_iap_{ohc_slug}", TIMESERIES_DIR)

        print("Evaluating monthly OHC detrended deseasonalized anomaly vs ARGO-IAP...")
        ohc_detrended_deseasonalized_metrics = _plot_ohc_detrended_deseasonalized_anomaly_timeseries(
            obs_ohc,
            sim_ohc,
            emulators,
            ohc_label,
            obs_climatology_ohc=obs_ohc_climatology_source,
            climatology_label=obs_clim_label,
        )
        save_current_fig(f"ohc_detrended_deseasonalized_anomaly_vs_argo_iap_{ohc_slug}", TIMESERIES_DIR)

    layer_mean_metrics = []
    for spec in ARGO_STRUCTURE_VARIABLES:
        print(f"Evaluating monthly layer-mean {spec['title']} vs ARGO-IAP...")
        obs_layer_mean, sim_layer_mean = _collect_layer_mean_timeseries(argo_iap, emulators, spec, ohc_start, ohc_end)
        obs_layer_mean_climatology_source = _compute_obs_layer_mean_timeseries(
            argo_iap,
            spec,
            obs_clim_start,
            obs_clim_end,
        )
        layer_mean_metrics.append(
            _plot_layer_mean_timeseries(obs_layer_mean, sim_layer_mean, emulators, spec, ohc_label)
        )
        save_current_fig(f"{spec['filename']}_monthly_vs_argo_iap_{ohc_slug}", TIMESERIES_DIR)

        print(f"Evaluating monthly layer-mean {spec['title']} mean-removed anomaly vs ARGO-IAP...")
        layer_mean_metrics.append(
            _plot_layer_mean_anomaly_timeseries(obs_layer_mean, sim_layer_mean, emulators, spec, ohc_label)
        )
        save_current_fig(f"{spec['filename']}_anomaly_vs_argo_iap_{ohc_slug}", TIMESERIES_DIR)

        print(f"Evaluating monthly layer-mean {spec['title']} detrended anomaly vs ARGO-IAP...")
        layer_mean_metrics.append(
            _plot_layer_mean_detrended_anomaly_timeseries(obs_layer_mean, sim_layer_mean, emulators, spec, ohc_label)
        )
        save_current_fig(f"{spec['filename']}_detrended_anomaly_vs_argo_iap_{ohc_slug}", TIMESERIES_DIR)

        print(f"Evaluating monthly layer-mean {spec['title']} climatology vs ARGO-IAP...")
        layer_mean_metrics.append(
            _plot_layer_mean_monthly_climatology(
                obs_layer_mean,
                sim_layer_mean,
                emulators,
                spec,
                ohc_label,
                obs_climatology_by_layer=obs_layer_mean_climatology_source,
                climatology_label=obs_clim_label,
            )
        )
        save_current_fig(f"{spec['filename']}_monthly_climatology_vs_argo_iap_{ohc_slug}", TIMESERIES_DIR)

        if args.plot_deseasoned_anomaly:
            print(f"Evaluating monthly layer-mean {spec['title']} deseasoned anomaly vs ARGO-IAP...")
            layer_mean_metrics.append(
                _plot_layer_mean_deseasoned_anomaly_timeseries(
                    obs_layer_mean,
                    sim_layer_mean,
                    emulators,
                    spec,
                    ohc_label,
                    obs_climatology_by_layer=obs_layer_mean_climatology_source,
                    climatology_label=obs_clim_label,
                )
            )
            save_current_fig(f"{spec['filename']}_deseasoned_anomaly_vs_argo_iap_{ohc_slug}", TIMESERIES_DIR)

            print(f"Evaluating monthly layer-mean {spec['title']} detrended deseasonalized anomaly vs ARGO-IAP...")
            layer_mean_metrics.append(
                _plot_layer_mean_detrended_deseasonalized_anomaly_timeseries(
                    obs_layer_mean,
                    sim_layer_mean,
                    emulators,
                    spec,
                    ohc_label,
                    obs_climatology_by_layer=obs_layer_mean_climatology_source,
                    climatology_label=obs_clim_label,
                )
            )
            save_current_fig(f"{spec['filename']}_detrended_deseasonalized_anomaly_vs_argo_iap_{ohc_slug}", TIMESERIES_DIR)

    structure_metrics = []
    for spec in ARGO_STRUCTURE_VARIABLES:
        print(f"Evaluating annual-mean {spec['title']} zonal mean vs ARGO-IAP...")
        structure_metrics.append(_plot_argo_variable_zonal_mean(argo_iap, emulators, args.year, spec))
        save_current_fig(f"argo_{spec['filename']}_zonal_mean_{args.year}", STRUCTURE_DIR)

        print(f"Evaluating annual-mean {spec['title']} structure vs ARGO-IAP...")
        structure_metrics.append(_plot_argo_variable_zonal_mean_bias(argo_iap, emulators, args.year, spec))
        save_current_fig(f"argo_{spec['filename']}_zonal_mean_bias_{args.year}", STRUCTURE_DIR)

    print("Evaluating annual-mean OHC zonal mean vs ARGO-IAP...")
    structure_metrics.append(_plot_argo_ohc_zonal_mean(argo_iap, emulators, args.year))
    save_current_fig(f"argo_ohc_zonal_mean_{args.year}", STRUCTURE_DIR)

    print("Evaluating annual-mean OHC structure vs ARGO-IAP...")
    ohc_structure_metrics = _plot_argo_ohc_zonal_mean_bias(argo_iap, emulators, args.year)
    save_current_fig(f"argo_ohc_zonal_mean_bias_{args.year}", STRUCTURE_DIR)

    metrics = pd.concat(
        [
            eke_metrics,
            eke_lp_metrics,
            *argo_map_metrics,
            temp_spec_metrics,
            ke_spec_metrics,
            ohc_metrics,
            ohc_anomaly_metrics,
            ohc_detrended_metrics,
            ohc_climatology_metrics,
            ohc_deseasoned_metrics,
            ohc_detrended_deseasonalized_metrics,
            *layer_mean_metrics,
            *structure_metrics,
            ohc_structure_metrics,
        ],
        ignore_index=True,
    )
    metrics_path = os.path.join(METRICS_DIR, f"observation_metrics_{args.year}.csv")
    metrics.to_csv(metrics_path, index=False)
    print(f"  -> Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
