"""
Observation-based evaluation driver for Samudra rollouts.

This script mirrors the existing notebook-exported evaluation drivers, but it
compares emulator rollouts against observational products instead of OM4 truth.

Implemented evaluations
-----------------------
1. Complete-year 2015-2022 total RMSE metrics and spatial RMSE maps for surface geostrophic
   velocity, instantaneous surface EKE, SST, and OHC, with annual uncertainty diagnostics and
   a retained target-year snapshot.
2. Velocity diagnostics: surface EKE maps, open-ocean North Pacific / Gulf Stream / Agulhas EKE
   spatial spectra, and Global / Gulf Stream / Agulhas temporal KE spectra, including per-year
   geometric-mean and log-standard-deviation diagnostics.
3. Temperature diagnostics: open-ocean North Pacific / Gulf Stream / Agulhas SST spatial spectra
   and OHC totals for upper 700m and 700-2000m.
4. Time-series anomaly diagnostics for SST, EKE, and upper-700m OHC: mean-removed anomalies,
   linear trend comparison, detrended+deseasonalized anomalies, and residual anomaly variance
   (SST/EKE retain 5-day cadence).
5. Maps of detrended+deseasonalized anomaly variance for SST and upper-700m OHC.

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
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")

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

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "32")))


parser = argparse.ArgumentParser(description="Observation-based evaluation")
parser.add_argument("--year", type=int, default=2022, help="Target evaluation year for map/time-series products")
parser.add_argument(
    "--rmse-start-date",
    default=None,
    help="Optional start date for primary RMSE diagnostics. Defaults to 2015-01-01.",
)
parser.add_argument(
    "--rmse-end-date",
    default=None,
    help="Optional end date for primary RMSE diagnostics. Defaults to 2022-12-31.",
)
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
    help=(
        "Legacy option retained for compatibility. The current main workflow estimates seasonal "
        "cycles from each diagnostic overlap window."
    ),
)
parser.add_argument(
    "--obs-climatology-end-date",
    default=None,
    help=(
        "Legacy option retained for compatibility. The current main workflow estimates seasonal "
        "cycles from each diagnostic overlap window."
    ),
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
    "--duacs-raw-path",
    type=Path,
    default=None,
    help=(
        "Optional raw daily DUACS zarr path for temporal spectra. Defaults to a "
        "2014-2022 DUACS zarr under --raw-obs-root when present, otherwise the "
        "legacy 2022-only DUACS zarr."
    ),
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
        "Legacy option retained for compatibility. The current main workflow always writes "
        "detrended+deseasonalized anomaly diagnostics."
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
OHC_UPPER_700_SLICE = DEPTH_SLICES[0]

SPECTRUM_REGIONS = [
    ("Gulf Stream", slice(300, 320), slice(25, 45)),
    ("Agulhas", slice(40, 60), slice(-50, -30)),
]
SPATIAL_SPECTRUM_PANELS = [
    ("North Pacific", slice(170, 200), slice(25, 45)),
    *SPECTRUM_REGIONS,
]
SPATIAL_SPECTRUM_PANEL_LABEL = "North Pacific, Gulf Stream, and Agulhas open-ocean boxes"
TEMPORAL_SPECTRUM_PANELS = [
    ("Global", slice(0, 360), slice(-90, 90)),
    *SPECTRUM_REGIONS,
]
SPECTRUM_TIME_BATCH = 32
RMSE_BOOTSTRAP_SAMPLES = 10_000

# Use the same complete calendar years for primary RMSE and interannual spectra so the
# point estimates, annual variability, and calendar-year block bootstrap share one period.
PRIMARY_EVALUATION_YEARS = list(range(2015, 2023))
PRIMARY_RMSE_START = pd.Timestamp(f"{PRIMARY_EVALUATION_YEARS[0]}-01-01")
PRIMARY_RMSE_END = pd.Timestamp(f"{PRIMARY_EVALUATION_YEARS[-1]}-12-31")
SPECTRA_INTERANNUAL_YEARS = PRIMARY_EVALUATION_YEARS

DUACS_RAW_ZARR = (
    "duacs/"
    "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_multi-vars_"
    "179.94W-179.94E_89.94S-89.94N_2022-01-01-2023-01-01.zarr"
)
DUACS_RAW_LONG_COLLECTION = (
    "duacs/"
    "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_velocity_"
    "2014-10-18-2022-12-26_yearly"
)
DUACS_RAW_LONG_ZARR = (
    "duacs/"
    "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_multi-vars_"
    "179.94W-179.94E_89.94S-89.94N_2014-10-18-2022-12-26.zarr"
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

RHO = 1035.0  # kg m^-3
CP = 3850.0  # J kg^-1 K^-1
OHC_REFERENCE = "temperature_integral_relative_to_0_degC"


def _attach_ohc_convention_metadata(metrics: pd.DataFrame) -> pd.DataFrame:
    """Record the OHC convention on OHC metric rows for reproducible external comparisons."""
    metrics = metrics.copy()
    ohc_rows = metrics.get("metric", pd.Series(index=metrics.index, dtype=str)).str.contains(
        "ohc", case=False, na=False
    )
    metrics["ohc_density_kg_m3"] = np.where(ohc_rows, RHO, np.nan)
    metrics["ohc_heat_capacity_j_kg_k"] = np.where(ohc_rows, CP, np.nan)
    metrics["ohc_volumetric_heat_capacity_j_m3_k"] = np.where(ohc_rows, RHO * CP, np.nan)
    metrics["ohc_reference"] = np.where(ohc_rows, OHC_REFERENCE, None)
    return metrics


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
    # bucketize maps values above the final edge to the final bin. Exclude those Fourier-corner
    # modes explicitly so they cannot contaminate the highest retained isotropic bin.
    retained_modes = k_mag_flat <= k_bins[-1]
    k_mag_binned = k_mag_flat[retained_modes]
    bin_indices = torch.bucketize(k_mag_binned, k_bins[1:-1], right=True)
    psd_flat = psd_2d.reshape(merged, k_mag_flat.shape[0])[:, retained_modes]
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


def _climatology_extra(climatology_label: str | None, period_label: str) -> str:
    if climatology_label is None:
        return f"climatologies: each series {period_label}"
    if climatology_label == period_label:
        return f"seasonal cycle over {climatology_label} (ARGO & models aligned)"
    return f"climatologies: ARGO {climatology_label}; models {period_label}"


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = "Figures_observations"
PDF_DIR = os.path.join(OUTPUT_DIR, "pdf_figures")
MAP_DIR = os.path.join(PDF_DIR, "maps")
RMSE_DIR = os.path.join(PDF_DIR, "rmse")
SPECTRUM_DIR = os.path.join(PDF_DIR, "spectrum")
TIMESERIES_DIR = os.path.join(PDF_DIR, "timeseries")
STRUCTURE_DIR = os.path.join(PDF_DIR, "structure")
VARIANCE_DIR = os.path.join(PDF_DIR, "variance")
METRICS_DIR = os.path.join(PDF_DIR, "metrics")

for directory in [OUTPUT_DIR, PDF_DIR, MAP_DIR, RMSE_DIR, SPECTRUM_DIR, TIMESERIES_DIR, STRUCTURE_DIR, VARIANCE_DIR, METRICS_DIR]:
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
    "duacs_raw": args.duacs_raw_path
    if args.duacs_raw_path is not None
    else next(
        (
            args.raw_obs_root / relative
            for relative in [DUACS_RAW_LONG_COLLECTION, DUACS_RAW_LONG_ZARR, DUACS_RAW_ZARR]
            if (args.raw_obs_root / relative).exists()
        ),
        args.raw_obs_root / DUACS_RAW_LONG_COLLECTION,
    ),
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
            if len(zarrs) > 1:
                datasets = [xr.open_zarr(zarr) for zarr in zarrs]
                ds = xr.concat(
                    datasets,
                    dim="time",
                    data_vars="minimal",
                    coords="minimal",
                    compat="override",
                    combine_attrs="override",
                )
                if "time" in ds.coords:
                    ds = ds.sortby("time")
                    times = pd.DatetimeIndex(ds.time.values)
                    keep = ~times.duplicated()
                    if not keep.all():
                        ds = ds.isel(time=keep)
                return ds
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

    timestamps = _coerce_datetime_values(values)
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


def _depth_edges_from_centers(depth_vals: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_vals, dtype=float)
    if depth.ndim != 1 or depth.size < 2:
        raise ValueError("Expected at least two 1D depth centers")
    if not np.all(np.isfinite(depth)) or np.any(np.diff(depth) <= 0):
        raise ValueError("Depth centers must be finite and strictly increasing")
    edges = np.empty(depth.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (depth[:-1] + depth[1:])
    edges[0] = max(0.0, depth[0] - 0.5 * (depth[1] - depth[0]))
    edges[-1] = depth[-1] + 0.5 * (depth[-1] - depth[-2])
    return edges


def _depth_thickness_from_centers(depth_vals: np.ndarray) -> np.ndarray:
    return np.diff(_depth_edges_from_centers(depth_vals))


def _layer_overlap_thickness(
    depth: xr.DataArray,
    min_depth: float,
    max_depth: float,
    native_dz: xr.DataArray | None = None,
) -> xr.DataArray:
    """Thickness of each native cell inside the half-open interval [min_depth, max_depth).

    Observation products provide depth centers, so their cell edges are reconstructed from
    adjacent centers. Model products additionally provide exact native layer thicknesses;
    those are used with the native centers to recover the model cell bounds. Partial boundary
    cells are prorated rather than included or excluded based only on their center depth.
    """
    min_depth = float(min_depth)
    max_depth = float(max_depth)
    if not np.isfinite(min_depth) or not np.isfinite(max_depth) or max_depth <= min_depth:
        raise ValueError(f"Invalid target depth interval [{min_depth}, {max_depth})")

    if native_dz is None:
        edges = _depth_edges_from_centers(depth.values)
        lower = xr.DataArray(edges[:-1], dims=depth.dims, coords=depth.coords)
        upper = xr.DataArray(edges[1:], dims=depth.dims, coords=depth.coords)
    else:
        if depth.ndim != 1:
            raise ValueError(f"Expected a 1D native depth coordinate, got dimensions {depth.dims}")
        lower = depth - 0.5 * native_dz
        upper = depth + 0.5 * native_dz

    clipped_lower = xr.where(lower > min_depth, lower, min_depth)
    clipped_upper = xr.where(upper < max_depth, upper, max_depth)
    overlap = (clipped_upper - clipped_lower).clip(min=0.0)
    overlap.name = "layer_overlap_dz"
    return overlap


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


def _require_exact_time_match(
    model: xr.Dataset | xr.DataArray,
    observation: xr.Dataset | xr.DataArray,
    context: str,
) -> None:
    """Fail rather than silently interpolate when model and observation times differ."""
    if "time" not in model.dims or "time" not in observation.dims:
        raise ValueError(f"{context}: both model and observation fields must have a time dimension")

    model_time = _coerce_datetime_values(model["time"].values)
    obs_time = _coerce_datetime_values(observation["time"].values)
    if model_time.equals(obs_time):
        return

    common = model_time.intersection(obs_time)
    model_only = model_time.difference(obs_time)
    obs_only = obs_time.difference(model_time)
    details = [
        f"model samples={len(model_time)}",
        f"observation samples={len(obs_time)}",
        f"common timestamps={len(common)}",
    ]
    if len(model_only):
        details.append(f"first model-only timestamp={model_only[0]}")
    if len(obs_only):
        details.append(f"first observation-only timestamp={obs_only[0]}")
    if not len(model_only) and not len(obs_only):
        details.append("the timestamp order differs")
    raise ValueError(
        f"{context}: model and observation timestamps must match exactly before spatial "
        f"interpolation ({'; '.join(details)}). Temporal interpolation is disabled to avoid "
        "silently smoothing model variability."
    )


def _common_time_window(*datasets: xr.Dataset | xr.DataArray) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Intersection of the time spans of all given datasets."""
    starts, ends = [], []
    for ds in datasets:
        lo, hi = _time_bounds(ds)
        starts.append(lo)
        ends.append(hi)
    start, end = max(starts), min(ends)
    if start > end:
        raise ValueError("No common time window across the provided datasets")
    return start, end


def _bounded_common_time_window(
    *datasets: xr.Dataset | xr.DataArray,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    common_start, common_end = _common_time_window(*datasets)
    bounded_start = max(common_start, start) if start is not None else common_start
    bounded_end = min(common_end, end) if end is not None else common_end
    if bounded_start > bounded_end:
        raise ValueError(
            "No common time samples within requested RMSE period: "
            f"requested={start} to {end}, common={common_start} to {common_end}"
        )
    return bounded_start, bounded_end


def _bounded_common_month_window(
    *datasets: xr.Dataset | xr.DataArray,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    starts, ends = zip(*[_time_bounds(ds) for ds in datasets], strict=True)
    common_start = max(timestamp.to_period("M") for timestamp in starts)
    common_end = min(timestamp.to_period("M") for timestamp in ends)
    if start is not None:
        common_start = max(common_start, start.to_period("M"))
    if end is not None:
        common_end = min(common_end, end.to_period("M"))
    if common_start > common_end:
        raise ValueError(f"No common monthly samples within requested RMSE period: {start} to {end}")
    return common_start.start_time, common_end.end_time


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
    # ARGO-IAP timestamps label monthly means at month start, while model timestamps
    # represent 5-day samples. Resolve the overlap by calendar month so a timestamp at
    # 2022-12-01 includes all model samples in December rather than truncating at day 1.
    return _bounded_common_month_window(
        argo_iap,
        *[emu["ds"] for emu in emulators.values()],
        start=_date_or_none(start_date),
        end=_date_or_none(end_date),
    )


def _resolve_obs_climatology_period(
    argo_iap: xr.Dataset,
    start_date: str | None,
    end_date: str | None,
    default_start: pd.Timestamp | None = None,
    default_end: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    obs_start, obs_end = _time_bounds(argo_iap)
    start = _date_or_none(start_date) or default_start or obs_start
    end = _date_or_none(end_date) or default_end or obs_end
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


def _median_time_step_days(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return float("nan")
    delta_days = np.diff(index.values).astype("timedelta64[s]").astype(float) / 86400.0
    delta_days = delta_days[np.isfinite(delta_days) & (delta_days > 0)]
    return float(np.median(delta_days)) if delta_days.size else float("nan")


def _pentad_index(index: pd.DatetimeIndex) -> np.ndarray:
    # Use a 365-day climatological calendar so dates after February remain aligned between
    # leap and non-leap years. Associate February 29 with the February 28 climatology bin.
    dayofyear = index.dayofyear.to_numpy().copy()
    is_leap = np.asarray(index.is_leap_year)
    month = index.month.to_numpy()
    day = index.day.to_numpy()
    dayofyear[is_leap & (month > 2)] -= 1
    dayofyear[is_leap & (month == 2) & (day == 29)] = 59
    return ((dayofyear - 1) // 5 + 1).astype(int)


def _series_without_pentad_climatology(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    pentad = pd.Series(_pentad_index(pd.DatetimeIndex(values.index)), index=values.index)
    climatology = values.groupby(pentad).mean()
    seasonal = pentad.map(climatology)
    return values - seasonal


def _series_without_seasonal_cycle(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    index = pd.DatetimeIndex(values.index)
    if _median_time_step_days(index) <= 10:
        counts = pd.Series(_pentad_index(index)).value_counts()
        if (counts >= 2).sum() >= 24:
            return _series_without_pentad_climatology(values)
    if _has_repeated_calendar_months(values):
        return _series_without_monthly_climatology(values)
    return values - values.mean()


def _series_linear_trend_fit_per_year(series: pd.Series) -> tuple[pd.Series, float]:
    values = series.astype(float)
    if values.empty:
        return values, float("nan")
    finite = np.isfinite(values.values)
    if finite.sum() < 2:
        return pd.Series(np.nan, index=values.index), float("nan")
    index = pd.DatetimeIndex(values.index)
    x = (index - index[0]).days.to_numpy(dtype=float) / 365.25
    slope, intercept = np.polyfit(x[finite], values.values[finite], 1)
    trend = pd.Series(slope * x + intercept, index=values.index)
    return trend, float(slope)


def _series_linear_trend_per_year(series: pd.Series) -> float:
    _, slope = _series_linear_trend_fit_per_year(series)
    return float(slope)


def _series_detrended_deseasonalized_anomaly(series: pd.Series) -> pd.Series:
    return _series_without_linear_trend(_series_without_seasonal_cycle(series))


def _series_variance(series: pd.Series) -> float:
    values = np.asarray(series.values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.var(finite))


def _has_repeated_calendar_months(series: pd.Series) -> bool:
    counts = series.groupby(series.index.month).size()
    return bool(len(counts) == 12 and (counts >= 2).all())


def _area_weighted_timeseries(
    field: xr.DataArray,
    area: xr.DataArray,
    spatial_dims: tuple[str, str],
) -> xr.DataArray:
    mask = np.isfinite(field)
    weighted_sum = (field.where(mask) * area).sum(spatial_dims, skipna=True)
    weight_sum = area.where(mask).sum(spatial_dims, skipna=True)
    return weighted_sum / weight_sum


def _resample_monthly_field(field: xr.DataArray, context: str = "field") -> xr.DataArray:
    if field.sizes.get("time", 0) == 0:
        raise ValueError(f"Cannot monthly-resample {context}: empty time coordinate")
    return field.resample(time="MS").mean()


def _detrend_linear_dataarray(field: xr.DataArray, time_dim: str = "time") -> xr.DataArray:
    if field.sizes.get(time_dim, 0) < 2:
        return field - field.mean(time_dim, skipna=True)

    x = xr.DataArray(
        np.arange(field.sizes[time_dim], dtype=float),
        dims=[time_dim],
        coords={time_dim: field[time_dim]},
    )
    finite = np.isfinite(field)
    x_mean = x.where(finite).mean(time_dim, skipna=True)
    y_mean = field.mean(time_dim, skipna=True)
    cov = ((x - x_mean) * (field - y_mean)).where(finite).sum(time_dim, skipna=True)
    var = ((x - x_mean) ** 2).where(finite).sum(time_dim, skipna=True)
    slope = xr.where(var > 0, cov / var, 0.0)
    intercept = y_mean - slope * x_mean
    return field - (slope * x + intercept)


def _detrended_deseasonalized_dataarray(field: xr.DataArray, time_dim: str = "time") -> xr.DataArray:
    if field.sizes.get(time_dim, 0) == 0:
        return field
    time_index = pd.DatetimeIndex(field[time_dim].values)
    if _median_time_step_days(time_index) <= 10:
        pentad_values = _pentad_index(time_index)
        counts = pd.Series(pentad_values).value_counts()
        if (counts >= 2).sum() >= 24:
            pentad = xr.DataArray(
                pentad_values,
                dims=[time_dim],
                coords={time_dim: field[time_dim]},
                name="pentad",
            )
            with_pentad = field.assign_coords(pentad=pentad)
            deseasonalized = with_pentad.groupby("pentad") - with_pentad.groupby("pentad").mean(
                time_dim, skipna=True
            )
            if "pentad" in deseasonalized.coords:
                deseasonalized = deseasonalized.drop_vars("pentad")
            return _detrend_linear_dataarray(deseasonalized, time_dim=time_dim)
    if _has_repeated_calendar_months(pd.Series(index=time_index, dtype=float)):
        deseasonalized = field.groupby(f"{time_dim}.month") - field.groupby(f"{time_dim}.month").mean(
            time_dim, skipna=True
        )
    else:
        deseasonalized = field - field.mean(time_dim, skipna=True)
    return _detrend_linear_dataarray(deseasonalized, time_dim=time_dim)


def _field_rmse_over_time(sim: xr.DataArray, obs: xr.DataArray, context: str = "field") -> xr.DataArray:
    sim_aligned, obs_aligned = xr.align(sim, obs, join="inner")
    if sim_aligned.sizes.get("time", 0) == 0:
        raise ValueError(f"No common time samples for {context} RMSE")
    return np.sqrt(((sim_aligned - obs_aligned) ** 2).mean("time", skipna=True))


def _complete_calendar_years(time: xr.DataArray, min_sample_fraction: float = 0.95) -> list[int]:
    index = pd.DatetimeIndex(time.values)
    if index.empty:
        return []
    counts = pd.Series(index.year).value_counts().sort_index()
    target_count = int(counts.max())
    step_days = _median_time_step_days(index)
    boundary_tolerance = pd.Timedelta(days=1.5 * step_days) if np.isfinite(step_days) else pd.Timedelta(0)
    years = []
    for year, count in counts.items():
        year_index = index[index.year == year]
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year}-12-31 23:59:59.999999999")
        covers_calendar = (
            year_index.min() <= year_start + boundary_tolerance
            and year_index.max() >= year_end - boundary_tolerance
        )
        if covers_calendar and count >= min_sample_fraction * target_count:
            years.append(int(year))
    return years


def _require_complete_calendar_years(time: xr.DataArray, context: str) -> list[int]:
    """Return selected years, failing if the primary period contains a partial year."""
    index = pd.DatetimeIndex(time.values)
    present_years = sorted({int(year) for year in index.year})
    complete_years = _complete_calendar_years(time)
    if complete_years != present_years:
        incomplete = sorted(set(present_years) - set(complete_years))
        raise ValueError(
            f"{context}: uncertainty-enabled primary RMSE requires complete calendar years; "
            f"incomplete years={incomplete}. Adjust --rmse-start-date/--rmse-end-date."
        )
    return complete_years


def _interannual_rmse_summary(
    annual_rmse: np.ndarray,
    bootstrap_samples: int = RMSE_BOOTSTRAP_SAMPLES,
) -> dict[str, float | int | str]:
    values = np.asarray(annual_rmse, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "annual_std": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_years": 0,
            "n_bootstrap": 0,
            "uncertainty_method": "unavailable",
            "block_aggregate_rmse": np.nan,
            "primary_aggregation_method": "root_mean_annual_mse_equal_calendar_years",
        }

    annual_mse = values ** 2
    block_aggregate_rmse = float(np.sqrt(annual_mse.mean()))
    annual_std = float(np.std(values, ddof=1)) if values.size > 1 else np.nan
    if values.size < 2 or bootstrap_samples <= 0:
        ci_low = ci_high = np.nan
        n_bootstrap = 0
    else:
        rng = np.random.default_rng(0)
        sampled = rng.choice(annual_mse, size=(bootstrap_samples, values.size), replace=True)
        bootstrap_rmse = np.sqrt(sampled.mean(axis=1))
        ci_low, ci_high = np.percentile(bootstrap_rmse, [2.5, 97.5]).astype(float)
        n_bootstrap = int(bootstrap_samples)

    return {
        "annual_std": annual_std,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_years": int(values.size),
        "n_bootstrap": n_bootstrap,
        "uncertainty_method": "calendar_year_block_bootstrap",
        "block_aggregate_rmse": block_aggregate_rmse,
        "primary_aggregation_method": "root_mean_annual_mse_equal_calendar_years",
    }


def _rmse_map_and_annual_rows(
    error_squared: xr.DataArray,
    area: xr.DataArray,
    spatial_dims: tuple[str, str],
    model: str,
    metric: str,
    depth: str | None = None,
    include_annual: bool = False,
) -> tuple[xr.DataArray, pd.DataFrame, dict[str, float | int | str]]:
    if error_squared.sizes.get("time", 0) == 0:
        raise ValueError(f"No time samples available for {model} {metric}")
    if not include_annual:
        rmse_map = np.sqrt(error_squared.mean("time", skipna=True))
        return rmse_map, pd.DataFrame(), {}

    years = _require_complete_calendar_years(error_squared["time"], f"{model} {metric}")
    if not years:
        rmse_map = np.sqrt(error_squared.mean("time", skipna=True))
        return rmse_map, pd.DataFrame(), _interannual_rmse_summary(np.array([]))

    annual_mse_maps = error_squared.groupby("time.year").mean("time", skipna=True).sel(year=years)
    # The primary map and scalar score use the same equal-year blocks as the bootstrap.
    rmse_map = np.sqrt(annual_mse_maps.mean("year", skipna=True))
    finite = np.isfinite(annual_mse_maps)
    weighted_sum = (annual_mse_maps.where(finite) * area).sum(spatial_dims, skipna=True)
    weight_sum = area.where(finite).sum(spatial_dims, skipna=True)
    annual_rmse = np.sqrt(weighted_sum / weight_sum)

    computed = xr.Dataset({"rmse_map": rmse_map, "annual_rmse": annual_rmse}).compute()
    rmse_map = computed["rmse_map"]
    annual_values = np.asarray(computed["annual_rmse"].values, dtype=float)
    time_index = pd.DatetimeIndex(error_squared["time"].values)
    rows = []
    for year, value in zip(years, annual_values, strict=True):
        year_index = time_index[time_index.year == year]
        rows.append(
            {
                "model": model,
                "metric": metric,
                "value": float(value),
                "depth": depth,
                "period_kind": "annual",
                "period_start": f"{year_index.min():%Y-%m-%d}",
                "period_end": f"{year_index.max():%Y-%m-%d}",
                "n_time_samples": int(year_index.size),
                "year": int(year),
                "is_primary": False,
            }
        )
    return rmse_map, pd.DataFrame(rows), _interannual_rmse_summary(annual_values)


def _annotate_rmse_period(
    rows: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    period_kind: str,
    n_time_samples: int,
    summaries: dict[tuple[str, str | None], dict[str, float | int | str]] | None = None,
) -> pd.DataFrame:
    rows = rows.copy()
    rows["period_kind"] = period_kind
    rows["period_start"] = f"{start:%Y-%m-%d}"
    rows["period_end"] = f"{end:%Y-%m-%d}"
    rows["n_time_samples"] = int(n_time_samples)
    rows["year"] = np.nan
    rows["is_primary"] = period_kind == "primary_complete_years"
    for column in [
        "annual_std",
        "ci_low",
        "ci_high",
        "n_years",
        "n_bootstrap",
        "uncertainty_method",
        "block_aggregate_rmse",
        "primary_aggregation_method",
    ]:
        rows[column] = None if column in {"uncertainty_method", "primary_aggregation_method"} else np.nan

    if summaries:
        for idx, row in rows.iterrows():
            depth = row.get("depth")
            key = (row["model"], None if pd.isna(depth) else str(depth))
            summary = summaries.get(key)
            if not summary:
                continue
            for column, value in summary.items():
                rows.at[idx, column] = value
    return rows


def _area_weighted_pattern_corr(
    model: xr.DataArray,
    observation: xr.DataArray,
    area: xr.DataArray,
) -> float:
    """Centered pattern correlation on the common finite, positive-area mask."""
    model, observation, area = xr.align(model, observation, area, join="inner")
    valid = np.isfinite(model) & np.isfinite(observation) & np.isfinite(area) & (area > 0)
    weights = area.where(valid)
    weight_sum = weights.sum(skipna=True)
    weight_sum_value = float(weight_sum.values)
    if not np.isfinite(weight_sum_value) or weight_sum_value <= 0.0:
        return float("nan")

    model_mean = (model.where(valid) * weights).sum(skipna=True) / weight_sum
    obs_mean = (observation.where(valid) * weights).sum(skipna=True) / weight_sum
    model_anomaly = model - model_mean
    obs_anomaly = observation - obs_mean
    covariance = (weights * model_anomaly * obs_anomaly).sum(skipna=True)
    model_variance = (weights * model_anomaly**2).sum(skipna=True)
    obs_variance = (weights * obs_anomaly**2).sum(skipna=True)
    denominator = np.sqrt(model_variance * obs_variance)
    denominator_value = float(denominator.values)
    if not np.isfinite(denominator_value) or denominator_value <= 0.0:
        return float("nan")
    return float((covariance / denominator).values)


def _time_mean_velocity_anomaly(u: xr.DataArray, v: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    return u - u.mean("time"), v - v.mean("time")


def _instantaneous_surface_eke_from_uv(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    u_anom, v_anom = _time_mean_velocity_anomaly(u, v)
    return 0.5 * (u_anom ** 2 + v_anom ** 2)


def _surface_eke_from_uv(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    return _instantaneous_surface_eke_from_uv(u, v).mean("time", skipna=True)


def _surface_eke_timeseries_from_uv(
    u: xr.DataArray,
    v: xr.DataArray,
    area: xr.DataArray,
    spatial_dims: tuple[str, str],
) -> xr.DataArray:
    eke = _instantaneous_surface_eke_from_uv(u, v)
    return _area_weighted_timeseries(eke, area, spatial_dims)


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


def _fully_observed_spectrum_block(
    field: xr.DataArray,
    lat_dim: str,
    lon_dim: str,
    min_size: int = 8,
) -> xr.DataArray:
    """Return a fully finite open-ocean box or reject the spatial FFT.

    Spatial FFTs assume a complete rectangular field. Filling land or persistent missing cells
    creates artificial boundary power, so the spectrum is not computed unless the curated box
    is at least ``min_size`` by ``min_size`` and finite at every time and grid point.
    """
    if lat_dim not in field.dims or lon_dim not in field.dims:
        raise ValueError(f"Spectrum field must contain {lat_dim!r} and {lon_dim!r} dimensions")
    n_lat = field.sizes[lat_dim]
    n_lon = field.sizes[lon_dim]
    if n_lat < min_size or n_lon < min_size:
        raise ValueError(
            f"Spatial-spectrum box has only {n_lat} x {n_lon} points "
            f"(minimum {min_size} x {min_size})."
        )
    if not bool(np.isfinite(field).all().compute().item()):
        raise ValueError(
            "Spatial spectrum requires a fully observed open-ocean rectangle; "
            "the selected box contains land or missing values."
        )
    return field


def _mean_isotropic_spectrum_over_time(
    field: xr.DataArray,
    dx: float,
    dy: float,
    batch_size: int = SPECTRUM_TIME_BATCH,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute isotropic spectra in time batches and average over time.

    Multi-year high-resolution fields are processed in bounded chunks rather than sent to
    torch as one (time, lat, lon) tensor.
    """
    ntime = int(field.sizes.get("time", 0))
    if ntime == 0:
        return np.array([]), np.array([])
    k_ref = None
    spec_sum = None
    spec_count = None
    for start in range(0, ntime, batch_size):
        chunk = field.isel(time=slice(start, min(start + batch_size, ntime)))
        values = np.asarray(chunk.values, dtype=np.float32)
        if values.size == 0:
            continue
        if not np.isfinite(values).all():
            raise ValueError("Spatial spectrum received non-finite values after open-ocean masking")
        k_cent, spec = compute_isotropic_spectrum_torch(
            torch.as_tensor(values),
            dx=dx,
            dy=dy,
            n_factor=2,
            detrend="linear",
            window="hann",
            cutoff_before_bins=False,
        )
        spec_np = np.asarray(spec.cpu().numpy(), dtype=float)
        if spec_np.ndim == 1:
            spec_np = spec_np[None, :]
        if k_ref is None:
            k_ref = k_cent.cpu().numpy() * 1000 * 2 * np.pi
            spec_sum = np.zeros(spec_np.shape[-1], dtype=float)
            spec_count = np.zeros(spec_np.shape[-1], dtype=float)
        finite = np.isfinite(spec_np)
        spec_sum += np.where(finite, spec_np, 0.0).sum(axis=0)
        spec_count += finite.sum(axis=0)
    if k_ref is None or spec_sum is None or spec_count is None:
        return np.array([]), np.array([])
    return k_ref, spec_sum / np.maximum(spec_count, 1.0)


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
    ssta = _calendar_day_climatology_anomaly(data)
    ssta = _fully_observed_spectrum_block(ssta, lat_dim, lon_dim)
    metric_selection = {lat_dim: ssta[lat_dim], lon_dim: ssta[lon_dim]}
    dx_mean = float(dx_2d.sel(metric_selection).mean().values)
    dy_mean = float(dy_2d.sel(metric_selection).mean().values)
    return _mean_isotropic_spectrum_over_time(ssta, dx_mean, dy_mean)


def _compute_isotropic_spectrum_from_map(
    field: xr.DataArray,
    dx_2d: xr.DataArray,
    dy_2d: xr.DataArray,
    lon_slice: slice,
    lat_slice: slice,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> tuple[np.ndarray, np.ndarray]:
    data = field.sel({lon_dim: lon_slice, lat_dim: lat_slice}).transpose(lat_dim, lon_dim)
    data = _fully_observed_spectrum_block(data, lat_dim, lon_dim)
    metric_selection = {lat_dim: data[lat_dim], lon_dim: data[lon_dim]}
    dx_mean = float(dx_2d.sel(metric_selection).mean().values)
    dy_mean = float(dy_2d.sel(metric_selection).mean().values)
    values = np.asarray(data.values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Spatial spectrum received non-finite map values after open-ocean masking")
    k_cent, spec = compute_isotropic_spectrum_torch(
        torch.as_tensor(values),
        dx=dx_mean,
        dy=dy_mean,
        n_factor=2,
        detrend="linear",
        window="hann",
        cutoff_before_bins=False,
    )
    return k_cent.cpu().numpy() * 1000 * 2 * np.pi, spec.cpu().numpy()


def _compute_temporal_spectrum_from_series(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    values = series.astype(float).dropna()
    if values.size < 4:
        return np.array([]), np.array([])
    index = pd.DatetimeIndex(values.index)
    delta_days = np.diff(index.values).astype("timedelta64[s]").astype(float) / 86400.0
    delta_days = delta_days[np.isfinite(delta_days) & (delta_days > 0)]
    if delta_days.size == 0:
        return np.array([]), np.array([])
    fs = 365.25 / float(np.median(delta_days))
    nperseg = min(256, values.size)
    freqs, psd = signal.welch(
        values.values,
        fs=fs,
        detrend="linear",
        window="hann",
        scaling="density",
        nperseg=nperseg,
    )
    valid = freqs > 0
    return freqs[valid], psd[valid]


def _weighted_region_mean_series(
    field: xr.DataArray,
    lat_dim: str,
    lon_dim: str,
    lat_chunk: int = 128,
    lon_chunk: int = 128,
) -> pd.Series:
    """Cos(latitude)-weighted regional mean time series computed in spatial blocks.

    Very large global grids are regularly sampled before the block loop so the
    temporal spectrum is practical on CPU interactive nodes. The Gulf Stream and
    Agulhas regional panels are small enough to keep at native resolution.
    """
    data = field.transpose("time", lat_dim, lon_dim)
    ntime = int(data.sizes["time"])
    weighted_sum = np.zeros(ntime, dtype=np.float64)
    weight_sum = np.zeros(ntime, dtype=np.float64)
    lat_values = np.asarray(data[lat_dim].values, dtype=float)

    for lat_start in range(0, data.sizes[lat_dim], lat_chunk):
        lat_stop = min(lat_start + lat_chunk, data.sizes[lat_dim])
        lat_weight = np.cos(np.deg2rad(lat_values[lat_start:lat_stop]))
        lat_weight = np.clip(lat_weight, 0.0, None)
        for lon_start in range(0, data.sizes[lon_dim], lon_chunk):
            lon_stop = min(lon_start + lon_chunk, data.sizes[lon_dim])
            block = data.isel(
                {
                    lat_dim: slice(lat_start, lat_stop),
                    lon_dim: slice(lon_start, lon_stop),
                }
            )
            values = np.asarray(block.values, dtype=np.float32)
            weights_2d = np.broadcast_to(lat_weight[:, None], values.shape[1:])
            finite = np.isfinite(values)
            weighted_sum += np.where(finite, values * weights_2d[None, :, :], 0.0).sum(axis=(1, 2))
            weight_sum += np.where(finite, weights_2d[None, :, :], 0.0).sum(axis=(1, 2))

    mean_values = np.full(ntime, np.nan, dtype=np.float64)
    valid = weight_sum > 0
    mean_values[valid] = weighted_sum[valid] / weight_sum[valid]
    return pd.Series(mean_values, index=pd.DatetimeIndex(data.time.values))


def _temporal_spectrum_spatial_strides(
    field: xr.DataArray,
    lat_dim: str,
    lon_dim: str,
    max_native_points: int = 100_000,
) -> tuple[int, int]:
    if field.sizes[lat_dim] * field.sizes[lon_dim] <= max_native_points:
        return 1, 1
    return (
        max(1, int(np.ceil(field.sizes[lat_dim] / 180))),
        max(1, int(np.ceil(field.sizes[lon_dim] / 360))),
    )


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
    lat_stride, lon_stride = _temporal_spectrum_spatial_strides(u_reg, lat_dim, lon_dim)
    if lat_stride > 1 or lon_stride > 1:
        stride_sel = {lat_dim: slice(None, None, lat_stride), lon_dim: slice(None, None, lon_stride)}
        u_reg = u_reg.isel(stride_sel)
        v_reg = v_reg.isel(stride_sel)
    ke = 0.5 * (u_reg ** 2 + v_reg ** 2)
    ke_series = _weighted_region_mean_series(ke, lat_dim, lon_dim)
    return _compute_temporal_spectrum_from_series(ke_series)


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


def _log10_rmse_between_spectra(
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    x_other: np.ndarray,
    y_other: np.ndarray,
) -> float:
    """Return spectral RMSE in log10-PSD space (dex) over the common positive domain.

    The comparison spectrum is interpolated onto the reference wavenumbers/frequencies in
    log-log space. Restricting both axes to finite, strictly positive values avoids undefined
    logarithms and gives multiplicative spectral errors comparable weight across PSD scales.
    """
    x_ref = np.asarray(x_ref, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    x_other = np.asarray(x_other, dtype=float)
    y_other = np.asarray(y_other, dtype=float)

    ref_valid = np.isfinite(x_ref) & np.isfinite(y_ref) & (x_ref > 0) & (y_ref > 0)
    other_valid = np.isfinite(x_other) & np.isfinite(y_other) & (x_other > 0) & (y_other > 0)
    if ref_valid.sum() == 0 or other_valid.sum() < 2:
        return np.nan

    order = np.argsort(x_other[other_valid])
    x_other_valid = x_other[other_valid][order]
    y_other_valid = y_other[other_valid][order]
    x_other_valid, unique_indices = np.unique(x_other_valid, return_index=True)
    y_other_valid = y_other_valid[unique_indices]
    if x_other_valid.size < 2:
        return np.nan

    domain = ref_valid & (x_ref >= x_other_valid[0]) & (x_ref <= x_other_valid[-1])
    if domain.sum() == 0:
        return np.nan

    log_y_other_on_ref = np.interp(
        np.log10(x_ref[domain]),
        np.log10(x_other_valid),
        np.log10(y_other_valid),
    )
    log_error = np.log10(y_ref[domain]) - log_y_other_on_ref
    log_error = log_error[np.isfinite(log_error)]
    if log_error.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(log_error**2)))


def _interannual_band(
    curves: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
    """Aggregate per-year spectra into a geometric mean and a +/- 1 log-std band.

    Each curve is (x, y). Curves are interpolated onto a common reference grid and the mean and
    spread are taken on log values, because spectra span orders of magnitude and the band must
    stay strictly positive on a log-log axis. Returns (x_ref, geom_mean, lower, upper,
    n_years_used); the array entries are None if no usable curve was provided.
    """
    cleaned = []
    for x, y in curves:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if mask.sum() >= 2:
            order = np.argsort(x[mask])
            cleaned.append((x[mask][order], y[mask][order]))
    if not cleaned:
        return None, None, None, None, 0
    x_ref = max((x for x, _ in cleaned), key=lambda arr: arr.size)
    log_ref = np.log(x_ref)
    log_rows = [np.interp(log_ref, np.log(x), np.log(y), left=np.nan, right=np.nan) for x, y in cleaned]
    log_rows = np.asarray(log_rows)
    with np.errstate(invalid="ignore"):
        log_mean = np.nanmean(log_rows, axis=0)
        log_std = np.nanstd(log_rows, axis=0, ddof=1) if log_rows.shape[0] > 1 else np.zeros_like(log_mean)
    return x_ref, np.exp(log_mean), np.exp(log_mean - log_std), np.exp(log_mean + log_std), log_rows.shape[0]


def _plot_spectra_interannual_panel(ax, x_ref, geom_mean, lower, upper, style: str, color, label: str) -> None:
    """Draw one interannual spectrum: a +/- 1 log-std shaded band plus the geometric-mean line."""
    ax.fill_between(x_ref, lower, upper, color=color, alpha=0.15, linewidth=0)
    ax.loglog(x_ref, geom_mean, style, color=color, linewidth=2, label=label)


def _interannual_spectrum_rmse_rows(
    model: str,
    region: str,
    metric: str,
    values_by_year: dict[int, float],
) -> list[dict]:
    """Return per-year log-spectrum RMSE plus interannual mean/std summary rows."""
    finite = {
        int(year): float(value)
        for year, value in values_by_year.items()
        if np.isfinite(value)
    }
    if not finite:
        return []
    rows = [
        {
            "model": model,
            "region": region,
            "metric": metric,
            "value": value,
            "period_kind": "annual_spectrum",
            "period_start": f"{year}-01-01",
            "period_end": f"{year}-12-31",
            "year": year,
            "is_primary": False,
        }
        for year, value in sorted(finite.items())
    ]
    values = np.asarray(list(finite.values()), dtype=float)
    summary_common = {
        "model": model,
        "region": region,
        "period_kind": "interannual_spectrum_summary",
        "period_start": f"{min(finite)}-01-01",
        "period_end": f"{max(finite)}-12-31",
        "year": np.nan,
        "n_years": int(values.size),
        "is_primary": False,
    }
    rows.append(
        {
            **summary_common,
            "metric": f"{metric}_interannual_mean",
            "value": float(values.mean()),
        }
    )
    rows.append(
        {
            **summary_common,
            "metric": f"{metric}_interannual_std",
            "value": float(values.std(ddof=1)) if values.size > 1 else np.nan,
        }
    )
    return rows


def _find_depth_name(ds: xr.Dataset | xr.DataArray) -> str:
    return _find_coord_name(ds, ["lev", "depth", "depth_0", "depth_std", "DEPTH", "z_t", "zlev"])


def _ohc_timeseries_from_obs_grid_field(
    field: xr.DataArray,
    area: xr.DataArray,
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
) -> xr.DataArray:
    """Global OHC (J) per depth layer from a (time, lat, lon, depth) ARGO temperature field
    (obs on its own grid, with exact overlap at the requested layer boundaries)."""
    out = {}
    for depth_slice in depth_slices:
        overlap_dz = _layer_overlap_thickness(
            field["depth"], depth_slice["min"], depth_slice["max"]
        )
        field_layer = field.where(overlap_dz > 0)
        ohc = (field_layer * overlap_dz * area * RHO * CP).sum(["depth", "lat", "lon"])
        out[depth_slice["title"]] = ohc
    series = xr.Dataset(out).to_array("depth_slice").transpose("time", "depth_slice")
    return series.resample(time="MS").mean()


def _obs_ohc_timeseries(
    argo_iap: xr.Dataset,
    start: pd.Timestamp,
    end: pd.Timestamp,
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
) -> xr.DataArray:
    if "area" not in argo_iap:
        argo_iap = _infer_obs_grid_metrics(argo_iap)
    obs_field = _reduce_obs_structure_field(
        argo_iap, ARGO_STRUCTURE_VARIABLES[0], slice(start, end), _identity_reducer
    )
    if obs_field.sizes.get("time", 0) == 0:
        raise ValueError(f"No ARGO-IAP OHC samples in requested period {start} to {end}")
    return _ohc_timeseries_from_obs_grid_field(obs_field, argo_iap["area"], depth_slices)


def _sim_ohc_timeseries(
    ds_sim: xr.Dataset,
    start: pd.Timestamp,
    end: pd.Timestamp,
    wet_mask: xr.DataArray | None = None,
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
) -> xr.DataArray:
    """Global OHC (J) per exact target depth interval on the model's native grid."""
    ds_year = ds_sim.sel(time=slice(start, end))
    if ds_year.sizes.get("time", 0) == 0:
        raise ValueError(f"No model OHC samples in requested period {start} to {end}")
    area = xr.DataArray(
        ds_year["areacello"].values,
        dims=["y", "x"],
        coords={"y": ds_year["y"], "x": ds_year["x"]},
    )
    field = _apply_native_wet_mask(ds_year["thetao"], wet_mask)
    out = {}
    for depth_slice in depth_slices:
        overlap_dz = _layer_overlap_thickness(
            ds_year["lev"],
            depth_slice["min"],
            depth_slice["max"],
            native_dz=ds_year["dz"],
        )
        thetao_layer = field.where(overlap_dz > 0)
        ohc = (thetao_layer * overlap_dz * area * RHO * CP).sum(["lev", "y", "x"])
        out[depth_slice["title"]] = ohc
    series = xr.Dataset(out).to_array("depth_slice").transpose("time", "depth_slice")
    return series.resample(time="MS").mean()


def _collect_ohc_timeseries(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    # Each product is reduced on its OWN native grid: a global integral/mean is grid-robust,
    # so obs (ARGO) and the models (native 1/4-degree) keep their own area, layer thickness,
    # and ocean mask. Native boundary cells are prorated so every product still integrates
    # the same exact physical intervals [0, 700) and [700, 2000). ARGO-IAP is a gap-filled
    # global product, so the horizontal domains nearly coincide.
    obs_ohc = _obs_ohc_timeseries(argo_iap, start, end)
    sim_ohc = {
        label: _sim_ohc_timeseries(emu["ds"], start, end, emu.get("wet_mask"))
        for label, emu in emulators.items()
    }
    return obs_ohc, sim_ohc


def _layer_weighted_mean_from_field(
    field: xr.DataArray,
    area: xr.DataArray,
    native_dz: xr.DataArray | None,
    depth_name: str,
    horizontal_dims: list[str],
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
) -> xr.DataArray:
    out = {}
    for depth_slice in depth_slices:
        overlap_dz = _layer_overlap_thickness(
            field[depth_name],
            depth_slice["min"],
            depth_slice["max"],
            native_dz=native_dz,
        )
        weights = area * overlap_dz
        mask = np.isfinite(field) & (overlap_dz > 0)
        field_layer = field.where(mask)
        weighted_sum = (field_layer * weights).sum([depth_name, *horizontal_dims], skipna=True)
        weight_sum = weights.where(mask).sum([depth_name, *horizontal_dims], skipna=True)
        out[depth_slice["title"]] = weighted_sum / weight_sum
    return xr.Dataset(out).to_array("depth_slice").transpose("time", "depth_slice")


def _layer_mean_timeseries_from_obs_grid_field(
    field: xr.DataArray,
    area: xr.DataArray,
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
) -> xr.DataArray:
    """Area- and thickness-weighted layer-mean timeseries from a (time, lat, lon, depth) ARGO
    field (obs on its own grid)."""
    out = _layer_weighted_mean_from_field(field, area, None, "depth", ["lat", "lon"], depth_slices)
    return out.resample(time="MS").mean()


def _obs_layer_mean_timeseries(
    argo_iap: xr.Dataset,
    spec: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
) -> xr.DataArray:
    if "area" not in argo_iap:
        argo_iap = _infer_obs_grid_metrics(argo_iap)
    obs_field = _reduce_obs_structure_field(argo_iap, spec, slice(start, end), _identity_reducer)
    if obs_field.sizes.get("time", 0) == 0:
        raise ValueError(f"No ARGO-IAP samples in requested period {start} to {end}")
    return _layer_mean_timeseries_from_obs_grid_field(obs_field, argo_iap["area"], depth_slices)


def _sim_layer_mean_timeseries(
    ds_sim: xr.Dataset,
    spec: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
    wet_mask: xr.DataArray | None = None,
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
) -> xr.DataArray:
    """Area- and thickness-weighted layer-mean timeseries on the model's own native grid."""
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
    out = _layer_weighted_mean_from_field(field, area, ds_year["dz"], "lev", ["y", "x"], depth_slices)
    return out.resample(time="MS").mean()


def _collect_layer_mean_timeseries(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    spec: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    # Each product is reduced on its OWN native grid (see _collect_ohc_timeseries): a global
    # area-weighted mean is grid-robust, so obs (ARGO) and the models (native 1/4-degree) keep
    # their own area, layer thickness, and ocean mask.
    obs_series = _obs_layer_mean_timeseries(argo_iap, spec, start, end)
    sim_series = {
        label: _sim_layer_mean_timeseries(emu["ds"], spec, start, end, emu.get("wet_mask"))
        for label, emu in emulators.items()
    }
    return obs_series, sim_series


def _time_mean_reducer(da: xr.DataArray) -> xr.DataArray:
    return da.mean("time", skipna=True)


def _time_var_reducer(da: xr.DataArray) -> xr.DataArray:
    return da.var("time", skipna=True)


def _identity_reducer(da: xr.DataArray) -> xr.DataArray:
    return da


def _reduce_obs_structure_field(
    ds_obs: xr.Dataset,
    spec: dict,
    time_slice: slice,
    reducer,
) -> xr.DataArray:
    obs_var = _find_var_name(ds_obs, spec["obs_aliases"])
    da = ds_obs[obs_var].sel(time=time_slice)
    depth_name = _find_depth_name(da)
    rename = {}
    if depth_name != "depth":
        rename[depth_name] = "depth"
    if rename:
        da = da.rename(rename)
    da = da.transpose("time", "lat", "lon", "depth")
    return reducer(da).sortby("lat").sortby("lon").sortby("depth")


def _annual_mean_obs_structure_field(ds_obs: xr.Dataset, spec: dict, year: int) -> xr.DataArray:
    return _reduce_obs_structure_field(
        ds_obs, spec, slice(f"{year}-01-01", f"{year}-12-31"), _time_mean_reducer
    )


def _reduce_sim_on_obs_grid(
    ds_sim: xr.Dataset,
    spec: dict,
    obs_field: xr.DataArray,
    time_slice: slice,
    reducer,
    wet_mask: xr.DataArray | None = None,
) -> xr.DataArray:
    model_var = spec["model_var"]
    if model_var not in ds_sim:
        raise KeyError(f"Model dataset does not contain required variable {model_var!r}")

    sim = reducer(ds_sim[model_var].sel(time=time_slice))
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
    sim = sim.transpose(..., "lat", "lon", "depth")
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


def _annual_mean_sim_on_obs_grid(
    ds_sim: xr.Dataset,
    spec: dict,
    obs_field: xr.DataArray,
    year: int,
    wet_mask: xr.DataArray | None = None,
) -> xr.DataArray:
    return _reduce_sim_on_obs_grid(
        ds_sim,
        spec,
        obs_field,
        slice(f"{year}-01-01", f"{year}-12-31"),
        _time_mean_reducer,
        wet_mask,
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


def _surface_model_field_on_obs_grid(
    ds_sim: xr.Dataset,
    model_var: str,
    obs_field: xr.DataArray,
    wet_mask: xr.DataArray | None = None,
) -> xr.DataArray:
    if model_var not in ds_sim:
        raise KeyError(f"Model dataset does not contain required variable {model_var!r}")
    sim = ds_sim[model_var]
    if "lev" in sim.dims:
        sim = sim.isel(lev=0)
    sim = _apply_native_wet_mask(sim, wet_mask)
    rename = {}
    if "y" in sim.dims:
        rename["y"] = "lat"
    if "x" in sim.dims:
        rename["x"] = "lon"
    if rename:
        sim = sim.rename(rename)
    sim = _normalize_lon(sim, "lon").sortby("lat").sortby("lon")

    if "time" in obs_field.dims or "time" in sim.dims:
        _require_exact_time_match(sim, obs_field, f"{model_var} surface-field comparison")
    # Preserve the validated model samples exactly; interpolation is horizontal only.
    return sim.interp(
        lat=obs_field["lat"],
        lon=obs_field["lon"],
        kwargs={"fill_value": np.nan},
    )


def _sim_geostrophic_velocity_on_obs_grid(
    ds_sim: xr.Dataset,
    obs_u: xr.DataArray,
    obs_v: xr.DataArray,
    wet_mask: xr.DataArray | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    sim_zos = _apply_native_wet_mask(ds_sim["zos"], wet_mask)
    sim_u, sim_v = _geostrophic_velocity_from_zos(sim_zos, lat_dim="y", lon_dim="x")
    sim_u = _apply_native_wet_mask(sim_u, wet_mask).rename({"y": "lat", "x": "lon"})
    sim_v = _apply_native_wet_mask(sim_v, wet_mask).rename({"y": "lat", "x": "lon"})
    sim_u = _normalize_lon(sim_u, "lon").sortby("lat").sortby("lon")
    sim_v = _normalize_lon(sim_v, "lon").sortby("lat").sortby("lon")
    _require_exact_time_match(sim_u, obs_u, "surface geostrophic eastward velocity comparison")
    _require_exact_time_match(sim_v, obs_v, "surface geostrophic northward velocity comparison")
    # Preserve the validated model samples exactly; interpolation is horizontal only.
    spatial_target = {"lat": obs_u["lat"], "lon": obs_u["lon"]}
    sim_u = sim_u.interp(**spatial_target, kwargs={"fill_value": np.nan}).where(np.isfinite(obs_u))
    sim_v = sim_v.interp(**spatial_target, kwargs={"fill_value": np.nan}).where(np.isfinite(obs_v))
    return sim_u, sim_v


def _global_sst_timeseries_from_field(
    sst: xr.DataArray,
    area: xr.DataArray,
    spatial_dims: tuple[str, str],
    resample_monthly: bool = False,
) -> pd.Series:
    series = _area_weighted_timeseries(sst, area, spatial_dims)
    if resample_monthly:
        series = _resample_monthly_field(series)
    return series.to_series()


def _collect_sst_timeseries(
    oisst: xr.Dataset,
    emulators: dict[str, dict],
) -> tuple[pd.Series, dict[str, pd.Series], str]:
    sst_name = _find_var_name(oisst, ["sst", "analysed_sst", "thetao", "tos"])
    window_start, window_end = _common_time_window(oisst, *[emu["ds"] for emu in emulators.values()])
    oisst_w = oisst.sel(time=slice(window_start, window_end))
    obs_series = _global_sst_timeseries_from_field(oisst_w[sst_name], oisst_w["area"], ("lat", "lon"))
    sim_series = {}
    for label, emu in emulators.items():
        sim = emu["ds"].sel(time=slice(window_start, window_end))
        sim_sst = _apply_native_wet_mask(sim["thetao"].isel(lev=0), emu.get("wet_mask"))
        sim_series[label] = _global_sst_timeseries_from_field(sim_sst, sim["areacello"], ("y", "x"))
    return obs_series, sim_series, _period_label(window_start, window_end)


def _collect_eke_timeseries(
    duacs: xr.Dataset,
    emulators: dict[str, dict],
    velocity_kind: str = "absolute",
) -> tuple[pd.Series, dict[str, pd.Series], str]:
    u_name, v_name = _duacs_velocity_names(duacs, velocity_kind)
    window_start, window_end = _common_time_window(duacs, *[emu["ds"] for emu in emulators.values()])
    duacs_w = duacs.sel(time=slice(window_start, window_end))
    obs_u = _apply_equatorial_mask(duacs_w[u_name], lat_dim="lat")
    obs_v = _apply_equatorial_mask(duacs_w[v_name], lat_dim="lat")
    obs_series = _surface_eke_timeseries_from_uv(obs_u, obs_v, duacs_w["area"], ("lat", "lon")).to_series()

    sim_series = {}
    for label, emu in emulators.items():
        sim = emu["ds"].sel(time=slice(window_start, window_end))
        sim_zos = _apply_native_wet_mask(sim["zos"], emu.get("wet_mask"))
        sim_u, sim_v = _geostrophic_velocity_from_zos(sim_zos, lat_dim="y", lon_dim="x")
        sim_u = _apply_equatorial_mask(_apply_native_wet_mask(sim_u, emu.get("wet_mask")), lat_dim="y")
        sim_v = _apply_equatorial_mask(_apply_native_wet_mask(sim_v, emu.get("wet_mask")), lat_dim="y")
        sim_series[label] = _surface_eke_timeseries_from_uv(sim_u, sim_v, sim["areacello"], ("y", "x")).to_series()
    return obs_series, sim_series, _period_label(window_start, window_end)


def _common_temporal_fields_on_argo_grid(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    spec: dict,
    time_slice: slice,
    reducer,
) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    obs_field = _reduce_obs_structure_field(argo_iap, spec, time_slice, reducer)
    sim_fields = {
        label: _reduce_sim_on_obs_grid(emu["ds"], spec, obs_field, time_slice, reducer, emu.get("wet_mask"))
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
    obs_valid_mask = np.isfinite(obs_common)
    if "time" in obs_valid_mask.dims:
        # ARGO structure fields are monthly, while model rollouts are commonly 5-day.
        # Use a spatial/depth ocean-valid mask here; exact time alignment is handled
        # by each metric after temporal aggregation or interpolation.
        obs_valid_mask = obs_valid_mask.any("time")
    sim_common = {}
    for label, sim_field in sim_fields.items():
        sim_interp = sim_field.interp(depth=common_depth, kwargs={"fill_value": np.nan})
        sim_common[label] = sim_interp.where(obs_valid_mask)
    return obs_common, sim_common


def _common_annual_fields_on_argo_grid(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    spec: dict,
    year: int,
) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    return _common_temporal_fields_on_argo_grid(
        argo_iap, emulators, spec, slice(f"{year}-01-01", f"{year}-12-31"), _time_mean_reducer
    )


def _depth_weighted_layer_maps_from_field(
    field: xr.DataArray,
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
) -> dict[str, xr.DataArray]:
    out = {}
    for depth_slice in depth_slices:
        overlap_dz = _layer_overlap_thickness(
            field["depth"], depth_slice["min"], depth_slice["max"]
        )
        mask = np.isfinite(field) & (overlap_dz > 0)
        weighted_sum = (field.where(mask) * overlap_dz).sum("depth", skipna=True)
        weight_sum = overlap_dz.where(mask).sum("depth", skipna=True)
        out[depth_slice["title"]] = weighted_sum / weight_sum
    return out


def _ohc_per_area_layer_maps_from_temperature_field(
    temp_field: xr.DataArray,
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
    native_dz: xr.DataArray | None = None,
    depth_name: str = "depth",
) -> dict[str, xr.DataArray]:
    out = {}
    for depth_slice in depth_slices:
        overlap_dz = _layer_overlap_thickness(
            temp_field[depth_name],
            depth_slice["min"],
            depth_slice["max"],
            native_dz=native_dz,
        )
        temp_layer = temp_field.where(overlap_dz > 0)
        valid = np.isfinite(temp_layer).any(depth_name)
        ohc = (temp_layer * overlap_dz * RHO * CP).sum(depth_name, skipna=True).where(valid)
        out[depth_slice["title"]] = ohc
    return out


def _model_field_on_obs_horizontal_grid(
    model_field: xr.DataArray,
    obs_field: xr.DataArray,
) -> xr.DataArray:
    """Interpolate a vertically reduced model field to an observation horizontal grid."""
    rename = {}
    if "y" in model_field.dims:
        rename["y"] = "lat"
    if "x" in model_field.dims:
        rename["x"] = "lon"
    if rename:
        model_field = model_field.rename(rename)
    model_field = _normalize_lon(model_field, "lon").sortby("lat").sortby("lon")
    interpolated = model_field.interp(
        lat=obs_field["lat"],
        lon=obs_field["lon"],
        kwargs={"fill_value": np.nan},
    )
    return interpolated.where(np.isfinite(obs_field))


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


def _plot_layer_bias_map_comparison(
    obs_maps: dict[str, xr.DataArray],
    sim_maps: dict[str, dict[str, xr.DataArray]],
    obs_area: xr.DataArray,
    emulators: dict[str, dict],
    title: str,
    units: str,
    metric_prefix: str,
    year: int,
) -> None:
    """Point-wise bias maps (model - ARGO-IAP) with area-weighted RMSE annotated.

    Mirrors the Samudra 2 paper bias panels (functions.plot_temperature_bias_comparison_together):
    rows = emulators, cols = depth layers, diverging colormap, per-layer symmetric limits,
    RMSE printed above each panel. RMSE values are already emitted to the metrics CSV by
    _plot_layer_map_comparison, so this figure only visualizes them and returns no metrics.
    """
    layer_titles = [depth_slice["title"] for depth_slice in OHC_DEPTH_SLICES]
    nrows = len(sim_maps)
    ncols = len(layer_titles)
    scientific = metric_prefix.startswith("annual_mean_ohc")

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.6 * ncols, 3.5 * nrows),
        squeeze=False,
        subplot_kw={"projection": ccrs.Robinson(central_longitude=210)},
        gridspec_kw={"hspace": 0.28, "wspace": 0.05},
    )

    limits = {}
    for layer_title in layer_titles:
        chunks = []
        for model_maps in sim_maps.values():
            bias = (model_maps[layer_title] - obs_maps[layer_title]).values
            chunks.append(bias[np.isfinite(bias)])
        flat = np.concatenate(chunks) if chunks else np.array([1.0])
        limit = float(np.nanpercentile(np.abs(flat), 99)) if flat.size else 1.0
        if not np.isfinite(limit) or limit == 0.0:
            limit = 1.0
        limits[layer_title] = limit

    for col_idx, layer_title in enumerate(layer_titles):
        limit = limits[layer_title]
        norm = colors.Normalize(vmin=-limit, vmax=limit)
        mesh = None
        for row_idx, (label, model_maps) in enumerate(sim_maps.items()):
            ax = axes[row_idx, col_idx]
            bias = model_maps[layer_title] - obs_maps[layer_title]
            mesh = bias.plot.pcolormesh(
                ax=ax,
                transform=ccrs.PlateCarree(),
                x="lon",
                y="lat",
                cmap=cmocean.cm.balance,
                norm=norm,
                add_colorbar=False,
                rasterized=True,
                add_labels=False,
            )
            _add_map_features(ax)
            if row_idx == 0:
                ax.set_title(layer_title, fontweight="bold", y=1.12)
            rmse = _area_weighted_map_rmse(bias, obs_area)
            ax.text(0.5, 1.02, f"RMSE: {rmse:.2e} {units}", transform=ax.transAxes,
                    fontsize=12, ha="center", va="bottom")
            if col_idx == 0:
                ax.text(-0.07, 0.55, label, va="center", ha="right", rotation=90,
                        transform=ax.transAxes, fontweight="bold")

        cbar = fig.colorbar(mesh, ax=axes[:, col_idx], orientation="horizontal",
                            shrink=0.7, pad=0.03, extend="both", label=f"Bias ({units})")
        if scientific:
            cbar.ax.ticklabel_format(style="scientific", scilimits=(0, 0), useMathText=True)

    fig.suptitle(f"Annual-Mean {title} Bias vs ARGO-IAP (model − obs, {year})", y=0.99, fontsize=15)


def _plot_rmse_map_comparison(
    rmse_maps: dict[str, xr.DataArray],
    obs_area: xr.DataArray,
    title: str,
    units: str,
    metric_prefix: str,
    period_label: str,
    total_values: dict[str, float] | None = None,
) -> pd.DataFrame:
    ncols = len(rmse_maps)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(5.8 * ncols, 4.8),
        squeeze=False,
        subplot_kw={"projection": ccrs.Robinson(central_longitude=210)},
    )
    axes = axes[0]

    flat = np.concatenate(
        [field.values[np.isfinite(field.values)] for field in rmse_maps.values()]
    ) if rmse_maps else np.array([])
    vmax = float(np.nanpercentile(flat, 98)) if flat.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    norm = colors.Normalize(vmin=0.0, vmax=vmax)

    rows = []
    mesh = None
    for ax, (label, rmse_map) in zip(axes, rmse_maps.items()):
        total_rmse = (
            float(total_values[label])
            if total_values is not None and label in total_values
            else _area_weighted_map_rmse(rmse_map, obs_area)
        )
        rows.append({"model": label, "metric": f"{metric_prefix}_total_rmse", "value": total_rmse})
        mesh = rmse_map.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            x="lon",
            y="lat",
            cmap=cmocean.cm.thermal,
            norm=norm,
            add_colorbar=False,
            rasterized=True,
            add_labels=False,
        )
        _add_map_features(ax)
        ax.set_title(label, fontsize=12, fontweight="bold", y=1.14)
        ax.text(
            0.5,
            1.035,
            f"Total RMSE={total_rmse:.3g} {units}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10.5,
        )

    fig.subplots_adjust(left=0.04, right=0.98, top=0.78, bottom=0.30, wspace=0.08)
    cax = fig.add_axes([0.27, 0.14, 0.46, 0.035])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="max")
    cbar.set_label(f"RMSE ({units})", fontsize=12)
    fig.suptitle(f"{title} Spatial RMSE vs Observations ({period_label})", y=0.97, fontsize=15)
    return pd.DataFrame(rows)


def _plot_layer_rmse_map_comparison(
    rmse_maps: dict[str, dict[str, xr.DataArray]],
    obs_area: xr.DataArray,
    title: str,
    units: str,
    metric_prefix: str,
    period_label: str,
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
    total_values: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    layer_titles = [depth_slice["title"] for depth_slice in depth_slices]
    nrows = len(layer_titles)
    ncols = len(rmse_maps)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.8 * ncols, 4.2 * nrows),
        squeeze=False,
        subplot_kw={"projection": ccrs.Robinson(central_longitude=210)},
        gridspec_kw={"hspace": 0.55, "wspace": 0.08},
    )

    rows = []
    row_meshes = []
    for row_idx, layer_title in enumerate(layer_titles):
        fields = [model_maps[layer_title] for model_maps in rmse_maps.values()]
        flat = np.concatenate([field.values[np.isfinite(field.values)] for field in fields])
        vmax = float(np.nanpercentile(flat, 98)) if flat.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        norm = colors.Normalize(vmin=0.0, vmax=vmax)
        mesh = None
        for col_idx, (label, model_maps) in enumerate(rmse_maps.items()):
            ax = axes[row_idx, col_idx]
            rmse_map = model_maps[layer_title]
            total_rmse = (
                float(total_values[label][layer_title])
                if total_values is not None
                and label in total_values
                and layer_title in total_values[label]
                else _area_weighted_map_rmse(rmse_map, obs_area)
            )
            rows.append(
                {
                    "model": label,
                    "depth": layer_title,
                    "metric": f"{metric_prefix}_total_rmse",
                    "value": total_rmse,
                }
            )
            mesh = rmse_map.plot.pcolormesh(
                ax=ax,
                transform=ccrs.PlateCarree(),
                x="lon",
                y="lat",
                cmap=cmocean.cm.thermal,
                norm=norm,
                add_colorbar=False,
                rasterized=True,
                add_labels=False,
            )
            _add_map_features(ax)
            if row_idx == 0:
                ax.set_title(label, fontweight="bold", y=1.18)
            ax.text(
                0.5,
                1.045,
                f"Total RMSE={total_rmse:.2e} {units}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=9.8,
            )
            if col_idx == 0:
                ax.text(
                    -0.08,
                    0.5,
                    layer_title,
                    ha="right",
                    va="center",
                    rotation=90,
                    transform=ax.transAxes,
                    fontsize=11,
                )

        row_meshes.append(mesh)

    fig.suptitle(f"{title} Spatial RMSE vs Observations ({period_label})", y=0.98, fontsize=15)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.12, wspace=0.08, hspace=0.78)
    for row_idx, mesh in enumerate(row_meshes):
        row_axes = axes[row_idx, :]
        positions = [ax.get_position() for ax in row_axes]
        x0 = min(pos.x0 for pos in positions)
        x1 = max(pos.x1 for pos in positions)
        y0 = min(pos.y0 for pos in positions)
        width = x1 - x0
        cax = fig.add_axes([x0 + 0.18 * width, max(0.02, y0 - 0.055), 0.64 * width, 0.026])
        cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="max")
        cbar.set_label(f"RMSE ({units})", fontsize=11)
    return pd.DataFrame(rows)


def _argo_variable_layer_maps_data(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    year: int,
    spec: dict,
) -> tuple[dict[str, xr.DataArray], dict[str, dict[str, xr.DataArray]], xr.DataArray]:
    if "area" not in argo_iap:
        argo_iap = _infer_obs_grid_metrics(argo_iap)
    obs_field, sim_fields = _common_annual_fields_on_argo_grid(argo_iap, emulators, spec, year)
    obs_maps = _depth_weighted_layer_maps_from_field(obs_field)
    sim_maps = {
        label: _depth_weighted_layer_maps_from_field(sim_field)
        for label, sim_field in sim_fields.items()
    }
    return obs_maps, sim_maps, argo_iap["area"]


def _argo_ohc_layer_maps_data(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    year: int,
) -> tuple[dict[str, xr.DataArray], dict[str, dict[str, xr.DataArray]], xr.DataArray]:
    temp_spec = ARGO_STRUCTURE_VARIABLES[0]
    if "area" not in argo_iap:
        argo_iap = _infer_obs_grid_metrics(argo_iap)
    time_slice = slice(f"{year}-01-01", f"{year}-12-31")
    obs_temp = _reduce_obs_structure_field(argo_iap, temp_spec, time_slice, _time_mean_reducer)
    obs_maps = _ohc_per_area_layer_maps_from_temperature_field(obs_temp)

    # Integrate vertically on each product's native grid before horizontal interpolation.
    # This preserves the exact target boundaries instead of inheriting ARGO center selection.
    sim_maps = {}
    for label, emu in emulators.items():
        ds_sim = emu["ds"]
        sim_temp = ds_sim["thetao"].sel(time=time_slice).mean("time", skipna=True)
        sim_temp = _apply_native_wet_mask(sim_temp, emu.get("wet_mask"))
        native_maps = _ohc_per_area_layer_maps_from_temperature_field(
            sim_temp,
            native_dz=ds_sim["dz"],
            depth_name="lev",
        )
        sim_maps[label] = {
            title: _model_field_on_obs_horizontal_grid(field, obs_maps[title])
            for title, field in native_maps.items()
        }
    return obs_maps, sim_maps, argo_iap["area"]


def _plot_argo_variable_layer_maps(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    year: int,
    spec: dict,
) -> pd.DataFrame:
    obs_maps, sim_maps, obs_area = _argo_variable_layer_maps_data(argo_iap, emulators, year, spec)
    return _plot_layer_map_comparison(
        obs_maps,
        sim_maps,
        obs_area,
        emulators,
        f"Layer-Mean {spec['title']}",
        spec["units"],
        spec["metric_prefix"],
        _argo_variable_cmap(spec),
        year,
    )


def _plot_argo_variable_layer_bias_maps(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    year: int,
    spec: dict,
) -> None:
    obs_maps, sim_maps, obs_area = _argo_variable_layer_maps_data(argo_iap, emulators, year, spec)
    _plot_layer_bias_map_comparison(
        obs_maps,
        sim_maps,
        obs_area,
        emulators,
        f"Layer-Mean {spec['title']}",
        spec["units"],
        spec["metric_prefix"],
        year,
    )


def _plot_argo_ohc_layer_maps(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    year: int,
) -> pd.DataFrame:
    obs_maps, sim_maps, obs_area = _argo_ohc_layer_maps_data(argo_iap, emulators, year)
    return _plot_layer_map_comparison(
        obs_maps,
        sim_maps,
        obs_area,
        emulators,
        "Layer-Integrated OHC per Unit Area",
        "J m$^{-2}$",
        "annual_mean_ohc_per_area",
        cmocean.cm.thermal,
        year,
    )


def _plot_argo_ohc_layer_bias_maps(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    year: int,
) -> None:
    obs_maps, sim_maps, obs_area = _argo_ohc_layer_maps_data(argo_iap, emulators, year)
    _plot_layer_bias_map_comparison(
        obs_maps,
        sim_maps,
        obs_area,
        emulators,
        "Layer-Integrated OHC per Unit Area",
        "J m$^{-2}$",
        "annual_mean_ohc_per_area",
        year,
    )


def _plot_surface_velocity_rmse_maps(
    duacs: xr.Dataset,
    emulators: dict[str, dict],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    period_kind: str = "primary_complete_years",
    include_annual: bool = False,
    velocity_kind: str = "absolute",
) -> pd.DataFrame:
    u_name, v_name = _duacs_velocity_names(duacs, velocity_kind)
    window_start, window_end = _bounded_common_time_window(
        duacs,
        *[emu["ds"] for emu in emulators.values()],
        start=start,
        end=end,
    )
    period_label = _period_label(window_start, window_end)
    duacs_w = duacs.sel(time=slice(window_start, window_end))
    obs_u = _apply_equatorial_mask(duacs_w[u_name], lat_dim="lat")
    obs_v = _apply_equatorial_mask(duacs_w[v_name], lat_dim="lat")

    rmse_maps = {}
    annual_frames = []
    summaries = {}
    total_values = {}
    metric = "surface_geostrophic_velocity_vector_total_rmse"
    for label, emu in emulators.items():
        sim_w = emu["ds"].sel(time=slice(window_start, window_end))
        sim_u, sim_v = _sim_geostrophic_velocity_on_obs_grid(sim_w, obs_u, obs_v, emu.get("wet_mask"))
        if velocity_kind == "anomaly":
            sim_u, sim_v = _time_mean_velocity_anomaly(sim_u, sim_v)
        du = sim_u - obs_u
        dv = sim_v - obs_v
        rmse_map, annual_rows, summary = _rmse_map_and_annual_rows(
            du ** 2 + dv ** 2,
            duacs_w["area"],
            ("lat", "lon"),
            label,
            metric,
            include_annual=include_annual,
        )
        rmse_maps[label] = rmse_map
        if not annual_rows.empty:
            annual_frames.append(annual_rows)
        summaries[(label, None)] = summary
        if include_annual:
            total_values[label] = float(summary["block_aggregate_rmse"])

    rows = _plot_rmse_map_comparison(
        rmse_maps,
        duacs_w["area"],
        "Surface Geostrophic Velocity Vector",
        "m s$^{-1}$",
        "surface_geostrophic_velocity_vector",
        period_label,
        total_values=total_values or None,
    )
    rows = _annotate_rmse_period(
        rows,
        pd.Timestamp(duacs_w.time.values[0]),
        pd.Timestamp(duacs_w.time.values[-1]),
        period_kind,
        duacs_w.sizes["time"],
        summaries,
    )
    return pd.concat([rows, *annual_frames], ignore_index=True, sort=False)


def _plot_eke_rmse_maps(
    duacs: xr.Dataset,
    emulators: dict[str, dict],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    period_kind: str = "primary_complete_years",
    include_annual: bool = False,
    velocity_kind: str = "absolute",
) -> pd.DataFrame:
    u_name, v_name = _duacs_velocity_names(duacs, velocity_kind)
    window_start, window_end = _bounded_common_time_window(
        duacs,
        *[emu["ds"] for emu in emulators.values()],
        start=start,
        end=end,
    )
    period_label = _period_label(window_start, window_end)
    duacs_w = duacs.sel(time=slice(window_start, window_end))
    obs_u = _apply_equatorial_mask(duacs_w[u_name], lat_dim="lat")
    obs_v = _apply_equatorial_mask(duacs_w[v_name], lat_dim="lat")
    obs_eke = _instantaneous_surface_eke_from_uv(obs_u, obs_v)

    rmse_maps = {}
    annual_frames = []
    summaries = {}
    total_values = {}
    metric = "surface_eke_total_rmse"
    for label, emu in emulators.items():
        sim_w = emu["ds"].sel(time=slice(window_start, window_end))
        sim_u, sim_v = _sim_geostrophic_velocity_on_obs_grid(sim_w, obs_u, obs_v, emu.get("wet_mask"))
        sim_eke = _instantaneous_surface_eke_from_uv(sim_u, sim_v)
        error_squared = (sim_eke - obs_eke) ** 2
        rmse_map, annual_rows, summary = _rmse_map_and_annual_rows(
            error_squared,
            duacs_w["area"],
            ("lat", "lon"),
            label,
            metric,
            include_annual=include_annual,
        )
        rmse_maps[label] = rmse_map
        if not annual_rows.empty:
            annual_frames.append(annual_rows)
        summaries[(label, None)] = summary
        if include_annual:
            total_values[label] = float(summary["block_aggregate_rmse"])

    rows = _plot_rmse_map_comparison(
        rmse_maps,
        duacs_w["area"],
        "Surface EKE",
        "m$^2$ s$^{-2}$",
        "surface_eke",
        period_label,
        total_values=total_values or None,
    )
    rows = _annotate_rmse_period(
        rows,
        pd.Timestamp(duacs_w.time.values[0]),
        pd.Timestamp(duacs_w.time.values[-1]),
        period_kind,
        duacs_w.sizes["time"],
        summaries,
    )
    return pd.concat([rows, *annual_frames], ignore_index=True, sort=False)


def _plot_sst_rmse_maps(
    oisst: xr.Dataset,
    emulators: dict[str, dict],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    period_kind: str = "primary_complete_years",
    include_annual: bool = False,
) -> pd.DataFrame:
    sst_name = _find_var_name(oisst, ["sst", "analysed_sst", "thetao", "tos"])
    window_start, window_end = _bounded_common_time_window(
        oisst,
        *[emu["ds"] for emu in emulators.values()],
        start=start,
        end=end,
    )
    period_label = _period_label(window_start, window_end)
    obs_sst = oisst[sst_name].sel(time=slice(window_start, window_end))

    rmse_maps = {}
    annual_frames = []
    summaries = {}
    total_values = {}
    metric = "surface_sst_total_rmse"
    for label, emu in emulators.items():
        sim_w = emu["ds"].sel(time=slice(window_start, window_end))
        sim_sst = _surface_model_field_on_obs_grid(sim_w, "thetao", obs_sst, emu.get("wet_mask"))
        rmse_map, annual_rows, summary = _rmse_map_and_annual_rows(
            (sim_sst - obs_sst) ** 2,
            oisst["area"],
            ("lat", "lon"),
            label,
            metric,
            include_annual=include_annual,
        )
        rmse_maps[label] = rmse_map
        if not annual_rows.empty:
            annual_frames.append(annual_rows)
        summaries[(label, None)] = summary
        if include_annual:
            total_values[label] = float(summary["block_aggregate_rmse"])

    rows = _plot_rmse_map_comparison(
        rmse_maps,
        oisst["area"],
        "SST",
        "$^\\circ$C",
        "surface_sst",
        period_label,
        total_values=total_values or None,
    )
    rows = _annotate_rmse_period(
        rows,
        pd.Timestamp(obs_sst.time.values[0]),
        pd.Timestamp(obs_sst.time.values[-1]),
        period_kind,
        obs_sst.sizes["time"],
        summaries,
    )
    return pd.concat([rows, *annual_frames], ignore_index=True, sort=False)


def _plot_ohc_rmse_maps(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    period_kind: str = "primary_complete_years",
    include_annual: bool = False,
    depth_slices: list[dict] = OHC_DEPTH_SLICES,
) -> pd.DataFrame:
    temp_spec = ARGO_STRUCTURE_VARIABLES[0]
    if "area" not in argo_iap:
        argo_iap = _infer_obs_grid_metrics(argo_iap)
    window_start, window_end = _bounded_common_month_window(
        argo_iap,
        *[emu["ds"] for emu in emulators.values()],
        start=start,
        end=end,
    )
    period_label = _period_label(window_start, window_end)
    time_slice = slice(window_start, window_end)
    obs_temp = _reduce_obs_structure_field(
        argo_iap, temp_spec, time_slice, _identity_reducer
    )
    obs_maps = {
        title: _resample_monthly_field(field, f"observed OHC {title}")
        for title, field in _ohc_per_area_layer_maps_from_temperature_field(obs_temp, depth_slices).items()
    }

    # Model OHC is integrated with native layer overlaps, then put on the ARGO horizontal grid.
    rmse_maps: dict[str, dict[str, xr.DataArray]] = {}
    annual_frames = []
    summaries = {}
    total_values: dict[str, dict[str, float]] = {}
    metric = "ohc_per_area_total_rmse"
    aligned_time: xr.DataArray | None = None
    for label, emu in emulators.items():
        ds_sim = emu["ds"]
        sim_temp = ds_sim["thetao"].sel(time=time_slice)
        sim_temp = _apply_native_wet_mask(sim_temp, emu.get("wet_mask"))
        sim_native_maps = _ohc_per_area_layer_maps_from_temperature_field(
            sim_temp,
            depth_slices,
            native_dz=ds_sim["dz"],
            depth_name="lev",
        )
        sim_maps = {
            title: _model_field_on_obs_horizontal_grid(
                _resample_monthly_field(field, f"{label} OHC {title}"),
                obs_maps[title],
            )
            for title, field in sim_native_maps.items()
        }
        rmse_maps[label] = {}
        total_values[label] = {}
        for title in obs_maps:
            sim_aligned, obs_aligned = xr.align(sim_maps[title], obs_maps[title], join="inner")
            if sim_aligned.sizes.get("time", 0) == 0:
                raise ValueError(f"No common monthly OHC samples for {label} {title} in {period_label}")
            rmse_map, annual_rows, summary = _rmse_map_and_annual_rows(
                (sim_aligned - obs_aligned) ** 2,
                argo_iap["area"],
                ("lat", "lon"),
                label,
                metric,
                depth=title,
                include_annual=include_annual,
            )
            rmse_maps[label][title] = rmse_map
            if not annual_rows.empty:
                annual_frames.append(annual_rows)
            summaries[(label, title)] = summary
            if include_annual:
                total_values[label][title] = float(summary["block_aggregate_rmse"])
            aligned_time = sim_aligned["time"]

    rows = _plot_layer_rmse_map_comparison(
        rmse_maps,
        argo_iap["area"],
        "OHC per Unit Area",
        "J m$^{-2}$",
        "ohc_per_area",
        period_label,
        depth_slices=depth_slices,
        total_values=total_values or None,
    )
    if aligned_time is None:
        raise ValueError(f"No aligned OHC samples in {period_label}")
    rows = _annotate_rmse_period(
        rows,
        pd.Timestamp(aligned_time.values[0]),
        pd.Timestamp(aligned_time.values[-1]),
        period_kind,
        aligned_time.size,
        summaries,
    )
    return pd.concat([rows, *annual_frames], ignore_index=True, sort=False)


def _plot_annual_rmse_timeseries(
    rmse_metrics: pd.DataFrame,
    emulators: dict[str, dict],
) -> None:
    panels = [
        (
            "surface_geostrophic_velocity_vector_total_rmse",
            None,
            "Surface Velocity Vector",
            "RMSE (m s$^{-1}$)",
        ),
        ("surface_eke_total_rmse", None, "Surface EKE", "RMSE (m$^2$ s$^{-2}$)"),
        ("surface_sst_total_rmse", None, "SST", "RMSE ($^\\circ$C)"),
        (
            "ohc_per_area_total_rmse",
            OHC_DEPTH_SLICES[0]["title"],
            "OHC per Unit Area, 0-700 m",
            "RMSE (J m$^{-2}$)",
        ),
        (
            "ohc_per_area_total_rmse",
            OHC_DEPTH_SLICES[1]["title"],
            "OHC per Unit Area, 700-2000 m",
            "RMSE (J m$^{-2}$)",
        ),
    ]
    annual = rmse_metrics[rmse_metrics["period_kind"] == "annual"].copy()
    if annual.empty:
        raise ValueError("No annual RMSE rows are available for the annual RMSE figure")
    primary_rows = rmse_metrics[
        rmse_metrics["period_kind"] == "primary_complete_years"
    ]
    if primary_rows.empty:
        raise ValueError("No primary complete-year RMSE rows are available")
    primary_start = pd.to_datetime(primary_rows["period_start"]).min()
    primary_end = pd.to_datetime(primary_rows["period_end"]).max()

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 11), sharex=True)
    axes_flat = axes.ravel()
    legend_handles = None
    legend_labels = None
    for ax, (metric, depth, title, ylabel) in zip(axes_flat, panels, strict=False):
        panel = annual[annual["metric"] == metric]
        full = rmse_metrics[
            (rmse_metrics["metric"] == metric)
            & (rmse_metrics["period_kind"] == "primary_complete_years")
        ]
        if depth is None:
            panel = panel[panel["depth"].isna()]
            full = full[full["depth"].isna()]
        else:
            panel = panel[panel["depth"] == depth]
            full = full[full["depth"] == depth]

        for label, emu in emulators.items():
            model_annual = panel[panel["model"] == label].sort_values("year")
            if model_annual.empty:
                continue
            years = model_annual["year"].astype(int).to_numpy()
            values = model_annual["value"].astype(float).to_numpy()
            color = emu["color"]
            ax.plot(years, values, marker="o", linewidth=2.0, markersize=4.5, color=color, label=label)

            model_full = full[full["model"] == label]
            if model_full.empty:
                continue
            row = model_full.iloc[0]
            ax.axhline(float(row["value"]), color=color, linewidth=1.4, linestyle="--", alpha=0.9)
            ci_low = pd.to_numeric(pd.Series([row.get("ci_low")]), errors="coerce").iloc[0]
            ci_high = pd.to_numeric(pd.Series([row.get("ci_high")]), errors="coerce").iloc[0]
            if np.isfinite(ci_low) and np.isfinite(ci_high):
                ax.fill_between(years, ci_low, ci_high, color=color, alpha=0.10, linewidth=0)

        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    axes_flat[-1].axis("off")
    axes[-1, 0].set_xlabel("Calendar year")
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=len(legend_labels),
            bbox_to_anchor=(0.5, 0.025),
            frameon=False,
        )
    fig.suptitle(
        "Annual Total RMSE vs Observations\n"
        f"Dashed lines: {primary_start:%Y}–{primary_end:%Y} primary RMSE; "
        "shading: calendar-year block-bootstrap 95% CI",
        y=0.985,
        fontsize=15,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.09, hspace=0.32, wspace=0.28)


def _plot_annual_variance_map_rmse_timeseries(
    variance_metrics: pd.DataFrame,
    emulators: dict[str, dict],
) -> None:
    panels = [
        (
            "surface_sst_residual_variance_map_rmse",
            "SST Residual-Variance Map",
            "RMSE (($^\\circ$C)$^2$)",
        ),
        (
            "ohc_upper700_per_area_residual_variance_map_rmse",
            "Upper-700m OHC Residual-Variance Map",
            "RMSE ((J m$^{-2}$)$^2$)",
        ),
    ]
    annual = variance_metrics[
        variance_metrics["period_kind"] == "annual_residual_variance"
    ]
    if annual.empty:
        raise ValueError("No annual residual-variance map RMSE rows are available")

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
    legend_handles = None
    legend_labels = None
    for ax, (metric, title, ylabel) in zip(axes, panels, strict=True):
        panel = annual[annual["metric"] == metric]
        full = variance_metrics[
            (variance_metrics["metric"] == metric)
            & (variance_metrics["period_kind"] == "full_rollout_residual_variance")
        ]
        for label, emu in emulators.items():
            model_annual = panel[panel["model"] == label].sort_values("year")
            if model_annual.empty:
                continue
            years = model_annual["year"].astype(int).to_numpy()
            values = model_annual["value"].astype(float).to_numpy()
            color = emu["color"]
            ax.plot(
                years,
                values,
                marker="o",
                linewidth=2,
                markersize=4.5,
                color=color,
                label=label,
            )
            model_full = full[full["model"] == label]
            if model_full.empty:
                continue
            summary = model_full.iloc[0]
            aggregate = float(summary["annual_variance_rmse_block_aggregate"])
            ci_low = float(summary["annual_variance_rmse_ci_low"])
            ci_high = float(summary["annual_variance_rmse_ci_high"])
            ax.axhline(aggregate, color=color, linewidth=1.4, linestyle="--", alpha=0.9)
            if np.isfinite(ci_low) and np.isfinite(ci_high):
                ax.fill_between(years, ci_low, ci_high, color=color, alpha=0.10, linewidth=0)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    axes[-1].set_xlabel("Calendar year")
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=len(legend_labels),
            bbox_to_anchor=(0.5, 0.02),
            frameon=False,
        )
    fig.suptitle(
        "Annual Residual-Variance Map RMSE\n"
        "Dashed lines: RMS across annual scores; shading: year-block bootstrap 95% CI",
        y=0.98,
        fontsize=14,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.12, hspace=0.32)


def _argo_variable_variance_layer_maps_data(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    spec: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict[str, xr.DataArray], dict[str, dict[str, xr.DataArray]], xr.DataArray]:
    if "area" not in argo_iap:
        argo_iap = _infer_obs_grid_metrics(argo_iap)
    obs_field, sim_fields = _common_temporal_fields_on_argo_grid(
        argo_iap, emulators, spec, slice(start, end), _time_var_reducer
    )
    obs_maps = _depth_weighted_layer_maps_from_field(obs_field)
    sim_maps = {
        label: _depth_weighted_layer_maps_from_field(sim_field)
        for label, sim_field in sim_fields.items()
    }
    return obs_maps, sim_maps, argo_iap["area"]


def _plot_layer_variance_map_comparison(
    obs_maps: dict[str, xr.DataArray],
    sim_maps: dict[str, dict[str, xr.DataArray]],
    obs_area: xr.DataArray,
    title: str,
    units: str,
    metric_prefix: str,
    period_label: str,
) -> pd.DataFrame:
    """Temporal-variance maps: ARGO-IAP row + one row per emulator, columns are depth layers.

    Mirrors functions.plot_variance_comparison_together (thermal colormap, per-layer linear
    scale to the 95th percentile). Each emulator panel is annotated with the global pattern
    correlation and area-weighted RMSE of the variance field against ARGO-IAP.
    """
    layer_titles = [depth_slice["title"] for depth_slice in OHC_DEPTH_SLICES]
    nrows = 1 + len(sim_maps)
    ncols = len(layer_titles)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols, 3.3 * nrows),
        squeeze=False,
        subplot_kw={"projection": ccrs.Robinson(central_longitude=210)},
        gridspec_kw={"hspace": 0.12, "wspace": 0.05},
    )

    rows = []
    for col_idx, layer_title in enumerate(layer_titles):
        obs_map = obs_maps[layer_title]
        col_fields = [obs_map] + [model_maps[layer_title] for model_maps in sim_maps.values()]
        flat = np.concatenate([f.values[np.isfinite(f.values)] for f in col_fields])
        vmax = float(np.nanpercentile(flat, 95)) if flat.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        norm = colors.Normalize(vmin=0.0, vmax=vmax)

        ax = axes[0, col_idx]
        mesh = obs_map.plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(), x="lon", y="lat",
            cmap=cmocean.cm.thermal, norm=norm, add_colorbar=False, rasterized=True, add_labels=False,
        )
        _add_map_features(ax)
        ax.set_title(layer_title, fontweight="bold", y=1.04)
        if col_idx == 0:
            ax.text(-0.07, 0.55, "ARGO-IAP", va="center", ha="right", rotation=90,
                    transform=ax.transAxes, fontweight="bold")

        for row_idx, (label, model_maps) in enumerate(sim_maps.items(), start=1):
            ax = axes[row_idx, col_idx]
            sim_map = model_maps[layer_title]
            mesh = sim_map.plot.pcolormesh(
                ax=ax, transform=ccrs.PlateCarree(), x="lon", y="lat",
                cmap=cmocean.cm.thermal, norm=norm, add_colorbar=False, rasterized=True, add_labels=False,
            )
            _add_map_features(ax)
            if col_idx == 0:
                ax.text(-0.07, 0.55, label, va="center", ha="right", rotation=90,
                        transform=ax.transAxes, fontweight="bold")

            bias = sim_map - obs_map
            rmse = _area_weighted_map_rmse(bias, obs_area)
            corr = _area_weighted_pattern_corr(sim_map, obs_map, obs_area)
            ax.text(0.02, 0.95, f"Area-wtd corr: {corr:.3f}", transform=ax.transAxes,
                    fontsize=8.5, fontweight="bold", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), zorder=20)
            rows.append({"model": label, "depth": layer_title,
                         "metric": f"{metric_prefix}_temporal_variance_layer_map_rmse", "value": rmse})
            rows.append({"model": label, "depth": layer_title,
                         "metric": f"{metric_prefix}_temporal_variance_area_weighted_pattern_corr", "value": corr})

        cbar = fig.colorbar(mesh, ax=axes[:, col_idx], orientation="horizontal",
                            shrink=0.7, pad=0.03, extend="max", label=f"Variance ({units})")
        cbar.ax.tick_params(labelsize=9)

    fig.suptitle(f"Temporal Variance of {title} vs ARGO-IAP ({period_label})", y=0.99, fontsize=15)
    return pd.DataFrame(rows)


def _plot_variance_map_comparison(
    obs_map: xr.DataArray,
    sim_maps: dict[str, xr.DataArray],
    obs_area: xr.DataArray,
    title: str,
    units: str,
    metric_prefix: str,
    period_label: str,
    obs_label: str,
    annual_rmse_rows: pd.DataFrame | None = None,
    annual_rmse_summaries: dict[str, dict[str, float | int | str]] | None = None,
) -> pd.DataFrame:
    print(f"  Loading {obs_label} residual variance map for plotting/metrics...", flush=True)
    obs_map = obs_map.load()
    loaded_sim_maps = {}
    for label, sim_map in sim_maps.items():
        print(f"  Loading {label} residual variance map for plotting/metrics...", flush=True)
        loaded_sim_maps[label] = sim_map.load()
    sim_maps = loaded_sim_maps

    ncols = 1 + len(sim_maps)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(5.4 * ncols, 4.9),
        squeeze=False,
        subplot_kw={"projection": ccrs.Robinson(central_longitude=210)},
    )
    axes = axes[0]
    fields = [obs_map] + list(sim_maps.values())
    flat = np.concatenate([field.values[np.isfinite(field.values)] for field in fields])
    vmax = float(np.nanpercentile(flat, 95)) if flat.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    norm = colors.Normalize(vmin=0.0, vmax=vmax)

    mesh = obs_map.plot.pcolormesh(
        ax=axes[0],
        transform=ccrs.PlateCarree(),
        x="lon",
        y="lat",
        cmap=cmocean.cm.thermal,
        norm=norm,
        add_colorbar=False,
        rasterized=True,
        add_labels=False,
    )
    _add_map_features(axes[0])
    axes[0].set_title(obs_label, fontsize=12)

    rows = []
    for ax, (label, sim_map) in zip(axes[1:], sim_maps.items()):
        mesh = sim_map.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            x="lon",
            y="lat",
            cmap=cmocean.cm.thermal,
            norm=norm,
            add_colorbar=False,
            rasterized=True,
            add_labels=False,
        )
        _add_map_features(ax)
        rmse = _area_weighted_map_rmse(sim_map - obs_map, obs_area)
        corr = _area_weighted_pattern_corr(sim_map, obs_map, obs_area)
        rmse_row = {
            "model": label,
            "metric": f"{metric_prefix}_residual_variance_map_rmse",
            "value": rmse,
            "period_kind": "full_rollout_residual_variance",
            "is_primary": True,
        }
        if annual_rmse_summaries and label in annual_rmse_summaries:
            rmse_row.update(annual_rmse_summaries[label])
        rows.append(rmse_row)
        rows.append(
            {
                "model": label,
                "metric": f"{metric_prefix}_residual_variance_area_weighted_pattern_corr",
                "value": corr,
                "period_kind": "full_rollout_residual_variance",
                "is_primary": True,
            }
        )
        ax.set_title(label, fontsize=12)
        ax.text(
            0.02,
            0.95,
            f"Area-wtd corr: {corr:.3f}",
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            zorder=20,
        )

    fig.subplots_adjust(left=0.04, right=0.98, top=0.76, bottom=0.30, wspace=0.08)
    cax = fig.add_axes([0.28, 0.145, 0.44, 0.035])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="max")
    cbar.set_label(f"Variance ({units})", fontsize=12)
    fig.suptitle(
        f"{title}\nDetrended Deseasonalized Anomaly Variance vs {obs_label} ({period_label})",
        y=0.98,
        fontsize=15,
        linespacing=1.15,
    )
    frames = [pd.DataFrame(rows)]
    if annual_rmse_rows is not None and not annual_rmse_rows.empty:
        frames.append(annual_rmse_rows)
    return pd.concat(frames, ignore_index=True, sort=False)


def _annual_variance_maps(
    residual: xr.DataArray,
    time_dim: str = "time",
) -> tuple[xr.DataArray, dict[int, int]]:
    """Variance maps for complete calendar years after one full-period residual fit."""
    years = _complete_calendar_years(residual[time_dim])
    if not years:
        raise ValueError("Residual-variance uncertainty requires at least one complete calendar year")
    index = pd.DatetimeIndex(residual[time_dim].values)
    counts = {year: int((index.year == year).sum()) for year in years}
    annual = residual.groupby(f"{time_dim}.year").var(time_dim, skipna=True).sel(year=years)
    return annual, counts


def _annual_variance_map_rmse_diagnostics(
    obs_annual_variance: xr.DataArray,
    sim_annual_variances: dict[str, xr.DataArray],
    area: xr.DataArray,
    metric_prefix: str,
    sample_counts: dict[int, int],
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int | str]]]:
    """Annual variance-map RMSE rows and year-block uncertainty summaries."""
    rows = []
    summaries = {}
    spatial_dims = tuple(area.dims)
    metric = f"{metric_prefix}_residual_variance_map_rmse"
    for label, sim_annual in sim_annual_variances.items():
        sim_aligned, obs_aligned, area_aligned = xr.align(
            sim_annual, obs_annual_variance, area, join="inner"
        )
        if sim_aligned.sizes.get("year", 0) == 0:
            continue
        bias_squared = (sim_aligned - obs_aligned) ** 2
        finite = np.isfinite(bias_squared)
        weighted_sum = (bias_squared.where(finite) * area_aligned).sum(spatial_dims, skipna=True)
        weight_sum = area_aligned.where(finite).sum(spatial_dims, skipna=True)
        annual_rmse = np.sqrt(weighted_sum / weight_sum).compute()
        years = np.asarray(annual_rmse["year"].values, dtype=int)
        values = np.asarray(annual_rmse.values, dtype=float)
        for year, value in zip(years, values, strict=True):
            rows.append(
                {
                    "model": label,
                    "metric": metric,
                    "value": float(value),
                    "period_kind": "annual_residual_variance",
                    "period_start": f"{year}-01-01",
                    "period_end": f"{year}-12-31",
                    "n_time_samples": sample_counts.get(int(year), np.nan),
                    "year": int(year),
                    "is_primary": False,
                }
            )
        summary = _interannual_rmse_summary(values)
        summaries[label] = {
            "annual_variance_rmse_std": summary["annual_std"],
            "annual_variance_rmse_ci_low": summary["ci_low"],
            "annual_variance_rmse_ci_high": summary["ci_high"],
            "annual_variance_rmse_block_aggregate": summary["block_aggregate_rmse"],
            "annual_variance_n_years": summary["n_years"],
            "annual_variance_n_bootstrap": summary["n_bootstrap"],
            "annual_variance_uncertainty_method": summary["uncertainty_method"],
        }
    return pd.DataFrame(rows), summaries


def _plot_sst_residual_variance_maps(
    oisst: xr.Dataset,
    emulators: dict[str, dict],
) -> pd.DataFrame:
    sst_name = _find_var_name(oisst, ["sst", "analysed_sst", "thetao", "tos"])
    window_start, window_end = _common_time_window(oisst, *[emu["ds"] for emu in emulators.values()])
    period_label = _period_label(window_start, window_end)
    obs_sst = oisst[sst_name].sel(time=slice(window_start, window_end))
    variance_chunks = {
        "time": -1,
        "lat": min(90, obs_sst.sizes["lat"]),
        "lon": min(180, obs_sst.sizes["lon"]),
    }
    obs_sst = obs_sst.chunk(variance_chunks)
    print(f"  SST residual variance chunks: {variance_chunks}", flush=True)
    obs_residual = _detrended_deseasonalized_dataarray(obs_sst)
    obs_var = obs_residual.var("time", skipna=True)
    obs_annual_var, annual_sample_counts = _annual_variance_maps(obs_residual)

    sim_vars = {}
    sim_annual_vars = {}
    for label, emu in emulators.items():
        print(f"  Computing native-grid SST residual variance for {label}", flush=True)
        sim_w = emu["ds"].sel(time=slice(window_start, window_end))
        sim_sst = _apply_native_wet_mask(sim_w["thetao"].isel(lev=0), emu.get("wet_mask"))
        sim_sst = sim_sst.rename({"y": "lat", "x": "lon"})
        sim_sst = _normalize_lon(sim_sst, "lon").sortby("lat").sortby("lon")
        sim_sst = sim_sst.chunk(
            {
                "time": -1,
                "lat": min(90, sim_sst.sizes["lat"]),
                "lon": min(180, sim_sst.sizes["lon"]),
            }
        )
        sim_residual = _detrended_deseasonalized_dataarray(sim_sst)
        sim_var_native = sim_residual.var("time", skipna=True)
        sim_vars[label] = sim_var_native.interp(
            lat=obs_var["lat"],
            lon=obs_var["lon"],
            kwargs={"fill_value": np.nan},
        ).where(np.isfinite(obs_var))
        sim_annual_native, _ = _annual_variance_maps(sim_residual)
        sim_annual_vars[label] = sim_annual_native.interp(
            lat=obs_annual_var["lat"],
            lon=obs_annual_var["lon"],
            kwargs={"fill_value": np.nan},
        ).where(np.isfinite(obs_annual_var))

    annual_rows, annual_summaries = _annual_variance_map_rmse_diagnostics(
        obs_annual_var,
        sim_annual_vars,
        oisst["area"],
        "surface_sst",
        annual_sample_counts,
    )

    return _plot_variance_map_comparison(
        obs_var,
        sim_vars,
        oisst["area"],
        "SST",
        "($^\\circ$C)$^2$",
        "surface_sst",
        period_label,
        "OISST",
        annual_rmse_rows=annual_rows,
        annual_rmse_summaries=annual_summaries,
    )


def _plot_ohc_upper700_residual_variance_maps(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    start: pd.Timestamp,
    end: pd.Timestamp,
    period_label: str,
) -> pd.DataFrame:
    temp_spec = ARGO_STRUCTURE_VARIABLES[0]
    if "area" not in argo_iap:
        argo_iap = _infer_obs_grid_metrics(argo_iap)
    time_slice = slice(start, end)
    obs_temp = _reduce_obs_structure_field(
        argo_iap, temp_spec, time_slice, _identity_reducer
    )
    depth_slices = [OHC_UPPER_700_SLICE]
    obs_ohc = _ohc_per_area_layer_maps_from_temperature_field(obs_temp, depth_slices)[OHC_UPPER_700_SLICE["title"]]
    obs_monthly = _resample_monthly_field(obs_ohc)
    obs_residual = _detrended_deseasonalized_dataarray(obs_monthly)
    obs_var = obs_residual.var("time", skipna=True)
    obs_annual_var, annual_sample_counts = _annual_variance_maps(obs_residual)

    sim_vars = {}
    sim_annual_vars = {}
    for label, emu in emulators.items():
        ds_sim = emu["ds"]
        sim_temp = ds_sim["thetao"].sel(time=time_slice)
        sim_temp = _apply_native_wet_mask(sim_temp, emu.get("wet_mask"))
        sim_ohc = _ohc_per_area_layer_maps_from_temperature_field(
            sim_temp,
            depth_slices,
            native_dz=ds_sim["dz"],
            depth_name="lev",
        )[OHC_UPPER_700_SLICE["title"]]
        sim_monthly = _resample_monthly_field(sim_ohc)
        sim_residual = _detrended_deseasonalized_dataarray(sim_monthly)
        sim_var_native = sim_residual.var("time", skipna=True)
        sim_vars[label] = _model_field_on_obs_horizontal_grid(sim_var_native, obs_var)
        sim_annual_native, _ = _annual_variance_maps(sim_residual)
        sim_annual_vars[label] = _model_field_on_obs_horizontal_grid(
            sim_annual_native, obs_annual_var
        )

    annual_rows, annual_summaries = _annual_variance_map_rmse_diagnostics(
        obs_annual_var,
        sim_annual_vars,
        argo_iap["area"],
        "ohc_upper700_per_area",
        annual_sample_counts,
    )

    return _plot_variance_map_comparison(
        obs_var,
        sim_vars,
        argo_iap["area"],
        "Upper-700m OHC per Unit Area",
        "(J m$^{-2}$)$^2$",
        "ohc_upper700_per_area",
        period_label,
        "ARGO-IAP",
        annual_rmse_rows=annual_rows,
        annual_rmse_summaries=annual_summaries,
    )


def _plot_argo_variable_variance_maps(
    argo_iap: xr.Dataset,
    emulators: dict[str, dict],
    spec: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
    period_label: str,
) -> pd.DataFrame:
    obs_maps, sim_maps, obs_area = _argo_variable_variance_layer_maps_data(
        argo_iap, emulators, spec, start, end
    )
    return _plot_layer_variance_map_comparison(
        obs_maps,
        sim_maps,
        obs_area,
        spec["title"],
        f"({spec['units']})$^2$",
        spec["metric_prefix"],
        period_label,
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
    # Restrict obs (DUACS) and the models to their common overlap within the year so the
    # eddy KE (time-mean of the squared velocity anomaly) is computed over the same period.
    window_start, window_end = _common_time_window(
        obs_year, *[_year_slice(emu["ds"], year) for emu in emulators.values()]
    )
    obs_year = obs_year.sel(time=slice(window_start, window_end))
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
        sim_year = _year_slice(emu["ds"], year).sel(time=slice(window_start, window_end))
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
    # Restrict obs (DUACS) and the models to their common overlap within the year.
    window_start, window_end = _common_time_window(
        obs_year, *[_year_slice(emu["ds"], year) for emu in emulators.values()]
    )
    obs_year = obs_year.sel(time=slice(window_start, window_end))
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
        sim_year = _year_slice(emu["ds"], year).sel(time=slice(window_start, window_end))
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


def _plot_eke_spatial_spectra(
    duacs: xr.Dataset,
    emulators: dict[str, dict],
    velocity_kind: str = "absolute",
) -> pd.DataFrame:
    u_name, v_name = _duacs_velocity_names(duacs, velocity_kind)
    velocity_label = _duacs_velocity_label(velocity_kind)
    window_start, window_end = _common_time_window(duacs, *[emu["ds"] for emu in emulators.values()])
    period_label = _period_label(window_start, window_end)
    duacs_w = duacs.sel(time=slice(window_start, window_end))
    obs_u = _apply_equatorial_mask(duacs_w[u_name], lat_dim="lat")
    obs_v = _apply_equatorial_mask(duacs_w[v_name], lat_dim="lat")
    obs_eke_map = _surface_eke_from_uv(obs_u, obs_v)

    rows = []
    sim_eke_maps = {}
    for label, emu in emulators.items():
        sim = emu["ds"].sel(time=slice(window_start, window_end))
        sim_zos = _apply_native_wet_mask(sim["zos"], emu.get("wet_mask"))
        sim_u, sim_v = _geostrophic_velocity_from_zos(sim_zos, lat_dim="y", lon_dim="x")
        sim_u = _apply_equatorial_mask(_apply_native_wet_mask(sim_u, emu.get("wet_mask")), lat_dim="y")
        sim_v = _apply_equatorial_mask(_apply_native_wet_mask(sim_v, emu.get("wet_mask")), lat_dim="y")
        sim_eke_maps[label] = (_surface_eke_from_uv(sim_u, sim_v), sim)

    ncols = len(SPATIAL_SPECTRUM_PANELS)
    fig, axes = plt.subplots(1, ncols, figsize=(4.8 * ncols, 4.2))
    axes = np.atleast_1d(axes)

    for idx, (region_name, lon_slice, lat_slice) in enumerate(SPATIAL_SPECTRUM_PANELS):
        ax = axes[idx]
        k_obs, spec_obs = _compute_isotropic_spectrum_from_map(
            obs_eke_map, duacs_w["dx"], duacs_w["dy"], lon_slice, lat_slice
        )
        ax.loglog(k_obs, spec_obs, "--", color="black", linewidth=2, label="DUACS")
        for label, (sim_eke_map, sim) in sim_eke_maps.items():
            k_sim, spec_sim = _compute_isotropic_spectrum_from_map(
                sim_eke_map, sim["dx"], sim["dy"], lon_slice, lat_slice, lat_dim="y", lon_dim="x"
            )
            rows.append(
                {
                    "model": label,
                    "region": region_name,
                    "metric": "surface_eke_spatial_spectrum_log10_rmse_vs_duacs",
                    "value": _log10_rmse_between_spectra(k_obs, spec_obs, k_sim, spec_sim),
                }
            )
            ax.loglog(k_sim, spec_sim, "-", color=emulators[label]["color"], linewidth=2, label=label)
        ax.set_title(f"{region_name}\nspatial spectrum")
        ax.set_xlabel(r"Wavenumber $\kappa$ (km$^{-1}$)")
        ax.set_ylabel("PSD")
        ax.grid(True, which="both", alpha=0.3)

    handles, labels, ncol = _legend_truth_first(*axes[0].get_legend_handles_labels(), truth_labels={"DUACS"})
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.005), frameon=False)
    fig.suptitle(
        _figure_title_with_context(
            "Surface EKE Spatial Spectra vs DUACS",
            period_label,
            f"{velocity_label}; spatial spectra over {SPATIAL_SPECTRUM_PANEL_LABEL}",
        ),
        y=0.98,
        fontsize=15,
        linespacing=1.15,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.18, wspace=0.28)
    return pd.DataFrame(rows)


def _yearly_eke_spatial_spectra(
    u: xr.DataArray,
    v: xr.DataArray,
    dx_2d: xr.DataArray,
    dy_2d: xr.DataArray,
    lon_slice: slice,
    lat_slice: slice,
    years: list[int],
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> tuple[np.ndarray | None, dict[int, np.ndarray]]:
    """Per-year spectra of annual-mean EKE using one shared multi-year velocity mean."""
    u_region = u.sel({lon_dim: lon_slice, lat_dim: lat_slice})
    v_region = v.sel({lon_dim: lon_slice, lat_dim: lat_slice})
    eke = _instantaneous_surface_eke_from_uv(u_region, v_region)
    eke = _fully_observed_spectrum_block(eke, lat_dim, lon_dim)
    metric_selection = {lat_dim: eke[lat_dim], lon_dim: eke[lon_dim]}
    dx_mean = float(dx_2d.sel(metric_selection).mean().values)
    dy_mean = float(dy_2d.sel(metric_selection).mean().values)

    k_common = None
    by_year = {}
    for year in years:
        annual_map = eke.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).mean(
            "time", skipna=True
        )
        if annual_map.sizes.get(lat_dim, 0) == 0 or annual_map.sizes.get(lon_dim, 0) == 0:
            continue
        values = np.asarray(annual_map.values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"EKE spatial spectrum contains non-finite values in {year}")
        k_centers, spectrum = compute_isotropic_spectrum_torch(
            torch.as_tensor(values),
            dx=dx_mean,
            dy=dy_mean,
            n_factor=2,
            detrend="linear",
            window="hann",
            cutoff_before_bins=False,
        )
        if k_common is None:
            k_common = k_centers.cpu().numpy() * 1000 * 2 * np.pi
        by_year[year] = spectrum.cpu().numpy()
    return k_common, by_year


def _plot_eke_spatial_spectra_interannual(
    duacs: xr.Dataset,
    emulators: dict[str, dict],
    years: list[int] = SPECTRA_INTERANNUAL_YEARS,
    velocity_kind: str = "absolute",
) -> pd.DataFrame:
    """Per-year EKE spatial spectra summarized by geometric mean and +/-1 log-std."""
    u_name, v_name = _duacs_velocity_names(duacs, velocity_kind)
    velocity_label = _duacs_velocity_label(velocity_kind)
    y0, y1 = years[0], years[-1]
    window = slice(f"{y0}-01-01", f"{y1}-12-31")
    duacs_w = duacs.sel(time=window)
    obs_u = _apply_equatorial_mask(duacs_w[u_name], lat_dim="lat")
    obs_v = _apply_equatorial_mask(duacs_w[v_name], lat_dim="lat")

    sim_velocities = {}
    for label, emu in emulators.items():
        sim = emu["ds"].sel(time=window)
        sim_zos = _apply_native_wet_mask(sim["zos"], emu.get("wet_mask"))
        sim_u, sim_v = _geostrophic_velocity_from_zos(sim_zos, lat_dim="y", lon_dim="x")
        sim_u = _apply_equatorial_mask(
            _apply_native_wet_mask(sim_u, emu.get("wet_mask")), lat_dim="y"
        )
        sim_v = _apply_equatorial_mask(
            _apply_native_wet_mask(sim_v, emu.get("wet_mask")), lat_dim="y"
        )
        sim_velocities[label] = (sim_u, sim_v, sim)

    rows = []
    ncols = len(SPATIAL_SPECTRUM_PANELS)
    fig, axes = plt.subplots(1, ncols, figsize=(4.8 * ncols, 4.2))
    axes = np.atleast_1d(axes)
    for ax, (region_name, lon_slice, lat_slice) in zip(
        axes, SPATIAL_SPECTRUM_PANELS, strict=True
    ):
        k_obs, obs_by_year = _yearly_eke_spatial_spectra(
            obs_u,
            obs_v,
            duacs_w["dx"],
            duacs_w["dy"],
            lon_slice,
            lat_slice,
            years,
        )
        if obs_by_year:
            x_ref, gm, lo, hi, _ = _interannual_band(
                [(k_obs, spectrum) for spectrum in obs_by_year.values()]
            )
            if x_ref is not None:
                _plot_spectra_interannual_panel(ax, x_ref, gm, lo, hi, "--", "black", "DUACS")

        for label, (sim_u, sim_v, sim) in sim_velocities.items():
            k_sim, sim_by_year = _yearly_eke_spatial_spectra(
                sim_u,
                sim_v,
                sim["dx"],
                sim["dy"],
                lon_slice,
                lat_slice,
                years,
                lat_dim="y",
                lon_dim="x",
            )
            if not sim_by_year:
                continue
            x_ref, gm, lo, hi, _ = _interannual_band(
                [(k_sim, spectrum) for spectrum in sim_by_year.values()]
            )
            if x_ref is not None:
                _plot_spectra_interannual_panel(
                    ax, x_ref, gm, lo, hi, "-", emulators[label]["color"], label
                )
            per_year = {
                year: _log10_rmse_between_spectra(
                    k_obs, obs_by_year[year], k_sim, sim_by_year[year]
                )
                for year in sorted(set(obs_by_year) & set(sim_by_year))
            }
            rows.extend(
                _interannual_spectrum_rmse_rows(
                    label,
                    region_name,
                    "surface_eke_spatial_spectrum_log10_rmse_vs_duacs",
                    per_year,
                )
            )
        ax.set_title(region_name)
        ax.set_xlabel(r"Wavenumber $\kappa$ (km$^{-1}$)")
        ax.set_ylabel("PSD")
        ax.grid(True, which="both", alpha=0.3)

    handles, labels, ncol = _legend_truth_first(
        *axes[0].get_legend_handles_labels(), truth_labels={"DUACS"}
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, -0.005),
        frameon=False,
    )
    fig.suptitle(
        "Open-Ocean Box Surface EKE Spatial Spectra vs DUACS\n"
        f"interannual geometric mean $\pm$1 log-std over {y0}–{y1} ({velocity_label})",
        y=0.975,
        fontsize=14,
        linespacing=1.15,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.18, wspace=0.28)
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
    # Restrict OISST and the models to their common overlap so the SSTA calendar-day
    # climatology and spectra are estimated over an identical period; the figure label is
    # derived from this window rather than hardcoded.
    window_start, window_end = _common_time_window(
        oisst_5day, *[emu["ds"] for emu in emulators.values()]
    )
    period_label = _period_label(window_start, window_end)
    oisst_5day = oisst_5day.sel(time=slice(window_start, window_end))
    if oisst_daily is not None:
        oisst_daily = oisst_daily.sel(time=slice(window_start, window_end))
    sim_windows = {
        label: emu["ds"].sel(time=slice(window_start, window_end)) for label, emu in emulators.items()
    }
    rows = []
    ncols = len(SPATIAL_SPECTRUM_PANELS)
    fig, axes = plt.subplots(1, ncols, figsize=(4.8 * ncols, 4.2))
    axes = np.ravel(axes)

    for idx, (region_name, lon_slice, lat_slice) in enumerate(SPATIAL_SPECTRUM_PANELS):
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
            sim = sim_windows[label]
            sim_sst = _apply_native_wet_mask(sim["thetao"].isel(lev=0), emu.get("wet_mask"))
            k_sim, spec_sim = _compute_isotropic_spectrum_from_surface(
                sim_sst, sim["dx"], sim["dy"], lon_slice, lat_slice,
                lat_dim="y", lon_dim="x",
            )
            rmse_5day = _log10_rmse_between_spectra(k_5day, spec_5day, k_sim, spec_sim)
            rows.append(
                {
                    "model": label,
                    "region": region_name,
                    "metric": "surface_ssta_isotropic_spectrum_log10_rmse_vs_oisst_5day_avg",
                    "value": rmse_5day,
                }
            )
            if k_daily is not None and spec_daily is not None:
                rmse_daily = _log10_rmse_between_spectra(k_daily, spec_daily, k_sim, spec_sim)
                rows.append(
                    {
                        "model": label,
                        "region": region_name,
                        "metric": "surface_ssta_isotropic_spectrum_log10_rmse_vs_oisst_raw_daily",
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
        "Open-Ocean Box Surface Temperature SSTA Spatial Spectra vs OISST\n"
        f"{period_label} calendar-day climatology removed",
        y=0.975,
        fontsize=15,
        linespacing=1.15,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.18, wspace=0.28)
    return pd.DataFrame(rows)


def _yearly_isotropic_spectra(
    field: xr.DataArray,
    dx_2d: xr.DataArray,
    dy_2d: xr.DataArray,
    lon_slice: slice,
    lat_slice: slice,
    years: list[int],
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> tuple[np.ndarray | None, dict[int, np.ndarray]]:
    """Per-year isotropic SSTA spectra for a region.

    The SSTA is taken relative to the calendar-day climatology of the FULL multi-year window
    (so every year shares one baseline), the region is cropped to a fully-observed block once,
    then the isotropic spectrum is computed per calendar year. Returns the common wavenumber
    grid (None if nothing usable) and {year: spectrum}.
    """
    squeeze_dims = [dim for dim, size in field.sizes.items() if size == 1 and dim not in {"time", lat_dim, lon_dim}]
    if squeeze_dims:
        field = field.squeeze(squeeze_dims, drop=True)
    data = field.sel({lon_dim: lon_slice, lat_dim: lat_slice}).transpose("time", lat_dim, lon_dim)
    ssta = _calendar_day_climatology_anomaly(data)
    ssta = _fully_observed_spectrum_block(ssta, lat_dim, lon_dim)
    metric_selection = {lat_dim: ssta[lat_dim], lon_dim: ssta[lon_dim]}
    dx_mean = float(dx_2d.sel(metric_selection).mean().values)
    dy_mean = float(dy_2d.sel(metric_selection).mean().values)
    k_common: np.ndarray | None = None
    by_year: dict[int, np.ndarray] = {}
    for year in years:
        ssta_yr = ssta.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        if ssta_yr.sizes.get("time", 0) == 0:
            continue
        values = np.asarray(ssta_yr.values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"Spatial spectrum contains non-finite values in {year}")
        k_cent, spec = compute_isotropic_spectrum_torch(
            torch.as_tensor(values),
            dx=dx_mean,
            dy=dy_mean,
            n_factor=2,
            detrend="linear",
            window="hann",
            cutoff_before_bins=False,
        )
        by_year[year] = spec.cpu().numpy().mean(0)
        if k_common is None:
            k_common = k_cent.cpu().numpy() * 1000 * 2 * np.pi
    return k_common, by_year


def _plot_temperature_spectra_interannual(
    oisst_5day: xr.Dataset,
    emulators: dict[str, dict],
    years: list[int] = SPECTRA_INTERANNUAL_YEARS,
) -> pd.DataFrame:
    """Isotropic SSTA spectra as a geometric mean +/- 1 log-std band: one spectrum per calendar
    year over `years`, aggregated across years. Complements the pooled single-curve figure."""
    sst_name = _find_var_name(oisst_5day, ["sst", "analysed_sst", "thetao", "tos"])
    y0, y1 = years[0], years[-1]
    window = slice(f"{y0}-01-01", f"{y1}-12-31")
    oisst_w = oisst_5day.sel(time=window)
    sim_w = {label: emu["ds"].sel(time=window) for label, emu in emulators.items()}

    rows = []
    ncols = len(SPATIAL_SPECTRUM_PANELS)
    fig, axes = plt.subplots(1, ncols, figsize=(4.8 * ncols, 4.2))
    axes = np.ravel(axes)

    for idx, (region_name, lon_slice, lat_slice) in enumerate(SPATIAL_SPECTRUM_PANELS):
        ax = axes[idx]
        k_obs, obs_by_year = _yearly_isotropic_spectra(
            oisst_w[sst_name], oisst_w["dx"], oisst_w["dy"], lon_slice, lat_slice, years
        )
        if obs_by_year:
            x_ref, gm, lo, hi, _ = _interannual_band([(k_obs, s) for s in obs_by_year.values()])
            if x_ref is not None:
                _plot_spectra_interannual_panel(ax, x_ref, gm, lo, hi, "--", "black", "OISST 5-day avg")
        for label, emu in emulators.items():
            sim = sim_w[label]
            sim_sst = _apply_native_wet_mask(sim["thetao"].isel(lev=0), emu.get("wet_mask"))
            k_sim, sim_by_year = _yearly_isotropic_spectra(
                sim_sst, sim["dx"], sim["dy"], lon_slice, lat_slice, years, lat_dim="y", lon_dim="x"
            )
            if not sim_by_year:
                continue
            x_ref_s, gm_s, lo_s, hi_s, _ = _interannual_band([(k_sim, s) for s in sim_by_year.values()])
            if x_ref_s is not None:
                _plot_spectra_interannual_panel(ax, x_ref_s, gm_s, lo_s, hi_s, "-", emu["color"], label)
            per_year = {
                yr: _log10_rmse_between_spectra(k_obs, obs_by_year[yr], k_sim, sim_by_year[yr])
                for yr in sorted(set(obs_by_year) & set(sim_by_year))
            }
            rows.extend(
                _interannual_spectrum_rmse_rows(
                    label,
                    region_name,
                    "surface_ssta_isotropic_spectrum_log10_rmse_vs_oisst_5day_avg",
                    per_year,
                )
            )
        ax.set_title(region_name)
        ax.set_xlabel(r"Wavenumber $\kappa$ (km$^{-1}$)")
        ax.set_ylabel("PSD")
        ax.grid(True, which="both", alpha=0.3)

    handles, labels, ncol = _legend_truth_first(
        *axes[0].get_legend_handles_labels(), truth_labels={"OISST 5-day avg"}
    )
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=11)
    fig.suptitle(
        "Open-Ocean Box Surface Temperature SSTA Spatial Spectra vs OISST\n"
        f"interannual geometric mean $\\pm$1 log-std over {y0}–{y1} (per-year spectra)",
        y=0.975,
        fontsize=14,
        linespacing=1.15,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.18, wspace=0.28)
    return pd.DataFrame(rows)


def _plot_temporal_ke_spectra(
    duacs_5day: xr.Dataset,
    duacs_raw: xr.Dataset,
    emulators: dict[str, dict],
    velocity_kind: str = "absolute",
) -> pd.DataFrame:
    span_start, span_end = _time_bounds(duacs_5day)
    raw_start = span_start.normalize()
    raw_end = span_end.normalize()
    duacs_raw = duacs_raw.sel(time=slice(raw_start, raw_end))
    if duacs_raw.sizes.get("time", 0) == 0:
        raise ValueError(f"No raw daily DUACS data found over 5-day DUACS span {raw_start} to {raw_end}")

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
    ncols = len(TEMPORAL_SPECTRUM_PANELS)
    fig, axes = plt.subplots(1, ncols, figsize=(4.8 * ncols, 4.2))
    axes = np.ravel(axes)

    for idx, (region_name, lon_slice, lat_slice) in enumerate(TEMPORAL_SPECTRUM_PANELS):
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
            rmse_raw = _log10_rmse_between_spectra(freq_raw, psd_raw, freq_sim, psd_sim)
            rmse_5day = _log10_rmse_between_spectra(freq_5day, psd_5day, freq_sim, psd_sim)
            rows.append(
                {
                    "model": label,
                    "region": region_name,
                    "metric": "surface_geostrophic_ke_temporal_spectrum_log10_rmse_vs_duacs_raw_daily",
                    "value": rmse_raw,
                }
            )
            rows.append(
                {
                    "model": label,
                    "region": region_name,
                    "metric": "surface_geostrophic_ke_temporal_spectrum_log10_rmse_vs_duacs_5day_avg",
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
            "Global and Regional Surface Geostrophic KE Temporal Spectra vs DUACS",
            extra=f"{velocity_label}; {_period_label(span_start, span_end)}; raw daily and 5-day averaged DUACS",
        ),
        y=0.975,
        fontsize=14,
        linespacing=1.15,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.18, wspace=0.28)
    return pd.DataFrame(rows)


def _yearly_temporal_ke_spectra(
    u: xr.DataArray,
    v: xr.DataArray,
    lon_slice: slice,
    lat_slice: slice,
    years: list[int],
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    min_samples: int = 8,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-year surface KE temporal (Welch) spectra for a region: {year: (freq, psd)}.

    Each full year is a separate Welch estimate, so the per-year spectra only resolve down to
    about 1 cycle/year; the lowest frequencies are intentionally left to the full-period figure.
    """
    by_year: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for year in years:
        u_yr = u.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        v_yr = v.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        if u_yr.sizes.get("time", 0) < min_samples:
            continue
        freq, psd = _compute_temporal_ke_spectrum_from_surface(u_yr, v_yr, lon_slice, lat_slice, lat_dim, lon_dim)
        if freq.size:
            by_year[year] = (freq, psd)
    return by_year


def _plot_temporal_ke_spectra_interannual(
    duacs_5day: xr.Dataset,
    emulators: dict[str, dict],
    years: list[int] = SPECTRA_INTERANNUAL_YEARS,
    velocity_kind: str = "absolute",
) -> pd.DataFrame:
    """Temporal KE spectra as a geometric mean +/- 1 log-std band (one Welch PSD per calendar
    year). Complements the pooled single-curve figure, which still covers the lowest frequencies."""
    u_name, v_name = _duacs_velocity_names(duacs_5day, velocity_kind)
    velocity_label = _duacs_velocity_label(velocity_kind)
    y0, y1 = years[0], years[-1]
    window = slice(f"{y0}-01-01", f"{y1}-12-31")
    duacs_w = duacs_5day.sel(time=window)

    rows = []
    ncols = len(TEMPORAL_SPECTRUM_PANELS)
    fig, axes = plt.subplots(1, ncols, figsize=(4.8 * ncols, 4.2))
    axes = np.ravel(axes)

    for idx, (region_name, lon_slice, lat_slice) in enumerate(TEMPORAL_SPECTRUM_PANELS):
        ax = axes[idx]
        obs_by_year = _yearly_temporal_ke_spectra(duacs_w[u_name], duacs_w[v_name], lon_slice, lat_slice, years)
        if obs_by_year:
            x_ref, gm, lo, hi, _ = _interannual_band(list(obs_by_year.values()))
            if x_ref is not None:
                _plot_spectra_interannual_panel(ax, x_ref, gm, lo, hi, "--", "black", "DUACS 5-day avg")
        for label, emu in emulators.items():
            sim = emu["ds"].sel(time=window)
            sim_region = sim.sel(
                x=_expand_numeric_slice(lon_slice, 1.0),
                y=_expand_numeric_slice(lat_slice, 1.0),
            )
            sim_zos = _apply_native_wet_mask(sim_region["zos"], emu.get("wet_mask"))
            sim_ug, sim_vg = _geostrophic_velocity_from_zos(sim_zos, lat_dim="y", lon_dim="x")
            if velocity_kind == "anomaly":
                sim_ug, sim_vg = _time_mean_velocity_anomaly(sim_ug, sim_vg)
            sim_by_year = _yearly_temporal_ke_spectra(sim_ug, sim_vg, lon_slice, lat_slice, years, lat_dim="y", lon_dim="x")
            if not sim_by_year:
                continue
            x_ref_s, gm_s, lo_s, hi_s, _ = _interannual_band(list(sim_by_year.values()))
            if x_ref_s is not None:
                _plot_spectra_interannual_panel(ax, x_ref_s, gm_s, lo_s, hi_s, "-", emu["color"], label)
            per_year = {}
            for yr in sorted(set(obs_by_year) & set(sim_by_year)):
                fo, po = obs_by_year[yr]
                fs, ps = sim_by_year[yr]
                per_year[yr] = _log10_rmse_between_spectra(fo, po, fs, ps)
            rows.extend(
                _interannual_spectrum_rmse_rows(
                    label,
                    region_name,
                    "surface_geostrophic_ke_temporal_spectrum_log10_rmse_vs_duacs_5day_avg",
                    per_year,
                )
            )
        ax.set_title(region_name)
        ax.set_xlabel("Frequency (cycles/year)")
        ax.set_ylabel("PSD")
        ax.grid(True, which="both", alpha=0.3)

    handles, labels, ncol = _legend_truth_first(
        *axes[0].get_legend_handles_labels(), truth_labels={"DUACS 5-day avg"}
    )
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=11)
    fig.suptitle(
        "Global and Regional Surface Geostrophic KE Temporal Spectra vs DUACS\n"
        f"interannual geometric mean $\\pm$1 log-std over {y0}–{y1} "
        f"({velocity_label}, per-year Welch)",
        y=0.975,
        fontsize=13,
        linespacing=1.15,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.18, wspace=0.28)
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


def _plot_scalar_anomaly_timeseries(
    obs_series: pd.Series,
    sim_series: dict[str, pd.Series],
    emulators: dict[str, dict],
    title: str,
    units: str,
    metric_prefix: str,
    period_label: str,
    obs_label: str,
) -> pd.DataFrame:
    fig, ax = plt.subplots(1, 1, figsize=(11.5, 4.2))
    obs_anom = obs_series.astype(float) - obs_series.astype(float).mean()
    ax.plot(obs_anom.index, obs_anom.values, "--", color="black", linewidth=2, label=obs_label)

    rows = []
    for label, series in sim_series.items():
        sim_anom = series.astype(float) - series.astype(float).mean()
        rmse = _rmse_between_series(obs_anom, sim_anom)
        rows.append({"model": label, "metric": f"{metric_prefix}_mean_removed_anomaly_rmse", "value": rmse})
        ax.plot(sim_anom.index, sim_anom.values, "-", color=emulators[label]["color"], linewidth=2, label=label)

    ax.set_ylabel(f"Anomaly ({units})", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.3)
    handles, labels, ncol = _legend_truth_first(*ax.get_legend_handles_labels(), truth_labels={obs_label})
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=12)
    _finish_timeseries_figure(fig, [ax], f"{title} Mean-Removed Anomaly vs {obs_label}", period_label, "Time")
    return pd.DataFrame(rows)


def _plot_scalar_trend_comparison(
    obs_series: pd.Series,
    sim_series: dict[str, pd.Series],
    emulators: dict[str, dict],
    title: str,
    units: str,
    metric_prefix: str,
    period_label: str,
    obs_label: str,
) -> pd.DataFrame:
    def centered_trend(trend: pd.Series) -> pd.Series:
        values = trend.astype(float)
        finite = np.isfinite(values.values)
        if not finite.any():
            return values
        return values - values.values[finite][0]

    def legend_label(label: str, slope: float) -> str:
        if not np.isfinite(slope):
            return f"{label}: slope=nan {units} yr$^{{-1}}$"
        return f"{label}: slope={slope:.3g} {units} yr$^{{-1}}$"

    obs_trend, obs_slope = _series_linear_trend_fit_per_year(obs_series)
    obs_plot_label = legend_label(obs_label, obs_slope)
    fig, ax = plt.subplots(1, 1, figsize=(11.5, 4.8))
    ax.plot(
        obs_trend.index,
        centered_trend(obs_trend).values,
        "--",
        color="black",
        linewidth=2.5,
        label=obs_plot_label,
    )

    rows = []
    for label, series in sim_series.items():
        trend, slope = _series_linear_trend_fit_per_year(series)
        rows.append({"model": label, "metric": f"{metric_prefix}_linear_trend_per_year", "value": slope})
        rows.append({"model": label, "metric": f"{metric_prefix}_linear_trend_bias_per_year", "value": slope - obs_slope})
        ax.plot(
            trend.index,
            centered_trend(trend).values,
            "-",
            color=emulators[label]["color"],
            linewidth=2.5,
            label=legend_label(label, slope),
        )

    ax.axhline(0.0, color="0.25", linewidth=1)
    ax.set_ylabel(f"OLS fitted trend change ({units})", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time", fontsize=13)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=1,
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
        fontsize=11,
    )
    fig.suptitle(
        _figure_title_with_context(
            f"{title} OLS Linear Trend Fit Comparison",
            period_label,
            "fitted lines only; each line centered at its first fitted value",
        ),
        y=0.955,
        fontsize=15,
        linespacing=1.15,
    )
    fig.tight_layout(rect=[0.025, 0.25, 0.99, 0.90])
    return pd.DataFrame(rows)


def _plot_scalar_detrended_deseasonalized_anomaly_timeseries(
    obs_series: pd.Series,
    sim_series: dict[str, pd.Series],
    emulators: dict[str, dict],
    title: str,
    units: str,
    metric_prefix: str,
    period_label: str,
    obs_label: str,
) -> pd.DataFrame:
    fig, ax = plt.subplots(1, 1, figsize=(11.5, 4.2))
    obs_resid = _series_detrended_deseasonalized_anomaly(obs_series)
    ax.plot(obs_resid.index, obs_resid.values, "--", color="black", linewidth=2, label=obs_label)

    rows = []
    for label, series in sim_series.items():
        sim_resid = _series_detrended_deseasonalized_anomaly(series)
        rmse = _rmse_between_series(obs_resid, sim_resid)
        rows.append(
            {
                "model": label,
                "metric": f"{metric_prefix}_detrended_deseasonalized_anomaly_rmse",
                "value": rmse,
            }
        )
        ax.plot(sim_resid.index, sim_resid.values, "-", color=emulators[label]["color"], linewidth=2, label=label)

    ax.set_ylabel(f"Residual anomaly ({units})", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.3)
    handles, labels, ncol = _legend_truth_first(*ax.get_legend_handles_labels(), truth_labels={obs_label})
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=12)
    _finish_timeseries_figure(
        fig,
        [ax],
        f"{title} Detrended Deseasonalized Anomaly vs {obs_label}",
        period_label,
        "Time",
    )
    return pd.DataFrame(rows)


def _plot_scalar_residual_variance_comparison(
    obs_series: pd.Series,
    sim_series: dict[str, pd.Series],
    emulators: dict[str, dict],
    title: str,
    units: str,
    metric_prefix: str,
    period_label: str,
    obs_label: str,
) -> pd.DataFrame:
    obs_resid = _series_detrended_deseasonalized_anomaly(obs_series)
    obs_var = _series_variance(obs_resid)
    labels = [obs_label]
    variances = [obs_var]
    colors_list = ["black"]
    rows = []

    for label, series in sim_series.items():
        sim_resid = _series_detrended_deseasonalized_anomaly(series)
        sim_var = _series_variance(sim_resid)
        labels.append(label)
        variances.append(sim_var)
        colors_list.append(emulators[label]["color"])
        rows.append({"model": label, "metric": f"{metric_prefix}_detrended_deseasonalized_anomaly_variance", "value": sim_var})
        rows.append({"model": label, "metric": f"{metric_prefix}_detrended_deseasonalized_anomaly_variance_bias", "value": sim_var - obs_var})
        rows.append({"model": label, "metric": f"{metric_prefix}_detrended_deseasonalized_anomaly_variance_ratio", "value": sim_var / obs_var if obs_var != 0 else np.nan})

    fig, ax = plt.subplots(1, 1, figsize=(max(7.0, 1.8 * len(labels)), 4.2))
    ax.bar(np.arange(len(labels)), variances, color=colors_list)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(f"Variance ({units}$^2$)", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        _figure_title_with_context(f"{title} Detrended Deseasonalized Anomaly Variance", period_label),
        y=0.96,
        fontsize=15,
    )
    fig.tight_layout(rect=[0.03, 0.03, 0.99, 0.92])
    return pd.DataFrame(rows)


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
    extra = _climatology_extra(climatology_label, period_label)
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
    extra = _climatology_extra(climatology_label, period_label)
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
    extra = _climatology_extra(climatology_label, period_label)
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
    extra = _climatology_extra(climatology_label, period_label)
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
    duacs_raw = _infer_obs_grid_metrics(
        _drop_singleton_nonspatial_dims(
            _standardize_latlon_time(_maybe_open_dataset(OBS_CONFIG["duacs_raw"]))
        )
    )
    oisst = _infer_obs_grid_metrics(
        _drop_singleton_nonspatial_dims(
            _standardize_latlon_time(_maybe_open_dataset(OBS_CONFIG["oisst"]))
        )
    )
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

    ohc_start, ohc_end = _resolve_ohc_period(
        argo_iap,
        emulators,
        args.ohc_start_date,
        args.ohc_end_date,
    )
    ohc_label = _period_label(ohc_start, ohc_end)
    ohc_slug = _period_slug(ohc_start, ohc_end)

    rmse_start = _date_or_none(args.rmse_start_date) or PRIMARY_RMSE_START
    rmse_end = _date_or_none(args.rmse_end_date) or PRIMARY_RMSE_END
    if rmse_start is not None and rmse_end is not None and rmse_start > rmse_end:
        raise ValueError(f"Invalid RMSE period: start={rmse_start}, end={rmse_end}")
    rmse_period_slug = f"{rmse_start:%Y%m}_{rmse_end:%Y%m}"

    print(f"\n[Total RMSE + spatial RMSE maps ({rmse_start:%Y}–{rmse_end:%Y} complete years)]")
    print("Evaluating primary surface geostrophic velocity vector RMSE vs DUACS...")
    velocity_rmse_full = _plot_surface_velocity_rmse_maps(
        duacs,
        emulators,
        start=rmse_start,
        end=rmse_end,
        period_kind="primary_complete_years",
        include_annual=True,
        velocity_kind=args.duacs_velocity_kind,
    )
    save_current_fig(
        f"surface_geostrophic_velocity_vector_rmse_vs_duacs_{rmse_period_slug}", RMSE_DIR
    )

    print("Evaluating primary instantaneous surface EKE RMSE vs DUACS...")
    eke_rmse_full = _plot_eke_rmse_maps(
        duacs,
        emulators,
        start=rmse_start,
        end=rmse_end,
        period_kind="primary_complete_years",
        include_annual=True,
        velocity_kind=args.duacs_velocity_kind,
    )
    save_current_fig(f"surface_eke_rmse_vs_duacs_{rmse_period_slug}", RMSE_DIR)

    print("Evaluating primary SST RMSE vs OISST...")
    sst_rmse_full = _plot_sst_rmse_maps(
        oisst,
        emulators,
        start=rmse_start,
        end=rmse_end,
        period_kind="primary_complete_years",
        include_annual=True,
    )
    save_current_fig(f"sst_rmse_maps_vs_oisst_{rmse_period_slug}", RMSE_DIR)

    print("Evaluating primary OHC RMSE vs ARGO-IAP...")
    ohc_rmse_full = _plot_ohc_rmse_maps(
        argo_iap,
        emulators,
        start=rmse_start,
        end=rmse_end,
        period_kind="primary_complete_years",
        include_annual=True,
    )
    save_current_fig(f"ohc_per_area_rmse_maps_vs_argo_iap_{rmse_period_slug}", RMSE_DIR)

    snapshot_start = pd.Timestamp(f"{args.year}-01-01")
    snapshot_end = pd.Timestamp(f"{args.year}-12-31")
    print(f"\n[Retained RMSE snapshot ({args.year})]")
    velocity_rmse_snapshot = _plot_surface_velocity_rmse_maps(
        duacs,
        emulators,
        start=snapshot_start,
        end=snapshot_end,
        period_kind="snapshot",
        velocity_kind=args.duacs_velocity_kind,
    )
    save_current_fig(f"surface_geostrophic_velocity_vector_rmse_vs_duacs_{args.year}", RMSE_DIR)

    eke_rmse_snapshot = _plot_eke_rmse_maps(
        duacs,
        emulators,
        start=snapshot_start,
        end=snapshot_end,
        period_kind="snapshot",
        velocity_kind=args.duacs_velocity_kind,
    )
    save_current_fig(f"surface_eke_rmse_vs_duacs_{args.year}", RMSE_DIR)

    sst_rmse_snapshot = _plot_sst_rmse_maps(
        oisst,
        emulators,
        start=snapshot_start,
        end=snapshot_end,
        period_kind="snapshot",
    )
    save_current_fig(f"sst_rmse_maps_vs_oisst_{args.year}", RMSE_DIR)

    ohc_rmse_snapshot = _plot_ohc_rmse_maps(
        argo_iap,
        emulators,
        start=snapshot_start,
        end=snapshot_end,
        period_kind="snapshot",
    )
    save_current_fig(f"ohc_per_area_rmse_maps_vs_argo_iap_{args.year}", RMSE_DIR)

    rmse_metrics = pd.concat(
        [
            velocity_rmse_full,
            eke_rmse_full,
            sst_rmse_full,
            ohc_rmse_full,
            velocity_rmse_snapshot,
            eke_rmse_snapshot,
            sst_rmse_snapshot,
            ohc_rmse_snapshot,
        ],
        ignore_index=True,
        sort=False,
    )
    _plot_annual_rmse_timeseries(rmse_metrics, emulators)
    save_current_fig("annual_total_rmse_timeseries_vs_observations", RMSE_DIR)

    print("\n[Velocities]")
    print("Evaluating EKE maps vs DUACS...")
    eke_metrics = _plot_eke_map_comparison(
        duacs,
        emulators,
        args.year,
        velocity_kind=args.duacs_velocity_kind,
    )
    save_current_fig(f"eke_mean_vs_duacs_{args.year}", MAP_DIR)

    print("Evaluating EKE spatial spectra vs DUACS (North Pacific, Gulf Stream, Agulhas)...")
    eke_spatial_spectrum_metrics = _plot_eke_spatial_spectra(
        duacs,
        emulators,
        velocity_kind=args.duacs_velocity_kind,
    )
    save_current_fig("eke_spatial_spectra_open_ocean_boxes_vs_duacs", SPECTRUM_DIR)

    print("Evaluating interannual EKE spatial spectra and log-standard-deviation bands...")
    eke_spatial_interannual_metrics = _plot_eke_spatial_spectra_interannual(
        duacs,
        emulators,
        velocity_kind=args.duacs_velocity_kind,
    )
    save_current_fig("eke_spatial_spectra_interannual_vs_duacs", SPECTRUM_DIR)

    print("Evaluating temporal KE spectra vs raw and 5-day DUACS (Global, Gulf Stream, Agulhas)...")
    ke_spec_metrics = _plot_temporal_ke_spectra(
        duacs,
        duacs_raw,
        emulators,
        velocity_kind=args.duacs_velocity_kind,
    )
    save_current_fig("ke_temporal_spectra_vs_duacs", SPECTRUM_DIR)

    print("Evaluating interannual temporal KE spectra and log-standard-deviation bands...")
    ke_spec_interannual_metrics = _plot_temporal_ke_spectra_interannual(
        duacs,
        emulators,
        velocity_kind=args.duacs_velocity_kind,
    )
    save_current_fig("ke_temporal_spectra_interannual_vs_duacs", SPECTRUM_DIR)

    print("\n[Temperature]")
    print("Evaluating isotropic SST spectra vs 5-day OISST (North Pacific, Gulf Stream, Agulhas)...")
    temp_spec_metrics = _plot_temperature_spectra(oisst, emulators)
    save_current_fig("temperature_isotropic_spectra_vs_oisst", SPECTRUM_DIR)

    print("Evaluating interannual SST spatial spectra and log-standard-deviation bands...")
    temp_spec_interannual_metrics = _plot_temperature_spectra_interannual(oisst, emulators)
    save_current_fig("temperature_isotropic_spectra_interannual_vs_oisst", SPECTRUM_DIR)

    print(f"Evaluating monthly total OHC for upper 700m and 700-2000m vs ARGO-IAP over {ohc_label}...")
    obs_ohc, sim_ohc = _collect_ohc_timeseries(argo_iap, emulators, ohc_start, ohc_end)
    ohc_metrics = _plot_ohc_timeseries(obs_ohc, sim_ohc, emulators, ohc_label)
    save_current_fig(f"ohc_monthly_vs_argo_iap_{ohc_slug}", TIMESERIES_DIR)

    print("Evaluating annual-mean OHC maps for upper 700m and 700-2000m vs ARGO-IAP...")
    ohc_map_metrics = _plot_argo_ohc_layer_maps(argo_iap, emulators, args.year)
    save_current_fig(f"ohc_layer_integrated_maps_vs_argo_iap_{args.year}", MAP_DIR)

    print("Evaluating annual-mean OHC point-wise bias/RMSE maps vs ARGO-IAP...")
    _plot_argo_ohc_layer_bias_maps(argo_iap, emulators, args.year)
    save_current_fig(f"ohc_layer_integrated_bias_maps_vs_argo_iap_{args.year}", MAP_DIR)

    print("\n[Anomalies]")
    anomaly_metrics = []

    print("Evaluating SST time-series anomalies, trends, and residual variance vs OISST...")
    obs_sst_series, sim_sst_series, sst_label = _collect_sst_timeseries(oisst, emulators)
    anomaly_metrics.append(
        _plot_scalar_anomaly_timeseries(
            obs_sst_series, sim_sst_series, emulators, "Global SST", "$^\\circ$C",
            "global_sst", sst_label, "OISST"
        )
    )
    save_current_fig("global_sst_mean_removed_anomaly_vs_oisst", TIMESERIES_DIR)
    anomaly_metrics.append(
        _plot_scalar_trend_comparison(
            obs_sst_series, sim_sst_series, emulators, "Global SST", "$^\\circ$C",
            "global_sst", sst_label, "OISST"
        )
    )
    save_current_fig("global_sst_linear_trend_comparison_vs_oisst", TIMESERIES_DIR)
    anomaly_metrics.append(
        _plot_scalar_detrended_deseasonalized_anomaly_timeseries(
            obs_sst_series, sim_sst_series, emulators, "Global SST", "$^\\circ$C",
            "global_sst", sst_label, "OISST"
        )
    )
    save_current_fig("global_sst_detrended_deseasonalized_anomaly_vs_oisst", TIMESERIES_DIR)
    anomaly_metrics.append(
        _plot_scalar_residual_variance_comparison(
            obs_sst_series, sim_sst_series, emulators, "Global SST", "$^\\circ$C",
            "global_sst", sst_label, "OISST"
        )
    )
    save_current_fig("global_sst_residual_anomaly_variance_vs_oisst", VARIANCE_DIR)

    print("Evaluating EKE time-series anomalies, trends, and residual variance vs DUACS...")
    obs_eke_series, sim_eke_series, eke_label = _collect_eke_timeseries(
        duacs, emulators, velocity_kind=args.duacs_velocity_kind
    )
    anomaly_metrics.append(
        _plot_scalar_anomaly_timeseries(
            obs_eke_series, sim_eke_series, emulators, "Global Surface EKE", "m$^2$ s$^{-2}$",
            "global_surface_eke", eke_label, "DUACS"
        )
    )
    save_current_fig("global_surface_eke_mean_removed_anomaly_vs_duacs", TIMESERIES_DIR)
    anomaly_metrics.append(
        _plot_scalar_trend_comparison(
            obs_eke_series, sim_eke_series, emulators, "Global Surface EKE", "m$^2$ s$^{-2}$",
            "global_surface_eke", eke_label, "DUACS"
        )
    )
    save_current_fig("global_surface_eke_linear_trend_comparison_vs_duacs", TIMESERIES_DIR)
    anomaly_metrics.append(
        _plot_scalar_detrended_deseasonalized_anomaly_timeseries(
            obs_eke_series, sim_eke_series, emulators, "Global Surface EKE", "m$^2$ s$^{-2}$",
            "global_surface_eke", eke_label, "DUACS"
        )
    )
    save_current_fig("global_surface_eke_detrended_deseasonalized_anomaly_vs_duacs", TIMESERIES_DIR)
    anomaly_metrics.append(
        _plot_scalar_residual_variance_comparison(
            obs_eke_series, sim_eke_series, emulators, "Global Surface EKE", "m$^2$ s$^{-2}$",
            "global_surface_eke", eke_label, "DUACS"
        )
    )
    save_current_fig("global_surface_eke_residual_anomaly_variance_vs_duacs", VARIANCE_DIR)

    print("Evaluating upper-700m OHC time-series anomalies, trends, and residual variance vs ARGO-IAP...")
    upper_title = OHC_UPPER_700_SLICE["title"]
    obs_ohc_upper_series = obs_ohc.sel(depth_slice=upper_title).to_series()
    sim_ohc_upper_series = {
        label: sim.sel(depth_slice=upper_title).to_series()
        for label, sim in sim_ohc.items()
    }
    anomaly_metrics.append(
        _plot_scalar_anomaly_timeseries(
            obs_ohc_upper_series, sim_ohc_upper_series, emulators, "Global Upper-700m OHC", "J",
            "global_ohc_upper700", ohc_label, "ARGO-IAP"
        )
    )
    save_current_fig(f"global_ohc_upper700_mean_removed_anomaly_vs_argo_iap_{ohc_slug}", TIMESERIES_DIR)
    anomaly_metrics.append(
        _plot_scalar_trend_comparison(
            obs_ohc_upper_series, sim_ohc_upper_series, emulators, "Global Upper-700m OHC", "J",
            "global_ohc_upper700", ohc_label, "ARGO-IAP"
        )
    )
    save_current_fig(f"global_ohc_upper700_linear_trend_comparison_vs_argo_iap_{ohc_slug}", TIMESERIES_DIR)
    anomaly_metrics.append(
        _plot_scalar_detrended_deseasonalized_anomaly_timeseries(
            obs_ohc_upper_series, sim_ohc_upper_series, emulators, "Global Upper-700m OHC", "J",
            "global_ohc_upper700", ohc_label, "ARGO-IAP"
        )
    )
    save_current_fig(f"global_ohc_upper700_detrended_deseasonalized_anomaly_vs_argo_iap_{ohc_slug}", TIMESERIES_DIR)
    anomaly_metrics.append(
        _plot_scalar_residual_variance_comparison(
            obs_ohc_upper_series, sim_ohc_upper_series, emulators, "Global Upper-700m OHC", "J",
            "global_ohc_upper700", ohc_label, "ARGO-IAP"
        )
    )
    save_current_fig(f"global_ohc_upper700_residual_anomaly_variance_vs_argo_iap_{ohc_slug}", VARIANCE_DIR)

    print("Evaluating SST residual-anomaly variance maps vs OISST...")
    sst_variance_map_metrics = _plot_sst_residual_variance_maps(oisst, emulators)
    save_current_fig("sst_residual_anomaly_variance_maps_vs_oisst", VARIANCE_DIR)

    print("Evaluating upper-700m OHC residual-anomaly variance maps vs ARGO-IAP...")
    ohc_upper_variance_map_metrics = _plot_ohc_upper700_residual_variance_maps(
        argo_iap, emulators, ohc_start, ohc_end, ohc_label
    )
    save_current_fig(f"ohc_upper700_residual_anomaly_variance_maps_vs_argo_iap_{ohc_slug}", VARIANCE_DIR)

    variance_map_metrics = pd.concat(
        [sst_variance_map_metrics, ohc_upper_variance_map_metrics],
        ignore_index=True,
        sort=False,
    )
    _plot_annual_variance_map_rmse_timeseries(variance_map_metrics, emulators)
    save_current_fig("annual_residual_variance_map_rmse_timeseries", VARIANCE_DIR)

    rmse_metrics = _attach_ohc_convention_metadata(rmse_metrics)
    metric_frames = [
        rmse_metrics,
        eke_metrics,
        eke_spatial_spectrum_metrics,
        eke_spatial_interannual_metrics,
        ke_spec_metrics,
        ke_spec_interannual_metrics,
        temp_spec_metrics,
        temp_spec_interannual_metrics,
        ohc_metrics,
        ohc_map_metrics,
        *anomaly_metrics,
        sst_variance_map_metrics,
        ohc_upper_variance_map_metrics,
    ]
    metric_frames = [frame for frame in metric_frames if frame is not None and not frame.empty]
    metrics = _attach_ohc_convention_metadata(pd.concat(metric_frames, ignore_index=True))
    rmse_metrics_path = os.path.join(
        METRICS_DIR,
        f"rmse_metrics_primary_{rmse_period_slug}_and_{args.year}_snapshot.csv",
    )
    rmse_metrics.to_csv(rmse_metrics_path, index=False)
    print(f"  -> Saved RMSE metrics and uncertainty: {rmse_metrics_path}")
    metrics_path = os.path.join(METRICS_DIR, f"observation_metrics_{args.year}.csv")
    metrics.to_csv(metrics_path, index=False)
    print(f"  -> Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
