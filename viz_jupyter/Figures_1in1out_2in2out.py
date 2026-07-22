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
parser.add_argument('--eval-year', type=int, default=2022,
                    help='Year used for RMSE maps and annual map diagnostics')
parser.add_argument('--legacy-extra', action='store_true',
                    help='After the four-part evaluation, also run the older extra plotting sections')
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
from scipy import signal
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
RMSE_DIR = os.path.join(PDF_DIR, "rmse")
MAP_DIR = os.path.join(PDF_DIR, "maps")
METRICS_DIR = os.path.join(PDF_DIR, "metrics")
ZONAL_BIAS_DIR = os.path.join(PDF_DIR, "zonal_bias")
SPATIAL_BIAS_DIR = os.path.join(PDF_DIR, "spatial_bias")
ZONAL_MEAN_DIR = os.path.join(PDF_DIR, "zonal_mean")
SPATIAL_DISTRIBUTION_DIR = os.path.join(PDF_DIR, "spatial_distribution")

for d in [SPECTRUM_DIR, TIMESERIES_DIR, VARIANCE_DIR, RMSE_DIR, MAP_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# Four-part evaluation helpers
# ============================================================
EVAL_SPECTRUM_REGIONS = [
    ("Gulf Stream", slice(300, 320), slice(25, 45)),
    ("Agulhas", slice(40, 60), slice(-50, -30)),
]
EVAL_SPATIAL_SPECTRUM_PANELS = [
    ("North Pacific", slice(170, 200), slice(25, 45)),
    *EVAL_SPECTRUM_REGIONS,
]
EVAL_TEMPORAL_SPECTRUM_PANELS = [
    ("Global", slice(0, 360), slice(-90, 90)),
    *EVAL_SPECTRUM_REGIONS,
]
EVAL_OHC_DEPTH_SLICES = DEPTH_SLICES[:2]
EVAL_OHC_UPPER_700_SLICE = EVAL_OHC_DEPTH_SLICES[0]
SPATIAL_DIMS = ("y", "x")


def _metric_row(model, section, metric, value, **extra):
    row = {"model": model, "section": section, "metric": metric, "value": value}
    row.update(extra)
    return row


def _time_label(ds):
    time = ds.time.values
    if len(time) == 0:
        return "empty"
    return f"{str(time[0])[:10]} to {str(time[-1])[:10]}"


def _select_year(ds, year):
    return ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))


def _align_time(*arrays):
    aligned = xr.align(*arrays, join="inner")
    if aligned[0].sizes.get("time", 0) == 0:
        raise ValueError("No common time samples after alignment")
    return aligned


def _surface(ds, var_name):
    return ds[var_name].isel(lev=0)


def _area_weighted_mean(field, area):
    mask = np.isfinite(field)
    weights = area.where(mask)
    denom = weights.sum(SPATIAL_DIMS, skipna=True)
    if float(denom.values) == 0.0:
        return float("nan")
    return float(((field.where(mask) * weights).sum(SPATIAL_DIMS, skipna=True) / denom).values)


def _area_weighted_map_rmse(diff_map, area):
    return float(np.sqrt(_area_weighted_mean(diff_map ** 2, area)))


def _field_rmse_over_time(model, truth):
    truth_a, model_a = _align_time(truth, model)
    return np.sqrt(((model_a - truth_a) ** 2).mean("time", skipna=True))


def _series_rmse(truth_series, model_series):
    common = truth_series.index.intersection(model_series.index)
    if common.empty:
        return float("nan")
    diff = model_series.loc[common].astype(float) - truth_series.loc[common].astype(float)
    diff = diff[np.isfinite(diff)]
    return float(np.sqrt(np.mean(diff ** 2))) if len(diff) else float("nan")


def _pattern_corr(a, b):
    a_vals = np.asarray(a.values).ravel()
    b_vals = np.asarray(b.values).ravel()
    valid = np.isfinite(a_vals) & np.isfinite(b_vals)
    if valid.sum() < 2:
        return float("nan")
    return float(np.corrcoef(a_vals[valid], b_vals[valid])[0, 1])


def _vector_pattern_corr(model_u, model_v, truth_u, truth_v):
    model_vals = np.concatenate([
        np.asarray(model_u.values).ravel(),
        np.asarray(model_v.values).ravel(),
    ])
    truth_vals = np.concatenate([
        np.asarray(truth_u.values).ravel(),
        np.asarray(truth_v.values).ravel(),
    ])
    valid = np.isfinite(model_vals) & np.isfinite(truth_vals)
    if valid.sum() < 2:
        return float("nan")
    return float(np.corrcoef(model_vals[valid], truth_vals[valid])[0, 1])


def _format_rmse_corr(rmse=None, corr=None):
    parts = []
    if rmse is not None:
        parts.append(f"RMSE = {rmse:.3g}")
    if corr is not None:
        parts.append(f"Corr = {corr:.3f}" if np.isfinite(corr) else "Corr = nan")
    return "\n".join(parts)


def _add_map_features(ax):
    ax.set_global()
    ax.add_feature(cfeature.LAND, zorder=10, edgecolor="black", linewidth=0.4, facecolor="lightgray")
    ax.coastlines(zorder=11, linewidth=0.5)


def _plot_single_row_maps(fields, title, units, cmap, vmin=None, vmax=None, extend="max", annotations=None):
    ncols = len(fields)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(5.4 * ncols, 4.5),
        squeeze=False,
        subplot_kw={"projection": ccrs.Robinson(central_longitude=210)},
    )
    axes = axes[0]
    if vmin is None or vmax is None:
        flat = np.concatenate([np.asarray(f.values).ravel()[np.isfinite(np.asarray(f.values).ravel())] for f in fields.values()])
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = float(np.nanpercentile(flat, 98)) if flat.size else 1.0
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mesh = None
    for ax, (label, field) in zip(axes, fields.items()):
        mesh = field.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            x="x",
            y="y",
            cmap=cmap,
            norm=norm,
            add_colorbar=False,
            rasterized=True,
            add_labels=False,
        )
        _add_map_features(ax)
        ax.set_title(label, fontsize=12, fontweight="bold", y=1.08)
        if annotations and label in annotations:
            ax.text(
                0.5,
                1.01,
                annotations[label],
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=9.5,
            )
    fig.subplots_adjust(left=0.04, right=0.98, top=0.80, bottom=0.24, wspace=0.08)
    cax = fig.add_axes([0.28, 0.11, 0.44, 0.035])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend=extend)
    cbar.set_label(units)
    fig.suptitle(title, y=0.96, fontsize=15)


def _plot_layer_maps(field_by_model, title, units, cmap, annotations_by_layer=None, centered=False):
    layer_titles = list(next(iter(field_by_model.values())).keys())
    model_labels = list(field_by_model.keys())
    nrows = len(layer_titles)
    ncols = len(model_labels)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols, 3.8 * nrows),
        squeeze=False,
        subplot_kw={"projection": ccrs.Robinson(central_longitude=210)},
        gridspec_kw={"hspace": 0.42, "wspace": 0.08},
    )
    row_meshes = []
    for row_idx, layer_title in enumerate(layer_titles):
        row_fields = [field_by_model[label][layer_title] for label in model_labels]
        flat = np.concatenate([np.asarray(f.values).ravel()[np.isfinite(np.asarray(f.values).ravel())] for f in row_fields])
        if centered:
            limit = float(np.nanpercentile(np.abs(flat), 98)) if flat.size else 1.0
            norm = colors.Normalize(vmin=-limit, vmax=limit)
            extend = "both"
        else:
            vmax = float(np.nanpercentile(flat, 98)) if flat.size else 1.0
            norm = colors.Normalize(vmin=0.0, vmax=vmax)
            extend = "max"
        mesh = None
        for col_idx, label in enumerate(model_labels):
            ax = axes[row_idx, col_idx]
            field = field_by_model[label][layer_title]
            mesh = field.plot.pcolormesh(
                ax=ax,
                transform=ccrs.PlateCarree(),
                x="x",
                y="y",
                cmap=cmap,
                norm=norm,
                add_colorbar=False,
                rasterized=True,
                add_labels=False,
            )
            _add_map_features(ax)
            if row_idx == 0:
                ax.set_title(label, fontsize=12, fontweight="bold", y=1.12)
            if col_idx == 0:
                ax.text(
                    -0.08,
                    0.5,
                    layer_title,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=11,
                )
            if annotations_by_layer:
                text = annotations_by_layer.get(layer_title, {}).get(label)
                if text:
                    ax.text(0.5, 1.02, text, transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
        row_meshes.append((mesh, extend))
    fig.suptitle(title, y=0.98, fontsize=15)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.10, hspace=0.55, wspace=0.08)
    for row_idx, (mesh, extend) in enumerate(row_meshes):
        row_axes = axes[row_idx, :]
        positions = [ax.get_position() for ax in row_axes]
        x0 = min(pos.x0 for pos in positions)
        x1 = max(pos.x1 for pos in positions)
        y0 = min(pos.y0 for pos in positions)
        width = x1 - x0
        cax = fig.add_axes([x0 + 0.18 * width, max(0.02, y0 - 0.052), 0.64 * width, 0.026])
        fig.colorbar(mesh, cax=cax, orientation="horizontal", extend=extend, label=units)


def _eke_field_from_uv(u, v):
    u_anom = u - u.mean("time", skipna=True)
    v_anom = v - v.mean("time", skipna=True)
    return 0.5 * (u_anom ** 2 + v_anom ** 2)


def _eke_map_from_uv(u, v):
    return _eke_field_from_uv(u, v).mean("time", skipna=True)


def _eke_map(ds):
    u = _surface(ds, "uo")
    v = _surface(ds, "vo")
    return _eke_map_from_uv(u, v)


def _eke_timeseries(ds):
    u = _surface(ds, "uo")
    v = _surface(ds, "vo")
    u_anom = u - u.mean("time")
    v_anom = v - v.mean("time")
    eke = 0.5 * (u_anom ** 2 + v_anom ** 2)
    return _area_weighted_timeseries(eke, ds["areacello"]).to_series()


def _area_weighted_timeseries(field, area):
    mask = np.isfinite(field)
    weights = area.where(mask)
    return (field.where(mask) * weights).sum(SPATIAL_DIMS, skipna=True) / weights.sum(SPATIAL_DIMS, skipna=True)


def _ohc_per_area_field(ds, depth_slice):
    thetao = ds["thetao"].sel(lev=slice(depth_slice["min"], depth_slice["max"]))
    dz = ds["dz"].sel(lev=slice(depth_slice["min"], depth_slice["max"]))
    valid = np.isfinite(thetao).any("lev")
    return (thetao * dz * RHO * CP).sum("lev", skipna=True).where(valid)


def _ohc_total_series(ds, depth_slice):
    ohc_area = _ohc_per_area_field(ds, depth_slice)
    total = (ohc_area * ds["areacello"]).sum(SPATIAL_DIMS, skipna=True)
    return total.resample(time="MS").mean().to_series()


def _global_sst_series(ds):
    return _area_weighted_timeseries(_surface(ds, "thetao"), ds["areacello"]).to_series()


def _monthly(da):
    return da.resample(time="MS").mean()


def _calendar_day_anomaly(da):
    clim = da.groupby("time.dayofyear").mean("time")
    anom = da.groupby("time.dayofyear") - clim
    return anom.drop_vars("dayofyear") if "dayofyear" in anom.coords else anom


def _datetime_index(index):
    try:
        return pd.DatetimeIndex(index)
    except Exception:
        return pd.to_datetime([str(value) for value in index])


def _median_time_step_days(index):
    dt_index = _datetime_index(index)
    if len(dt_index) < 2:
        return float("nan")
    delta = np.diff(dt_index.values).astype("timedelta64[s]").astype(float) / 86400.0
    delta = delta[np.isfinite(delta) & (delta > 0)]
    return float(np.median(delta)) if delta.size else float("nan")


def _pentad_index(index):
    dt_index = _datetime_index(index)
    return ((dt_index.dayofyear.to_numpy() - 1) // 5 + 1).astype(int)


def _series_without_seasonal_cycle(series):
    values = series.astype(float)
    if values.empty:
        return values
    if _median_time_step_days(values.index) <= 10:
        pentad = pd.Series(_pentad_index(values.index), index=values.index)
        counts = pentad.value_counts()
        if (counts >= 2).sum() >= 24:
            return values - pentad.map(values.groupby(pentad).mean())
    month = pd.Series(_datetime_index(values.index).month, index=values.index)
    counts = month.value_counts()
    if len(counts) == 12 and (counts >= 2).all():
        return values - month.map(values.groupby(month).mean())
    return values - values.mean()


def _series_linear_trend_fit_per_year(series):
    values = series.astype(float)
    finite = np.isfinite(values.values)
    if finite.sum() < 2:
        return pd.Series(np.nan, index=values.index), float("nan")
    dt_index = _datetime_index(values.index)
    x = (dt_index - dt_index[0]).days.to_numpy(dtype=float) / 365.25
    slope, intercept = np.polyfit(x[finite], values.values[finite], 1)
    return pd.Series(slope * x + intercept, index=values.index), float(slope)


def _series_without_linear_trend(series):
    trend, _ = _series_linear_trend_fit_per_year(series)
    return series.astype(float) - trend


def _series_residual_anomaly(series):
    return _series_without_linear_trend(_series_without_seasonal_cycle(series))


def _series_variance(series):
    values = np.asarray(series.values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.var(values)) if values.size else float("nan")


def _detrend_linear_dataarray(field, time_dim="time"):
    x = xr.DataArray(np.arange(field.sizes[time_dim], dtype=float), dims=[time_dim], coords={time_dim: field[time_dim]})
    finite = np.isfinite(field)
    x_mean = x.where(finite).mean(time_dim, skipna=True)
    y_mean = field.mean(time_dim, skipna=True)
    cov = ((x - x_mean) * (field - y_mean)).where(finite).sum(time_dim, skipna=True)
    var = ((x - x_mean) ** 2).where(finite).sum(time_dim, skipna=True)
    slope = xr.where(var > 0, cov / var, 0.0)
    intercept = y_mean - slope * x_mean
    return field - (slope * x + intercept)


def _detrended_deseasonalized_dataarray(field, time_dim="time"):
    if field.sizes.get(time_dim, 0) == 0:
        return field
    time_index = field[time_dim].values
    if _median_time_step_days(time_index) <= 10:
        pentad_values = _pentad_index(time_index)
        counts = pd.Series(pentad_values).value_counts()
        if (counts >= 2).sum() >= 24:
            pentad = xr.DataArray(pentad_values, dims=[time_dim], coords={time_dim: field[time_dim]}, name="pentad")
            with_pentad = field.assign_coords(pentad=pentad)
            deseasoned = with_pentad.groupby("pentad") - with_pentad.groupby("pentad").mean(time_dim, skipna=True)
            if "pentad" in deseasoned.coords:
                deseasoned = deseasoned.drop_vars("pentad")
            return _detrend_linear_dataarray(deseasoned, time_dim=time_dim)
    month = xr.DataArray(_datetime_index(time_index).month, dims=[time_dim], coords={time_dim: field[time_dim]}, name="month")
    with_month = field.assign_coords(month=month)
    counts = pd.Series(month.values).value_counts()
    if len(counts) == 12 and (counts >= 2).all():
        deseasoned = with_month.groupby("month") - with_month.groupby("month").mean(time_dim, skipna=True)
        if "month" in deseasoned.coords:
            deseasoned = deseasoned.drop_vars("month")
    else:
        deseasoned = field - field.mean(time_dim, skipna=True)
    return _detrend_linear_dataarray(deseasoned, time_dim=time_dim)


def _curve_rmse(x_ref, y_ref, x_other, y_other):
    x_ref = np.asarray(x_ref, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    x_other = np.asarray(x_other, dtype=float)
    y_other = np.asarray(y_other, dtype=float)
    ref_valid = np.isfinite(x_ref) & np.isfinite(y_ref)
    other_valid = np.isfinite(x_other) & np.isfinite(y_other)
    if ref_valid.sum() == 0 or other_valid.sum() < 2:
        return float("nan")
    order = np.argsort(x_other[other_valid])
    xo = x_other[other_valid][order]
    yo = y_other[other_valid][order]
    domain = ref_valid & (x_ref >= xo[0]) & (x_ref <= xo[-1])
    if domain.sum() == 0:
        return float("nan")
    interp = np.interp(x_ref[domain], xo, yo)
    valid = np.isfinite(y_ref[domain]) & np.isfinite(interp)
    return float(np.sqrt(np.mean((y_ref[domain][valid] - interp[valid]) ** 2))) if valid.sum() else float("nan")


def _fully_observed_spectrum_box(field, min_size=8):
    if "y" not in field.dims or "x" not in field.dims:
        raise ValueError("Spectrum field must contain 'y' and 'x' dimensions")
    if field.sizes["y"] < min_size or field.sizes["x"] < min_size:
        raise ValueError(
            f"Spatial-spectrum box has only {field.sizes['y']} x {field.sizes['x']} points "
            f"(minimum {min_size} x {min_size})."
        )
    if not bool(np.isfinite(field).all().compute().item()):
        raise ValueError(
            "Spatial spectrum requires a fully observed open-ocean rectangle; "
            "the selected box contains land or missing values."
        )
    return field


def _mean_isotropic_spectrum(field, dx, dy, batch_size=32):
    ntime = int(field.sizes.get("time", 0))
    if ntime == 0:
        return np.array([]), np.array([])
    k_ref = None
    spec_sum = None
    spec_count = None
    for start in range(0, ntime, batch_size):
        chunk = field.isel(time=slice(start, min(start + batch_size, ntime)))
        values = np.asarray(chunk.values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Spatial spectrum received non-finite values after open-ocean validation")
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
    return k_ref, spec_sum / np.maximum(spec_count, 1.0)


def _spectrum_from_surface_field(field, ds, lon_slice, lat_slice):
    data = field.sel(x=lon_slice, y=lat_slice).transpose("time", "y", "x")
    data = _calendar_day_anomaly(data)
    data = _fully_observed_spectrum_box(data)
    dx = float(ds["dx"].sel(x=data["x"], y=data["y"]).mean().values)
    dy = float(ds["dy"].sel(x=data["x"], y=data["y"]).mean().values)
    return _mean_isotropic_spectrum(data, dx, dy)


def _spectrum_from_map(field, ds, lon_slice, lat_slice):
    data = field.sel(x=lon_slice, y=lat_slice).transpose("y", "x")
    data = _fully_observed_spectrum_box(data)
    dx = float(ds["dx"].sel(x=data["x"], y=data["y"]).mean().values)
    dy = float(ds["dy"].sel(x=data["x"], y=data["y"]).mean().values)
    values = np.asarray(data.values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Spatial spectrum received non-finite map values after open-ocean validation")
    k_cent, spec = compute_isotropic_spectrum_torch(
        torch.as_tensor(values),
        dx=dx,
        dy=dy,
        n_factor=2,
        detrend="linear",
        window="hann",
        cutoff_before_bins=False,
    )
    return k_cent.cpu().numpy() * 1000 * 2 * np.pi, np.asarray(spec.cpu().numpy(), dtype=float)


def _regional_mean_series(field, area):
    mask = np.isfinite(field)
    weights = area.where(mask)
    return ((field.where(mask) * weights).sum(SPATIAL_DIMS, skipna=True) / weights.sum(SPATIAL_DIMS, skipna=True)).to_series()


def _welch_spectrum(series):
    values = series.astype(float).dropna()
    if len(values) < 4:
        return np.array([]), np.array([])
    dt_days = _median_time_step_days(values.index)
    fs = 365.25 / dt_days if np.isfinite(dt_days) and dt_days > 0 else 1.0
    freqs, psd = signal.welch(
        values.values,
        fs=fs,
        detrend="linear",
        window="hann",
        scaling="density",
        nperseg=min(256, len(values)),
    )
    valid = freqs > 0
    return freqs[valid], psd[valid]


def _ke_temporal_spectrum(ds, lon_slice, lat_slice):
    u = _surface(ds, "uo").sel(x=lon_slice, y=lat_slice)
    v = _surface(ds, "vo").sel(x=lon_slice, y=lat_slice)
    ke = 0.5 * (u ** 2 + v ** 2)
    area = ds["areacello"].sel(x=lon_slice, y=lat_slice)
    return _welch_spectrum(_regional_mean_series(ke, area))


def _finish_line_figure(fig, ax, title):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), bbox_to_anchor=(0.5, 0.02), frameon=False)
    fig.suptitle(title, y=0.96, fontsize=15)
    fig.tight_layout(rect=[0.03, 0.12, 0.99, 0.90])


def _finish_panel_figure(fig, axes, title, bottom=0.16):
    axes = np.asarray(axes).ravel()
    handles = []
    labels = []
    seen = set()
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in seen:
                handles.append(handle)
                labels.append(label)
                seen.add(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(labels)),
            bbox_to_anchor=(0.5, 0.025),
            frameon=False,
        )
    fig.suptitle(title, y=0.96, fontsize=15)
    fig.tight_layout(rect=[0.03, bottom, 0.99, 0.90])


def _with_color(series, color):
    series = series.copy()
    series.attrs["color"] = color
    return series


def _plot_timeseries(ax, series, values=None, *args, **kwargs):
    y_values = series.values if values is None else values
    return ax.plot(_datetime_index(series.index), y_values, *args, **kwargs)


def _plot_anomaly_timeseries(series_by_model, title, units, obs_label, prefix):
    fig, ax = plt.subplots(1, 1, figsize=(11.5, 4.2))
    rows = []
    truth_anom = series_by_model[obs_label].astype(float) - series_by_model[obs_label].astype(float).mean()
    _plot_timeseries(ax, truth_anom, truth_anom.values, "--", color="black", linewidth=2, label=obs_label)
    for label, item in series_by_model.items():
        if label == obs_label:
            continue
        color = item.attrs.get("color", None) if hasattr(item, "attrs") else None
        model_anom = item.astype(float) - item.astype(float).mean()
        rows.append(_metric_row(label, "Anomalies", f"{prefix}_mean_removed_anomaly_rmse", _series_rmse(truth_anom, model_anom)))
        _plot_timeseries(ax, model_anom, model_anom.values, "-", linewidth=2, label=label, color=color)
    ax.set_ylabel(f"Anomaly ({units})")
    ax.grid(True, alpha=0.3)
    _finish_line_figure(fig, ax, f"{title} Mean-Removed Anomaly vs {obs_label}")
    return rows


def _plot_trend_comparison(series_by_model, title, units, obs_label, prefix):
    def _center(trend):
        finite = np.isfinite(trend.values)
        return trend - trend.values[finite][0] if finite.any() else trend

    def _label(label, slope):
        return f"{label}: slope={slope:.3g} {units} yr$^{{-1}}$" if np.isfinite(slope) else f"{label}: slope=nan"

    fig, ax = plt.subplots(1, 1, figsize=(11.5, 4.8))
    rows = []
    truth_trend, truth_slope = _series_linear_trend_fit_per_year(series_by_model[obs_label])
    _plot_timeseries(ax, truth_trend, _center(truth_trend).values, "--", color="black", linewidth=2.5, label=_label(obs_label, truth_slope))
    for label, item in series_by_model.items():
        if label == obs_label:
            continue
        color = item.attrs.get("color", None) if hasattr(item, "attrs") else None
        trend, slope = _series_linear_trend_fit_per_year(item)
        rows.append(_metric_row(label, "Anomalies", f"{prefix}_linear_trend_per_year", slope))
        rows.append(_metric_row(label, "Anomalies", f"{prefix}_linear_trend_bias_per_year", slope - truth_slope))
        _plot_timeseries(ax, trend, _center(trend).values, "-", linewidth=2.5, label=_label(label, slope), color=color)
    ax.axhline(0.0, color="0.25", linewidth=1)
    ax.set_ylabel(f"OLS fitted trend change ({units})")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=10)
    fig.suptitle(f"{title} OLS Linear Trend Fit Comparison", y=0.96, fontsize=15)
    fig.tight_layout(rect=[0.03, 0.25, 0.99, 0.90])
    return rows


def _plot_residual_anomaly_timeseries(series_by_model, title, units, obs_label, prefix):
    fig, ax = plt.subplots(1, 1, figsize=(11.5, 4.2))
    rows = []
    truth_resid = _series_residual_anomaly(series_by_model[obs_label])
    _plot_timeseries(ax, truth_resid, truth_resid.values, "--", color="black", linewidth=2, label=obs_label)
    for label, item in series_by_model.items():
        if label == obs_label:
            continue
        color = item.attrs.get("color", None) if hasattr(item, "attrs") else None
        resid = _series_residual_anomaly(item)
        rows.append(_metric_row(label, "Anomalies", f"{prefix}_detrended_deseasonalized_anomaly_rmse", _series_rmse(truth_resid, resid)))
        _plot_timeseries(ax, resid, resid.values, "-", linewidth=2, label=label, color=color)
    ax.set_ylabel(f"Residual anomaly ({units})")
    ax.grid(True, alpha=0.3)
    _finish_line_figure(fig, ax, f"{title} Detrended Deseasonalized Anomaly")
    return rows


def _plot_residual_variance_bars(series_by_model, title, units, obs_label, prefix):
    labels = []
    vals = []
    colors_list = []
    rows = []
    truth_var = _series_variance(_series_residual_anomaly(series_by_model[obs_label]))
    for label, item in series_by_model.items():
        labels.append(label)
        vals.append(_series_variance(_series_residual_anomaly(item)))
        colors_list.append("black" if label == obs_label else getattr(item, "attrs", {}).get("color", None))
    for label, value in zip(labels, vals):
        if label == obs_label:
            continue
        rows.append(_metric_row(label, "Anomalies", f"{prefix}_detrended_deseasonalized_anomaly_variance", value))
        rows.append(_metric_row(label, "Anomalies", f"{prefix}_detrended_deseasonalized_anomaly_variance_bias", value - truth_var))
        rows.append(_metric_row(label, "Anomalies", f"{prefix}_detrended_deseasonalized_anomaly_variance_ratio", value / truth_var if truth_var != 0 else np.nan))
    fig, ax = plt.subplots(1, 1, figsize=(max(7, 1.8 * len(labels)), 4.2))
    ax.bar(np.arange(len(labels)), vals, color=colors_list)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(f"Variance ({units}$^2$)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"{title} Detrended Deseasonalized Anomaly Variance", y=0.96, fontsize=15)
    fig.tight_layout(rect=[0.03, 0.03, 0.99, 0.92])
    return rows


def _depth_label(depth_slice):
    if depth_slice["min"] == 0 and depth_slice["max"] == 700:
        return "Upper 700 m"
    if depth_slice["min"] == 700 and depth_slice["max"] == 2000:
        return "700-2000 m"
    return depth_slice["title"]


def _plot_loglog_curve(ax, x, y, *args, **kwargs):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if valid.any():
        ax.loglog(x[valid], y[valid], *args, **kwargs)


def _run_total_rmse_evaluation(emulators_dict, eval_year):
    rows = []
    velocity_maps = {}
    velocity_ann = {}
    eke_maps = {}
    eke_ann = {}
    sst_maps = {}
    sst_ann = {}
    ohc_maps = {}
    ohc_ann = {_depth_label(depth_slice): {} for depth_slice in EVAL_OHC_DEPTH_SLICES}

    for label, emu_data in emulators_dict.items():
        truth = _select_year(emu_data["truth"], eval_year)
        model = _select_year(emu_data["ds"], eval_year)
        period = _time_label(model)
        area = truth["areacello"]

        truth_u, truth_v, model_u, model_v = _align_time(
            _surface(truth, "uo"),
            _surface(truth, "vo"),
            _surface(model, "uo"),
            _surface(model, "vo"),
        )
        velocity_rmse_map = np.sqrt(((model_u - truth_u) ** 2 + (model_v - truth_v) ** 2).mean("time", skipna=True))
        velocity_total_rmse = _area_weighted_map_rmse(velocity_rmse_map, area)
        velocity_corr = _vector_pattern_corr(
            model_u.mean("time", skipna=True),
            model_v.mean("time", skipna=True),
            truth_u.mean("time", skipna=True),
            truth_v.mean("time", skipna=True),
        )
        velocity_maps[label] = velocity_rmse_map
        velocity_ann[label] = f"Total RMSE = {velocity_total_rmse:.3g}"
        rows.append(_metric_row(
            label,
            "Total RMSE + spatial RMSE maps",
            "surface_velocity_vector_total_rmse",
            velocity_total_rmse,
            period=period,
            units="m s^-1",
            year=eval_year,
        ))
        rows.append(_metric_row(
            label,
            "Total RMSE + spatial RMSE maps",
            "surface_velocity_vector_annual_mean_pattern_correlation",
            velocity_corr,
            period=period,
            units="1",
            year=eval_year,
        ))

        truth_eke_field = _eke_field_from_uv(truth_u, truth_v)
        model_eke_field = _eke_field_from_uv(model_u, model_v)
        truth_eke_mean = truth_eke_field.mean("time", skipna=True)
        model_eke_mean = model_eke_field.mean("time", skipna=True)
        eke_rmse_map = np.sqrt(((model_eke_field - truth_eke_field) ** 2).mean("time", skipna=True))
        eke_total_rmse = _area_weighted_map_rmse(eke_rmse_map, area)
        eke_corr = _pattern_corr(model_eke_mean, truth_eke_mean)
        eke_maps[label] = eke_rmse_map
        eke_ann[label] = f"Total RMSE = {eke_total_rmse:.3g}"
        rows.append(_metric_row(
            label,
            "Total RMSE + spatial RMSE maps",
            "surface_eke_total_rmse",
            eke_total_rmse,
            period=period,
            units="m^2 s^-2",
            year=eval_year,
        ))
        rows.append(_metric_row(
            label,
            "Total RMSE + spatial RMSE maps",
            "surface_eke_annual_mean_pattern_correlation",
            eke_corr,
            period=period,
            units="1",
            year=eval_year,
        ))

        truth_sst, model_sst = _align_time(_surface(truth, "thetao"), _surface(model, "thetao"))
        sst_rmse_map = np.sqrt(((model_sst - truth_sst) ** 2).mean("time", skipna=True))
        sst_total_rmse = _area_weighted_map_rmse(sst_rmse_map, area)
        sst_corr = _pattern_corr(model_sst.mean("time", skipna=True), truth_sst.mean("time", skipna=True))
        sst_maps[label] = sst_rmse_map
        sst_ann[label] = f"Total RMSE = {sst_total_rmse:.3g}"
        rows.append(_metric_row(
            label,
            "Total RMSE + spatial RMSE maps",
            "surface_sst_total_rmse",
            sst_total_rmse,
            period=period,
            units="degC",
            year=eval_year,
        ))
        rows.append(_metric_row(
            label,
            "Total RMSE + spatial RMSE maps",
            "surface_sst_annual_mean_pattern_correlation",
            sst_corr,
            period=period,
            units="1",
            year=eval_year,
        ))

        ohc_maps[label] = {}
        for depth_slice in EVAL_OHC_DEPTH_SLICES:
            layer = _depth_label(depth_slice)
            truth_ohc, model_ohc = _align_time(
                _ohc_per_area_field(truth, depth_slice),
                _ohc_per_area_field(model, depth_slice),
            )
            ohc_rmse_map = np.sqrt(((model_ohc - truth_ohc) ** 2).mean("time", skipna=True))
            ohc_total_rmse = _area_weighted_map_rmse(ohc_rmse_map, area)
            ohc_corr = _pattern_corr(model_ohc.mean("time", skipna=True), truth_ohc.mean("time", skipna=True))
            ohc_maps[label][layer] = ohc_rmse_map
            ohc_ann[layer][label] = f"Total RMSE = {ohc_total_rmse:.3g}"
            rows.append(_metric_row(
                label,
                "Total RMSE + spatial RMSE maps",
                "ohc_per_unit_area_total_rmse",
                ohc_total_rmse,
                period=period,
                units="J m^-2",
                year=eval_year,
                layer=layer,
            ))
            rows.append(_metric_row(
                label,
                "Total RMSE + spatial RMSE maps",
                "ohc_per_unit_area_annual_mean_pattern_correlation",
                ohc_corr,
                period=period,
                units="1",
                year=eval_year,
                layer=layer,
            ))

    _plot_single_row_maps(
        velocity_maps,
        f"Surface Velocity Vector Spatial RMSE vs OM4 ({eval_year})",
        "m s$^{-1}$",
        cmocean.cm.thermal,
        annotations=velocity_ann,
    )
    save_current_fig(f"surface_velocity_vector_spatial_rmse_vs_om4_{eval_year}", RMSE_DIR)

    _plot_single_row_maps(
        eke_maps,
        f"Surface EKE Spatial RMSE vs OM4 ({eval_year})",
        "m$^2$ s$^{-2}$",
        cmocean.cm.thermal,
        annotations=eke_ann,
    )
    save_current_fig(f"surface_eke_spatial_rmse_vs_om4_{eval_year}", RMSE_DIR)

    _plot_single_row_maps(
        sst_maps,
        f"SST Spatial RMSE vs OM4 ({eval_year})",
        "$^\\circ$C",
        cmocean.cm.thermal,
        annotations=sst_ann,
    )
    save_current_fig(f"sst_spatial_rmse_vs_om4_{eval_year}", RMSE_DIR)

    _plot_layer_maps(
        ohc_maps,
        f"OHC per Unit Area Spatial RMSE vs OM4 ({eval_year})",
        "J m$^{-2}$",
        cmocean.cm.thermal,
        annotations_by_layer=ohc_ann,
    )
    save_current_fig(f"ohc_per_unit_area_spatial_rmse_vs_om4_{eval_year}", RMSE_DIR)
    return rows


def _run_velocity_evaluation(emulators_dict, eval_year):
    rows = []
    first_data = next(iter(emulators_dict.values()))
    first_truth = _select_year(first_data["truth"], eval_year)
    eke_mean_fields = {"OM4": _eke_map(first_truth)}
    eke_mean_ann = {}

    for label, emu_data in emulators_dict.items():
        truth = _select_year(emu_data["truth"], eval_year)
        model = _select_year(emu_data["ds"], eval_year)
        truth_u, truth_v, model_u, model_v = _align_time(
            _surface(truth, "uo"),
            _surface(truth, "vo"),
            _surface(model, "uo"),
            _surface(model, "vo"),
        )
        truth_eke = _eke_map_from_uv(truth_u, truth_v)
        model_eke = _eke_map_from_uv(model_u, model_v)
        rmse = _area_weighted_map_rmse(model_eke - truth_eke, truth["areacello"])
        corr = _pattern_corr(model_eke, truth_eke)
        eke_mean_fields[label] = model_eke
        eke_mean_ann[label] = f"Corr = {corr:.3f}" if np.isfinite(corr) else "Corr = nan"
        rows.append(_metric_row(
            label,
            "Velocities",
            "surface_eke_mean_map_rmse",
            rmse,
            period=_time_label(model),
            units="m^2 s^-2",
            year=eval_year,
        ))
        rows.append(_metric_row(
            label,
            "Velocities",
            "surface_eke_mean_map_pattern_correlation",
            corr,
            period=_time_label(model),
            units="1",
            year=eval_year,
        ))

    _plot_single_row_maps(
        eke_mean_fields,
        f"Surface EKE Mean Map vs OM4 ({eval_year})",
        "m$^2$ s$^{-2}$",
        cmocean.cm.thermal,
        annotations=eke_mean_ann,
    )
    save_current_fig(f"surface_eke_mean_map_vs_om4_{eval_year}", MAP_DIR)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), sharey=False)
    for ax, (region, lon_slice, lat_slice) in zip(axes, EVAL_SPATIAL_SPECTRUM_PANELS):
        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            truth_u, truth_v, model_u, model_v = _align_time(
                _surface(emu_data["truth"], "uo"),
                _surface(emu_data["truth"], "vo"),
                _surface(emu_data["ds"], "uo"),
                _surface(emu_data["ds"], "vo"),
            )
            truth_eke = _eke_map_from_uv(truth_u, truth_v)
            model_eke = _eke_map_from_uv(model_u, model_v)
            truth_k, truth_spec = _spectrum_from_map(truth_eke, emu_data["truth"], lon_slice, lat_slice)
            model_k, model_spec = _spectrum_from_map(model_eke, emu_data["ds"], lon_slice, lat_slice)
            if idx == 0:
                _plot_loglog_curve(ax, truth_k, truth_spec, "--", color="black", linewidth=2.3, label="OM4")
            _plot_loglog_curve(ax, model_k, model_spec, "-", color=emu_data["color"], linewidth=2.0, label=label)
            rows.append(_metric_row(
                label,
                "Velocities",
                "surface_eke_spatial_spectrum_rmse",
                _curve_rmse(truth_k, truth_spec, model_k, model_spec),
                region=region,
                units="spectrum density",
            ))
        ax.set_title(region, fontsize=12, fontweight="bold")
        ax.set_xlabel("Wavenumber (rad km$^{-1}$)")
        ax.set_ylabel("EKE spectrum")
        ax.grid(True, which="both", alpha=0.25)
    _finish_panel_figure(fig, axes, "Surface EKE Spatial Spectra vs OM4")
    save_current_fig("surface_eke_spatial_spectra_vs_om4", SPECTRUM_DIR)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), sharey=False)
    for ax, (region, lon_slice, lat_slice) in zip(axes, EVAL_TEMPORAL_SPECTRUM_PANELS):
        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            truth_freq, truth_psd = _ke_temporal_spectrum(emu_data["truth"], lon_slice, lat_slice)
            model_freq, model_psd = _ke_temporal_spectrum(emu_data["ds"], lon_slice, lat_slice)
            if idx == 0:
                _plot_loglog_curve(ax, truth_freq, truth_psd, "--", color="black", linewidth=2.3, label="OM4")
            _plot_loglog_curve(ax, model_freq, model_psd, "-", color=emu_data["color"], linewidth=2.0, label=label)
            rows.append(_metric_row(
                label,
                "Velocities",
                "surface_ke_temporal_spectrum_rmse",
                _curve_rmse(truth_freq, truth_psd, model_freq, model_psd),
                region=region,
                units="PSD",
            ))
        ax.set_title(region, fontsize=12, fontweight="bold")
        ax.set_xlabel("Frequency (cycles yr$^{-1}$)")
        ax.set_ylabel("KE PSD")
        ax.set_xlim(right=100)
        ax.grid(True, which="both", alpha=0.25)
    _finish_panel_figure(fig, axes, "Surface KE Temporal Spectra vs OM4")
    save_current_fig("surface_ke_temporal_spectra_vs_om4", SPECTRUM_DIR)
    return rows


def _run_temperature_evaluation(emulators_dict, eval_year):
    rows = []

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), sharey=False)
    for ax, (region, lon_slice, lat_slice) in zip(axes, EVAL_SPATIAL_SPECTRUM_PANELS):
        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            truth_sst, model_sst = _align_time(_surface(emu_data["truth"], "thetao"), _surface(emu_data["ds"], "thetao"))
            truth_k, truth_spec = _spectrum_from_surface_field(truth_sst, emu_data["truth"], lon_slice, lat_slice)
            model_k, model_spec = _spectrum_from_surface_field(model_sst, emu_data["ds"], lon_slice, lat_slice)
            if idx == 0:
                _plot_loglog_curve(ax, truth_k, truth_spec, "--", color="black", linewidth=2.3, label="OM4")
            _plot_loglog_curve(ax, model_k, model_spec, "-", color=emu_data["color"], linewidth=2.0, label=label)
            rows.append(_metric_row(
                label,
                "Temperature",
                "surface_sst_spatial_spectrum_rmse",
                _curve_rmse(truth_k, truth_spec, model_k, model_spec),
                region=region,
                units="spectrum density",
            ))
        ax.set_title(region, fontsize=12, fontweight="bold")
        ax.set_xlabel("Wavenumber (rad km$^{-1}$)")
        ax.set_ylabel("SST spectrum")
        ax.grid(True, which="both", alpha=0.25)
    _finish_panel_figure(fig, axes, "SST Spatial Spectra vs OM4")
    save_current_fig("sst_spatial_spectra_vs_om4", SPECTRUM_DIR)

    fig, axes = plt.subplots(len(EVAL_OHC_DEPTH_SLICES), 1, figsize=(11.5, 6.8), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, depth_slice in zip(axes, EVAL_OHC_DEPTH_SLICES):
        layer = _depth_label(depth_slice)
        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            truth_series = _ohc_total_series(emu_data["truth"], depth_slice)
            model_series = _with_color(_ohc_total_series(emu_data["ds"], depth_slice), emu_data["color"])
            if idx == 0:
                _plot_timeseries(ax, truth_series, truth_series.values / SCALE_FACTOR, "--", color="black", linewidth=2.2, label="OM4")
            _plot_timeseries(ax, model_series, model_series.values / SCALE_FACTOR, "-", color=emu_data["color"], linewidth=2.0, label=label)
            rows.append(_metric_row(
                label,
                "Temperature",
                "monthly_ohc_total_rmse",
                _series_rmse(truth_series, model_series),
                layer=layer,
                units="J",
            ))
        ax.set_ylabel(f"{layer}\nOHC (ZJ)")
        ax.grid(True, alpha=0.3)
    _finish_panel_figure(fig, axes, "Monthly Total OHC Time Series vs OM4")
    save_current_fig("monthly_total_ohc_timeseries_vs_om4", TIMESERIES_DIR)

    first_data = next(iter(emulators_dict.values()))
    first_truth_year = _select_year(first_data["truth"], eval_year)
    ohc_maps = {"OM4": {}}
    ohc_bias_maps = {}
    ohc_map_ann = {_depth_label(depth_slice): {} for depth_slice in EVAL_OHC_DEPTH_SLICES}
    ohc_bias_ann = {_depth_label(depth_slice): {} for depth_slice in EVAL_OHC_DEPTH_SLICES}
    for depth_slice in EVAL_OHC_DEPTH_SLICES:
        layer = _depth_label(depth_slice)
        ohc_maps["OM4"][layer] = _ohc_per_area_field(first_truth_year, depth_slice).mean("time", skipna=True)

    for label, emu_data in emulators_dict.items():
        truth = _select_year(emu_data["truth"], eval_year)
        model = _select_year(emu_data["ds"], eval_year)
        ohc_maps[label] = {}
        ohc_bias_maps[label] = {}
        for depth_slice in EVAL_OHC_DEPTH_SLICES:
            layer = _depth_label(depth_slice)
            truth_map = _ohc_per_area_field(truth, depth_slice).mean("time", skipna=True)
            model_map = _ohc_per_area_field(model, depth_slice).mean("time", skipna=True)
            bias_map = model_map - truth_map
            rmse = _area_weighted_map_rmse(bias_map, truth["areacello"])
            corr = _pattern_corr(model_map, truth_map)
            ohc_maps[label][layer] = model_map
            ohc_bias_maps[label][layer] = bias_map
            ohc_map_ann[layer][label] = f"Corr = {corr:.3f}" if np.isfinite(corr) else "Corr = nan"
            ohc_bias_ann[layer][label] = f"Corr = {corr:.3f}" if np.isfinite(corr) else "Corr = nan"
            rows.append(_metric_row(
                label,
                "Temperature",
                "annual_mean_ohc_per_unit_area_map_rmse",
                rmse,
                period=_time_label(model),
                units="J m^-2",
                year=eval_year,
                layer=layer,
            ))
            rows.append(_metric_row(
                label,
                "Temperature",
                "annual_mean_ohc_per_unit_area_map_pattern_correlation",
                corr,
                period=_time_label(model),
                units="1",
                year=eval_year,
                layer=layer,
            ))

    _plot_layer_maps(
        ohc_maps,
        f"Annual Mean OHC per Unit Area vs OM4 ({eval_year})",
        "J m$^{-2}$",
        cmocean.cm.thermal,
        annotations_by_layer=ohc_map_ann,
    )
    save_current_fig(f"annual_mean_ohc_per_unit_area_maps_vs_om4_{eval_year}", MAP_DIR)

    _plot_layer_maps(
        ohc_bias_maps,
        f"Annual Mean OHC per Unit Area Bias vs OM4 ({eval_year})",
        "J m$^{-2}$",
        cmocean.cm.balance,
        annotations_by_layer=ohc_bias_ann,
        centered=True,
    )
    save_current_fig(f"annual_mean_ohc_per_unit_area_bias_maps_vs_om4_{eval_year}", MAP_DIR)
    return rows


def _series_group(emulators_dict, series_func, obs_label="OM4"):
    first_data = next(iter(emulators_dict.values()))
    series_by_model = {obs_label: series_func(first_data["truth"])}
    for label, emu_data in emulators_dict.items():
        series_by_model[label] = _with_color(series_func(emu_data["ds"]), emu_data["color"])
    return series_by_model


def _run_anomaly_evaluation(emulators_dict):
    rows = []
    obs_label = "OM4"
    specs = [
        ("sst", "SST", "$^\\circ$C", _global_sst_series),
        ("surface_eke", "Surface EKE", "m$^2$ s$^{-2}$", _eke_timeseries),
        ("ohc_upper700", "Upper-700 m OHC", "J", lambda ds: _ohc_total_series(ds, EVAL_OHC_UPPER_700_SLICE)),
    ]

    for prefix, title, units, series_func in specs:
        series_by_model = _series_group(emulators_dict, series_func, obs_label=obs_label)
        rows.extend(_plot_anomaly_timeseries(series_by_model, title, units, obs_label, prefix))
        save_current_fig(f"{prefix}_mean_removed_anomaly_timeseries_vs_om4", TIMESERIES_DIR)
        rows.extend(_plot_trend_comparison(series_by_model, title, units, obs_label, prefix))
        save_current_fig(f"{prefix}_linear_trend_fit_comparison_vs_om4", TIMESERIES_DIR)
        rows.extend(_plot_residual_anomaly_timeseries(series_by_model, title, units, obs_label, prefix))
        save_current_fig(f"{prefix}_detrended_deseasonalized_anomaly_timeseries_vs_om4", TIMESERIES_DIR)
        rows.extend(_plot_residual_variance_bars(series_by_model, title, units, obs_label, prefix))
        save_current_fig(f"{prefix}_detrended_deseasonalized_anomaly_variance_vs_om4", VARIANCE_DIR)

    first_data = next(iter(emulators_dict.values()))
    first_truth = first_data["truth"]
    sst_var_maps = {"OM4": _detrended_deseasonalized_dataarray(_surface(first_truth, "thetao")).var("time", skipna=True)}
    ohc_upper_first = _monthly(_ohc_per_area_field(first_truth, EVAL_OHC_UPPER_700_SLICE))
    ohc_var_maps = {"OM4": _detrended_deseasonalized_dataarray(ohc_upper_first).var("time", skipna=True)}
    sst_var_ann = {}
    ohc_var_ann = {}

    for label, emu_data in emulators_dict.items():
        truth_sst, model_sst = _align_time(_surface(emu_data["truth"], "thetao"), _surface(emu_data["ds"], "thetao"))
        truth_sst_var = _detrended_deseasonalized_dataarray(truth_sst).var("time", skipna=True)
        model_sst_var = _detrended_deseasonalized_dataarray(model_sst).var("time", skipna=True)
        sst_var_rmse = _area_weighted_map_rmse(model_sst_var - truth_sst_var, emu_data["truth"]["areacello"])
        sst_var_corr = _pattern_corr(model_sst_var, truth_sst_var)
        sst_var_maps[label] = model_sst_var
        sst_var_ann[label] = f"Corr = {sst_var_corr:.3f}" if np.isfinite(sst_var_corr) else "Corr = nan"
        rows.append(_metric_row(
            label,
            "Anomalies",
            "sst_residual_anomaly_variance_map_rmse",
            sst_var_rmse,
            units="degC^2",
        ))
        rows.append(_metric_row(
            label,
            "Anomalies",
            "sst_residual_anomaly_variance_map_pattern_correlation",
            sst_var_corr,
            units="1",
        ))

        truth_ohc, model_ohc = _align_time(
            _monthly(_ohc_per_area_field(emu_data["truth"], EVAL_OHC_UPPER_700_SLICE)),
            _monthly(_ohc_per_area_field(emu_data["ds"], EVAL_OHC_UPPER_700_SLICE)),
        )
        truth_ohc_var = _detrended_deseasonalized_dataarray(truth_ohc).var("time", skipna=True)
        model_ohc_var = _detrended_deseasonalized_dataarray(model_ohc).var("time", skipna=True)
        ohc_var_rmse = _area_weighted_map_rmse(model_ohc_var - truth_ohc_var, emu_data["truth"]["areacello"])
        ohc_var_corr = _pattern_corr(model_ohc_var, truth_ohc_var)
        ohc_var_maps[label] = model_ohc_var
        ohc_var_ann[label] = f"Corr = {ohc_var_corr:.3f}" if np.isfinite(ohc_var_corr) else "Corr = nan"
        rows.append(_metric_row(
            label,
            "Anomalies",
            "ohc_upper700_residual_anomaly_variance_map_rmse",
            ohc_var_rmse,
            units="(J m^-2)^2",
        ))
        rows.append(_metric_row(
            label,
            "Anomalies",
            "ohc_upper700_residual_anomaly_variance_map_pattern_correlation",
            ohc_var_corr,
            units="1",
        ))

    _plot_single_row_maps(
        sst_var_maps,
        "SST Detrended Deseasonalized Anomaly Variance vs OM4",
        "$^\\circ$C$^2$",
        cmocean.cm.thermal,
        annotations=sst_var_ann,
    )
    save_current_fig("sst_detrended_deseasonalized_anomaly_variance_maps_vs_om4", VARIANCE_DIR)

    _plot_single_row_maps(
        ohc_var_maps,
        "Upper-700 m OHC per Unit Area Detrended Deseasonalized Anomaly Variance vs OM4",
        "(J m$^{-2}$)$^2$",
        cmocean.cm.thermal,
        annotations=ohc_var_ann,
    )
    save_current_fig("ohc_upper700_detrended_deseasonalized_anomaly_variance_maps_vs_om4", VARIANCE_DIR)
    return rows


def _run_four_part_evaluation(emulators_dict, eval_year):
    print("\n" + "=" * 60)
    print("1. Total RMSE + spatial RMSE maps")
    print("=" * 60)
    rows = _run_total_rmse_evaluation(emulators_dict, eval_year)

    print("\n" + "=" * 60)
    print("2. Velocities")
    print("=" * 60)
    rows.extend(_run_velocity_evaluation(emulators_dict, eval_year))

    print("\n" + "=" * 60)
    print("3. Temperature")
    print("=" * 60)
    rows.extend(_run_temperature_evaluation(emulators_dict, eval_year))

    print("\n" + "=" * 60)
    print("4. Anomalies")
    print("=" * 60)
    rows.extend(_run_anomaly_evaluation(emulators_dict))

    metrics = pd.DataFrame(rows)
    csv_path = os.path.join(METRICS_DIR, f"1in1out_2in2out_four_part_metrics_{eval_year}.csv")
    metrics.to_csv(csv_path, index=False)
    print(f"  -> Saved metrics: {csv_path}")
    return metrics

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

four_part_metrics = _run_four_part_evaluation(emulators_to_analyze, args.eval_year)
print("\nFour-part evaluation complete.")
if not args.legacy_extra:
    print("Skipping legacy extra diagnostics. Use --legacy-extra to run the older auxiliary sections.")
    sys.exit(0)

for d in [ZONAL_BIAS_DIR, SPATIAL_BIAS_DIR, ZONAL_MEAN_DIR, SPATIAL_DISTRIBUTION_DIR]:
    os.makedirs(d, exist_ok=True)

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
