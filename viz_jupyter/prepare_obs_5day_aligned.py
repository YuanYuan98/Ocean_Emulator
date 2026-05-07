"""
Prepare observation products for ``obs_evaluation.py``.

What this script does
---------------------
- Reads the Samudra rollout (``predictions.zarr``), extracts the year's 5-day
  reference timestamps, and validates the cadence (start ``YYYY-01-03 12:00:00``,
  step 5 days).
- DUACS and OISST (daily): centered 5-sample rolling mean in time, sampled on
  the rollout dates and relabeled to the exact rollout timestamps.
- ARGO-IAP (monthly): kept on its native monthly axis (no temporal alignment),
  because forcing it to 5 days would invent variability.
- Coordinate housekeeping only: rename ``latitude``/``longitude`` to
  ``lat``/``lon``, normalize longitude to ``[0, 360)`` and sort, sort by
  latitude.

What this script does NOT do
----------------------------
- No spatial coarsening or interpolation. Each product is written on its
  source native grid (DUACS ~0.125 deg, OISST ~0.25 deg, ARGO-IAP ~0.5 deg).
  ``obs_evaluation.py`` computes diagnostics on each side's own grid and
  reconciles in diagnostic space (global means, spectral curves, etc.).

If a future diagnostic needs the obs and model on a common grid, add the
regridding in the evaluation script -- do not bake it in here.

See ``obs_data_process.md`` for the full rationale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


RAW_OBS_ROOT = Path("/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/obs_data")
DEFAULT_OUTPUT_ROOT = RAW_OBS_ROOT / "nativegrid"
DEFAULT_ROLLOUT = Path(
    "/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/checkpoints/ablation_studies/2025-12-02-samudra_om4_v1_onedeg_eval/predictions.zarr"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare rollout-aligned 5-day observation products")
    parser.add_argument("--raw-obs-root", type=Path, default=RAW_OBS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rollout-path", type=Path, default=DEFAULT_ROLLOUT)
    parser.add_argument("--year", type=int, default=2022)
    return parser.parse_args()


def _coerce_datetime_index(values: np.ndarray) -> pd.DatetimeIndex:
    flat = np.asarray(values)
    if np.issubdtype(flat.dtype, np.datetime64):
        return pd.to_datetime(flat)
    first = flat.flat[0]
    if hasattr(first, "year") and hasattr(first, "month") and hasattr(first, "day"):
        return pd.to_datetime(
            [
                f"{val.year:04d}-{val.month:02d}-{val.day:02d} "
                f"{getattr(val, 'hour', 0):02d}:{getattr(val, 'minute', 0):02d}:{getattr(val, 'second', 0):02d}"
                for val in flat
            ]
        )
    numeric = flat.astype(float)
    if np.all((numeric >= 100001) & (numeric <= 999912)):
        years = (numeric // 100).astype(int)
        months = np.round(numeric - years * 100).astype(int)
        return pd.to_datetime([f"{yy:04d}-{mm:02d}-01" for yy, mm in zip(years, months, strict=True)])
    return pd.to_datetime(flat.astype(str))


def _rollout_times(rollout_path: Path, year: int) -> pd.DatetimeIndex:
    ds = xr.open_zarr(rollout_path)
    times = _coerce_datetime_index(ds.time.values)
    mask = (times >= pd.Timestamp(f"{year}-01-01")) & (times <= pd.Timestamp(f"{year}-12-31 23:59:59"))
    target = times[mask]
    if len(target) == 0:
        raise ValueError(f"No rollout timestamps found in {year}")
    deltas = np.unique((target[1:] - target[:-1]).days)
    if target[0] != pd.Timestamp(f"{year}-01-03 12:00:00") or not np.array_equal(deltas, np.array([5])):
        raise ValueError(f"Unexpected rollout cadence: first={target[0]!s}, deltas={deltas}")
    return target


def _standardize_daily_ds(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    ds = ds.rename(rename)
    ds = ds.assign_coords(time=_coerce_datetime_index(ds.time.values))
    if float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon % 360)).sortby("lon")
    return ds.sortby("lat")


def _align_daily_with_centered_rolling(ds: xr.Dataset, target_times: pd.DatetimeIndex) -> xr.Dataset:
    rolled = ds.rolling(time=5, center=True, min_periods=5).mean()
    rolled_dates = pd.DatetimeIndex(rolled.time.values).normalize()
    target_dates = target_times.normalize()
    rolled = rolled.assign_coords(time=("time", rolled_dates))
    aligned = rolled.sel(time=target_dates)
    aligned = aligned.assign_coords(time=("time", target_times.values))
    return aligned


def _chunk_for_write(ds: xr.Dataset) -> xr.Dataset:
    chunks = {}
    if "time" in ds.dims:
        chunks["time"] = min(int(ds.sizes["time"]), 24)
    if "lat" in ds.dims:
        chunks["lat"] = min(int(ds.sizes["lat"]), 180)
    if "lon" in ds.dims:
        chunks["lon"] = min(int(ds.sizes["lon"]), 360)
    if "depth_std" in ds.dims:
        chunks["depth_std"] = min(int(ds.sizes["depth_std"]), 16)
    return ds.chunk(chunks)


def _prepare_duacs(
    raw_root: Path,
    output_root: Path,
    target_times: pd.DatetimeIndex,
) -> None:
    input_path = raw_root / "duacs" / "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_multi-vars_179.94W-179.94E_89.94S-89.94N_2022-01-01-2023-01-01.zarr"
    ds = xr.open_zarr(input_path)[["ugos", "vgos"]]
    ds = _standardize_daily_ds(ds)
    ds = _align_daily_with_centered_rolling(ds, target_times)
    _chunk_for_write(ds).to_zarr(output_root / "duacs.zarr", mode="w")


def _prepare_oisst(
    raw_root: Path,
    output_root: Path,
    target_times: pd.DatetimeIndex,
) -> None:
    ds = xr.open_mfdataset(str(raw_root / "oisst" / "*.nc"), combine="by_coords")[["sst"]]
    ds = _standardize_daily_ds(ds).squeeze(drop=True)
    ds = _align_daily_with_centered_rolling(ds, target_times)
    _chunk_for_write(ds).to_zarr(output_root / "oisst.zarr", mode="w")


def _prepare_argo_iap(
    raw_root: Path,
    output_root: Path,
) -> None:
    ds = xr.open_mfdataset(str(raw_root / "argo-iap" / "*.nc"), combine="by_coords")[["temp"]]
    ds = ds.assign_coords(time=_coerce_datetime_index(ds.time.values)).sortby("time")
    if float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon % 360)).sortby("lon")
    ds = ds.sortby("lat")
    _chunk_for_write(ds).to_zarr(output_root / "argo-iap.zarr", mode="w")


def _write_readme(output_root: Path, target_times: pd.DatetimeIndex) -> None:
    text = (
        "Observation products prepared for obs_evaluation.py on the native observation grid.\n\n"
        f"Target timeline: {target_times[0]} through {target_times[-1]} every 5 days ({len(target_times)} samples).\n"
        "DUACS and OISST were converted from daily products using a centered 5-sample rolling mean,\n"
        "then sampled on the rollout dates and relabeled to the exact rollout timestamps.\n"
        "Spatial resolution is preserved at the source product's native grid:\n"
        "  - DUACS: ~0.125 degree\n"
        "  - OISST: ~0.25 degree\n"
        "  - ARGO-IAP: monthly, native grid\n"
        "obs_evaluation.py compares each side on its own native grid (no spatial resampling).\n"
    )
    (output_root / "README.txt").write_text(text)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    target_times = _rollout_times(args.rollout_path, args.year)

    print(f"Preparing rollout-aligned products in {args.output_root}")
    _prepare_duacs(args.raw_obs_root, args.output_root, target_times)
    print("  -> wrote duacs.zarr")
    _prepare_oisst(args.raw_obs_root, args.output_root, target_times)
    print("  -> wrote oisst.zarr")
    _prepare_argo_iap(args.raw_obs_root, args.output_root)
    print("  -> wrote argo-iap.zarr")
    _write_readme(args.output_root, target_times)
    print("  -> wrote README.txt")


if __name__ == "__main__":
    main()
