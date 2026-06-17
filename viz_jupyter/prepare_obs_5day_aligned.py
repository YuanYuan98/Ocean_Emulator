"""
Prepare observation products for ``obs_evaluation.py``.

What this script does
---------------------
- Reads the Samudra rollout (``predictions.zarr``), extracts the year's 5-day
  reference timestamps, and validates the cadence (start ``YYYY-01-03 12:00:00``,
  step 5 days).
- DUACS and OISST (daily): centered 5-sample rolling mean in time, sampled on
  the rollout dates and relabeled to the exact rollout timestamps.
- OISST daily can also be written separately as ``oisst_daily.zarr`` for
  diagnostics that compare daily and 5-day-smoothed observational spectra.
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
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


RAW_OBS_ROOT = Path("/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/obs_data")
DEFAULT_OUTPUT_ROOT = RAW_OBS_ROOT / "nativegrid"
DEFAULT_ROLLOUT = Path(
    "/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/checkpoints/ablation_studies/2025-12-02-samudra_om4_v1_onedeg_eval/predictions.zarr"
)

DUACS_KEEP_VARS = ["ugos", "vgos", "ugosa", "vgosa", "adt", "sla"]
DUACS_RAW_LONG_COLLECTION = (
    "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_velocity_"
    "2014-10-18-2022-12-26_yearly"
)
DUACS_RAW_LONG_ZARR = (
    "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_multi-vars_"
    "179.94W-179.94E_89.94S-89.94N_2014-10-18-2022-12-26.zarr"
)
DUACS_RAW_2022_ZARR = (
    "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_multi-vars_"
    "179.94W-179.94E_89.94S-89.94N_2022-01-01-2023-01-01.zarr"
)
ARGO_TEMP_ALIASES = ["temp", "thetao", "temperature", "T"]
ARGO_SALT_ALIASES = ["so", "salinity", "salt", "sal", "psal", "S"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare rollout-aligned 5-day observation products")
    parser.add_argument("--raw-obs-root", type=Path, default=RAW_OBS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rollout-path", type=Path, default=DEFAULT_ROLLOUT)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument(
        "--start-date",
        default=None,
        help=(
            "Optional rollout start date for DUACS/OISST alignment. "
            "When set, --end-date must also be set and --year is only used for products not using rollout times."
        ),
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional rollout end date for DUACS/OISST alignment. Requires --start-date.",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        choices=["all", "duacs", "oisst", "oisst-daily", "argo-iap"],
        default=["all"],
        help="Products to rebuild. Use --products argo-iap after downloading a longer ARGO-IAP record.",
    )
    parser.add_argument("--argo-start-year", type=int, default=None)
    parser.add_argument("--argo-end-year", type=int, default=None)
    parser.add_argument(
        "--duacs-raw-path",
        type=Path,
        default=None,
        help=(
            "Optional raw daily DUACS zarr or yearly-zarr collection path. Defaults "
            "to the 2014-2022 DUACS collection under --raw-obs-root when present, "
            "otherwise the legacy 2022 zarr."
        ),
    )
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


def _rollout_times(
    rollout_path: Path,
    year: int,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DatetimeIndex:
    ds = xr.open_zarr(rollout_path)
    times = _coerce_datetime_index(ds.time.values)
    if (start_date is None) != (end_date is None):
        raise ValueError("--start-date and --end-date must be provided together")
    if start_date is not None and end_date is not None:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if end == end.normalize():
            end = end + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    else:
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-31 23:59:59")
    mask = (times >= start) & (times <= end)
    target = times[mask]
    if len(target) == 0:
        raise ValueError(f"No rollout timestamps found between {start} and {end}")
    deltas = np.unique((target[1:] - target[:-1]).days)
    if start_date is None:
        if target[0] != pd.Timestamp(f"{year}-01-03 12:00:00") or not np.array_equal(deltas, np.array([5])):
            raise ValueError(f"Unexpected rollout cadence: first={target[0]!s}, deltas={deltas}")
    elif not set(deltas).issubset({5, 6}):
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


def _find_var_name(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in ds.data_vars:
            return name
    return None


def _argo_file_time(path: Path) -> pd.Timestamp | None:
    match = re.search(r"_year_(\d{4})_month_(\d{2})\.nc$", path.name, flags=re.IGNORECASE)
    if match is None:
        return None
    year = int(match.group(1))
    month = int(match.group(2))
    if not 1 <= month <= 12:
        return None
    return pd.Timestamp(year=year, month=month, day=1)


def _open_argo_files(
    raw_root: Path,
    keywords: list[str],
    start_year: int | None = None,
    end_year: int | None = None,
) -> xr.Dataset | None:
    candidates: list[tuple[pd.Timestamp, Path]] = []
    for path in (raw_root / "argo-iap").glob("*.nc"):
        if not any(keyword in path.name.lower() for keyword in keywords):
            continue
        timestamp = _argo_file_time(path)
        if timestamp is None:
            continue
        if start_year is not None and timestamp.year < start_year:
            continue
        if end_year is not None and timestamp.year > end_year:
            continue
        candidates.append((timestamp, path))
    candidates.sort()
    files = [path for _, path in candidates]
    if not files:
        return None
    ds = xr.open_mfdataset([str(p) for p in files], combine="nested", concat_dim="time")
    ds = ds.assign_coords(time=pd.DatetimeIndex([timestamp for timestamp, _ in candidates]))
    return ds


def _validate_monthly_argo_times(ds: xr.Dataset, label: str) -> None:
    times = pd.DatetimeIndex(ds.time.values)
    if times.empty:
        raise ValueError(f"No {label} times found")
    expected = pd.date_range(times[0], times[-1], freq="MS")
    missing = expected.difference(times)
    duplicates = times[times.duplicated()]
    if len(missing) or len(duplicates):
        msg = [f"{label} does not have a complete unique monthly time axis."]
        if len(missing):
            msg.append(f"Missing first entries: {[str(val.date()) for val in missing[:10]]}")
        if len(duplicates):
            msg.append(f"Duplicate first entries: {[str(val.date()) for val in duplicates[:10]]}")
        raise ValueError(" ".join(msg))


def _open_zarr_or_collection(path: Path) -> xr.Dataset:
    if path.is_dir() and path.suffix == ".zarr":
        return xr.open_zarr(path)
    if path.is_dir():
        zarrs = sorted(path.glob("*.zarr"))
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
            ds = ds.sortby("time")
            times = pd.DatetimeIndex(ds.time.values)
            keep = ~times.duplicated()
            if not keep.all():
                ds = ds.isel(time=keep)
            return ds
    return xr.open_zarr(path)


def _prepare_duacs(
    raw_root: Path,
    output_root: Path,
    target_times: pd.DatetimeIndex,
    duacs_raw_path: Path | None = None,
) -> None:
    first_date = target_times.min().normalize() - pd.Timedelta(days=2)
    last_date = target_times.max().normalize() + pd.Timedelta(days=2)
    input_path = duacs_raw_path
    if input_path is None:
        needs_long_raw = first_date < pd.Timestamp("2022-01-01") or last_date > pd.Timestamp("2023-01-01")
        candidate_names = (
            [DUACS_RAW_LONG_COLLECTION, DUACS_RAW_LONG_ZARR]
            if needs_long_raw
            else [DUACS_RAW_LONG_COLLECTION, DUACS_RAW_LONG_ZARR, DUACS_RAW_2022_ZARR]
        )
        for candidate_name in candidate_names:
            candidate = raw_root / "duacs" / candidate_name
            if candidate.exists():
                input_path = candidate
                break
    if input_path is None:
        input_path = raw_root / "duacs" / DUACS_RAW_LONG_COLLECTION
    if not input_path.exists():
        raise FileNotFoundError(f"Raw DUACS zarr not found: {input_path}")

    ds_raw = _open_zarr_or_collection(input_path)
    keep_vars = [name for name in DUACS_KEEP_VARS if name in ds_raw.data_vars]
    missing = sorted(set(DUACS_KEEP_VARS) - set(keep_vars))
    if missing:
        print(f"  -> DUACS fields not found and not written: {missing}")
    ds = ds_raw[keep_vars]
    ds = _standardize_daily_ds(ds)
    ds = ds.sel(time=slice(first_date, last_date))
    ds = _align_daily_with_centered_rolling(ds, target_times)
    _chunk_for_write(ds).to_zarr(output_root / "duacs.zarr", mode="w")


def _prepare_oisst(
    raw_root: Path,
    output_root: Path,
    target_times: pd.DatetimeIndex,
) -> None:
    first_date = target_times.min().normalize() - pd.Timedelta(days=2)
    last_date = target_times.max().normalize() + pd.Timedelta(days=2)
    needed_dates = pd.date_range(first_date, last_date, freq="D")
    files = [raw_root / "oisst" / f"oisst-avhrr-v02r01.{date:%Y%m%d}.nc" for date in needed_dates]
    missing = [path for path in files if not path.exists()]
    if missing:
        first_missing = ", ".join(path.name for path in missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} OISST daily files needed for centered 5-day alignment. "
            f"First missing files: {first_missing}"
        )
    ds = xr.open_mfdataset([str(path) for path in files], combine="by_coords")[["sst"]]
    ds = _standardize_daily_ds(ds).squeeze(drop=True)
    ds = _align_daily_with_centered_rolling(ds, target_times)
    _chunk_for_write(ds).to_zarr(output_root / "oisst.zarr", mode="w")


def _prepare_oisst_daily(
    raw_root: Path,
    output_root: Path,
    target_times: pd.DatetimeIndex,
) -> None:
    first_date = target_times.min().normalize()
    last_date = target_times.max().normalize()
    needed_dates = pd.date_range(first_date, last_date, freq="D")
    files = [raw_root / "oisst" / f"oisst-avhrr-v02r01.{date:%Y%m%d}.nc" for date in needed_dates]
    missing = [path for path in files if not path.exists()]
    if missing:
        first_missing = ", ".join(path.name for path in missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} OISST daily files needed for daily output. "
            f"First missing files: {first_missing}"
        )
    ds = xr.open_mfdataset([str(path) for path in files], combine="by_coords")[["sst"]]
    ds = _standardize_daily_ds(ds).squeeze(drop=True)
    _chunk_for_write(ds).to_zarr(output_root / "oisst_daily.zarr", mode="w")


def _prepare_argo_iap(
    raw_root: Path,
    output_root: Path,
    start_year: int | None = None,
    end_year: int | None = None,
) -> None:
    ds_temp = _open_argo_files(raw_root, ["temperature", "temp"], start_year, end_year)
    if ds_temp is None:
        ds_temp = xr.open_mfdataset(str(raw_root / "argo-iap" / "*.nc"), combine="by_coords")

    ds_salt = _open_argo_files(raw_root, ["salinity", "salt", "psal"], start_year, end_year)

    temp_name = _find_var_name(ds_temp, ARGO_TEMP_ALIASES)
    salt_name = _find_var_name(ds_salt, ARGO_SALT_ALIASES) if ds_salt is not None else None
    if temp_name is None:
        raise KeyError(f"Could not find ARGO temperature variable in {list(ds_temp.data_vars)}")
    _validate_monthly_argo_times(ds_temp, "ARGO-IAP temperature")

    data_vars = {"temp": ds_temp[temp_name]}
    if salt_name is None:
        print(
            "  -> ARGO salinity was not found in the raw files; "
            "temperature/salinity structure evaluation will require adding it later."
        )
    else:
        _validate_monthly_argo_times(ds_salt, "ARGO-IAP salinity")
        temp_times = pd.DatetimeIndex(ds_temp.time.values)
        salt_times = pd.DatetimeIndex(ds_salt.time.values)
        if not temp_times.equals(salt_times):
            raise ValueError("ARGO-IAP temperature and salinity files do not cover the same monthly time axis")
        data_vars["salt"] = ds_salt[salt_name]

    ds = xr.Dataset(data_vars)
    ds = ds.sortby("time")
    if float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon % 360)).sortby("lon")
    ds = ds.sortby("lat")
    for name in list(ds.data_vars):
        if np.issubdtype(ds[name].dtype, np.floating):
            ds[name] = ds[name].astype("float32")
    _chunk_for_write(ds).to_zarr(output_root / "argo-iap.zarr", mode="w")


def _write_readme(output_root: Path, target_times: pd.DatetimeIndex | None) -> None:
    if target_times is None:
        timeline_text = "DUACS/OISST target timeline was not rebuilt in this run.\n"
    else:
        timeline_text = (
            f"Target timeline: {target_times[0]} through {target_times[-1]} every 5 days "
            f"({len(target_times)} samples).\n"
        )
    text = (
        "Observation products prepared for obs_evaluation.py on the native observation grid.\n\n"
        f"{timeline_text}"
        "DUACS and OISST were converted from daily products using a centered 5-sample rolling mean,\n"
        "then sampled on the rollout dates and relabeled to the exact rollout timestamps.\n"
        "OISST daily, when written, is kept at the original daily cadence over the target timeline.\n"
        "Spatial resolution is preserved at the source product's native grid:\n"
        "  - DUACS: ~0.125 degree\n"
        "  - OISST: ~0.25 degree\n"
        "  - ARGO-IAP: monthly, native grid; salinity is included when present in the raw files\n"
        "obs_evaluation.py compares each side on its own native grid (no spatial resampling).\n"
    )
    (output_root / "README.txt").write_text(text)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    products = set(args.products)
    if "all" in products:
        products = {"duacs", "oisst", "oisst-daily", "argo-iap"}
    needs_rollout = bool(products & {"duacs", "oisst", "oisst-daily"})
    target_times = (
        _rollout_times(args.rollout_path, args.year, args.start_date, args.end_date)
        if needs_rollout
        else None
    )

    print(f"Preparing rollout-aligned products in {args.output_root}")
    if "duacs" in products:
        _prepare_duacs(args.raw_obs_root, args.output_root, target_times, args.duacs_raw_path)
        print("  -> wrote duacs.zarr")
    if "oisst" in products:
        _prepare_oisst(args.raw_obs_root, args.output_root, target_times)
        print("  -> wrote oisst.zarr")
    if "oisst-daily" in products:
        _prepare_oisst_daily(args.raw_obs_root, args.output_root, target_times)
        print("  -> wrote oisst_daily.zarr")
    if "argo-iap" in products:
        _prepare_argo_iap(args.raw_obs_root, args.output_root, args.argo_start_year, args.argo_end_year)
        print("  -> wrote argo-iap.zarr")
    _write_readme(args.output_root, target_times)
    print("  -> wrote README.txt")


if __name__ == "__main__":
    main()
