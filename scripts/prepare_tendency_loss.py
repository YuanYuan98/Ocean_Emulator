"""Compute training-only temporal-increment scales for the revision loss baseline."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cftime
import numpy as np
import xarray as xr

from ocean_emulators.constants import PROGNOSTIC_VARS


def increment_moments(
    field: xr.DataArray, wet: np.ndarray, lag: int, chunk: int = 64
) -> tuple[float, float, int]:
    """Population mean/variance of increments, using wet cells and no wraparound."""
    if lag < 1 or field.sizes["time"] <= lag or not wet.any():
        raise ValueError("Insufficient time samples or wet cells")
    count, total, squared = 0, 0.0, 0.0
    for start in range(lag, field.sizes["time"], chunk):
        stop = min(start + chunk, field.sizes["time"])
        block = np.asarray(
            field.isel(time=slice(start - lag, stop)).values, dtype=np.float64
        )
        deltas = (block[lag:] - block[:-lag])[:, wet]
        if not np.isfinite(deltas).all():
            raise ValueError(f"Non-finite wet-cell increment in {field.name}")
        count += deltas.size
        total += deltas.sum(dtype=np.float64)
        squared += np.square(deltas).sum(dtype=np.float64)
    mean = total / count
    variance = squared / count - mean * mean
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError(f"Nonpositive increment variance for {field.name}: {variance}")
    return mean, variance, count


def validate_time_spacing(times: np.ndarray) -> dict[str, int]:
    """Validate the original OM4 cadence, including Julian leap-year boundaries."""
    five_day = leap_boundary = 0
    for a, b in zip(times[:-1], times[1:], strict=True):
        days = (b - a).total_seconds() / 86400
        if days == 5:
            five_day += 1
        elif (
            days == 6
            and a.year % 4 == 0
            and b.year == a.year + 1
            and (a.month, a.day) == (12, 28)
            and (b.month, b.day) == (1, 3)
        ):
            leap_boundary += 1
        else:
            raise ValueError(f"Unexpected OM4 time spacing: {a} -> {b} ({days} days)")
    return {"five_day_intervals": five_day, "six_day_leap_boundaries": leap_boundary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-start", default="1975-01-03")
    parser.add_argument("--train-end", default="2013-10-05")
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--output-frames", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    ds = xr.open_zarr(args.data_root / "OM4.zarr", chunks=None)
    stds = xr.open_zarr(args.data_root / "OM4_stds.zarr", chunks=None)
    start = cftime.DatetimeJulian(*map(int, args.train_start.split("-")))
    end = cftime.DatetimeJulian(*map(int, args.train_end.split("-")))
    selected = np.flatnonzero((ds.time.values >= start) & (ds.time.values < end))
    if len(selected) < 2 or not np.all(np.diff(selected) == 1):
        raise ValueError("Training time selection must be contiguous")
    train = ds.isel(time=slice(int(selected[0]), int(selected[-1]) + 1))
    cadence = validate_time_spacing(train.time.values)
    print(f"Training time cadence: {cadence}", flush=True)
    channels = PROGNOSTIC_VARS["thermo_dynamic_all"]

    def compute(channel: str) -> dict:
        level = channel.rsplit("_", 1)[-1] if channel != "zos" else "0"
        wet = np.asarray(ds[f"mask_{level}"].values, dtype=bool)
        if wet.ndim != 2:
            raise ValueError("Expected a static two-dimensional wet mask")
        mean, variance, count = increment_moments(train[channel], wet, args.lag)
        state_std = float(stds[channel].item())
        if not np.isfinite(state_std) or state_std <= 0:
            raise ValueError(f"Invalid existing state normalization for {channel}")
        weight = state_std**2 / variance
        print(
            f"{channel}: increment_std={variance**0.5:.8g}, loss_weight={weight:.8g}",
            flush=True,
        )
        return {
            "channel": channel,
            "increment_mean": mean,
            "increment_variance": variance,
            "count": count,
            "state_std": state_std,
            "weight": weight,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        stats = list(pool.map(compute, channels))
    weights = [row["weight"] for row in stats]
    payload = {
        "method": "MSE normalized by training-period temporal-increment standard deviations",
        "data_root": str(args.data_root.resolve()),
        "fit_start_inclusive": str(train.time.values[0]),
        "fit_end_inclusive": str(train.time.values[-1]),
        "training_states": int(train.sizes["time"]),
        "lag_steps": args.lag,
        "lag_days": 5 * args.lag,
        "lag_days_convention": "Nominal five-day sample lag; original six-day Julian leap-year boundaries retained",
        "training_time_cadence": cadence,
        "output_frames": args.output_frames,
        "channel_names": channels,
        "channel_weights": weights,
        "output_weights": weights * args.output_frames,
        "statistics": stats,
        "normalization": "Existing state statistics and full-grid masked MSE unchanged; only increment scales are fitted on training data. No latitude weighting or ratio cap.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
