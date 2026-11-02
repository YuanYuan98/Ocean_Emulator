"""Cache exact legacy heat anomalies and training-only seasonal input climatology."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import xarray as xr

from ocean_emulators.config import TrainConfig
from ocean_emulators.utils.forcing_revision import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = TrainConfig.from_yaml_and_cli([args.config])
    assets = Path(cfg.data.forcing_revision_assets)
    assets.mkdir(parents=True, exist_ok=True)
    manifest_path = assets / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            "Assets already exist; refuse to silently rebuild frozen preprocessing"
        )
    root = cfg.experiment.resolved_data_root
    sources = {
        name: str(root.resolve(location))
        for name, location in [
            ("data", cfg.data.data_location),
            ("means", cfg.data.data_means_location),
            ("stds", cfg.data.data_stds_location),
        ]
    }
    ds = xr.open_zarr(sources["data"], chunks=None)
    if ds.hfds.shape != (4745, 180, 360):
        raise ValueError(f"Unexpected paper-era one-degree shape: {ds.hfds.shape}")
    times = ds.time.values
    train_indices = np.flatnonzero(
        (times >= cfg.train_time.start.datetime) & (times < cfg.train_time.end.datetime)
    )
    if len(train_indices) != 2829 or not np.all(np.diff(train_indices) == 1):
        raise ValueError(f"Unexpected training split: {len(train_indices)}")
    print(
        "Reading physical heat flux and calculating exact original full-record anomalies",
        flush=True,
    )
    heat = ds.hfds.load()
    full_clim = heat.groupby("time.dayofyear").mean("time").compute()
    anomaly = (heat - full_clim.sel(dayofyear=heat.time.dt.dayofyear)).compute()
    anomaly = anomaly.drop_vars("dayofyear")
    anomaly_mean = anomaly.mean().compute()
    anomaly_std = anomaly.std().compute()
    anom_ds = anomaly.rename("hfds_anomalies").to_dataset()
    anom_ds.hfds_anomalies.encoding = {"chunks": (1, 180, 360)}
    anom_ds.to_zarr(assets / "original_anomaly.zarr", mode="w", consolidated=True)
    climatology = {}
    for variable in ("tauuo", "tauvo", "hfds"):
        print(f"Fitting training-only seasonal climatology: {variable}", flush=True)
        field = heat if variable == "hfds" else ds[variable]
        climatology[variable] = (
            field.isel(time=train_indices)
            .groupby("time.dayofyear")
            .mean("time")
            .compute()
        )
    climatology["hfds_anomalies"] = climatology["hfds"] - full_clim
    clim = xr.Dataset(climatology)
    if clim.sizes["dayofyear"] != 73:
        raise ValueError("Expected 73 five-day seasonal slots")
    for variable in clim:
        clim[variable].encoding = {"chunks": (1, 180, 360)}
    clim.to_zarr(assets / "climatology.zarr", mode="w", consolidated=True)
    full_clim.rename("hfds").to_dataset().to_zarr(
        assets / "original_heat_climatology.zarr", mode="w", consolidated=True
    )
    reopened = xr.open_zarr(assets / "original_anomaly.zarr", chunks=None)
    for index in (0, 1241, 2810, 4744):
        np.testing.assert_array_equal(
            reopened.hfds_anomalies.isel(time=index), anomaly.isel(time=index)
        )
    payload = {
        "status": "complete",
        "sources": sources,
        "source_metadata_sha256": {
            str(Path(p) / ".zmetadata"): sha256(Path(p) / ".zmetadata")
            for p in sources.values()
        },
        "train_start": str(cfg.train_time.start),
        "train_end_exclusive": str(cfg.train_time.end),
        "train_frames": len(train_indices),
        "train_first": str(times[train_indices[0]]),
        "train_last": str(times[train_indices[-1]]),
        "full_frames": len(times),
        "full_first": str(times[0]),
        "full_last": str(times[-1]),
        "calendar": "julian",
        "seasonal_slots": list(map(int, clim.dayofyear.values)),
        "anomaly_stats": {
            "mean": anomaly_mean.item(),
            "std": anomaly_std.item(),
            "dtype": str(anomaly.dtype),
        },
        "anomaly_definition": "hfds minus original full-record dayofyear mean; legacy frozen mean/std",
        "climatology_definition": "training-only physical forcing; heat anomaly equals train seasonal Q minus original full-record seasonal Q",
        "cache_metadata_sha256": {
            name: sha256(assets / name / ".zmetadata")
            for name in (
                "original_anomaly.zarr",
                "climatology.zarr",
                "original_heat_climatology.zarr",
            )
        },
    }
    temp = manifest_path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, manifest_path)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
