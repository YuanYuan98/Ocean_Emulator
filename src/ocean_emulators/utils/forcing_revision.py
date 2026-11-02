"""Training-consistent forcing information ablations; original data remain immutable."""

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

from ocean_emulators.utils.data import DataSource

FORCING = ("tauuo", "tauvo", "hfds", "hfds_anomalies")
EXCLUDED = {
    "full": (),
    "wind_only": ("hfds", "hfds_anomalies"),
    "heat_only": ("tauuo", "tauvo"),
    "climatology": (),
    "no_forcing": FORCING,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def apply_fields(data, means, stds, mode, climatology=None):
    """Replace physical forcing fields; preserve original normalization and masks.

    Excluded inputs are cheap constant views so the loader need not read them.
    Dataset tensor preparation additionally enforces exact normalized zeros, avoiding
    float32/float64 rounding artifacts. The same code serves train/val/inference.
    """
    excluded = EXCLUDED[mode]
    data = data.copy(deep=False)
    if mode == "climatology":
        if climatology is None:
            raise ValueError("Missing training-only climatology")
        selected = climatology.sel(dayofyear=data.time.dt.dayofyear)
        for variable in FORCING:
            data[variable] = selected[variable].drop_vars("dayofyear", errors="ignore")
    for variable in excluded:
        template = data[variable]
        # Zero strides: no time x space materialization or original field reads.
        constant = np.broadcast_to(np.asarray(means[variable].item()), template.shape)
        data[variable] = xr.DataArray(
            constant, dims=template.dims, coords=template.coords, attrs=template.attrs
        )
    return data, means, stds


def build_source(
    *,
    data_location,
    means_location,
    stds_location,
    prognostic_var_names,
    boundary_var_names,
    static_data_vars,
    use_dask,
    mode,
    assets,
):
    if tuple(boundary_var_names) != FORCING:
        raise ValueError("Expected unchanged four-channel forcing layout")
    assets = Path(assets)
    manifest = json.loads((assets / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Forcing preprocessing cache is incomplete")
    for name, location in [
        ("data", data_location),
        ("means", means_location),
        ("stds", stds_location),
    ]:
        if Path(str(location)).resolve() != Path(manifest["sources"][name]).resolve():
            raise ValueError(f"Forcing cache belongs to another {name} source")
    # This campaign intentionally uses only local, audited paper-era data.
    for path, expected in manifest["source_metadata_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Data metadata changed since cache preparation: {path}")
    for name, expected in manifest["cache_metadata_sha256"].items():
        if sha256(assets / name / ".zmetadata") != expected:
            raise ValueError(f"Frozen preprocessing cache metadata changed: {name}")
    chunks = {} if use_dask else None
    data = data_location.open(chunks)
    means = means_location.open(chunks).copy()
    stds = stds_location.open(chunks).copy()
    anomaly = xr.open_zarr(assets / "original_anomaly.zarr", chunks=chunks)
    if not data.time.equals(anomaly.time):
        raise ValueError("Frozen anomaly and raw data time coordinates differ")
    if data.hfds.shape != anomaly.hfds_anomalies.shape:
        raise ValueError("Frozen anomaly and raw heat flux shapes differ")
    data["hfds_anomalies"] = anomaly.hfds_anomalies
    # Preserve scalar dtype as well as values from original compute_anomalies.
    dtype = manifest["anomaly_stats"]["dtype"]
    means["hfds_anomalies"] = np.asarray(manifest["anomaly_stats"]["mean"], dtype=dtype)
    stds["hfds_anomalies"] = np.asarray(manifest["anomaly_stats"]["std"], dtype=dtype)
    climatology = None
    if mode == "climatology":
        # Dask indexing keeps the full repeated seasonal record lazy.
        climatology = xr.open_zarr(assets / "climatology.zarr", chunks={})
    data, means, stds = apply_fields(data, means, stds, mode, climatology)
    source = DataSource.from_datasets(
        data,
        means,
        stds,
        prognostic_var_names=prognostic_var_names,
        boundary_var_names=boundary_var_names,
        static_data_vars=static_data_vars,
        name=f"forcing_revision:{mode}:{data_location}:{use_dask}",
    )
    return dataclasses.replace(source, forcing_revision_excluded=EXCLUDED[mode])
