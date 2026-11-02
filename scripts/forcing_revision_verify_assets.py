"""Compare frozen assets to both original loaders and audit real input ablations."""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from ocean_emulators.config import TimeConfig, TrainConfig
from ocean_emulators.constants import BOUNDARY_VARS, PROGNOSTIC_VARS
from ocean_emulators.datasets import InferenceDataset
from ocean_emulators.utils.data import DataSource
from ocean_emulators.utils.forcing_revision import EXCLUDED


def main():
    campaign = Path(sys.argv[1])
    cfg = TrainConfig.from_yaml_and_cli(
        [str(campaign / "configs/train_climatology.yaml")]
    )
    root = cfg.experiment.resolved_data_root
    prog = PROGNOSTIC_VARS[cfg.experiment.prognostic_vars_key]
    bound = BOUNDARY_VARS[cfg.experiment.boundary_vars_key]
    assets = campaign / "assets"
    manifest = json.loads((assets / "manifest.json").read_text())
    original_anomaly = xr.open_zarr(
        assets / "original_anomaly.zarr", chunks=None
    ).hfds_anomalies
    report = {"status": "running", "original_loader_comparison": {}}
    expected_inputs = {}
    for use_dask in (False, True):
        print(f"Comparing ORIGINAL DataSource(use_dask={use_dask})", flush=True)
        source = DataSource.from_locations(
            data_location=root.resolve(cfg.data.data_location),
            means_location=root.resolve(cfg.data.data_means_location),
            stds_location=root.resolve(cfg.data.data_stds_location),
            prognostic_var_names=prog,
            boundary_var_names=bound,
            static_data_vars=[],
            use_dask=use_dask,
        )
        maximum = 0.0
        for start in range(0, len(original_anomaly.time), 64):
            a = original_anomaly.isel(time=slice(start, start + 64)).values
            b = source.data.hfds_anomalies.isel(time=slice(start, start + 64)).values
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                raise ValueError("Frozen anomaly changed NaN geometry")
            maximum = max(maximum, float(np.nanmax(np.abs(a - b))))
        mean = float(source.means.hfds_anomalies.item())
        std = float(source.stds.hfds_anomalies.item())
        cached = manifest["anomaly_stats"]
        # Formal train + standalone eval both use the NumPy OM4_TORCH source.
        # The old optional in-training inference source uses a different float32
        # reduction tree. Report that backend difference; never claim bit equality.
        # The actual selected backend remains an exact-equality gate below.
        diagnostic_limit = 2e-4 if use_dask else 0.0
        assert maximum / std <= diagnostic_limit, (use_dask, maximum, std)
        assert abs(mean - cached["mean"]) / std < 1e-5
        assert abs(std - cached["std"]) / std < 1e-5
        if not use_dask:
            assert maximum == 0
            assert mean == cached["mean"] and std == cached["std"]
            for label, start, end in [
                ("train", "1975-01-03", "1975-02-03"),
                ("heldout", "2015-01-03", "2015-02-03"),
            ]:
                dataset = InferenceDataset(
                    source.slice(TimeConfig(start=start, end=end)),
                    prog,
                    bound,
                    hist=1,
                    normalize_before_mask=True,
                    masked_fill_value=0.0,
                    long_rollout=True,
                )
                expected_inputs[label] = dataset.get_initial_input()
        report["original_loader_comparison"][str(use_dask)] = {
            "max_abs_anomaly_difference": maximum,
            "max_difference_in_original_std_units": maximum / std,
            "original_mean": mean,
            "original_std": std,
            "cache_mean": cached["mean"],
            "cache_std": cached["std"],
            "max_difference_std_units_limit": diagnostic_limit,
            "used_for_training_and_standalone_eval": not use_dask,
        }
        del source
        gc.collect()
    report["variants"] = {}
    for mode in ("wind_only", "heat_only", "climatology", "no_forcing"):
        c = TrainConfig.from_yaml_and_cli(
            [str(campaign / f"configs/train_{mode}.yaml")]
        )
        container = c.data.build(root, prog, bound)
        checks = {}
        for label, start, end in [
            ("train", "1975-01-03", "1975-02-03"),
            ("heldout", "2015-01-03", "2015-02-03"),
        ]:
            dataset = InferenceDataset(
                container.source.slice(TimeConfig(start=start, end=end)),
                prog,
                bound,
                hist=1,
                normalize_before_mask=True,
                masked_fill_value=0.0,
                long_rollout=True,
            )
            actual = dataset.get_initial_input()
            expected = expected_inputs[label]
            assert torch.equal(actual[:, :154], expected[:, :154])
            for index, name in enumerate(bound):
                slots = [154 + index, 158 + index]
                if name in EXCLUDED[mode]:
                    assert torch.count_nonzero(actual[:, slots]) == 0
                elif mode != "climatology":
                    torch.testing.assert_close(
                        actual[:, slots], expected[:, slots], rtol=0, atol=0
                    )
            checks[label] = {
                "ocean_input_exactly_unchanged": True,
                "excluded_forcing_exact_zero": True,
            }
        if mode == "climatology":
            data = container.source.data
            first = (
                data[list(bound)]
                .sel(time=slice("2015-01-01", "2015-01-31"))
                .to_array()
                .values
            )
            later = (
                data[list(bound)]
                .sel(time=slice("2017-01-01", "2017-01-31"))
                .to_array()
                .values
            )
            np.testing.assert_array_equal(first, later)
            checks["same_season_across_heldout_years_exactly_equal"] = True
        report["variants"][mode] = checks
        del container
        gc.collect()
    report["training_only_climatology"] = {
        k: manifest[k]
        for k in (
            "train_start",
            "train_end_exclusive",
            "train_frames",
            "train_first",
            "train_last",
        )
    }
    report["status"] = "passed"
    (campaign / "preflight/assets_verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
