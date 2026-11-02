import dataclasses

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from ocean_emulators.datasets import InferenceDataset, TorchTrainDataset
from ocean_emulators.utils.data import DataSource, Masks
from ocean_emulators.utils.forcing_revision import EXCLUDED, FORCING, apply_fields
from ocean_emulators.utils.loss import DynamicLoss


def source():
    times = [cftime.DatetimeJulian(2001, 1, 3 + 5 * i) for i in range(6)]
    shape = (6, 2, 3)
    fields = {
        name: (
            ("time", "lat", "lon"),
            np.arange(36, dtype="float32").reshape(shape) + i,
        )
        for i, name in enumerate(("zos", *FORCING))
    }
    data = xr.Dataset(
        fields, coords={"time": times, "lat": [-45.0, 45.0], "lon": [0.0, 120.0, 240.0]}
    )
    means = xr.Dataset({v: np.float64(0.123456789) for v in fields})
    stds = xr.Dataset({v: np.float64(3.0) for v in fields})
    mask = torch.tensor([[True, True, False], [True, False, True]])
    return DataSource("synthetic", data, means, stds, Masks(mask[None], mask))


@pytest.mark.parametrize("mode", list(EXCLUDED))
def test_same_input_ablation_for_training_and_inference(mode, monkeypatch):
    monkeypatch.setattr(
        "ocean_emulators.datasets.get_device", lambda: torch.device("cpu")
    )
    monkeypatch.setattr("ocean_emulators.datasets.using_gpu", lambda: False)
    original = source()
    climatology = (
        original.data[list(FORCING)]
        .assign_coords(
            dayofyear=("time", [t.dayofyr for t in original.data.time.values])
        )
        .swap_dims({"time": "dayofyear"})
        .drop_vars("time")
    )
    fields, means, stds = apply_fields(
        original.data, original.means, original.stds, mode, climatology
    )
    altered = dataclasses.replace(
        original,
        data=fields,
        means=means,
        stds=stds,
        forcing_revision_excluded=EXCLUDED[mode],
    )
    args = dict(
        src=altered,
        prognostic_var_names=["zos"],
        boundary_var_names=list(FORCING),
        hist=1,
        normalize_before_mask=True,
        masked_fill_value=0.0,
    )
    inference = InferenceDataset(**args, long_rollout=True)
    training = TorchTrainDataset(**args, steps=1)
    raw = fields.isel(time=slice(0, 2))
    ocean = torch.from_numpy(
        raw[["zos"]].to_array().transpose("time", "variable", "lat", "lon").values
    ).float()[None]
    boundary = torch.from_numpy(
        raw[list(FORCING)].to_array().transpose("time", "variable", "lat", "lon").values
    ).float()[None]
    prepared = training._prep_tensor_steps(ocean, boundary)
    expected = inference.get_initial_input()
    torch.testing.assert_close(prepared, expected, atol=1e-6, rtol=1e-6)
    for index, name in enumerate(FORCING):
        slots = [2 + index, 2 + 4 + index]
        if name in EXCLUDED[mode]:
            assert torch.count_nonzero(prepared[:, slots]) == 0
            assert (
                torch.count_nonzero(
                    inference.merge_prognostic_and_boundary(prepared[:, :2], 1)[
                        :, slots
                    ]
                )
                == 0
            )
    # Raw original objects and normalization must remain untouched.
    xr.testing.assert_identical(original.means, means)
    xr.testing.assert_identical(original.stds, stds)
    xr.testing.assert_identical(original.data.zos, fields.zos)
    assert original.forcing_revision_excluded == ()


def test_climatology_repeats_season_and_keeps_nonzero_heat_feature():
    original = source()
    seasonal = original.data[list(FORCING)].groupby("time.dayofyear").mean("time")
    seasonal["hfds_anomalies"] = seasonal.hfds - 10.5
    first, _, _ = apply_fields(
        original.data, original.means, original.stds, "climatology", seasonal
    )
    later = original.data.assign_coords(
        time=[
            cftime.DatetimeJulian(2005, t.month, t.day)
            for t in original.data.time.values
        ]
    )
    second, _, _ = apply_fields(
        later, original.means, original.stds, "climatology", seasonal
    )
    for variable in FORCING:
        np.testing.assert_array_equal(first[variable], second[variable])
    np.testing.assert_array_equal(first.hfds_anomalies, first.hfds - 10.5)
    assert np.any(first.hfds_anomalies != 0)


def test_paper_dynamic_loss_matches_time_major_reference_and_preserves_legacy():
    # Strongly distinguish same-variable temporal pairing from adjacent variables.
    errors = torch.tensor([1.0, 4.0, 25.0, 9.0, 16.0, 100.0])

    def metric(pred, target):
        return errors

    scales = torch.tensor([2.0, 3.0, 7.0])
    paper = DynamicLoss(
        metric,
        limit=20.0,
        device=torch.device("cpu"),
        num_channels=3,
        history_grouping="time_major",
    )
    legacy = DynamicLoss(metric, limit=20.0, device=torch.device("cpu"), num_channels=3)
    for loss in (paper, legacy):
        loss.load_state_dict({"per_channel_scale": scales.clone()})
    torch.testing.assert_close(paper(None, None), errors * scales.repeat(2))
    torch.testing.assert_close(legacy(None, None), errors * scales.repeat_interleave(2))
    inverse = errors.reciprocal().reshape(2, 3).mean(0)
    inverse = inverse.clamp(inverse.min(), inverse.min() * 20.0)
    expected = (scales * 24 + inverse) / 25
    paper.update(None, None)
    torch.testing.assert_close(paper.loss_scale_per_channel(), expected)
    restored = DynamicLoss(
        metric,
        limit=20.0,
        device=torch.device("cpu"),
        num_channels=3,
        history_grouping="time_major",
    )
    restored.load_state_dict(paper.state_dict())
    torch.testing.assert_close(restored(None, None), paper(None, None))
    with pytest.raises(ValueError, match="grouping"):
        legacy.load_state_dict(paper.state_dict())
