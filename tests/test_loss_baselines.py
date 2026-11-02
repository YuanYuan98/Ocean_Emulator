from functools import partial

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from ocean_emulators.utils.loss import DynamicLoss, decomposed_mse
from ocean_emulators.utils.loss_baselines import AdaLoss, FixedOutputWeightedLoss
from scripts.prepare_tendency_loss import increment_moments, validate_time_spacing


def mse(outputs=6):
    return partial(decomposed_mse, wet=torch.ones(outputs, 1, 1))


def test_frozen_final_weights_match_legacy_dynamic_value_and_gradient():
    dynamic = DynamicLoss(mse(), limit=20, device=torch.device("cpu"), num_channels=3)
    dynamic.load_state_dict({"per_channel_scale": torch.tensor([2.0, 5.0, 11.0])})
    frozen = FixedOutputWeightedLoss(
        mse(), torch.tensor([2.0, 2.0, 5.0, 5.0, 11.0, 11.0])
    )
    pred = torch.arange(6.0).reshape(1, 6, 1, 1).requires_grad_()
    target = torch.zeros_like(pred)
    assert torch.equal(dynamic(pred, target), frozen(pred, target))
    assert torch.equal(
        torch.autograd.grad(dynamic(pred, target).sum(), pred)[0],
        torch.autograd.grad(frozen(pred, target).sum(), pred)[0],
    )
    before = frozen.weights.clone()
    frozen.update(pred, target)
    assert torch.equal(before, frozen.weights)


def test_tendency_loss_is_normalized_increment_error_without_changing_outputs():
    state_std = torch.tensor([2.0, 3.0, 4.0]).repeat(2).reshape(1, 6, 1, 1)
    delta_std = torch.tensor([0.5, 1.0, 2.0]).repeat(2).reshape(1, 6, 1, 1)
    reference = torch.arange(6.0).reshape(1, 6, 1, 1)
    truth = reference + 1
    pred = (truth + 0.25).requires_grad_()
    loss = FixedOutputWeightedLoss(mse(), (state_std / delta_std).square().flatten())
    actual = loss(pred / state_std, truth / state_std)
    expected = (
        ((pred - reference) / delta_std - (truth - reference) / delta_std) ** 2
    ).flatten()
    assert torch.allclose(actual, expected)
    assert torch.allclose(
        torch.autograd.grad(actual.sum(), pred)[0],
        torch.autograd.grad(expected.sum(), pred)[0],
    )


def test_adaloss_inverts_smoothed_error_and_resumes_exactly():
    loss = AdaLoss(
        mse(2), num_channels=2, num_outputs=2, device=torch.device("cpu"), ema_window=2
    )
    target = torch.zeros(1, 2, 1, 1)
    loss.update(torch.tensor([2.0, 4.0]).reshape_as(target), target)
    assert torch.allclose(loss.weights, torch.tensor([1 / 2.5, 1 / 8.5]))
    restored = AdaLoss(
        mse(2), num_channels=2, num_outputs=2, device=torch.device("cpu"), ema_window=2
    )
    restored.load_state_dict(loss.state_dict())
    for obj in [loss, restored]:
        obj.update(torch.tensor([1.0, 2.0]).reshape_as(target), target)
    assert torch.equal(loss.weights, restored.weights)
    assert torch.equal(loss.ema_losses, restored.ema_losses)


def test_adaloss_reduces_group_errors_before_inverting(monkeypatch):
    def doubled_global_mean(errors):
        errors.mul_(2)

    monkeypatch.setattr(
        "ocean_emulators.utils.distributed.all_reduce_mean", doubled_global_mean
    )
    loss = AdaLoss(
        mse(4), num_channels=2, num_outputs=4, device=torch.device("cpu"), ema_window=1
    )
    target = torch.zeros(1, 4, 1, 1)
    loss.update(torch.tensor([1.0, 3.0, 2.0, 4.0]).reshape_as(target), target)
    assert torch.allclose(loss.weights, torch.tensor([0.1, 0.1, 0.05, 0.05]))


def test_increment_statistics_cross_chunks_and_exclude_land():
    values = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    field = xr.DataArray(
        np.stack([values, [np.nan] * 5], axis=-1)[:, None, :], dims=("time", "y", "x")
    )
    for lag in [1, 2]:
        mean, var, count = increment_moments(
            field, np.array([[True, False]]), lag, chunk=2
        )
        expected = values[lag:] - values[:-lag]
        assert mean == pytest.approx(expected.mean())
        assert var == pytest.approx(expected.var())
        assert count == len(expected)


def test_fixed_loss_refuses_different_resume_weights():
    loss = FixedOutputWeightedLoss(mse(2), torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match="differ"):
        loss.load_state_dict({"output_weights": torch.tensor([2.0, 1.0])})


def test_om4_cadence_retains_julian_leap_boundary_without_changing_sample_lag():
    times = np.array(
        [
            cftime.DatetimeJulian(1976, 12, 23, 12),
            cftime.DatetimeJulian(1976, 12, 28, 12),
            cftime.DatetimeJulian(1977, 1, 3, 12),
            cftime.DatetimeJulian(1977, 1, 8, 12),
        ]
    )
    assert validate_time_spacing(times) == {
        "five_day_intervals": 2,
        "six_day_leap_boundaries": 1,
    }


@pytest.mark.parametrize("end_day", [3, 4, 8])
def test_om4_cadence_rejects_missing_or_irregular_states(end_day):
    times = np.array(
        [cftime.DatetimeJulian(1977, 1, 2), cftime.DatetimeJulian(1977, 1, end_day)]
    )
    with pytest.raises(ValueError, match="Unexpected OM4 time spacing"):
        validate_time_spacing(times)
