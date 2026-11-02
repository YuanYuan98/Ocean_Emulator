import copy
import datetime
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from pydantic import TypeAdapter, ValidationError

from ocean_emulators.config import (
    AnyModelConfig,
    DynamicLossConfig,
    EvalConfig,
    NeptuneConfig,
    SamudraConfig,
    TrainConfig,
)
from ocean_emulators.datasets import TrainData
from ocean_emulators.models.neptune import (
    FourierPosition,
    Neptune,
    SphericalConvolution,
    spherical_grid,
)
from ocean_emulators.stepper import Stepper
from ocean_emulators.train import Trainer

pytest.importorskip("torch_harmonics")


def coordinates(nlat=8, nlon=16):
    lat = torch.tensor(
        np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(nlat)[0])),
        dtype=torch.float32,
    )
    lon = torch.arange(nlon) * (360 / nlon) + (180 / nlon)
    return lat, lon


def make_model(**kwargs):
    lat, lon = coordinates()
    wet = torch.ones(4, 8, 16, dtype=torch.bool)
    wet[:, 2:4, 3:6] = False
    return Neptune(
        in_channels=6,
        out_channels=4,
        hist=1,
        wet=wet,
        lat=lat,
        lon=lon,
        hidden_channels=8,
        condition_dim=12,
        noise_dim=4,
        spatial_fourier_bands=3,
        temporal_fourier_bands=2,
        day_length_fourier_bands=2,
        **kwargs,
    )


@pytest.mark.parametrize("ascending", [True, False])
def test_sht_preserves_constant_and_degree_one(ascending):
    lat, lon = coordinates()
    if not ascending:
        lat = lat.flip(0)
    spectral = SphericalConvolution(
        1, 8, 16, (2, 2), "legendre-gauss", spherical_grid(lat, lon, "legendre-gauss")
    )
    with torch.no_grad():
        spectral.weight.zero_()
        spectral.weight[..., 0] = 1
    latitude = torch.deg2rad(lat)[:, None]
    longitude = torch.deg2rad(lon)[None, :]
    field = (1 + 0.4 * latitude.sin() + 0.7 * latitude.cos() * longitude.cos())[
        None, None
    ]
    torch.testing.assert_close(spectral(field), field, atol=2e-6, rtol=2e-6)
    # The truncation must reject degree 3, including its non-zonal component.
    high_degree = (latitude.cos() ** 3 * (3 * longitude).cos())[None, None]
    torch.testing.assert_close(
        spectral(high_degree), torch.zeros_like(high_degree), atol=2e-6, rtol=0
    )


def test_grid_validation_and_day_length():
    lat, lon = coordinates()
    with pytest.raises(ValueError, match="Latitude"):
        spherical_grid(torch.linspace(-89, 89, 8), lon, "legendre-gauss")
    with pytest.raises(ValueError, match="Longitude"):
        spherical_grid(lat, torch.linspace(0, 360, 16), "legendre-gauss")
    position = FourierPosition(torch.tensor([-90.0, 0.0, 90.0]), lon, 8, 3, 2, 2)
    hours = position.day_length(
        torch.tensor([80 / 365.2425, 172 / 365.2425, 355 / 365.2425])
    )
    torch.testing.assert_close(hours[:, 1], torch.full((3,), 12.0))
    torch.testing.assert_close(hours[1], torch.tensor([0.0, 12.0, 24.0]))
    torch.testing.assert_close(hours[2], torch.tensor([24.0, 12.0, 0.0]))


def test_noise_dates_masks_and_state_dict():
    torch.manual_seed(11)
    model = make_model().eval()
    x = torch.randn(2, 6, 8, 16)
    phase = torch.tensor([0.1, 0.7])
    a = model.forward_once(x, phase)
    assert a.shape == (2, 4, 8, 16)
    assert torch.count_nonzero(a[:, ~model.wet]) == 0
    torch.testing.assert_close(a, model.forward_once(x, phase), rtol=0, atol=0)
    assert not torch.allclose(a, model.forward_once(x, phase + 0.05))
    assert not torch.allclose(a, model.forward_once(x, phase, torch.ones(2, 4)))
    clone = make_model().eval()
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(a, clone.forward_once(x, phase), rtol=0, atol=0)
    assert all(not p.is_complex() for p in model.parameters())
    with pytest.raises(ValueError, match="calendar"):
        model.forward_once(x)


def test_separate_fourier_bands_and_polar_day_length():
    lat, lon = coordinates()
    phase = torch.tensor([0.0, 0.4])
    position = FourierPosition(lat, lon, 8, 8, 4, 4)
    other = FourierPosition(lat, lon, 8, 8, 2, 2)
    position.projection = torch.nn.Identity()
    other.projection = torch.nn.Identity()
    encoded = position(phase)
    assert encoded.shape[1] == 48
    # Reducing temporal bands must leave every spatial feature unchanged.
    torch.testing.assert_close(encoded[:, :32], other(phase)[:, :32])
    torch.testing.assert_close(
        position(phase), position(phase + 1), atol=2e-5, rtol=1e-5
    )
    polar = FourierPosition(torch.tensor([-90.0, 90.0]), lon, 8, 8, 4, 4)
    polar.projection = torch.nn.Identity()
    summer = polar(torch.tensor([172 / 365.2425]))
    # Last eight features encode day length. 0 and 24 hours cannot alias.
    assert not torch.allclose(summer[:, -8:, 0], summer[:, -8:, 1])


def test_fixed_noise_member_is_reproducible_and_batch_independent():
    model = make_model(inference_noise="fixed", inference_noise_seed=15).eval()
    x = torch.randn(2, 6, 8, 16)
    phase = torch.tensor([0.1, 0.7])
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        prediction = model.forward_once(x, phase)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        partitioned = torch.cat(
            [model.forward_once(x[i : i + 1], phase[i : i + 1]) for i in range(2)]
        )
        torch.testing.assert_close(prediction, partitioned)
        explicit = model.forward_once(x, phase, model.fixed_noise.expand(2, -1))
        torch.testing.assert_close(prediction, explicit, atol=0, rtol=0)
        assert not torch.allclose(
            prediction, model.forward_once(x, phase, torch.zeros(2, 4))
        )
        another = make_model(inference_noise="fixed", inference_noise_seed=16).eval()
        another.load_state_dict(model.state_dict())
        assert not torch.allclose(model.fixed_noise, another.fixed_noise)
        assert not torch.allclose(prediction, another.forward_once(x, phase))


@pytest.mark.parametrize("mode", ["zero", "sample"])
def test_other_noise_modes_remain_available(mode):
    model = make_model(inference_noise=mode).eval()
    x, phase = torch.randn(1, 6, 8, 16), torch.tensor([0.2])
    with torch.no_grad():
        a, b = model.forward_once(x, phase), model.forward_once(x, phase)
        if mode == "zero":
            torch.testing.assert_close(
                a, model.forward_once(x, phase, torch.zeros(1, 4))
            )
        else:
            assert not torch.allclose(a, b)


@pytest.mark.parametrize("checkpointing", [None, "all"])
@pytest.mark.parametrize("decoder_mid_channels", [None, 16])
def test_multistep_backward_and_forcing(checkpointing, decoder_mid_channels):
    model = make_model(
        checkpointing=checkpointing, decoder_mid_channels=decoder_mid_channels
    )
    data = TrainData(4)
    for step in range(4):
        data.append(
            torch.randn(2, 6, 8, 16, requires_grad=True),
            torch.randn(2, 4, 8, 16),
            torch.tensor([0.1, 0.7]) + step * 0.03,
        )
    loss = model(data, torch.nn.MSELoss())
    loss.backward()
    assert torch.isfinite(loss)
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    # Later prognostic truth must not enter student rollout, but prescribed forcing must.
    later_grad = data.get_input(1).grad
    first_grad = data.get_input(0).grad
    assert later_grad is not None and first_grad is not None
    assert torch.count_nonzero(later_grad[:, :4]) == 0
    assert torch.count_nonzero(later_grad[:, 4:]) > 0
    assert torch.count_nonzero(first_grad[:, :4]) > 0


def test_om4_parameter_budget():
    """Keep the disclosed 256-wide lift while matching the rounded 13M budget."""
    lat, lon = coordinates()
    config_path = Path(__file__).resolve().parents[1] / "configs/neptune_om4/model.yaml"
    config = NeptuneConfig.model_validate(yaml.safe_load(config_path.read_text()))
    model = Neptune(
        in_channels=162,
        out_channels=154,
        hist=1,
        wet=torch.ones(154, 8, 16, dtype=torch.bool),
        lat=lat,
        lon=lon,
        **config.model_dump(exclude={"type", "add_3d_coordinates"}),
    )
    assert sum(p.numel() for p in model.parameters()) == 13_281_946
    assert model.lift.conv.out_channels == 256
    for block in model.decoders:
        assert block.convs[0].conv.in_channels == 512
        assert block.convs[0].conv.out_channels == 512
        assert block.convs[1].conv.out_channels == 256


def test_checkpoint_replay_preserves_noise_and_dropout():
    model = make_model(dropout=0.2)
    other = copy.deepcopy(model)
    other.checkpointing = "all"
    x = torch.randn(2, 6, 8, 16)
    phase = torch.tensor([0.1, 0.7])
    torch.manual_seed(123)
    a = model.forward_once(x, phase)
    a.square().mean().backward()
    torch.manual_seed(123)
    b = other.forward_once(x, phase)
    b.square().mean().backward()
    torch.testing.assert_close(a, b)
    for p, q in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad)


def test_consecutive_residual_reconstruction():
    model = make_model(pred_residuals=True)
    inputs = torch.zeros(1, 6, 8, 16)
    inputs[:, :2] = 100  # Older state must not enter either reconstruction.
    inputs[:, 2:4] = 10
    increments = torch.ones(1, 4, 8, 16)
    increments[:, 2:] = 2
    prediction = model.decode_prediction(inputs, increments)
    expected = torch.full_like(prediction, 11)
    expected[:, 2:] = 13
    torch.testing.assert_close(prediction, torch.where(model.wet, expected, 0))


@pytest.mark.parametrize("residuals", [False, True])
def test_validation_and_loss_update_keep_calendar(residuals):
    model = make_model(pred_residuals=residuals).eval()
    data = TrainData(4)
    data.append(
        torch.randn(2, 6, 8, 16), torch.randn(2, 4, 8, 16), torch.tensor([0.2, 0.6])
    )
    expected = model(data)[0]
    validation = Stepper.validate_batch(model, data, torch.nn.MSELoss())
    torch.testing.assert_close(validation.gen_data, expected)
    trainer = Mock(spec=Trainer)
    trainer.model = model
    trainer.loss_fn = Mock()
    Trainer._call_loss_update(trainer, data)
    trainer.loss_fn.update.assert_called_once()
    torch.testing.assert_close(trainer.loss_fn.update.call_args.args[0], expected)


def test_config_selection_and_om4_protocol():
    assert isinstance(TypeAdapter(AnyModelConfig).validate_python({}), SamudraConfig)
    config: AnyModelConfig = TypeAdapter(AnyModelConfig).validate_python(
        {"type": "neptune"}
    )
    assert isinstance(config, NeptuneConfig)
    with pytest.raises(ValidationError):
        NeptuneConfig.model_validate({"type": "neptune", "add_3d_coordinates": True})
    folder = Path(__file__).resolve().parents[1] / "configs/neptune_om4"
    train = TrainConfig.model_validate(
        yaml.safe_load((folder / "train.yaml").read_text())
    )
    evaluate = EvalConfig.model_validate(
        yaml.safe_load((folder / "eval.yaml").read_text())
    )
    assert isinstance(train.model, NeptuneConfig)
    assert train.model == evaluate.model
    assert train.model == NeptuneConfig.model_validate(
        yaml.safe_load((folder / "model.yaml").read_text())
    )
    assert isinstance(
        EvalConfig.from_yaml_and_cli([str(folder / "eval.yaml")]).model, NeptuneConfig
    )
    assert (
        train.epochs == 70
        and train.batch_size * 2 * train.gradient_accumulation_steps == 32
    )
    assert train.steps == [4] and not train.model.pred_residuals
    assert isinstance(train.loss, DynamicLossConfig)
    assert (
        train.data.hist == 1 and train.loss.type == "dynamic" and train.loss.limit == 20
    )


def _ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=90),
    )
    try:
        torch.manual_seed(10 + rank)
        model = torch.nn.parallel.DistributedDataParallel(
            make_model(checkpointing="all", decoder_mid_channels=16)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        noise_inputs = []
        handle = model.module.condition.register_forward_pre_hook(
            lambda module, args: noise_inputs.append(args[0][:, -4:].detach().clone())
        )
        data = TrainData(4)
        for step in range(2):
            data.append(
                torch.randn(1, 6, 8, 16),
                torch.randn(1, 4, 8, 16),
                torch.tensor([0.1 + step * 0.03]),
            )
        torch.manual_seed(
            321
        )  # The production trainer uses the same seed on every rank.
        for _ in range(2):
            optimizer.zero_grad()
            loss = model(data, torch.nn.MSELoss())
            loss.backward()
            optimizer.step()
        parameters = torch.cat([p.detach().flatten() for p in model.parameters()])
        reference = parameters.clone()
        dist.broadcast(reference, src=0)
        torch.testing.assert_close(parameters, reference, rtol=0, atol=0)
        rank_noises = [torch.zeros_like(noise_inputs[0]) for _ in range(2)]
        dist.all_gather(rank_noises, noise_inputs[0])
        assert not torch.equal(rank_noises[0], rank_noises[1])
        handle.remove()
    finally:
        dist.destroy_process_group()


def test_two_rank_ddp(tmp_path):
    mp.spawn(
        _ddp_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True
    )
