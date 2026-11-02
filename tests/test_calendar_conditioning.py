import datetime

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from ocean_emulators.datasets import InferenceDataset, RawTrainData, TorchTrainDataset
from ocean_emulators.models.base import BaseModel
from ocean_emulators.utils.calendar import calendar_phase
from ocean_emulators.utils.data import DataSource, Masks, Normalize
from ocean_emulators.utils.device import set_device
from ocean_emulators.utils.multiton import MultitonScope
from ocean_emulators.utils.train import collate_raw_train_data


def test_cf_and_numpy_calendars():
    julian = [
        cftime.DatetimeJulian(1900, 2, 28, 12),
        cftime.DatetimeJulian(1900, 3, 1, 12),
    ]
    np.testing.assert_allclose(calendar_phase(julian), np.array([58.5, 60.5]) / 366)
    gregorian = np.array(
        ["1900-02-28T12:00", "1900-03-01T12:00"], dtype="datetime64[m]"
    )
    np.testing.assert_allclose(calendar_phase(gregorian), np.array([58.5, 59.5]) / 365)
    np.testing.assert_allclose(
        calendar_phase([cftime.Datetime360Day(2000, 7, 1)]), [0.5]
    )
    np.testing.assert_allclose(calendar_phase([cftime.DatetimeNoLeap(2001, 1, 1)]), [0])
    with pytest.raises(ValueError, match="NaT"):
        calendar_phase([np.datetime64("NaT")])


@pytest.fixture
def dated_source():
    set_device(torch.device("cpu"))
    # Includes a Julian leap-day crossing; indices, not elapsed-time arithmetic,
    # determine the cadence and the last input timestamp at each AR call.
    times = [
        cftime.DatetimeJulian(2000, 2, 24, 12) + datetime.timedelta(days=i * 5)
        for i in range(20)
    ]
    fields = {
        name: (("time", "lat", "lon"), np.ones((20, 4, 8), dtype=np.float32))
        for name in ("p", "f")
    }
    data = xr.Dataset(
        fields, coords={"time": times, "lat": np.arange(4), "lon": np.arange(8)}
    )
    means = xr.Dataset({"p": 0.0, "f": 0.0})
    stds = xr.Dataset({"p": 1.0, "f": 1.0})
    source = DataSource(
        "dated", data, means, stds, masks=Masks(torch.ones(1, 4, 8), torch.ones(4, 8))
    )
    with MultitonScope():
        Normalize.init_instance(
            source, prognostic_var_names=["p"], boundary_var_names=["f"]
        )
        yield source


@pytest.mark.parametrize("stride", [1, 2])
def test_train_dates_follow_hist_stride_and_batch(dated_source, stride):
    dataset = TorchTrainDataset(
        dated_source,
        ["p"],
        ["f"],
        hist=1,
        steps=3,
        normalize_before_mask=True,
        masked_fill_value=0.0,
        stride=stride,
    )
    raw = collate_raw_train_data([dataset[0], dataset[1]])
    raw.to(torch.device("cpu"))
    data = dataset.to_train_data(raw)
    data.to(torch.device("cpu"))
    for step in range(3):
        indices = np.array([0, 1]) + (step * 2 + 1) * stride
        expected = calendar_phase(dated_source.data.time.values[indices])
        np.testing.assert_allclose(data.get_calendar_phase(step).numpy(), expected)


def test_mixed_dated_batch_rejected():
    a, b = RawTrainData("same"), RawTrainData("same")
    a.insert(torch.zeros(1), torch.zeros(1), torch.tensor(0.1))
    b.insert(torch.zeros(1), torch.zeros(1))
    with pytest.raises(ValueError, match="mix dated"):
        collate_raw_train_data([a, b])


def test_inference_chunk_uses_absolute_input_dates(dated_source):
    dataset = InferenceDataset(
        dated_source,
        ["p"],
        ["f"],
        hist=1,
        normalize_before_mask=True,
        masked_fill_value=0.0,
        long_rollout=True,
    )

    class CalendarRecorder(BaseModel):
        requires_calendar = True

        def __init__(self):
            super().__init__(
                4, 2, torch.ones(2, 4, 8), 1, False, 3, "circular", None, 0
            )
            self.phases = []

        def forward_step(self, x, phase=None):
            self.phases.append(phase.clone())
            return x[:, :2] + phase[:, None, None, None]

    model = CalendarRecorder()
    whole = model.inference(dataset, dataset.initial_prognostic, num_steps=4)
    all_phases = torch.cat(model.phases)
    expected = calendar_phase(dated_source.data.time.values[[1, 3, 5, 7]])
    np.testing.assert_allclose(all_phases.numpy(), expected)
    # Run two chunks and compare predictions. Attribute name comes from the
    # common inference output, so this also checks the standard eval path.
    model.phases.clear()
    first = model.inference(dataset, dataset.initial_prognostic, num_steps=2)
    second = model.inference(
        dataset, first.prediction[-1:].clone(), steps_completed=2, num_steps=2
    )
    torch.testing.assert_close(
        whole.prediction, torch.cat((first.prediction, second.prediction))
    )
    torch.testing.assert_close(torch.cat(model.phases), all_phases)
