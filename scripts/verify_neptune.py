"""Check one real OM4 example on a compute node, without optimizer updates."""

import argparse
import json
import time
from pathlib import Path

import torch

from ocean_emulators.config import NeptuneConfig, TrainConfig
from ocean_emulators.constants import BOUNDARY_VARS, PROGNOSTIC_VARS, TensorMap
from ocean_emulators.datasets import TorchTrainDataset
from ocean_emulators.utils.data import Normalize
from ocean_emulators.utils.device import set_device
from ocean_emulators.utils.multiton import MultitonScope
from ocean_emulators.utils.train import collate_raw_train_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = TrainConfig.from_yaml_and_cli([args.config])
    if not isinstance(config.model, NeptuneConfig):
        raise ValueError("This smoke check requires a Neptune config.")
    device = torch.device(args.device)
    set_device(device)
    torch.manual_seed(config.experiment.rand_seed)
    prognostic = PROGNOSTIC_VARS[config.experiment.prognostic_vars_key]
    boundary = BOUNDARY_VARS[config.experiment.boundary_vars_key]
    with MultitonScope():
        TensorMap.init_instance(
            config.experiment.prognostic_vars_key, config.experiment.boundary_vars_key
        )
        container = config.data.build(
            config.experiment.resolved_data_root, prognostic, boundary
        )
        src = container.source
        Normalize.init_instance(src, prognostic, boundary)
        dataset = TorchTrainDataset(
            src.map_data(lambda ds: ds.sel(time=config.train_time.time_slice)),
            prognostic,
            boundary,
            hist=1,
            steps=1,
            normalize_before_mask=config.data.normalize_before_mask,
            masked_fill_value=config.data.masked_fill_value,
        )
        data = dataset.to_train_data(collate_raw_train_data([dataset[0]]))
        wet = src.masks.prognostic_with_hist(1).to(device)
        model = config.model.build(
            in_channels=2 * (len(prognostic) + len(boundary)),
            out_channels=2 * len(prognostic),
            hist=1,
            wet=wet,
            area_weights=torch.ones_like(wet[0]),
            static_data=container.static_data,
            lat=torch.from_numpy(src.data.lat.values),
            lon=torch.from_numpy(src.data.lon.values),
        ).to(device)
        model.train(args.backward)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.set_grad_enabled(args.backward):
            output = model(data)[0]
            if not torch.isfinite(output).all() or torch.count_nonzero(output[:, ~wet]):
                raise AssertionError("Non-finite output or nonzero land output.")
            loss = (output[:, wet] - data.get_label(0)[:, wet]).square().mean()
            if args.backward:
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is None or not torch.isfinite(param.grad).all():
                        raise AssertionError(f"Invalid gradient for {name}")
        if device.type == "cuda":
            torch.cuda.synchronize()
        report = dict(
            device=str(device),
            parameters=sum(p.numel() for p in model.parameters()),
            input_shape=list(data.get_input(0).shape),
            output_shape=list(output.shape),
            phase=data.get_calendar_phase(0).tolist(),
            backward=args.backward,
            seconds=time.perf_counter() - started,
            random_initialization_mse=loss.item(),
            peak_cuda_bytes=torch.cuda.max_memory_allocated()
            if device.type == "cuda"
            else None,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
