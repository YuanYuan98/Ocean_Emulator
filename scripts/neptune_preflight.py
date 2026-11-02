"""Exercise the actual distributed trainer for two batches before long training."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from ocean_emulators.config import NeptuneConfig, TrainConfig
from ocean_emulators.stepper import Stepper
from ocean_emulators.train import Trainer
from ocean_emulators.utils.distributed import get_rank, get_world_size
from ocean_emulators.utils.logging import get_model_summary, handle_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = TrainConfig.from_yaml_and_cli([args.config])
    if not isinstance(cfg.model, NeptuneConfig):
        raise ValueError("Expected a Neptune configuration.")
    cfg.experiment.name = f"preflight_batch{cfg.batch_size}"
    cfg.experiment.base_output_dir = str(args.output.parent / "preflight")
    cfg.preemptible = False
    cfg.resume_ckpt_path = None
    cfg.prepare_output_dirs()
    handle_logging(False, cfg.experiment.output_dir)
    trainer = Trainer(cfg)
    try:
        assert get_world_size() * cfg.batch_size * cfg.gradient_accumulation_steps == 32
        trainer.init_data_loaders(4)
        iterator = iter(trainer.train_loader)
        torch.cuda.reset_peak_memory_stats()
        timings = []
        losses = []
        trainer.optimizer.zero_grad(set_to_none=True)
        optimizer_steps = 0
        for iteration in range(2 * cfg.gradient_accumulation_steps):
            data = next(iterator)
            if iteration == 0:
                get_model_summary(trainer.model, data, False)
            trainer.model.train()
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = Stepper.train_batch(trainer.model, data, trainer.loss_fn)
            if not torch.isfinite(output.loss):
                raise RuntimeError("Non-finite preflight loss.")
            (output.loss / cfg.gradient_accumulation_steps).backward()
            for name, parameter in trainer.model.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"Non-finite or missing gradient: {name}")
            if (iteration + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
                trainer.optimizer.step()
                trainer.optimizer.zero_grad(set_to_none=True)
                trainer._ema(model=trainer.model)
                optimizer_steps += 1
            trainer._call_loss_update(data)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
            losses.append(output.loss.item())
            del data, output
        trainer.model.eval()
        with torch.no_grad(), trainer._test_context():
            validation = Stepper.validate_batch(
                trainer.model, next(iter(trainer.val_loader)), trainer.loss_fn
            )
        if not torch.isfinite(validation.loss):
            raise RuntimeError("Non-finite validation loss.")
        report = dict(
            rank=get_rank(),
            world_size=get_world_size(),
            batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            autoregressive_calls=4,
            parameters=sum(p.numel() for p in trainer.model.parameters()),
            model_config=cfg.model.model_dump(),
            optimizer_steps=optimizer_steps,
            losses=losses,
            validation_loss=validation.loss.item(),
            batch_seconds=timings,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            device=torch.cuda.get_device_name(),
            status="passed",
        )
        reports: list = [None] * get_world_size()
        dist.all_gather_object(reports, report)
        if get_rank() == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(reports, indent=2) + "\n")
            print(json.dumps(reports, indent=2), flush=True)
    finally:
        trainer.finish()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
