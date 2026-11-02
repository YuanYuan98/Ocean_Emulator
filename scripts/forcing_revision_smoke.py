"""Two-GPU smoke of the original accumulation path and one validation batch."""

import argparse
import json
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist
from forcing_revision_preflight import check_data, digest_state

from ocean_emulators.config import TrainConfig
from ocean_emulators.stepper import Stepper
from ocean_emulators.train import Trainer
from ocean_emulators.utils.distributed import get_rank, get_world_size
from ocean_emulators.utils.forcing_revision import sha256
from ocean_emulators.utils.logging import handle_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    cfg = TrainConfig.from_yaml_and_cli([args.config])
    variant = cfg.data.forcing_revision_mode
    work = args.campaign / "preflight" / variant
    work.mkdir(parents=True, exist_ok=True)
    cfg.experiment.base_output_dir = str(args.campaign / "preflight/trainer")
    cfg.experiment.name = variant + "_shared_smoke"
    cfg.preemptible, cfg.resume_ckpt_path = False, None
    cfg.prepare_output_dirs()
    handle_logging(False, cfg.experiment.output_dir)
    trainer = Trainer(cfg)
    trainer.forcing_mode_checked = variant
    try:
        assert get_world_size() == 2 and cfg.batch_size == 8
        assert (
            cfg.gradient_accumulation_steps == trainer.gradient_accumulation_steps == 2
        )
        assert torch.cuda.device_count() == 2
        assert (
            torch.cuda.get_device_properties(trainer.device).total_memory > 75 * 2**30
        )
        assert trainer.num_in == 162 and trainer.num_out == 154
        assert sum(p.numel() for p in trainer.model.parameters()) == 84066474
        assert not cfg.model.use_bfloat16
        assert trainer.loss_fn.history_grouping == "time_major"
        assert int(trainer.loss_fn.state_dict()["history_grouping_code"]) == 1
        assert trainer.scheduler.T_max == 70
        assert not trainer.model.module.requires_calendar
        initial_hash = digest_state(trainer.model)
        trainer.init_data_loaders(4)
        trainer.train_sampler.set_epoch(1)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        original_loader = trainer.train_loader

        class TwoMicrobatches:
            def __len__(self):
                return 2

            def __iter__(self):
                iterator = iter(original_loader)
                for _ in range(2):
                    data = next(iterator)
                    check_data(data, variant)
                    yield data

        counts = {"optimizer_updates": 0, "dynamic_loss_updates": 0}

        def before_step(optimizer, args, kwargs):
            for name, parameter in trainer.model.named_parameters():
                assert (
                    parameter.grad is not None and torch.isfinite(parameter.grad).all()
                ), name
            counts["optimizer_updates"] += 1

        hook = trainer.optimizer.register_step_pre_hook(before_step)
        original_loss_update = trainer._call_loss_update

        def tracked_loss_update(data):
            counts["dynamic_loss_updates"] += 1
            original_loss_update(data)

        trainer._call_loss_update = tracked_loss_update
        trainer.train_loader = TwoMicrobatches()
        train_logs = trainer.train_one_epoch(1)
        hook.remove()
        assert trainer.num_batches_seen == 2
        assert counts == {"optimizer_updates": 1, "dynamic_loss_updates": 2}
        assert int(trainer._ema.num_updates) == 1
        counts["ema_updates"] = int(trainer._ema.num_updates)
        assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
        train_loss = train_logs["train/mean/loss"]
        assert torch.isfinite(torch.tensor(train_loss))
        trainer.model.eval()
        validation_data = next(iter(trainer.val_loader))
        check_data(validation_data, variant, steps=1)
        with torch.no_grad(), trainer._test_context():
            validation = Stepper.validate_batch(
                trainer.model, validation_data, trainer.loss_fn
            )
        assert torch.isfinite(validation.loss)
        report = {
            "status": "passed",
            "variant": variant,
            "phase": "shared_accumulation_smoke",
            "scope": "Unmodified Trainer.train_one_epoch over exactly two microbatches: each four-step autoregression, loss/2 backward, one optimizer/EMA update, two dynamic-loss updates; one validation batch. No checkpoint or resume retest.",
            "config_sha256": sha256(args.config),
            "source_manifest_sha256": sha256(args.campaign / "manifest.json"),
            "assets_manifest_sha256": sha256(args.campaign / "assets/manifest.json"),
            "rank": get_rank(),
            "hostname": socket.gethostname(),
            "world_size": 2,
            "batch_per_gpu": 8,
            "smoke_global_batch": 16,
            "gradient_accumulation_steps": 2,
            "production_world_size": 2,
            "production_global_batch": 32,
            "observed_update_counts": counts,
            "initial_model_sha256": initial_hash,
            "train_loss": train_loss,
            "validation_loss": validation.loss.item(),
            "seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device": torch.cuda.get_device_name(),
            "fp32": True,
            "production_backend_unchanged": True,
        }
        reports = [None] * get_world_size()
        dist.all_gather_object(reports, report)
        if get_rank() == 0:
            target = work / "first.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(reports, indent=2) + "\n")
            temporary.replace(target)
            print(json.dumps(reports, indent=2), flush=True)
    finally:
        trainer.finish()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
