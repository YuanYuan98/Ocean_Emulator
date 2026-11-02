"""Run the unchanged trainer in resumable Slurm allocations.

The scheduler always targets all 70 epochs. Stop only between complete epochs;
resume from ckpt.pt, never from the legacy inference export ema_ckpt.pt.
"""

import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ocean_emulators.config import TrainConfig
from ocean_emulators.train import Trainer
from ocean_emulators.utils.distributed import get_rank, is_main_process
from ocean_emulators.utils.logging import handle_logging, handle_warnings

logger = logging.getLogger(__name__)


class SegmentFinished(Exception):
    """A complete epoch is saved and another allocation is needed."""


def save_rng(path: Path, next_epoch: int, device: torch.device) -> None:
    numpy_state = np.random.get_state()
    assert isinstance(numpy_state, tuple)
    state = {
        "next_epoch": next_epoch,
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def restore_rng(path: Path, next_epoch: int, device: torch.device) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["next_epoch"] != next_epoch:
        raise ValueError("RNG state does not match the resumed epoch")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


class SegmentTrainer(Trainer):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        self.segment_start_epoch = self.start_epoch
        self.completed_epoch = self.start_epoch - 1
        self.segment_deadline = float(os.environ["REVISION_SEGMENT_DEADLINE"])
        self.previous_epoch_started: float | None = None
        self.longest_epoch_seconds = 0.0
        self.resumed_best = (
            (self.best_val_loss, self.best_inf_loss) if self.start_epoch > 1 else None
        )
        self.epoch_metrics: dict = {}

    def train_one_epoch(self, epoch):
        now = time.time()
        if self.previous_epoch_started is not None:
            self.longest_epoch_seconds = max(
                self.longest_epoch_seconds, now - self.previous_epoch_started
            )
            self.completed_epoch = epoch - 1

        # All ranks meet here after rank 0 finishes the previous checkpoint writes.
        # Leave 25% over the longest observed epoch, in addition to the batch
        # script's 15-minute safety margin before Slurm's hard time limit.
        reserve = max(600.0, 1.25 * self.longest_epoch_seconds)
        stop = torch.tensor(
            int(now + reserve >= self.segment_deadline), device=self.device
        )
        if dist.is_initialized():
            dist.broadcast(stop, src=0)

        rng_path = (
            self.output_dir / "resume_rng" / f"epoch_{epoch:03d}_rank_{get_rank()}.pt"
        )
        if epoch == self.start_epoch and self.start_epoch > 1:
            if rng_path.exists():
                restore_rng(rng_path, epoch, torch.device(self.device))
            else:
                # A hard interruption may happen after the checkpoint is renamed
                # but before the next epoch begins. We never silently reseed.
                raise FileNotFoundError(f"Missing resume RNG state: {rng_path}")
            # Trainer.run initializes these counters; retain their loaded values
            # when continuing this same experiment.
            assert self.resumed_best is not None
            self.best_val_loss, self.best_inf_loss = self.resumed_best
        save_rng(rng_path, epoch, torch.device(self.device))

        if stop.item():
            if self.completed_epoch < self.segment_start_epoch:
                raise RuntimeError("Allocation ended without completing an epoch")
            logger.info("Pausing after epoch %s for requeue", self.completed_epoch)
            raise SegmentFinished
        self.previous_epoch_started = time.time()
        self.epoch_metrics = dict(super().train_one_epoch(epoch))
        return self.epoch_metrics

    def validate_one_epoch(self, epoch):
        metrics = super().validate_one_epoch(epoch)
        self.epoch_metrics.update(metrics)
        return metrics

    def save_all_checkpoints(self, epoch, v_loss, inf_loss):
        super().save_all_checkpoints(epoch, v_loss, inf_loss)
        # Keep scalar diagnostics locally even when the W&B integration is disabled.
        scalar_metrics = {
            key: float(value)
            for key, value in self.epoch_metrics.items()
            if isinstance(value, (int, float))
            or isinstance(value, torch.Tensor)
            and value.numel() == 1
        }
        path = self.output_dir / "metrics" / f"epoch_{epoch:03d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"epoch": epoch, "metrics": scalar_metrics}, indent=2) + "\n"
        )
        os.replace(temporary, path)


def main() -> None:
    cfg = TrainConfig.from_yaml_and_cli()
    if not cfg.preemptible or cfg.finetune or cfg.epochs != 70:
        raise ValueError(
            "Segments require preemptible training, 70 epochs, no finetune"
        )
    cfg.prepare_output_dirs()
    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()
    trainer = SegmentTrainer(cfg)
    try:
        try:
            trainer.run()
            trainer.completed_epoch = trainer.epochs
        except SegmentFinished:
            trainer.finish()
        if dist.is_initialized():
            dist.barrier()
        if is_main_process():
            result = {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
                "start_epoch": trainer.segment_start_epoch,
                "completed_epoch": trainer.completed_epoch,
                "complete": trainer.completed_epoch == trainer.epochs,
            }
            path = trainer.output_dir / "segment_result.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            os.replace(temporary, path)
            logger.info("Segment result: %s", result)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
