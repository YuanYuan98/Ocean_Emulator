import random
from functools import partial

import numpy as np
import pytest
import torch

from ocean_emulators.train import Trainer
from ocean_emulators.utils.ema import EMATracker
from ocean_emulators.utils.loss import decomposed_mse
from ocean_emulators.utils.loss_baselines import AdaLoss, FixedOutputWeightedLoss
from scripts.revision_loss_segment import (
    SegmentFinished,
    SegmentTrainer,
    restore_rng,
    save_rng,
)


def tiny_trainer(variant):
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=6e-4)
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=70
    )
    metric = partial(decomposed_mse, wet=torch.ones(2, 1, 1))
    if variant == "adaloss":
        trainer.loss_fn = AdaLoss(
            metric, num_channels=2, num_outputs=2, device=trainer.device
        )
    else:
        weights = [2.0, 2.0] if variant == "frozen_final" else [0.5, 4.0]
        trainer.loss_fn = FixedOutputWeightedLoss(metric, torch.tensor(weights))
    trainer._ema = EMATracker(trainer.model)
    trainer.best_val_loss = 0.1
    trainer.best_inf_loss = 0.2
    trainer.num_batches_seen = 0
    trainer.wandb_id = trainer.wandb_name = None
    return trainer


def step(trainer):
    inputs = torch.tensor([[1.0, 2.0], [3.0, -1.0], [-1.0, 0.5]])
    target = torch.zeros(3, 2, 1, 1)
    trainer.optimizer.zero_grad()
    prediction = trainer.model(inputs).reshape_as(target)
    trainer.loss_fn(prediction, target).mean().backward()
    trainer.optimizer.step()
    trainer._ema(trainer.model)
    with torch.no_grad():
        trainer.loss_fn.update(trainer.model(inputs).reshape_as(target), target)
    trainer.scheduler.step()
    trainer.num_batches_seen += 1


@pytest.mark.parametrize("variant", ["adaloss", "frozen_final", "tendency"])
def test_checkpoint_resume_preserves_next_update_and_legacy_export(tmp_path, variant):
    uninterrupted = tiny_trainer(variant)
    step(uninterrupted)
    path = tmp_path / "ckpt.pt"
    # Exercise the prefix handling used by distributed training checkpoints.
    wrapper = torch.nn.Module()
    wrapper.add_module("module", uninterrupted.model)
    uninterrupted.model = wrapper
    uninterrupted.save_checkpoint(1, path)
    uninterrupted.model = wrapper.module

    resumed = tiny_trainer(variant)
    resumed.load_checkpoint(path)
    assert resumed.start_epoch == 2
    assert resumed.scheduler.T_max == 70
    assert resumed._ema == uninterrupted._ema
    assert resumed.num_batches_seen == uninterrupted.num_batches_seen
    assert resumed.best_val_loss == uninterrupted.best_val_loss
    assert resumed.best_inf_loss == uninterrupted.best_inf_loss
    step(uninterrupted)
    step(resumed)
    for name, tensor in uninterrupted.model.state_dict().items():
        assert torch.equal(tensor, resumed.model.state_dict()[name])
    assert resumed._ema == uninterrupted._ema
    assert resumed.scheduler.state_dict() == uninterrupted.scheduler.state_dict()
    for name, tensor in uninterrupted.loss_fn.state_dict().items():
        assert torch.equal(tensor, resumed.loss_fn.state_dict()[name])

    exported = tmp_path / "ema_ckpt.pt"
    resumed.save_checkpoint(2, exported, for_inference=True)
    state = torch.load(exported, weights_only=True)
    assert "ema_params" not in state["ema"]
    for name, tensor in resumed.model.state_dict().items():
        assert torch.equal(tensor, state["model"][name])


def test_rng_state_restores_all_cpu_generators(tmp_path):
    path = tmp_path / "rng.pt"
    device = torch.device("cpu")
    save_rng(path, 17, device)
    expected = (random.random(), np.random.random(), torch.rand(5))
    restore_rng(path, 17, device)
    actual = (random.random(), np.random.random(), torch.rand(5))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])
    with pytest.raises(ValueError, match="epoch"):
        restore_rng(path, 18, device)


def test_segment_stops_between_epochs_and_resumes_without_restarting_schedule(
    tmp_path, monkeypatch
):
    clock = [1000.0]
    monkeypatch.setattr("scripts.revision_loss_segment.time.time", lambda: clock[0])
    executed = []
    monkeypatch.setattr(
        Trainer, "train_one_epoch", lambda self, epoch: executed.append(epoch) or {}
    )
    trainer = SegmentTrainer.__new__(SegmentTrainer)
    trainer.device = torch.device("cpu")
    trainer.output_dir = tmp_path
    trainer.segment_start_epoch = trainer.start_epoch = 1
    trainer.completed_epoch = 0
    trainer.previous_epoch_started = None
    trainer.longest_epoch_seconds = 0.0
    trainer.segment_deadline = 2000.0
    trainer.resumed_best = None
    trainer.train_one_epoch(1)
    clock[0] = 1700.0
    with pytest.raises(SegmentFinished):
        trainer.train_one_epoch(2)
    assert executed == [1]
    assert trainer.completed_epoch == 1
    expected_rng = torch.rand(5)

    trainer.start_epoch = trainer.segment_start_epoch = 2
    trainer.previous_epoch_started = None
    trainer.segment_deadline = 3000.0
    trainer.resumed_best = (0.1, 0.2)
    trainer.train_one_epoch(2)
    assert executed == [1, 2]
    assert torch.equal(torch.rand(5), expected_rng)
    assert trainer.best_val_loss == 0.1


def test_segment_refuses_requeue_without_any_progress(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.revision_loss_segment.time.time", lambda: 1000.0)
    trainer = SegmentTrainer.__new__(SegmentTrainer)
    trainer.device = torch.device("cpu")
    trainer.output_dir = tmp_path
    trainer.segment_start_epoch = trainer.start_epoch = 1
    trainer.completed_epoch = 0
    trainer.previous_epoch_started = None
    trainer.longest_epoch_seconds = 0.0
    trainer.segment_deadline = 1100.0
    with pytest.raises(RuntimeError, match="without completing"):
        trainer.train_one_epoch(1)
