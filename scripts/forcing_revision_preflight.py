"""Real single-node/four-GPU smoke and optimizer/RNG continuation check."""

import argparse
import hashlib
import json
import os
import pickle
import random
import socket
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from revision_loss_segment import restore_rng, save_rng

from ocean_emulators.config import TrainConfig
from ocean_emulators.stepper import Stepper
from ocean_emulators.train import Trainer
from ocean_emulators.utils.distributed import get_rank, get_world_size
from ocean_emulators.utils.forcing_revision import EXCLUDED, FORCING, sha256
from ocean_emulators.utils.logging import handle_logging


def digest_state(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def compare(a, b, path="state", exact=False, stats=None):
    if isinstance(a, torch.Tensor):
        actual, reference = a.detach().cpu(), b.detach().cpu()
        if stats is not None:
            assert torch.isfinite(actual).all() and torch.isfinite(reference).all(), (
                path
            )
            delta = actual.double() - reference.double()
            difference = delta.abs().max().item() if delta.numel() else 0.0
            stats["tensor_count"] += 1
            stats["numel"] += delta.numel()
            stats["squared_difference"] += delta.square().sum().item()
            stats["squared_reference"] += reference.double().square().sum().item()
            if difference > stats["max_abs"]:
                stats["max_abs"] = difference
                stats["max_abs_tensor"] = path
        try:
            torch.testing.assert_close(
                actual,
                reference,
                rtol=0 if exact else 1e-6,
                atol=0 if exact else 1e-7,
                msg=lambda msg: path + ": " + msg,
            )
        except AssertionError:
            if stats is None:
                raise
            stats["outside_strict_tolerance"].append(path)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            compare(a[key], b[key], f"{path}/{key}", exact=exact, stats=stats)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            compare(x, y, f"{path}/{i}", exact=exact, stats=stats)
    else:
        assert a == b, (path, a, b)


def check_data(data, mode, steps=4):
    assert len(data) == steps
    excluded = EXCLUDED[mode]
    for inp, target in data.values():
        assert inp.shape[1:] == (162, 180, 360)
        assert target.shape[1:] == (154, 180, 360)
        assert torch.isfinite(inp).all() and torch.isfinite(target).all()
        for index, name in enumerate(FORCING):
            if name in excluded:
                assert torch.count_nonzero(inp[:, [154 + index, 158 + index]]) == 0


def update(trainer, data):
    check_data(data, trainer.forcing_mode_checked)
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    output = Stepper.train_batch(trainer.model, data, trainer.loss_fn)
    assert torch.isfinite(output.loss), "Nonfinite preflight loss"
    output.loss.backward()
    for name, parameter in trainer.model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer._ema(trainer.model)
    trainer._call_loss_update(data)
    trainer.num_batches_seen += 1
    # A short synthetic epoch for checking scheduler state, still T_max=70.
    trainer.scheduler.step()
    return output.loss.item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("phase", choices=["first", "resume"])
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    # Production retains the historical nondeterministic backend. This diagnostic
    # isolates serialization/RNG correctness from cuDNN algorithm selection.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["NCCL_ALGO"] = "Ring"
    os.environ["NCCL_PROTO"] = "Simple"
    os.environ["NCCL_MAX_NCHANNELS"] = "1"
    cfg = TrainConfig.from_yaml_and_cli([args.config])
    mode = cfg.data.forcing_revision_mode
    work = args.campaign / "preflight" / mode
    work.mkdir(parents=True, exist_ok=True)
    cfg.experiment.base_output_dir = str(args.campaign / "preflight" / "trainer")
    cfg.experiment.name = mode + "_" + args.phase
    cfg.preemptible = False
    cfg.resume_ckpt_path = str(work / "resume.pt") if args.phase == "resume" else None
    cfg.prepare_output_dirs()
    handle_logging(False, cfg.experiment.output_dir)
    ddp_init = torch.nn.parallel.DistributedDataParallel.__init__

    def diagnostic_ddp_init(self, *args, **kwargs):
        # Default DDP changes from one initial bucket to rebuilt smaller buckets.
        # A single bucket keeps reduction ordering comparable across processes.
        kwargs["bucket_cap_mb"] = 1024
        ddp_init(self, *args, **kwargs)

    with patch.object(
        torch.nn.parallel.DistributedDataParallel, "__init__", diagnostic_ddp_init
    ):
        trainer = Trainer(cfg)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    trainer.forcing_mode_checked = mode
    try:
        assert get_world_size() == 4 and cfg.batch_size == 8
        assert int(os.environ["GROUP_WORLD_SIZE"]) == 1
        assert int(os.environ["LOCAL_WORLD_SIZE"]) == 4
        assert trainer.num_in == 162 and trainer.num_out == 154
        assert sum(p.numel() for p in trainer.model.parameters()) == 84066474
        assert not cfg.model.use_bfloat16
        assert not trainer.model.module.requires_calendar
        assert cfg.model.pos_channels == 0 and not cfg.model.add_3d_coordinates
        assert not cfg.model.corrector.ocean_heat_corrector
        assert trainer.loss_fn.history_grouping == "time_major"
        assert int(trainer.loss_fn.state_dict()["history_grouping_code"]) == 1
        assert trainer.scheduler.T_max == 70
        initial_hash = digest_state(trainer.model) if args.phase == "first" else None
        if args.phase == "resume":
            saved = torch.load(
                work / "resume.pt", weights_only=True, mmap=True, map_location="cpu"
            )
            compare(
                trainer.model.state_dict(), saved["model"], "restored_model", exact=True
            )
            compare(
                trainer.optimizer.state_dict(),
                saved["optimizer"],
                "restored_optimizer",
                exact=True,
            )
            compare(
                trainer.scheduler.state_dict(),
                saved["scheduler"],
                "restored_scheduler",
                exact=True,
            )
            compare(
                trainer.loss_fn.state_dict(),
                saved["loss_fn_state"],
                "restored_loss",
                exact=True,
            )
            compare(
                trainer._ema.get_state(include_ema_params=True),
                saved["ema"],
                "restored_ema",
                exact=True,
            )
            assert trainer.num_batches_seen == saved["num_batches_seen"] == 1
            del saved
        trainer.init_data_loaders(4)
        trainer.train_sampler.set_epoch(1)
        iterator = iter(trainer.train_loader)
        first_data = next(iterator)
        losses = []
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        if args.phase == "first":
            trainer.best_val_loss, trainer.best_inf_loss = 1e8, 1e8
            losses.append(update(trainer, first_data))
            if get_rank() == 0:
                trainer.save_checkpoint(0, work / "resume.pt")
            dist.barrier()
            save_rng(work / f"rng_rank{get_rank()}.pt", 1, torch.device(trainer.device))
        del first_data
        second_data = next(iterator)
        if args.phase == "resume":
            restore_rng(
                work / f"rng_rank{get_rank()}.pt", 1, torch.device(trainer.device)
            )
        batch_digest = hashlib.sha256()
        for pair in second_data.values():
            for tensor in pair:
                batch_digest.update(tensor.cpu().numpy().tobytes())
        input_proof = {
            "batch_sha256": batch_digest.hexdigest(),
            "python_rng_sha256": hashlib.sha256(
                pickle.dumps(random.getstate())
            ).hexdigest(),
            "numpy_rng_sha256": hashlib.sha256(
                pickle.dumps(np.random.get_state())
            ).hexdigest(),
            "torch_rng_sha256": hashlib.sha256(
                torch.get_rng_state().numpy().tobytes()
            ).hexdigest(),
            "cuda_rng_sha256": hashlib.sha256(
                torch.cuda.get_rng_state().cpu().numpy().tobytes()
            ).hexdigest(),
        }
        proof_path = work / f"input_proof_rank{get_rank()}.json"
        if args.phase == "first":
            proof_path.write_text(json.dumps(input_proof, indent=2) + "\n")
        else:
            assert input_proof == json.loads(proof_path.read_text()), (
                "Resumed batch/RNG mismatch"
            )
        losses.append(update(trainer, second_data))
        del second_data
        continuation_metrics = None
        if args.phase == "first":
            if get_rank() == 0:
                trainer.save_checkpoint(1, work / "expected.pt")
        else:
            expected = torch.load(
                work / "expected.pt", weights_only=True, mmap=True, map_location="cpu"
            )
            continuation_metrics = {}
            for name, actual, reference in (
                ("model", trainer.model.state_dict(), expected["model"]),
                ("optimizer", trainer.optimizer.state_dict(), expected["optimizer"]),
                ("scheduler", trainer.scheduler.state_dict(), expected["scheduler"]),
                (
                    "dynamic_loss",
                    trainer.loss_fn.state_dict(),
                    expected["loss_fn_state"],
                ),
                (
                    "ema",
                    trainer._ema.get_state(include_ema_params=True),
                    expected["ema"],
                ),
            ):
                stats = {
                    "tensor_count": 0,
                    "numel": 0,
                    "max_abs": 0.0,
                    "max_abs_tensor": None,
                    "squared_difference": 0.0,
                    "squared_reference": 0.0,
                    "outside_strict_tolerance": [],
                }
                compare(actual, reference, name, stats=stats)
                stats["relative_l2"] = (
                    stats["squared_difference"]
                    / max(stats["squared_reference"], 1e-300)
                ) ** 0.5
                continuation_metrics[name] = stats
            assert trainer.num_batches_seen == expected["num_batches_seen"] == 2
            del expected
            first_reports = json.loads((work / "first.json").read_text())
            expected_loss = first_reports[get_rank()]["losses"][-1]
            diagnostic = {
                "restored_state_bitwise_exact": True,
                "batch_and_rng_exact": True,
                "rank": get_rank(),
                "continuous_second_loss": expected_loss,
                "resumed_second_loss": losses[-1],
                "loss_abs_difference": abs(expected_loss - losses[-1]),
                "comparison": continuation_metrics,
            }
            (work / f"continuation_metrics_rank{get_rank()}.json").write_text(
                json.dumps(diagnostic, indent=2) + "\n"
            )
            review = json.loads(
                (
                    args.campaign / "preflight/primary_continuation_review.json"
                ).read_text()
            )
            assert review["status"] == "approved_numerical_continuation_evidence"
            for name, limits in review["post_update_limits"].items():
                assert continuation_metrics[name]["max_abs"] <= limits["max_abs"]
                assert (
                    continuation_metrics[name]["relative_l2"] <= limits["relative_l2"]
                )
            assert abs(expected_loss - losses[-1]) <= review[
                "post_update_loss_relative_tolerance"
            ] * max(abs(expected_loss), 1e-12)
        dist.barrier()
        trainer.model.eval()
        validation_data = next(iter(trainer.val_loader))
        check_data(validation_data, mode, steps=1)
        with torch.no_grad(), trainer._test_context():
            validation = Stepper.validate_batch(
                trainer.model, validation_data, trainer.loss_fn
            )
        assert torch.isfinite(validation.loss)
        report = {
            "status": "passed",
            "variant": mode,
            "phase": args.phase,
            "config_sha256": sha256(args.config),
            "source_manifest_sha256": sha256(args.campaign / "manifest.json"),
            "assets_manifest_sha256": sha256(args.campaign / "assets/manifest.json"),
            "rank": get_rank(),
            "hostname": socket.gethostname(),
            "node_rank": int(os.environ["GROUP_RANK"]),
            "world_size": get_world_size(),
            "batch_per_gpu": 8,
            "initial_model_sha256": initial_hash,
            "losses": losses,
            "validation_loss": validation.loss.item(),
            "seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device": torch.cuda.get_device_name(),
            "fp32": True,
            "diagnostic_deterministic_backend": True,
            "diagnostic_ddp_bucket_cap_mb": 1024,
            "diagnostic_nccl": {
                "algorithm": "Ring",
                "protocol": "Simple",
                "max_channels": 1,
            },
            "production_backend_unchanged": True,
            "restored_state_bitwise_exact": args.phase == "resume",
            "input_proof": input_proof,
            "continuation_metrics": continuation_metrics,
            "strict_elementwise_diagnostic": {"rtol": 1e-6, "atol": 1e-7},
            "numerical_continuation_review": "preflight/primary_continuation_review.json",
        }
        reports = [None] * get_world_size()
        dist.all_gather_object(reports, report)
        if get_rank() == 0:
            target = work / (args.phase + ".json")
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
