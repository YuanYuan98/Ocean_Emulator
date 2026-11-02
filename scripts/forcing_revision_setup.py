"""Create isolated forcing campaign configs and freeze the reviewed runtime source."""

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml

BASE = Path("/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace")
REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = BASE / "experiments/forcing_revision_20261019"
VARIANTS = ("wind_only", "heat_only", "climatology", "no_forcing")


def main():
    if any(CAMPAIGN.glob("training/*/saved_nets/ckpt.pt")):
        raise RuntimeError("Refuse to overwrite campaign source after training starts")
    for directory in (
        "configs",
        "assets",
        "runtime",
        "source",
        "logs",
        "controllers",
        "preflight",
        "training",
    ):
        (CAMPAIGN / directory).mkdir(parents=True, exist_ok=True)
    reference = (
        BASE / "experiments/loss_revision_20261003/configs/train_frozen_final.yaml"
    )
    base = yaml.safe_load(reference.read_text())
    base["batch_size"] = 8
    base["loss"] = {
        "type": "dynamic",
        "metric": "mse",
        "limit": 20.0,
        "history_grouping": "time_major",
    }
    base["model"]["unet"]["up_sampling_block"] = "cell_centered_periodic_upsample"
    base["experiment"]["base_output_dir"] = str(CAMPAIGN / "training")
    base["experiment"]["data_root"] = str(BASE / "datasets/onedeg")
    base["data"]["forcing_revision_assets"] = str(CAMPAIGN / "assets")
    for variant in VARIANTS:
        cfg = copy.deepcopy(base)
        cfg["experiment"]["name"] = variant
        cfg["data"]["forcing_revision_mode"] = variant
        cfg["gradient_accumulation_steps"] = (
            2 if variant in ("climatology", "no_forcing") else 1
        )
        (CAMPAIGN / "configs" / f"train_{variant}.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False)
        )
        evaluation = yaml.safe_load(
            (
                BASE
                / "experiments/loss_revision_20261003/configs/eval_frozen_final.yaml"
            ).read_text()
        )
        evaluation["model"] = copy.deepcopy(cfg["model"])
        evaluation["data"] = copy.deepcopy(cfg["data"])
        evaluation["experiment"] = copy.deepcopy(cfg["experiment"])
        evaluation["experiment"]["base_output_dir"] = str(CAMPAIGN / "inference")
        evaluation["ckpt_path"] = str(
            CAMPAIGN / f"training/{variant}/saved_nets/ema_ckpt.pt"
        )
        (CAMPAIGN / "configs" / f"eval_{variant}.yaml").write_text(
            yaml.safe_dump(evaluation, sort_keys=False)
        )
    shutil.copytree(
        REPO / "src/ocean_emulators",
        CAMPAIGN / "source/ocean_emulators",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for path in (REPO / "scripts").glob("forcing_revision_*"):
        shutil.copy2(path, CAMPAIGN / "runtime" / path.name)
    for name in ("revision_loss_segment.py", "verify_loss_checkpoint.py"):
        shutil.copy2(REPO / "scripts" / name, CAMPAIGN / "runtime" / name)
    audit = json.loads(
        (BASE / "experiments/multistart_20261019/input_audit.json").read_text()
    )
    (CAMPAIGN / "input_audit_onedeg.json").write_text(
        json.dumps(audit["resolutions"]["onedeg"], indent=2) + "\n"
    )
    manifest = {
        "status": "prepared_not_started",
        "variants": VARIANTS,
        "existing_full_control_not_retrained": True,
        "resolution": "onedeg",
        "source_repository": str(REPO),
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "reference_training_recipe": str(reference),
        "global_batch": 32,
        "nodes": 1,
        "gpus_per_node": 4,
        "gpus": 4,
        "batch_per_gpu": 8,
        "hardware_evidence": "Paper uses eight GPUs/batch4; user explicitly authorizes four GPUs/batch8 to match global32",
        "epochs": 70,
        "seed": 15,
        "loss": {
            "type": "dynamic",
            "limit": 20.0,
            "history_grouping": "time_major",
            "reference_commit": "a217a1c20117d7992375fd6d511581dd1a3b00ea",
            "indexing_fix_commit": "aeff3736",
            "limit_evidence": "paper section 3.4",
        },
        "parameter_count": 84066474,
        "sync_batchnorm": True,
        "fp32": True,
        "upsampling": {
            "type": "cell_centered_periodic_upsample",
            "reference_commit": "a217a1c20117d7992375fd6d511581dd1a3b00ea",
            "semantics": "circular pad, bilinear interpolate align_corners=False, crop",
        },
        "checkpoint_policy": "final ordinary weights; unchanged legacy EMA export",
        "historical_training_provenance": {
            "wandb_run": "ocean_emulators/default/3o112ot4",
            "wandb_readback": "401 Unauthorized using existing environment; no credential changes",
            "checkpoint_evidence": {
                "epoch": 70,
                "num_batches_seen": 6160,
                "scheduler_T_max": 70,
                "scheduler_base_lr": 0.0006,
            },
            "limitation": "Original training W&B config not directly accessible; recipe reconstructed from published checkpoint, paper section 3.4 and historical source/config. Eval source commit may postdate training.",
            "grouping_evidence": "Jan20 time-major fix precedes named Jan29 training; final 77 weight pattern consistent, but final state alone cannot uniquely prove grouping",
        },
        "source_sha256": {
            str(p.relative_to(CAMPAIGN / "source")): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted((CAMPAIGN / "source").rglob("*.py"))
        },
    }
    (CAMPAIGN / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(CAMPAIGN)


if __name__ == "__main__":
    main()
