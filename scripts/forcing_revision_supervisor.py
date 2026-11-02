"""Record reviewed shared gates and start four independently gated controllers."""

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from forcing_revision_controller import SCHEDULE


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    campaign = Path(sys.argv[1]).resolve()
    python, smoke_job = sys.argv[2:4]
    requested = sys.argv[4].split(",") if len(sys.argv) > 4 else list(SCHEDULE)
    lock = (campaign / "controllers/supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assets = json.loads((campaign / "preflight/assets_verification.json").read_text())
    assert assets["status"] == "passed"
    assert (
        assets["original_loader_comparison"]["False"]["max_abs_anomaly_difference"] == 0
    )
    review_path = campaign / "preflight/primary_continuation_review.json"
    review = json.loads(review_path.read_text())
    assert review["status"] == "approved_numerical_continuation_evidence"
    for name, expected in review["evidence_sha256"].items():
        assert digest(campaign / name) == expected
    source = json.loads((campaign / "manifest.json").read_text())
    for name, expected in source["source_sha256"].items():
        assert digest(campaign / "source" / name) == expected
    first = json.loads((campaign / "preflight/wind_only/first.json").read_text())
    assert len(first) == 4 and all(r["status"] == "passed" for r in first)
    initial = {r["initial_model_sha256"] for r in first}
    assert len(initial) == 1
    approval = {
        "approved_for_training": True,
        "reviewer_agent": "/root",
        "review_authorization": "Parent independently reviewed source/setup and numerical continuation. Latest user explicitly authorizes wind_only/heat_only each interactive four GPUs,batch8,accum1 and climatology/no_forcing each shared_interactive two GPUs,batch8,accum2; all effectiveglobal32 and independent4h continuation. Parent permits immediate wind/heat submission based on existing four-GPU model/continuation path and mode-specific real-input/mask tests. Shared variants submit now with an allocation-level hard gate: original Trainer two-microbatch accumulation smoke and validation must pass before fresh training in that same allocation; subsequent allocations skip only matching successful evidence. No untested full-run results are claimed.",
        "direct_submit_variants": ["wind_only", "heat_only"],
        "approved_at_unix": time.time(),
        "remaining_smoke_job": smoke_job,
        "remaining_smoke_scope": "Inside each shared formal allocation before first training: actual Trainer two microbatches,one optimizer/EMA update,two dynamic-loss updates,one validation; effectiveglobal32. Evidence must match source/config before skipping in later allocations. No separate preflight allocation or repeated resume/assets tests.",
        "initial_model_sha256": next(iter(initial)),
        "source_hashes": {
            str(p.relative_to(campaign)): digest(p)
            for p in sorted((campaign / "source").rglob("*.py"))
        },
        "config_hashes": {
            str(p.relative_to(campaign)): digest(p)
            for p in sorted((campaign / "configs").glob("*.yaml"))
        },
        "runtime_hashes": {
            str(p.relative_to(campaign)): digest(p)
            for p in sorted((campaign / "runtime").iterdir())
            if p.is_file()
        },
        "evidence_hashes": {
            **review["evidence_sha256"],
            str(review_path.relative_to(campaign)): digest(review_path),
            "manifest.json": digest(campaign / "manifest.json"),
            "assets/manifest.json": digest(campaign / "assets/manifest.json"),
            "preflight/assets_verification.json": digest(
                campaign / "preflight/assets_verification.json"
            ),
        },
        "known_limits": [
            "Original complete W&B training config unavailable401; reconstructed from checkpoint/paper/historical code.",
            "Paper used8GPUs/batch4. User chooses4GPU/batch8/accum1 for wind/heat and2GPU/batch8/accum2 for climate/none. All effectiveglobal32, but SyncBN microbatch size and dynamic-loss update frequency differ; original formulas are unchanged.",
            "Training cache exactly matches original NumPy training backend; historical standalone evaluation used Dask and needs a separately audited evaluation policy.",
            "Restart is state-complete and numerically consistent; bitwise-identical future trajectories are not claimed.",
        ],
    }
    approval_path = campaign / "preflight/review_approval.json"
    tmp = approval_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(approval, indent=2) + "\n")
    tmp.replace(approval_path)
    schedule_path = campaign / "controllers/schedule.json"
    controllers = (
        json.loads(schedule_path.read_text()) if schedule_path.exists() else {}
    )
    for variant in requested:
        qos, hours = SCHEDULE[variant]
        with (campaign / f"logs/controller_{variant}.log").open("a") as output:
            process = subprocess.Popen(
                [
                    python,
                    "-u",
                    str(campaign / "runtime/forcing_revision_controller.py"),
                    str(campaign),
                    python,
                    variant,
                ],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        controllers[variant] = {"pid": process.pid, "qos": qos, "wall_hours": hours}
    result = {
        "status": "independent_controllers_launched",
        "supervisor_pid": os.getpid(),
        "preflight_job": smoke_job,
        "controllers": controllers,
        "updated_at_unix": time.time(),
    }
    (campaign / "preflight/supervisor.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (campaign / "controllers/schedule.json").write_text(
        json.dumps(controllers, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
