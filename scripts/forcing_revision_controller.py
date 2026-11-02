"""Resume one independently scheduled forcing experiment to epoch 70."""

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

VARIANTS = ("wind_only", "heat_only", "climatology", "no_forcing")
SCHEDULE = {
    "wind_only": ("interactive", 4),
    "heat_only": ("interactive", 4),
    "climatology": ("shared_interactive", 4),
    "no_forcing": ("shared_interactive", 4),
}


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    campaign = Path(sys.argv[1]).resolve()
    python = sys.argv[2]
    selected = sys.argv[3]
    qos, wall_hours = SCHEDULE[selected]
    gpus = 2 if qos == "shared_interactive" else 4
    accumulation = 2 if gpus == 2 else 1
    lock = (campaign / f"controllers/{selected}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def status(state, **kwargs):
        content = {
            "status": state,
            "controller_pid": os.getpid(),
            "updated_at_unix": time.time(),
            "variant": selected,
            "qos": qos,
            "wall_hours": wall_hours,
            "gpus": gpus,
            "batch_per_gpu": 8,
            "gradient_accumulation_steps": accumulation,
            "effective_global_batch": 32,
            **kwargs,
        }
        path = campaign / f"controllers/{selected}_controller.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(content, indent=2) + "\n")
        temporary.replace(path)
        print(json.dumps(content), flush=True)

    try:
        approval = read(campaign / "preflight/review_approval.json", {})
        if approval.get("approved_for_training") is not True:
            raise RuntimeError(
                "Independent source/setup review has not approved formal training"
            )
        for group in (
            "source_hashes",
            "config_hashes",
            "runtime_hashes",
            "evidence_hashes",
        ):
            if not approval.get(group):
                raise RuntimeError(f"Independent review lacks {group}")
            for name, expected in approval[group].items():
                actual = hashlib.sha256((campaign / name).read_bytes()).hexdigest()
                if actual != expected:
                    raise RuntimeError(f"File changed after independent review: {name}")
        if (
            read(campaign / "preflight/assets_verification.json", {}).get("status")
            != "passed"
        ):
            raise RuntimeError(
                "Original-loader and real-input asset verification did not pass"
            )
        manifest = read(campaign / "manifest.json")
        for name, expected in manifest["source_sha256"].items():
            actual = hashlib.sha256(
                (campaign / "source" / name).read_bytes()
            ).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Frozen scientific source changed: {name}")
        continuation = read(campaign / "preflight/primary_continuation_review.json")
        if continuation["status"] != "approved_numerical_continuation_evidence":
            raise RuntimeError("Shared continuation mechanism was not reviewed")
        for name, expected in continuation["evidence_sha256"].items():
            assert (
                hashlib.sha256((campaign / name).read_bytes()).hexdigest() == expected
            )
        if gpus == 4:
            evidence_variant = "wind_only" if selected == "heat_only" else selected
            if selected == "heat_only":
                assert selected in approval["direct_submit_variants"]
                # Parent approved reuse of the identical four-GPU model/optimizer path;
                # heat-mode fields and both-history masks were independently verified.
            smoke_path = campaign / f"preflight/{evidence_variant}/first.json"
            while not smoke_path.exists():
                status(
                    "waiting_for_variant_smoke",
                    preflight_job=approval["remaining_smoke_job"],
                )
                active = subprocess.check_output(
                    [
                        "squeue",
                        "--noheader",
                        "-j",
                        approval["remaining_smoke_job"],
                        "--format=%T",
                    ],
                    text=True,
                ).strip()
                if not active and not smoke_path.exists():
                    raise RuntimeError(
                        f"Preflight ended without successful {selected} smoke"
                    )
                time.sleep(30)
            reports = read(smoke_path)
            smoke_world = len(reports)
            assert smoke_world in (2, 4) and {r["rank"] for r in reports} == set(
                range(smoke_world)
            )
            if smoke_world == 2:
                assert all(
                    r["phase"] == "shared_accumulation_smoke"
                    and r["smoke_global_batch"] == 16
                    and r["production_world_size"] == 2
                    and r["production_global_batch"] == 32
                    and r["gradient_accumulation_steps"] == 2
                    for r in reports
                )
            expected_cfg = hashlib.sha256(
                (campaign / f"configs/train_{evidence_variant}.yaml").read_bytes()
            ).hexdigest()
            expected_source = hashlib.sha256(
                (campaign / "manifest.json").read_bytes()
            ).hexdigest()
            expected_assets = hashlib.sha256(
                (campaign / "assets/manifest.json").read_bytes()
            ).hexdigest()
            assert all(
                r["status"] == "passed"
                and r["world_size"] == smoke_world
                and r["batch_per_gpu"] == 8
                and r["config_sha256"] == expected_cfg
                and r["source_manifest_sha256"] == expected_source
                and r["assets_manifest_sha256"] == expected_assets
                and r["initial_model_sha256"] == approval["initial_model_sha256"]
                for r in reports
            )

        else:
            status(
                "allocation_gpu_smoke_required",
                scope="two real microbatches and validation before fresh training",
            )

        def queued_jobs():
            output = subprocess.check_output(
                ["squeue", "--me", "--noheader", "--format=%i|%q|%T|%j"], text=True
            )
            return [line.split("|") for line in output.splitlines() if line]

        def wait_for_job(job):
            while any(row[0] == job for row in queued_jobs()):
                time.sleep(60)
            for attempt in range(10):
                result = subprocess.check_output(
                    ["sacct", "-X", "-n", "-P", "-j", job, "--format=State,ExitCode"],
                    text=True,
                ).strip()
                if result:
                    if result.splitlines()[0].split("|")[:2] != ["COMPLETED", "0:0"]:
                        raise RuntimeError(f"Allocation {job} failed: {result}")
                    return
                time.sleep(10)
            raise RuntimeError(f"No final accounting for {job}")

        while True:
            previous = read(
                campaign / f"training/{selected}/segment_result.json",
                {"completed_epoch": 0, "complete": False},
            )
            if previous["complete"]:
                status(
                    "training_complete",
                    result=previous,
                    checkpoint_policy="epoch70 ordinary weights",
                )
                return
            job_name = f"s2force_{selected}"
            existing = [row[0] for row in queued_jobs() if row[3] == job_name]
            if len(existing) > 1:
                raise RuntimeError(f"Duplicate jobs for {selected}: {existing}")
            allocation_args = [
                "--account=m4874_g",
                f"--qos={qos}",
                "--constraint=gpu&hbm80g",
                "--nodes=1",
                "--ntasks=1",
                "--ntasks-per-node=1",
                f"--gpus-per-node={gpus}",
                f"--cpus-per-task={16 * gpus}",
                f"--mem={55 * gpus}G",
                f"--time={wall_hours:02d}:00:00",
                f"--job-name={job_name}",
            ]
            script_args = [
                str(campaign / "runtime/forcing_revision_allocation.sh"),
                str(campaign),
                python,
                "train",
                selected,
                str(wall_hours * 3600),
                str(gpus),
                str(accumulation),
            ]
            if existing:
                status("reattached", job_id=existing[0], previous=previous)
                wait_for_job(existing[0])
            elif qos == "regular":
                job = (
                    subprocess.check_output(
                        [
                            "sbatch",
                            "--parsable",
                            *allocation_args,
                            f"--output={campaign}/logs/batch_{selected}_%j.log",
                            *script_args,
                        ],
                        text=True,
                    )
                    .strip()
                    .split(";")[0]
                )
                if not job.isdigit():
                    raise RuntimeError(f"Unexpected sbatch response: {job}")
                status(
                    "queued", job_id=job, previous=previous, gpus=gpus, batch_per_gpu=8
                )
                wait_for_job(job)
            else:
                # Serialize just quota-check/submission, never the training jobs.
                with (campaign / "controllers/interactive_submission.lock").open(
                    "a"
                ) as gate:
                    fcntl.flock(gate, fcntl.LOCK_EX)
                    interactive = [
                        row for row in queued_jobs() if row[1] in (qos, "gpu_" + qos)
                    ]
                    if len(interactive) >= 2:
                        process = None
                        status(
                            "waiting_for_interactive_quota",
                            jobs=interactive,
                            previous=previous,
                        )
                    else:
                        status(
                            "allocating", previous=previous, gpus=gpus, batch_per_gpu=8
                        )
                        process = subprocess.Popen(
                            [
                                "salloc",
                                *allocation_args,
                                "--immediate=86400",
                                "bash",
                                *script_args,
                            ]
                        )
                        for attempt in range(30):
                            submitted = [
                                row for row in queued_jobs() if row[3] == job_name
                            ]
                            if submitted:
                                status(
                                    "submitted",
                                    job_id=submitted[0][0],
                                    previous=previous,
                                )
                                break
                            if process.poll() is not None:
                                break
                            time.sleep(2)
                if process is None:
                    time.sleep(60)
                    continue
                returncode = process.wait()
                if returncode:
                    raise RuntimeError(
                        f"Allocation/trainer failed for {selected}: exit {returncode}"
                    )
            updated = read(campaign / f"training/{selected}/segment_result.json")
            if (
                updated is None
                or updated["completed_epoch"] <= previous["completed_epoch"]
            ):
                raise RuntimeError(f"No complete epoch progress for {selected}")
            status("allocation_finished", result=updated)
    except BaseException as exc:
        status("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
