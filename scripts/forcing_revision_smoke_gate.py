"""Accept only matching, successful evidence for the actual two-GPU accum2 path."""

import hashlib
import json
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    campaign, variant = Path(sys.argv[1]), sys.argv[2]
    target = campaign / f"preflight/{variant}/first.json"
    if not target.exists():
        print(f"Accumulation smoke required for {variant}: no evidence yet", flush=True)
        return 3
    reports = json.loads(target.read_text())
    approval = json.loads((campaign / "preflight/review_approval.json").read_text())
    expected = {
        "config_sha256": digest(campaign / f"configs/train_{variant}.yaml"),
        "source_manifest_sha256": digest(campaign / "manifest.json"),
        "assets_manifest_sha256": digest(campaign / "assets/manifest.json"),
    }
    valid = len(reports) == 2 and {r["rank"] for r in reports} == {0, 1}
    for report in reports:
        valid = valid and all(report.get(k) == v for k, v in expected.items())
        valid = valid and report.get("status") == "passed"
        valid = valid and report.get("variant") == variant
        valid = valid and report.get("phase") == "shared_accumulation_smoke"
        valid = valid and report.get("world_size") == 2
        valid = valid and report.get("batch_per_gpu") == 8
        valid = valid and report.get("gradient_accumulation_steps") == 2
        valid = valid and report.get("production_global_batch") == 32
        valid = (
            valid
            and report.get("initial_model_sha256") == approval["initial_model_sha256"]
        )
        valid = valid and report.get("observed_update_counts") == {
            "optimizer_updates": 1,
            "dynamic_loss_updates": 2,
            "ema_updates": 1,
        }
    print(
        f"Accumulation smoke {'passed' if valid else 'requires rerun'} for {variant}",
        flush=True,
    )
    return 0 if valid else 3


if __name__ == "__main__":
    sys.exit(main())
