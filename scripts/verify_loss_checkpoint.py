"""Require a complete, finite final-epoch checkpoint before dependent inference."""

import json
import sys
from pathlib import Path

import torch


def validate_loss_weights(loss_state: dict) -> str:
    if "output_weights" in loss_state:
        key, expected = "output_weights", 154
    elif "per_channel_scale" in loss_state:
        key, expected = "per_channel_scale", 77
    else:
        raise ValueError(f"Unrecognized saved loss state: {sorted(loss_state)}")
    weights = loss_state[key]
    if weights.shape != (expected,) or not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError(f"Invalid saved loss weights: {key}")
    return key


def main() -> None:
    torch.set_num_threads(4)
    path = Path(sys.argv[1])
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if state["epoch"] != 70:
        raise ValueError(f"Expected epoch 70, found {state['epoch']}")
    for name, tensor in state["model"].items():
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite checkpoint parameter: {name}")
    if "ema_params" in state["ema"]:
        raise ValueError(
            "Unexpected checkpoint format; preserve original inference export"
        )
    loss_weights_key = validate_loss_weights(state["loss_fn_state"])
    print(
        json.dumps(
            {
                "checkpoint": str(path),
                "epoch": state["epoch"],
                "loss_weights_key": loss_weights_key,
                "checkpoint_policy": "legacy final-epoch export; EMA export behavior intentionally unchanged",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
