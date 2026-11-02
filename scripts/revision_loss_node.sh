#!/usr/bin/env bash
set -euo pipefail
campaign="${1:?campaign directory}"
python_bin="${2:?Python interpreter}"
variant="${3:?loss variant}"
"$python_bin" - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 4
print('GPU preflight:', torch.__version__, torch.cuda.get_device_name(0), flush=True)
PY
exec "$python_bin" -u -m torch.distributed.run \
    --nnodes=2 --nproc-per-node=4 --node-rank="$SLURM_PROCID" \
    --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
    "$campaign/runtime/revision_loss_segment.py" "$campaign/configs/train_$variant.yaml"
