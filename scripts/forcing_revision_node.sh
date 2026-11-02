#!/usr/bin/env bash
set -euo pipefail
campaign="${1:?campaign}"
python_bin="${2:?python}"
stage="${3:?assets, preflight, train, or verify}"
variant="${4:-all}"
phase="${5:-}"
export PYTHONPATH="$campaign/source"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=disabled MPLBACKEND=Agg NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
cd "$campaign"
echo "$(date -u +%FT%TZ) job=$SLURM_JOB_ID node=$SLURM_NODEID stage=$stage variant=$variant phase=$phase"
if [[ "$stage" == assets ]]; then
    if [[ ! -f "$campaign/assets/manifest.json" ]]; then
        "$python_bin" "$campaign/runtime/forcing_revision_prepare_assets.py" "$campaign/configs/train_climatology.yaml"
    fi
    "$python_bin" "$campaign/runtime/forcing_revision_verify_assets.py" "$campaign"
elif [[ "$stage" == verify ]]; then
    if jq -e '.complete == true' "$campaign/training/$variant/segment_result.json" >/dev/null; then
        "$python_bin" "$campaign/runtime/verify_loss_checkpoint.py" "$campaign/training/$variant/saved_nets/ema_ckpt.pt"
    fi
else
    "$python_bin" - <<'PY'
import torch
import os
count = int(os.environ.get("FORCING_GPUS_PER_NODE", "4"))
assert torch.cuda.device_count() == count
assert all(torch.cuda.get_device_properties(i).total_memory > 75 * 2**30 for i in range(count))
print([torch.cuda.get_device_name(i) for i in range(count)], flush=True)
PY
    launch=("$python_bin" -u -m torch.distributed.run --nnodes=1 "--nproc-per-node=${FORCING_GPUS_PER_NODE:-4}"
        "--node-rank=$SLURM_NODEID" "--master-addr=${MASTER_ADDR:?master address}"
        "--master-port=${MASTER_PORT:?master port}")
    if [[ "$stage" == accum_smoke ]]; then
        "${launch[@]}" "$campaign/runtime/forcing_revision_smoke.py" \
            "$campaign/configs/train_$variant.yaml" --campaign "$campaign"
    elif [[ "$stage" == preflight ]]; then
        "${launch[@]}" "$campaign/runtime/forcing_revision_preflight.py" \
            "$campaign/configs/train_$variant.yaml" "$phase" --campaign "$campaign"
    elif [[ "$stage" == train ]]; then
        "${launch[@]}" "$campaign/runtime/revision_loss_segment.py" "$campaign/configs/train_$variant.yaml"
    else
        exit 2
    fi
fi
