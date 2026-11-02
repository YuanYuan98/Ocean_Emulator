#!/usr/bin/env bash
# Executed on a compute node inside a four-hour interactive allocation.
set -euo pipefail
campaign="${1:?campaign directory}"
python_bin="${2:?Python interpreter}"
variant="${3:?loss variant}"
gpus="${4:?GPU count}"
stage="${5:?train or eval}"
cd "$campaign"
export PYTHONPATH="$campaign/source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=disabled MPLBACKEND=Agg NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export REVISION_SEGMENT_DEADLINE=$(($(date +%s) + 4 * 3600 - 15 * 60))
log="$campaign/logs/${variant}_${stage}_${SLURM_JOB_ID}.log"
exec > "$log" 2>&1
echo "$(date -u +%FT%TZ) job=$SLURM_JOB_ID variant=$variant stage=$stage gpus=$gpus"
jq -n --arg job_id "$SLURM_JOB_ID" --arg stage "$stage" --argjson gpus "$gpus" \
    '{job_id:$job_id,stage:$stage,gpus:$gpus,status:"running"}' \
    > "$campaign/controllers/${variant}_allocation.json.tmp"
mv "$campaign/controllers/${variant}_allocation.json.tmp" \
    "$campaign/controllers/${variant}_allocation.json"
"$python_bin" - "$gpus" <<'PY'
import os
import sys
import torch
expected = int(sys.argv[1])
visible = torch.cuda.device_count()
if not torch.cuda.is_available() or visible != expected:
    raise RuntimeError(f'Expected {expected} CUDA devices, found {visible}; '
                       f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}')
print('GPU preflight:', torch.__version__, torch.cuda.get_device_name(0), flush=True)
PY
case "$stage" in
    train)
        "$python_bin" - "$campaign/configs/train_$variant.yaml" "$gpus" <<'PY'
import json
import sys
from ocean_emulators.config import TrainConfig
cfg = TrainConfig.from_yaml_and_cli([sys.argv[1]])
gpus = int(sys.argv[2])
assert cfg.batch_size * gpus == 32 and cfg.gradient_accumulation_steps == 1
assert cfg.epochs == 70 and cfg.experiment.rand_seed == 15
assert cfg.preemptible and not cfg.finetune and cfg.steps == [4]
assert cfg.learning_rate == 0.0006 and cfg.scheduler.type == 'cosine'
print(json.dumps({'setup': {'gpus': gpus, 'batch_size_per_gpu': cfg.batch_size,
    'global_batch_size': 32, 'epochs': cfg.epochs, 'seed': cfg.experiment.rand_seed,
    'checkpoint': 'final epoch ordinary parameters; legacy export unchanged'}}), flush=True)
PY
        if [[ "$variant" == tendency && ! -f "$campaign/assets/tendency_weights.json" ]]; then
            "$python_bin" "$campaign/runtime/prepare_tendency_loss.py" \
                --data-root /pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/onedeg \
                --output "$campaign/assets/tendency_weights.json" --workers 8
        fi
        "$python_bin" -u -m torch.distributed.run --standalone --nnodes=1 \
            --nproc-per-node="$gpus" "$campaign/runtime/revision_loss_segment.py" \
            "$campaign/configs/train_$variant.yaml"
        if jq -e '.complete == true' "$campaign/training/$variant/segment_result.json" > /dev/null; then
            "$python_bin" "$campaign/runtime/verify_loss_checkpoint.py" \
                "$campaign/training/$variant/saved_nets/ema_ckpt.pt"
        fi
        ;;
    eval)
        "$python_bin" "$campaign/runtime/verify_loss_checkpoint.py" \
            "$campaign/training/$variant/saved_nets/ema_ckpt.pt"
        "$python_bin" -u -m ocean_emulators.eval "$campaign/configs/eval_$variant.yaml"
        ;;
    *) exit 2 ;;
esac
