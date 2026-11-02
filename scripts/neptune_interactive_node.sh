#!/usr/bin/env bash
set -euo pipefail
campaign="${1:?campaign directory}"
python_bin="${2:?Python interpreter}"
stage="${3:?train or eval}"
cd "$campaign"
export PYTHONPATH="$campaign/source:$campaign/deps${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=disabled MPLBACKEND=Agg NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export REVISION_SEGMENT_DEADLINE=$(($(date +%s) + 4 * 3600 - 15 * 60))
log="$campaign/logs/${stage}_${SLURM_JOB_ID}.log"
exec > "$log" 2>&1
echo "$(date -u +%FT%TZ) job=$SLURM_JOB_ID stage=$stage"
jq -n --arg job_id "$SLURM_JOB_ID" --arg stage "$stage" \
    '{job_id:$job_id,stage:$stage,gpus:2,status:"running"}' \
    > "$campaign/controllers/allocation.json.tmp"
mv "$campaign/controllers/allocation.json.tmp" "$campaign/controllers/allocation.json"
"$python_bin" - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 2
print('GPU preflight:', torch.__version__, torch.cuda.get_device_name(0), flush=True)
PY
config="$campaign/configs/neptune_train.yaml"
case "$stage" in
    train)
        if [[ ! -f "$campaign/verification/gpu_preflight.json" ]]; then
            preflight_log="$campaign/logs/preflight_${SLURM_JOB_ID}.log"
            if ! "$python_bin" -u -m torch.distributed.run --standalone --nnodes=1 \
                --nproc-per-node=2 "$campaign/runtime/neptune_preflight.py" "$config" \
                --output "$campaign/verification/gpu_preflight.json" > "$preflight_log" 2>&1; then
                cat "$preflight_log"
                if ! rg -q 'CUDA out of memory|torch.OutOfMemoryError' "$preflight_log"; then
                    echo 'Preflight failed for a reason other than CUDA OOM; stopping.' >&2
                    exit 1
                fi
                # Only the preflight configuration is changed, before training starts.
                "$python_bin" - "$config" <<'PY'
import sys
from pathlib import Path
import yaml
path = Path(sys.argv[1])
config = yaml.safe_load(path.read_text())
assert config['batch_size'] == 16 and config['gradient_accumulation_steps'] == 1
config['batch_size'] = 8
config['gradient_accumulation_steps'] = 2
path.write_text(yaml.safe_dump(config, sort_keys=False))
print('CUDA OOM fallback: per-GPU batch 8, accumulation 2, global batch 32.', flush=True)
PY
                "$python_bin" -u -m torch.distributed.run --standalone --nnodes=1 \
                    --nproc-per-node=2 "$campaign/runtime/neptune_preflight.py" "$config" \
                    --output "$campaign/verification/gpu_preflight.json" \
                    > "$campaign/logs/preflight_retry_${SLURM_JOB_ID}.log" 2>&1
            fi
            cat "$campaign/verification/gpu_preflight.json"
        fi
        "$python_bin" - "$config" <<'PY'
import sys
from ocean_emulators.config import TrainConfig
cfg = TrainConfig.from_yaml_and_cli([sys.argv[1]])
assert cfg.batch_size * cfg.gradient_accumulation_steps * 2 == 32
assert cfg.epochs == 70 and cfg.steps == [4] and cfg.experiment.rand_seed == 15
assert cfg.preemptible and not cfg.finetune
print(f'Training: batch={cfg.batch_size}, accumulation={cfg.gradient_accumulation_steps}, GPUs=2, epochs=70', flush=True)
PY
        "$python_bin" -u -m torch.distributed.run --standalone --nnodes=1 \
            --nproc-per-node=2 "$campaign/runtime/revision_loss_segment.py" "$config"
        ;;
    eval)
        "$python_bin" "$campaign/runtime/verify_loss_checkpoint.py" \
            "$campaign/training/neptune_om4/saved_nets/ema_ckpt.pt"
        "$python_bin" -u -m ocean_emulators.eval "$campaign/configs/neptune_eval.yaml"
        ;;
    *) exit 2 ;;
esac
