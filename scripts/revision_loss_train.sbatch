#!/usr/bin/env bash
#SBATCH --account=m4874_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
campaign="${1:?campaign directory}"
python_bin="${2:?Python interpreter}"
variant="${3:?loss variant}"
case "$variant" in adaloss|frozen_final|tendency) ;; *) exit 2 ;; esac
cd "$campaign"
export PYTHONPATH="$campaign/source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=disabled MPLBACKEND=Agg NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
# Include setup and preprocessing in the four-hour allocation budget.
export REVISION_SEGMENT_DEADLINE=$(($(date +%s) + 4 * 3600 - 15 * 60))
echo "Starting training segment: job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0}"
if [[ "$variant" == tendency && ! -f "$campaign/assets/tendency_weights.json" ]]; then
    "$python_bin" "$campaign/runtime/prepare_tendency_loss.py" \
        --data-root /pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/onedeg \
        --output "$campaign/assets/tendency_weights.json" --workers 8
fi
srun --kill-on-bad-exit=1 --ntasks=2 --ntasks-per-node=1 \
    --gpus-per-task=4 --gpu-bind=none --cpu-bind=cores \
    bash "$campaign/runtime/revision_loss_node.sh" "$campaign" "$python_bin" "$variant"
complete="$($python_bin -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["complete"]))' \
    "$campaign/training/$variant/segment_result.json")"
if [[ "$complete" == 1 ]]; then
    "$python_bin" "$campaign/runtime/verify_loss_checkpoint.py" \
        "$campaign/training/$variant/saved_nets/ema_ckpt.pt"
else
    echo "Saved complete epochs; returning job $SLURM_JOB_ID to the queue."
    scontrol requeue "$SLURM_JOB_ID"
fi
