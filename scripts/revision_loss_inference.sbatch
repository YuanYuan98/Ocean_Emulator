#!/usr/bin/env bash
#SBATCH --account=m4874_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=55G
#SBATCH --time=06:00:00

set -euo pipefail
campaign="${1:?campaign directory}"
python_bin="${2:?Python interpreter}"
variant="${3:?loss variant}"
case "$variant" in adaloss|frozen_final|tendency) ;; *) exit 2 ;; esac
cd "$campaign"
export PYTHONPATH="$campaign/source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=disabled MPLBACKEND=Agg
"$python_bin" "$campaign/runtime/verify_loss_checkpoint.py" \
    "$campaign/training/$variant/saved_nets/ema_ckpt.pt"
srun --kill-on-bad-exit=1 --cpu-bind=cores --gpus-per-task=1 \
    "$python_bin" -u -m ocean_emulators.eval "$campaign/configs/eval_$variant.yaml"
