#!/usr/bin/env bash
set -euo pipefail
campaign="${1:?campaign}"
python_bin="${2:?python}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=disabled MPLBACKEND=Agg
export PYTHONPATH="$campaign/source"
deadline=$(($(date +%s) + 13800))
pids=()
for resolution in onedeg halfdeg quarterdeg; do
    case "$resolution" in
        onedeg) memory=35G ;;
        halfdeg) memory=65G ;;
        quarterdeg) memory=110G ;;
    esac
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=1 --gpus-per-task=1 \
        --mem="$memory" --cpu-bind=cores "$python_bin" -u "$campaign/runtime/multistart_worker.py" \
        "$campaign" "$resolution" --deadline "$deadline" \
        > "$campaign/logs/${resolution}_${SLURM_JOB_ID}.log" 2>&1 &
    pids+=("$!")
done
result=0
for pid in "${pids[@]}"; do
    wait "$pid" || result=1
done
exit "$result"
