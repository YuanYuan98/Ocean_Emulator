#!/usr/bin/env bash
set -euo pipefail
campaign="${1:?campaign}"
python_bin="${2:?python}"
export PYTHONPATH="$campaign/source"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=disabled MPLBACKEND=Agg NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec > "$campaign/logs/shared_smoke_${SLURM_JOB_ID}.log" 2>&1
echo "$(date -u +%FT%TZ) shared accumulation smoke job=$SLURM_JOB_ID,2GPU,batch8,accum2/effectiveglobal32"
jq -e '.status == "passed"' "$campaign/preflight/assets_verification.json" >/dev/null
for variant in climatology no_forcing; do
    srun --kill-on-bad-exit=1 --nodes=1 --ntasks=1 --gpus-per-task=2 \
        --cpus-per-task=32 --cpu-bind=cores \
        "$python_bin" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
        "$campaign/runtime/forcing_revision_smoke.py" "$campaign/configs/train_$variant.yaml" \
        --campaign "$campaign"
done
