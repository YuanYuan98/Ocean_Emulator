#!/usr/bin/env bash
# Allocation-level orchestration only; Python/data/GPU work runs in srun steps.
set -euo pipefail
campaign="${1:?campaign}"
python_bin="${2:?python}"
stage="${3:?preflight, smoke_remaining, or train}"
variant="${4:-all}"
wall_seconds="${5:-14400}"
export FORCING_GPUS_PER_NODE="${6:-4}"
accumulation="${7:-1}"
cpus_per_task=$((16 * FORCING_GPUS_PER_NODE))
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
allocation_end="${SLURM_JOB_END_TIME:-$(($(date +%s) + wall_seconds))}"
export REVISION_SEGMENT_DEADLINE=$((allocation_end - 15 * 60))
log="$campaign/logs/${stage}_${variant}_${SLURM_JOB_ID}.log"
exec > "$log" 2>&1
echo "$(date -u +%FT%TZ) job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST stage=$stage variant=$variant"
[[ "$SLURM_JOB_NUM_NODES" == 1 ]]
allocation_file="$campaign/controllers/${variant}_allocation.json"
write_allocation() {
    jq -n --arg job_id "$SLURM_JOB_ID" --arg variant "$variant" --arg stage "$stage" \
        --arg status "$1" --arg updated "$(date -u +%FT%TZ)" \
        --argjson deadline "$REVISION_SEGMENT_DEADLINE" --argjson wall_seconds "$wall_seconds" \
        --argjson gpus "$FORCING_GPUS_PER_NODE" --argjson accumulation "$accumulation" \
        '{job_id:$job_id,variant:$variant,stage:$stage,status:$status,
          updated_at_utc:$updated,nodes:1,gpus_per_node:$gpus,gpus:$gpus,batch_per_gpu:8,
          gradient_accumulation_steps:$accumulation,effective_global_batch:32,
          deadline_unix:$deadline,requested_wall_seconds:$wall_seconds}' > "$allocation_file.tmp"
    mv "$allocation_file.tmp" "$allocation_file"
}
write_allocation running
trap 'code=$?; if (( code == 0 )); then write_allocation finished; else write_allocation failed; fi' EXIT
single=(srun --kill-on-bad-exit=1 --nodes=1 --ntasks=1 --ntasks-per-node=1
    "--gpus-per-task=$FORCING_GPUS_PER_NODE" "--cpus-per-task=$cpus_per_task" --cpu-bind=cores bash "$campaign/runtime/forcing_revision_node.sh")
distributed=(srun --kill-on-bad-exit=1 --nodes=1 --ntasks=1 --ntasks-per-node=1
    "--gpus-per-task=$FORCING_GPUS_PER_NODE" "--cpus-per-task=$cpus_per_task" --cpu-bind=cores bash "$campaign/runtime/forcing_revision_node.sh")
if [[ "$stage" == smoke_remaining ]]; then
    jq -e '.status == "passed"' "$campaign/preflight/assets_verification.json" >/dev/null
    for experiment in heat_only climatology no_forcing; do
        "${distributed[@]}" "$campaign" "$python_bin" preflight "$experiment" first
    done
elif [[ "$stage" == preflight ]]; then
    "${single[@]}" "$campaign" "$python_bin" assets
    for experiment in wind_only heat_only climatology no_forcing; do
        for phase in first resume; do
            "${distributed[@]}" "$campaign" "$python_bin" preflight "$experiment" "$phase"
        done
    done
elif [[ "$stage" == train ]]; then
    if [[ "$FORCING_GPUS_PER_NODE" == 2 ]]; then
        if ! "$python_bin" "$campaign/runtime/forcing_revision_smoke_gate.py" "$campaign" "$variant"; then
            "${distributed[@]}" "$campaign" "$python_bin" accum_smoke "$variant"
            "$python_bin" "$campaign/runtime/forcing_revision_smoke_gate.py" "$campaign" "$variant"
        fi
    fi
    "${distributed[@]}" "$campaign" "$python_bin" train "$variant"
    "${single[@]}" "$campaign" "$python_bin" verify "$variant"
else
    exit 2
fi
