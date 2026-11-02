#!/usr/bin/env bash
# Shared-interactive, two-GPU allocations renewed after complete epochs.
set -euo pipefail
campaign="${1:?campaign directory}"
python_bin="${2:?Python interpreter}"
mkdir -p "$campaign/logs" "$campaign/controllers"
exec 9>"$campaign/controllers/neptune.lock"
flock -n 9 || { echo 'A Neptune controller is already active' >&2; exit 1; }
status_path="$campaign/controllers/neptune.json"
status() {
    jq -n --arg status "$1" --arg updated "$(date -u +%FT%TZ)" --argjson pid "$$" \
        '{status:$status,updated_at_utc:$updated,controller_pid:$pid,
          qos:"shared_interactive",gpus:2,allocation_hours:4,global_batch_size:32}' \
        > "$status_path.tmp"
    mv "$status_path.tmp" "$status_path"
}
trap 'code=$?; if (( code != 0 )); then status failed; fi' EXIT
allocate() {
    salloc --account=m4874_g --qos=shared_interactive --constraint='gpu&hbm80g' \
        --nodes=1 --ntasks=1 --gpus=2 --cpus-per-task=32 --mem=110G \
        --time=04:00:00 --immediate=86400 --job-name="neptune_$1" \
        srun --kill-on-bad-exit=1 --ntasks=1 --gpus-per-task=2 --cpu-bind=cores \
        bash "$campaign/runtime/neptune_interactive_node.sh" \
        "$campaign" "$python_bin" "$1" 9>&-
}
result="$campaign/training/neptune_om4/segment_result.json"
while [[ ! -f "$result" ]] || ! jq -e '.complete == true' "$result" > /dev/null; do
    status allocating_training
    allocate train
    jq -e '.completed_epoch >= .start_epoch' "$result" > /dev/null
    cat "$result"
done
status training_complete
allocate eval
status complete
