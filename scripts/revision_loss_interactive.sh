#!/usr/bin/env bash
# A persistent, lightweight controller; all numerical work runs through srun.
set -euo pipefail
campaign="${1:?campaign directory}"
python_bin="${2:?Python interpreter}"
variant="${3:?loss variant}"
case "$variant" in
    adaloss|frozen_final) qos=interactive; gpus=4; cpus=64; memory=220G ;;
    tendency) qos=shared_interactive; gpus=2; cpus=32; memory=110G ;;
    *) exit 2 ;;
esac
mkdir -p "$campaign/logs" "$campaign/controllers"
exec 9>"$campaign/controllers/$variant.lock"
if ! flock -w 60 9; then
    echo "Another controller still holds the $variant lock" >&2
    exit 1
fi
status_path="$campaign/controllers/$variant.json"
write_status() {
    jq -n --arg variant "$variant" --arg qos "$qos" --arg status "$1" \
        --arg updated "$(date -u +%FT%TZ)" --argjson pid "$$" \
        --argjson gpus "$gpus" \
        '{variant:$variant,qos:$qos,status:$status,updated_at_utc:$updated,
          controller_pid:$pid,gpus:$gpus,allocation_hours:4}' > "$status_path.tmp"
    mv "$status_path.tmp" "$status_path"
}
trap 'code=$?; if (( code != 0 )); then write_status failed; fi' EXIT
result="$campaign/training/$variant/segment_result.json"
while [[ ! -f "$result" ]] || ! jq -e '.complete == true' "$result" > /dev/null; do
    write_status allocating_training
    echo "$(date -u +%FT%TZ) Requesting four-hour $qos training allocation for $variant"
    salloc --account=m4874_g --qos="$qos" --constraint='gpu&hbm80g' \
        --nodes=1 --ntasks=1 --gpus="$gpus" --cpus-per-task="$cpus" \
        --mem="$memory" --time=04:00:00 --immediate=86400 \
        --job-name="s2loss_$variant" \
        srun --kill-on-bad-exit=1 --ntasks=1 --gpus-per-task="$gpus" \
        --cpu-bind=cores \
        bash "$campaign/runtime/revision_loss_interactive_node.sh" \
        "$campaign" "$python_bin" "$variant" "$gpus" train 9>&-
    # A failed allocation or trainer exits loudly; only a successful pause continues.
    jq -e '.completed_epoch >= .start_epoch' "$result" > /dev/null
    cat "$result"
done
write_status training_complete
# Final inference uses the original final-epoch export and prescribed forcing.
salloc --account=m4874_g --qos="$qos" --constraint='gpu&hbm80g' \
    --nodes=1 --ntasks=1 --gpus="$gpus" --cpus-per-task="$cpus" \
    --mem="$memory" --time=04:00:00 --immediate=86400 \
    --job-name="s2loss_${variant}_eval" \
    srun --kill-on-bad-exit=1 --ntasks=1 --gpus-per-task="$gpus" \
    --cpu-bind=cores \
    bash "$campaign/runtime/revision_loss_interactive_node.sh" \
    "$campaign" "$python_bin" "$variant" "$gpus" eval 9>&-
write_status complete
