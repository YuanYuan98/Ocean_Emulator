#!/bin/bash
#SBATCH --job-name=yy_eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=eval_%j.out

NAME_long="2025-12-03-samudra_om4_onedeg"
NAME_short="Samudra_om4_onedeg"
NAME_short2="Samudra_om4_onedeg_teacher"

DATA_PATH=/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/onedeg
CKPT=/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/checkpoints/$NAME_long/ema_ckpt.pt
CONFIG_PATH=configs/samudra_om4/eval.yaml

uv run --preview-features extra-build-dependencies -m ocean_emulators.viz configs/samudra_om4/viz.yaml \
    --data_root $DATA_PATH \
    --name "${NAME_short2}_cmp_viz" \
    --runs="[{\"name\": \"$NAME_short\", \"location\": \"/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/Ocean_Emulator/.LOCAL/${NAME_long}/predictions.zarr\"},{\"name\": \"$NAME_short2\", \"location\": \"/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/Ocean_Emulator/.LOCAL/${NAME_short2}/predictions.zarr\"}]" 


NAME_long="2025-12-02-samudra_om4_v1_onedeg"
NAME_short="Samudra_om4_v1_onedeg"
NAME_short2="Samudra_om4_v1_onedeg_teacher"

DATA_PATH=/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/onedeg
CONFIG_PATH=configs/samudra_om4_v1/eval.yaml
CKPT=/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/checkpoints/$NAME_long/ema_ckpt.pt


uv run --preview-features extra-build-dependencies -m ocean_emulators.viz configs/samudra_om4_v1/viz.yaml \
    --data_root $DATA_PATH \
    --name "${NAME_short2}_cmp_viz" \
    --runs="[{\"name\": \"$NAME_short\", \"location\": \"/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/Ocean_Emulator/.LOCAL/${NAME_long}/predictions.zarr\"},{\"name\": \"$NAME_short2\", \"location\": \"/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/Ocean_Emulator/.LOCAL/${NAME_short2}/predictions.zarr\"}]" 


# sbatch -A m4874 -C gpu -q regular scripts/run_eval_onedeg.sbatch
# 2025-12-02-samudra_om4_halfdeg  2025-12-03-samudra_om4_onedeg